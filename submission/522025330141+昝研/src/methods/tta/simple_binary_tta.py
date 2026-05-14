from dataclasses import dataclass
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SimpleTTAConfig:
    batch_size: int
    lr: float
    weight_decay: float
    steps_per_batch: int
    entropy_weight: float
    pseudo_label_weight: float
    confidence_threshold: float
    update_scope: str = "head"
    max_batches: int | None = None


@dataclass(frozen=True)
class SimpleTTAResult:
    num_batches: int
    num_updates: int
    mean_loss: float
    mean_entropy_loss: float
    mean_pseudo_label_loss: float
    pseudo_label_fraction: float

    def as_dict(self, prefix: str = "tta_") -> dict[str, float | int]:
        return {
            f"{prefix}num_batches": self.num_batches,
            f"{prefix}num_updates": self.num_updates,
            f"{prefix}mean_loss": self.mean_loss,
            f"{prefix}mean_entropy_loss": self.mean_entropy_loss,
            f"{prefix}mean_pseudo_label_loss": self.mean_pseudo_label_loss,
            f"{prefix}pseudo_label_fraction": self.pseudo_label_fraction,
        }


def adapt_binary_classifier(
    model: nn.Module,
    X_target: np.ndarray | torch.Tensor,
    config: SimpleTTAConfig,
    device: torch.device,
    logger: Logger | None = None,
    log_prefix: str = "",
) -> SimpleTTAResult:
    """Adapt a binary classifier on unlabeled target features only.

    This function deliberately accepts only X_target. Target labels are not part
    of the adaptation interface, which makes accidental OOD-label leakage harder.
    """
    if config.steps_per_batch < 1:
        raise ValueError("steps_per_batch must be >= 1.")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")

    trainable_parameters = configure_trainable_parameters(model, config.update_scope)
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters selected for update_scope={config.update_scope!r}.")

    loader = make_feature_loader(
        X_target,
        batch_size=config.batch_size,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_entropy = 0.0
    total_pseudo = 0.0
    total_updates = 0
    total_pseudo_selected = 0
    total_examples_seen = 0
    num_batches = 0

    for batch_idx, (X,) in enumerate(loader, start=1):
        if config.max_batches is not None and batch_idx > config.max_batches:
            break

        X = X.to(device)
        num_batches += 1
        total_examples_seen += int(X.shape[0]) * config.steps_per_batch

        for _ in range(config.steps_per_batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss, entropy_loss, pseudo_loss, selected = binary_tta_loss(logits, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered non-finite TTA loss.")

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_entropy += float(entropy_loss.detach().cpu())
            total_pseudo += float(pseudo_loss.detach().cpu())
            total_pseudo_selected += int(selected)
            total_updates += 1

    if logger:
        logger.info(
            "%sTTA adapted batches=%d updates=%d pseudo_fraction=%.4f mean_loss=%.6f",
            log_prefix,
            num_batches,
            total_updates,
            total_pseudo_selected / max(total_examples_seen, 1),
            total_loss / max(total_updates, 1),
        )

    return SimpleTTAResult(
        num_batches=num_batches,
        num_updates=total_updates,
        mean_loss=total_loss / max(total_updates, 1),
        mean_entropy_loss=total_entropy / max(total_updates, 1),
        mean_pseudo_label_loss=total_pseudo / max(total_updates, 1),
        pseudo_label_fraction=total_pseudo_selected / max(total_examples_seen, 1),
    )


def binary_tta_loss(
    logits: torch.Tensor,
    config: SimpleTTAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    probabilities = torch.sigmoid(logits)
    entropy_loss = binary_entropy(probabilities).mean()

    confidence = torch.maximum(probabilities.detach(), 1.0 - probabilities.detach())
    selected_mask = confidence >= config.confidence_threshold
    selected_count = int(selected_mask.sum().detach().cpu())

    if selected_count > 0:
        pseudo_targets = (probabilities.detach() >= 0.5).to(dtype=logits.dtype)
        pseudo_loss = nn.functional.binary_cross_entropy_with_logits(
            logits[selected_mask],
            pseudo_targets[selected_mask],
        )
    else:
        pseudo_loss = logits.new_tensor(0.0)

    loss = (
        config.entropy_weight * entropy_loss
        + config.pseudo_label_weight * pseudo_loss
    )
    return loss, entropy_loss, pseudo_loss, selected_count


def binary_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp(min=eps, max=1.0 - eps)
    return -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities)
    )


def configure_trainable_parameters(model: nn.Module, update_scope: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    update_scope = update_scope.lower()
    if update_scope == "all":
        modules = [model]
    elif update_scope in {"head", "classifier", "last_layer"}:
        modules = [_last_linear_layer(model)]
    else:
        raise ValueError(
            "Unsupported update_scope. Expected one of: head, classifier, last_layer, all."
        )

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_feature_loader(
    X: np.ndarray | torch.Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(_as_float_tensor(X))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _last_linear_layer(model: nn.Module) -> nn.Linear:
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("Cannot select update_scope='head': model has no nn.Linear layer.")
    return linear_layers[-1]


def _as_float_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(values, dtype=torch.float32)


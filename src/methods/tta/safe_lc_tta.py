from dataclasses import dataclass
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SafeLCTTAConfig:
    batch_size: int
    lr: float
    weight_decay: float
    steps_per_batch: int
    entropy_weight: float
    prior_weight: float
    anchor_weight: float
    confidence_threshold: float
    confident_prior_weight: float
    min_confident_fraction: float
    prior_strength: float
    max_prior_shift: float
    update_scope: str = "head_bn"
    max_batches: int | None = None


@dataclass(frozen=True)
class SafeLCTTAResult:
    num_batches: int
    num_updates: int
    mean_loss: float
    mean_entropy_loss: float
    mean_prior_loss: float
    mean_anchor_loss: float
    source_positive_prior: float
    target_positive_prior: float
    effective_target_positive_prior: float
    all_prediction_prior: float
    confident_prediction_prior: float
    prior_confident_fraction: float
    prior_calibration_enabled: float
    adaptation_enabled: float
    entropy_selected_fraction: float
    logit_offset: float

    def as_dict(self, prefix: str = "tta_") -> dict[str, float | int]:
        return {
            f"{prefix}num_batches": self.num_batches,
            f"{prefix}num_updates": self.num_updates,
            f"{prefix}mean_loss": self.mean_loss,
            f"{prefix}mean_entropy_loss": self.mean_entropy_loss,
            f"{prefix}mean_prior_loss": self.mean_prior_loss,
            f"{prefix}mean_anchor_loss": self.mean_anchor_loss,
            f"{prefix}source_positive_prior": self.source_positive_prior,
            f"{prefix}target_positive_prior": self.target_positive_prior,
            f"{prefix}effective_target_positive_prior": self.effective_target_positive_prior,
            f"{prefix}all_prediction_prior": self.all_prediction_prior,
            f"{prefix}confident_prediction_prior": self.confident_prediction_prior,
            f"{prefix}prior_confident_fraction": self.prior_confident_fraction,
            f"{prefix}prior_calibration_enabled": self.prior_calibration_enabled,
            f"{prefix}adaptation_enabled": self.adaptation_enabled,
            f"{prefix}entropy_selected_fraction": self.entropy_selected_fraction,
            f"{prefix}logit_offset": self.logit_offset,
        }


class BinaryLogitOffsetWrapper(nn.Module):
    def __init__(self, model: nn.Module, logit_offset: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "logit_offset",
            torch.tensor(float(logit_offset), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + self.logit_offset.to(device=x.device, dtype=x.dtype)


@dataclass(frozen=True)
class TargetPriorEstimate:
    target_positive_prior: float
    all_prediction_prior: float
    confident_prediction_prior: float
    confident_fraction: float


def adapt_safe_lc_binary_classifier(
    model: nn.Module,
    X_target: np.ndarray | torch.Tensor,
    source_positive_prior: float,
    config: SafeLCTTAConfig,
    device: torch.device,
    logger: Logger | None = None,
    log_prefix: str = "",
) -> tuple[BinaryLogitOffsetWrapper, SafeLCTTAResult]:
    """Conservative label-calibrated TTA on unlabeled target features only.

    The target labels are intentionally not part of this interface. The only
    supervised statistic is the source positive prior, which should be computed
    from source training labels before test-time adaptation.
    """
    _validate_config(config)
    source_positive_prior = _clip_prior(source_positive_prior)

    model.to(device)
    model.eval()

    prior_estimate = estimate_target_positive_prior(
        model=model,
        X_target=X_target,
        config=config,
        device=device,
    )
    prior_calibration_enabled = (
        prior_estimate.confident_fraction >= config.min_confident_fraction
    )
    effective_target_prior = (
        prior_estimate.target_positive_prior
        if prior_calibration_enabled
        else source_positive_prior
    )
    loss_target_prior = (
        prior_estimate.target_positive_prior
        if prior_calibration_enabled
        else prior_estimate.all_prediction_prior
    )
    logit_offset = compute_logit_offset(
        source_positive_prior=source_positive_prior,
        target_positive_prior=effective_target_prior,
        prior_strength=config.prior_strength,
        max_prior_shift=config.max_prior_shift,
    )
    wrapped_model = BinaryLogitOffsetWrapper(model, logit_offset).to(device)

    trainable_parameters = configure_trainable_parameters(model, config.update_scope)
    anchor_state = _clone_trainable_state(model)

    loader = make_feature_loader(X_target, batch_size=config.batch_size)
    adaptation_enabled = bool(trainable_parameters) and (
        prior_estimate.confident_fraction >= config.min_confident_fraction
    )
    optimizer = None
    if adaptation_enabled:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    target_prior_tensor = torch.tensor(
        loss_target_prior,
        dtype=torch.float32,
        device=device,
    )
    total_loss = 0.0
    total_entropy = 0.0
    total_prior = 0.0
    total_anchor = 0.0
    total_updates = 0
    total_entropy_selected = 0
    total_examples_seen = 0
    num_batches = 0

    if optimizer is not None:
        for batch_idx, (X,) in enumerate(loader, start=1):
            if config.max_batches is not None and batch_idx > config.max_batches:
                break

            X = X.to(device)
            num_batches += 1
            total_examples_seen += int(X.shape[0]) * config.steps_per_batch

            for _ in range(config.steps_per_batch):
                optimizer.zero_grad(set_to_none=True)
                logits = wrapped_model(X)
                loss, entropy_loss, prior_loss, anchor_loss, selected = safe_lc_loss(
                    logits=logits,
                    model=model,
                    anchor_state=anchor_state,
                    target_positive_prior=target_prior_tensor,
                    config=config,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Encountered non-finite SafeLC-TTA loss.")

                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_entropy += float(entropy_loss.detach().cpu())
                total_prior += float(prior_loss.detach().cpu())
                total_anchor += float(anchor_loss.detach().cpu())
                total_entropy_selected += int(selected)
                total_updates += 1
    else:
        num_batches = _count_batches(loader, config.max_batches)

    if logger:
        logger.info(
            "%sSafeLC-TTA target_prior=%.4f source_prior=%.4f "
            "effective_prior=%.4f offset=%.4f confident_fraction=%.4f "
            "prior_enabled=%s adaptation_enabled=%s batches=%d updates=%d "
            "mean_loss=%.6f",
            log_prefix,
            prior_estimate.target_positive_prior,
            source_positive_prior,
            effective_target_prior,
            logit_offset,
            prior_estimate.confident_fraction,
            prior_calibration_enabled,
            adaptation_enabled,
            num_batches,
            total_updates,
            total_loss / max(total_updates, 1),
        )

    result = SafeLCTTAResult(
        num_batches=num_batches,
        num_updates=total_updates,
        mean_loss=total_loss / max(total_updates, 1),
        mean_entropy_loss=total_entropy / max(total_updates, 1),
        mean_prior_loss=total_prior / max(total_updates, 1),
        mean_anchor_loss=total_anchor / max(total_updates, 1),
        source_positive_prior=source_positive_prior,
        target_positive_prior=prior_estimate.target_positive_prior,
        effective_target_positive_prior=effective_target_prior,
        all_prediction_prior=prior_estimate.all_prediction_prior,
        confident_prediction_prior=prior_estimate.confident_prediction_prior,
        prior_confident_fraction=prior_estimate.confident_fraction,
        prior_calibration_enabled=float(prior_calibration_enabled),
        adaptation_enabled=float(adaptation_enabled),
        entropy_selected_fraction=total_entropy_selected / max(total_examples_seen, 1),
        logit_offset=logit_offset,
    )
    return wrapped_model, result


@torch.no_grad()
def estimate_target_positive_prior(
    model: nn.Module,
    X_target: np.ndarray | torch.Tensor,
    config: SafeLCTTAConfig,
    device: torch.device,
) -> TargetPriorEstimate:
    loader = make_feature_loader(X_target, batch_size=config.batch_size)
    model.eval()
    probability_chunks: list[torch.Tensor] = []
    confident_chunks: list[torch.Tensor] = []

    for X, in loader:
        X = X.to(device)
        probabilities = torch.sigmoid(model(X)).detach().cpu()
        confidence = torch.maximum(probabilities, 1.0 - probabilities)
        probability_chunks.append(probabilities)
        confident_chunks.append(confidence >= config.confidence_threshold)

    probabilities = torch.cat(probability_chunks).reshape(-1)
    confident_mask = torch.cat(confident_chunks).reshape(-1)
    all_prior = float(probabilities.mean().item())
    confident_fraction = float(confident_mask.float().mean().item())

    if bool(confident_mask.any()):
        confident_prior = float(probabilities[confident_mask].mean().item())
    else:
        confident_prior = all_prior

    target_prior = (
        (1.0 - config.confident_prior_weight) * all_prior
        + config.confident_prior_weight * confident_prior
    )
    return TargetPriorEstimate(
        target_positive_prior=_clip_prior(target_prior),
        all_prediction_prior=_clip_prior(all_prior),
        confident_prediction_prior=_clip_prior(confident_prior),
        confident_fraction=confident_fraction,
    )


def compute_logit_offset(
    *,
    source_positive_prior: float,
    target_positive_prior: float,
    prior_strength: float,
    max_prior_shift: float,
) -> float:
    source_logit = _logit(_clip_prior(source_positive_prior))
    target_logit = _logit(_clip_prior(target_positive_prior))
    offset = prior_strength * (target_logit - source_logit)
    return float(np.clip(offset, -max_prior_shift, max_prior_shift))


def safe_lc_loss(
    *,
    logits: torch.Tensor,
    model: nn.Module,
    anchor_state: dict[str, torch.Tensor],
    target_positive_prior: torch.Tensor,
    config: SafeLCTTAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    probabilities = torch.sigmoid(logits)
    confidence = torch.maximum(probabilities.detach(), 1.0 - probabilities.detach())
    selected_mask = confidence >= config.confidence_threshold
    selected_count = int(selected_mask.sum().detach().cpu())

    if selected_count > 0:
        entropy_loss = binary_entropy(probabilities[selected_mask]).mean()
    else:
        entropy_loss = logits.new_tensor(0.0)

    prior_loss = (probabilities.mean() - target_positive_prior.to(probabilities.dtype)).pow(2)
    anchor_loss = parameter_anchor_loss(model, anchor_state)
    loss = (
        config.entropy_weight * entropy_loss
        + config.prior_weight * prior_loss
        + config.anchor_weight * anchor_loss
    )
    return loss, entropy_loss, prior_loss, anchor_loss, selected_count


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
    if update_scope == "none":
        return []
    if update_scope == "all":
        modules = [model]
    elif update_scope in {"head", "classifier", "last_layer"}:
        modules = [_last_linear_layer(model)]
    elif update_scope in {"bn", "bn_affine", "batchnorm"}:
        modules = _batchnorm_layers(model)
    elif update_scope in {"head_bn", "head_bn_affine", "classifier_bn"}:
        modules = [_last_linear_layer(model), *_batchnorm_layers(model)]
    else:
        raise ValueError(
            "Unsupported update_scope. Expected one of: none, head, bn_affine, "
            "head_bn, all."
        )

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def parameter_anchor_loss(
    model: nn.Module,
    anchor_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            losses.append((parameter - anchor_state[name].to(parameter.device)).pow(2).mean())
    if not losses:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(losses).mean()


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


def _validate_config(config: SafeLCTTAConfig) -> None:
    if config.steps_per_batch < 1:
        raise ValueError("steps_per_batch must be >= 1.")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")
    if not 0.0 <= config.confident_prior_weight <= 1.0:
        raise ValueError("confident_prior_weight must be in [0, 1].")
    if not 0.0 <= config.min_confident_fraction <= 1.0:
        raise ValueError("min_confident_fraction must be in [0, 1].")
    if config.prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative.")
    if config.max_prior_shift < 0.0:
        raise ValueError("max_prior_shift must be non-negative.")


def _clone_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _last_linear_layer(model: nn.Module) -> nn.Linear:
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("Cannot select update_scope='head': model has no nn.Linear layer.")
    return linear_layers[-1]


def _batchnorm_layers(model: nn.Module) -> list[nn.BatchNorm1d]:
    return [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]


def _as_float_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(values, dtype=torch.float32)


def _count_batches(loader: DataLoader, max_batches: int | None) -> int:
    count = len(loader)
    if max_batches is not None:
        return min(count, max_batches)
    return count


def _clip_prior(value: float, eps: float = 1e-4) -> float:
    return float(np.clip(float(value), eps, 1.0 - eps))


def _logit(value: float) -> float:
    value = _clip_prior(value)
    return float(np.log(value / (1.0 - value)))

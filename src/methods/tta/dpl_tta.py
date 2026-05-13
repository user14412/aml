from dataclasses import dataclass
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DPLTTAConfig:
    batch_size: int
    lr: float
    weight_decay: float
    adapt_steps: int
    confidence_threshold: float
    beta: float
    prior_shift_threshold: float
    anchor_weight: float
    update_scope: str = "head_bn_affine"
    max_pseudo_samples: int | None = None


@dataclass(frozen=True)
class DPLTTAResult:
    num_updates: int
    mean_loss: float
    mean_pseudo_loss: float
    mean_anchor_loss: float
    source_prior: float
    target_prior_raw: float
    target_prior: float
    prior_shift: float
    confident_fraction: float
    num_pseudo_pos: int
    num_pseudo_neg: int
    adapted_or_skipped: float

    def as_dict(self, prefix: str = "tta_") -> dict[str, float]:
        return {
            f"{prefix}num_updates": float(self.num_updates),
            f"{prefix}mean_loss": self.mean_loss,
            f"{prefix}mean_pseudo_loss": self.mean_pseudo_loss,
            f"{prefix}mean_anchor_loss": self.mean_anchor_loss,
            f"{prefix}source_prior": self.source_prior,
            f"{prefix}target_prior_raw": self.target_prior_raw,
            f"{prefix}target_prior": self.target_prior,
            f"{prefix}prior_shift": self.prior_shift,
            f"{prefix}confident_fraction": self.confident_fraction,
            f"{prefix}num_pseudo_pos": float(self.num_pseudo_pos),
            f"{prefix}num_pseudo_neg": float(self.num_pseudo_neg),
            f"{prefix}adapted_or_skipped": self.adapted_or_skipped,
        }


@dataclass(frozen=True)
class DPLPseudoSet:
    X: torch.Tensor
    y: torch.Tensor
    target_prior_raw: float
    target_prior: float
    prior_shift: float
    confident_fraction: float
    num_pseudo_pos: int
    num_pseudo_neg: int


def adapt_dpl_binary_classifier(
    model: nn.Module,
    X_target: np.ndarray | torch.Tensor,
    source_prior: float,
    config: DPLTTAConfig,
    device: torch.device,
    logger: Logger | None = None,
    log_prefix: str = "",
) -> tuple[nn.Module, DPLTTAResult]:
    """Distribution-guided pseudo-label TTA using unlabeled target features only."""
    _validate_config(config)
    source_prior = _clip_prior(source_prior)

    model.to(device)
    model.eval()
    pseudo_set = build_distribution_guided_pseudo_set(
        model=model,
        X_target=X_target,
        source_prior=source_prior,
        config=config,
        device=device,
    )

    should_skip = (
        pseudo_set.prior_shift < config.prior_shift_threshold
        or pseudo_set.num_pseudo_pos + pseudo_set.num_pseudo_neg == 0
    )
    if should_skip:
        result = DPLTTAResult(
            num_updates=0,
            mean_loss=0.0,
            mean_pseudo_loss=0.0,
            mean_anchor_loss=0.0,
            source_prior=source_prior,
            target_prior_raw=pseudo_set.target_prior_raw,
            target_prior=pseudo_set.target_prior,
            prior_shift=pseudo_set.prior_shift,
            confident_fraction=pseudo_set.confident_fraction,
            num_pseudo_pos=pseudo_set.num_pseudo_pos,
            num_pseudo_neg=pseudo_set.num_pseudo_neg,
            adapted_or_skipped=0.0,
        )
        if logger:
            logger.info(
                "%sDPL-TTA skipped source_prior=%.4f target_prior_raw=%.4f "
                "target_prior=%.4f prior_shift=%.4f confident_fraction=%.4f "
                "pseudo_pos=%d pseudo_neg=%d",
                log_prefix,
                result.source_prior,
                result.target_prior_raw,
                result.target_prior,
                result.prior_shift,
                result.confident_fraction,
                result.num_pseudo_pos,
                result.num_pseudo_neg,
            )
        return model, result

    trainable_parameters = configure_trainable_parameters(model, config.update_scope)
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters selected for update_scope={config.update_scope!r}.")

    anchor_state = _clone_trainable_state(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loader = make_pseudo_loader(
        pseudo_set.X,
        pseudo_set.y,
        batch_size=config.batch_size,
        shuffle=True,
    )

    total_loss = 0.0
    total_pseudo = 0.0
    total_anchor = 0.0
    total_updates = 0
    model.eval()

    for _ in range(config.adapt_steps):
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            pseudo_loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            anchor_loss = parameter_anchor_loss(model, anchor_state)
            loss = pseudo_loss + config.anchor_weight * anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered non-finite DPL-TTA loss.")

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_pseudo += float(pseudo_loss.detach().cpu())
            total_anchor += float(anchor_loss.detach().cpu())
            total_updates += 1

    result = DPLTTAResult(
        num_updates=total_updates,
        mean_loss=total_loss / max(total_updates, 1),
        mean_pseudo_loss=total_pseudo / max(total_updates, 1),
        mean_anchor_loss=total_anchor / max(total_updates, 1),
        source_prior=source_prior,
        target_prior_raw=pseudo_set.target_prior_raw,
        target_prior=pseudo_set.target_prior,
        prior_shift=pseudo_set.prior_shift,
        confident_fraction=pseudo_set.confident_fraction,
        num_pseudo_pos=pseudo_set.num_pseudo_pos,
        num_pseudo_neg=pseudo_set.num_pseudo_neg,
        adapted_or_skipped=1.0,
    )
    if logger:
        logger.info(
            "%sDPL-TTA adapted source_prior=%.4f target_prior_raw=%.4f "
            "target_prior=%.4f prior_shift=%.4f confident_fraction=%.4f "
            "pseudo_pos=%d pseudo_neg=%d updates=%d mean_loss=%.6f",
            log_prefix,
            result.source_prior,
            result.target_prior_raw,
            result.target_prior,
            result.prior_shift,
            result.confident_fraction,
            result.num_pseudo_pos,
            result.num_pseudo_neg,
            result.num_updates,
            result.mean_loss,
        )
    return model, result


@torch.no_grad()
def build_distribution_guided_pseudo_set(
    *,
    model: nn.Module,
    X_target: np.ndarray | torch.Tensor,
    source_prior: float,
    config: DPLTTAConfig,
    device: torch.device,
) -> DPLPseudoSet:
    X_tensor = _as_float_tensor(X_target)
    probabilities = predict_probabilities(model, X_tensor, config.batch_size, device)
    confidence = torch.maximum(probabilities, 1.0 - probabilities)
    confident_mask = confidence >= config.confidence_threshold
    confident_fraction = float(confident_mask.float().mean().item())

    if bool(confident_mask.any()):
        target_prior_raw = float(probabilities[confident_mask].mean().item())
    else:
        target_prior_raw = source_prior

    target_prior = _clip_prior(
        config.beta * target_prior_raw + (1.0 - config.beta) * source_prior
    )
    prior_shift = abs(target_prior - source_prior)
    num_confident = int(confident_mask.sum().item())
    if num_confident == 0:
        return DPLPseudoSet(
            X=torch.empty((0, X_tensor.shape[1]), dtype=torch.float32),
            y=torch.empty((0,), dtype=torch.float32),
            target_prior_raw=_clip_prior(target_prior_raw),
            target_prior=target_prior,
            prior_shift=prior_shift,
            confident_fraction=confident_fraction,
            num_pseudo_pos=0,
            num_pseudo_neg=0,
        )

    desired_total = num_confident
    if config.max_pseudo_samples is not None:
        desired_total = min(desired_total, config.max_pseudo_samples)
    desired_pos = int(round(target_prior * desired_total))
    desired_pos = min(max(desired_pos, 0), desired_total)
    desired_neg = desired_total - desired_pos

    pos_candidates = torch.nonzero(probabilities >= config.confidence_threshold, as_tuple=False).reshape(-1)
    neg_candidates = torch.nonzero(probabilities <= 1.0 - config.confidence_threshold, as_tuple=False).reshape(-1)

    pos_indices = _topk_indices(pos_candidates, probabilities, desired_pos, largest=True)
    neg_indices = _topk_indices(neg_candidates, probabilities, desired_neg, largest=False)

    selected_indices = []
    selected_labels = []
    if pos_indices.numel() > 0:
        selected_indices.append(pos_indices)
        selected_labels.append(torch.ones(pos_indices.numel(), dtype=torch.float32))
    if neg_indices.numel() > 0:
        selected_indices.append(neg_indices)
        selected_labels.append(torch.zeros(neg_indices.numel(), dtype=torch.float32))

    if not selected_indices:
        X_pseudo = torch.empty((0, X_tensor.shape[1]), dtype=torch.float32)
        y_pseudo = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.cat(selected_indices)
        labels = torch.cat(selected_labels)
        X_pseudo = X_tensor[indices].contiguous()
        y_pseudo = labels.contiguous()

    return DPLPseudoSet(
        X=X_pseudo,
        y=y_pseudo,
        target_prior_raw=_clip_prior(target_prior_raw),
        target_prior=target_prior,
        prior_shift=prior_shift,
        confident_fraction=confident_fraction,
        num_pseudo_pos=int(pos_indices.numel()),
        num_pseudo_neg=int(neg_indices.numel()),
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = make_feature_loader(X, batch_size=batch_size)
    chunks = []
    model.eval()
    for (batch_X,) in loader:
        logits = model(batch_X.to(device))
        chunks.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(chunks).reshape(-1)


def configure_trainable_parameters(model: nn.Module, update_scope: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    update_scope = update_scope.lower()
    if update_scope in {"head", "classifier", "last_layer"}:
        modules = [_last_linear_layer(model)]
    elif update_scope in {"bn", "bn_affine", "batchnorm"}:
        modules = _batchnorm_layers(model)
    elif update_scope in {"head_bn", "head_bn_affine", "classifier_bn"}:
        modules = [_last_linear_layer(model), *_batchnorm_layers(model)]
    elif update_scope == "all":
        modules = [model]
    else:
        raise ValueError(
            "Unsupported update_scope. Expected one of: head, bn_affine, "
            "head_bn_affine, all."
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
    X: torch.Tensor,
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


def make_pseudo_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(_as_float_tensor(X), _as_float_tensor(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _validate_config(config: DPLTTAConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if config.adapt_steps < 1:
        raise ValueError("adapt_steps must be >= 1.")
    if not 0.5 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1].")
    if not 0.0 <= config.beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if config.prior_shift_threshold < 0.0:
        raise ValueError("prior_shift_threshold must be non-negative.")
    if config.anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative.")


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


def _topk_indices(
    candidates: torch.Tensor,
    probabilities: torch.Tensor,
    count: int,
    largest: bool,
) -> torch.Tensor:
    if count <= 0 or candidates.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    count = min(count, int(candidates.numel()))
    candidate_scores = probabilities[candidates]
    order = torch.argsort(candidate_scores, descending=largest)
    return candidates[order[:count]].to(dtype=torch.long)


def _clip_prior(value: float, eps: float = 1e-4) -> float:
    return float(np.clip(float(value), eps, 1.0 - eps))

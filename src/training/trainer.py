from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluator import evaluate_binary_classifier


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    use_pos_weight: bool = True
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_val_balanced_accuracy: float
    best_val_loss: float


def make_loader(
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """创建 DataLoader：根据 array、超参数 创建"""
    dataset = TensorDataset(
        _as_float_tensor(X),
        _as_float_tensor(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_erm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    threshold: float = 0.5,
) -> TrainingResult:
    """一个经典的训练循环。加pos_weight解决类别不平衡；加validate早停防止过拟合；没什么神奇的"""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(train_loader, device, config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_balanced_accuracy = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_loss = _compute_loss(model, val_loader, criterion, device)
        val_metrics = evaluate_binary_classifier(model, val_loader, device, threshold=threshold)
        improved = val_metrics.balanced_accuracy > best_val_balanced_accuracy
        tied_with_lower_loss = (
            val_metrics.balanced_accuracy == best_val_balanced_accuracy
            and val_loss < best_val_loss
        )

        if improved or tied_with_lower_loss:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_balanced_accuracy = val_metrics.balanced_accuracy
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        best_val_balanced_accuracy=best_val_balanced_accuracy,
        best_val_loss=best_val_loss,
    )


def _as_float_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(values, dtype=torch.float32)


def _compute_pos_weight(
    train_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> torch.Tensor | None:
    if not config.use_pos_weight:
        return None

    labels = train_loader.dataset.tensors[1]
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0:
        return None

    return torch.tensor([float(negatives / positives)], dtype=torch.float32, device=device)


@torch.no_grad()
def _compute_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        batch_size = X.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)

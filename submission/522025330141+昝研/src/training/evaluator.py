import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.metrics import BinaryMetrics, compute_binary_metrics


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """简单推理，返回概率和目标标签"""
    model.eval()
    probabilities = []
    targets = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(probabilities), np.concatenate(targets)


def evaluate_binary_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """胶水函数：返回老师要求的 3 个评价指标"""
    probabilities, targets = predict_probabilities(model, loader, device)
    return compute_binary_metrics(targets, probabilities, threshold=threshold)

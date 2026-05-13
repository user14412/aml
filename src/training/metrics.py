from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    balanced_accuracy: float
    f1: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}balanced_accuracy": self.balanced_accuracy,
            f"{prefix}f1": self.f1,
        }


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = (np.asarray(probabilities).reshape(-1) >= threshold).astype(int)

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

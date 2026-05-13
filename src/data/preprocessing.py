from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.tableshift_loader import TableShiftDataset


@dataclass
class PreparedArrays:
    X_train: np.ndarray | torch.Tensor
    y_train: np.ndarray | torch.Tensor
    X_val: np.ndarray | torch.Tensor
    y_val: np.ndarray | torch.Tensor
    X_id_test: np.ndarray | torch.Tensor
    y_id_test: np.ndarray | torch.Tensor
    X_ood_test: np.ndarray | torch.Tensor
    y_ood_test: np.ndarray | torch.Tensor
    feature_names: list[str]


class TabularPreprocessor:
    """表格预处理类：封装了对表格数据进行预处理的方法。包括数值化、标准化等操作。"""
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        self.feature_names = list(X.columns)
        X_array = self._to_numeric_array(X)
        mean = np.nanmean(X_array, axis=0).astype(np.float32)
        std = np.nanstd(X_array, axis=0).astype(np.float32)

        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < 1e-6] = 1.0

        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform.")

        X_array = self._to_numeric_array(X)
        X_array -= self.mean_
        X_array /= self.std_
        return np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _to_numeric_array(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        return X_numeric.to_numpy(dtype=np.float32, copy=True)


def prepare_tableshift_arrays(dataset: TableShiftDataset) -> PreparedArrays:
    """数据预处理：将 TableShiftDataset 对象转换为预处理后的np数组"""
    preprocessor = TabularPreprocessor().fit(dataset.train.X)

    return PreparedArrays(
        X_train=preprocessor.transform(dataset.train.X),
        y_train=_prepare_binary_labels(dataset.train.y),
        X_val=preprocessor.transform(dataset.val.X),
        y_val=_prepare_binary_labels(dataset.val.y),
        X_id_test=preprocessor.transform(dataset.id_test.X),
        y_id_test=_prepare_binary_labels(dataset.id_test.y),
        X_ood_test=preprocessor.transform(dataset.ood_test.X),
        y_ood_test=_prepare_binary_labels(dataset.ood_test.y),
        feature_names=preprocessor.feature_names,
    )


def _prepare_binary_labels(y: pd.Series) -> np.ndarray:
    values = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(values).any():
        raise ValueError("Target labels contain NaN after numeric conversion.")

    unique_values = np.unique(values)
    if not set(unique_values.tolist()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Expected binary labels encoded as 0/1, got {unique_values[:10]}")

    return values.astype(np.float32)

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SPLIT_FILES = {
    "train": ("Xtrain", "ytrain"),
    "val": ("Xval", "yval"),
    "id_test": ("Xidtest", "yidtest"),
    "ood_test": ("Xood", "yood"),
}


@dataclass(frozen=True)
class TableShiftSplit:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class TableShiftDataset:
    name: str
    train: TableShiftSplit
    val: TableShiftSplit
    id_test: TableShiftSplit
    ood_test: TableShiftSplit

    @property
    def input_dim(self) -> int:
        return self.train.X.shape[1]


def _read_split(dataset_dir: Path, dataset_name: str, split: str) -> TableShiftSplit:
    x_suffix, y_suffix = SPLIT_FILES[split]
    x_path = dataset_dir / f"{dataset_name}_{x_suffix}.csv"
    y_path = dataset_dir / f"{dataset_name}_{y_suffix}.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing files for {dataset_name}/{split}: {x_path}, {y_path}")

    X = pd.read_csv(x_path)
    y_frame = pd.read_csv(y_path)
    if y_frame.shape[1] != 1:
        raise ValueError(f"Expected one target column in {y_path}, got {y_frame.shape[1]}")

    return TableShiftSplit(X=X, y=y_frame.iloc[:, 0])


def load_tableshift_dataset(data_root: str | Path, dataset_name: str) -> TableShiftDataset:
    """加载指定的数据集。返回一个 TableShiftDataset 对象，包含训练集、验证集、ID 测试集和 OOD 测试集的特征和标签。"""
    dataset_dir = Path(data_root) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    splits = {
        split_name: _read_split(dataset_dir, dataset_name, split_name)
        for split_name in SPLIT_FILES
    }
    return TableShiftDataset(name=dataset_name, **splits)

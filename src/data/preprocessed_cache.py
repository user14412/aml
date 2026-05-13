from pathlib import Path
from typing import Any

import torch

from src.data.preprocessing import PreparedArrays, prepare_tableshift_arrays
from src.data.tableshift_loader import load_tableshift_dataset
from src.utils.io import ensure_dir


CACHE_VERSION = 1
PREPROCESSING_ID = "train_mean_std_nan_to_zero_float32_v1"


def load_or_prepare_tableshift_arrays(
    data_root: str | Path,
    dataset_name: str,
    cache_dir: str | Path,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[PreparedArrays, str]:
    path = cache_path(cache_dir, dataset_name)

    if use_cache and path.exists() and not refresh_cache:
        return load_prepared_arrays(path, dataset_name), "cache_hit"

    dataset = load_tableshift_dataset(data_root, dataset_name)
    arrays = prepare_tableshift_arrays(dataset)

    if use_cache:
        save_prepared_arrays(arrays, path, dataset_name)
        return arrays, "cache_created"

    return arrays, "cache_disabled"


def cache_path(cache_dir: str | Path, dataset_name: str) -> Path:
    return Path(cache_dir) / f"{dataset_name}.pt"


def save_prepared_arrays(arrays: PreparedArrays, path: str | Path, dataset_name: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "cache_version": CACHE_VERSION,
        "preprocessing_id": PREPROCESSING_ID,
        "dataset_name": dataset_name,
        "feature_names": arrays.feature_names,
        "splits": {
            "X_train": _to_cpu_float_tensor(arrays.X_train),
            "y_train": _to_cpu_float_tensor(arrays.y_train),
            "X_val": _to_cpu_float_tensor(arrays.X_val),
            "y_val": _to_cpu_float_tensor(arrays.y_val),
            "X_id_test": _to_cpu_float_tensor(arrays.X_id_test),
            "y_id_test": _to_cpu_float_tensor(arrays.y_id_test),
            "X_ood_test": _to_cpu_float_tensor(arrays.X_ood_test),
            "y_ood_test": _to_cpu_float_tensor(arrays.y_ood_test),
        },
    }
    torch.save(payload, path)


def load_prepared_arrays(path: str | Path, dataset_name: str) -> PreparedArrays:
    payload = _torch_load_cpu(Path(path))
    _validate_payload(payload, dataset_name, Path(path))
    splits = payload["splits"]

    return PreparedArrays(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_id_test=splits["X_id_test"],
        y_id_test=splits["y_id_test"],
        X_ood_test=splits["X_ood_test"],
        y_ood_test=splits["y_ood_test"],
        feature_names=list(payload["feature_names"]),
    )


def _to_cpu_float_tensor(values: Any) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return torch.as_tensor(values, dtype=torch.float32).contiguous()


def _torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_payload(payload: dict[str, Any], dataset_name: str, path: Path) -> None:
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"Cache version mismatch in {path}. Re-run with --refresh-cache.")
    if payload.get("preprocessing_id") != PREPROCESSING_ID:
        raise ValueError(f"Preprocessing mismatch in {path}. Re-run with --refresh-cache.")
    if payload.get("dataset_name") != dataset_name:
        raise ValueError(f"Cache dataset mismatch in {path}. Re-run with --refresh-cache.")

    required_splits = {
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_id_test",
        "y_id_test",
        "X_ood_test",
        "y_ood_test",
    }
    missing = required_splits - set(payload.get("splits", {}))
    if missing:
        raise ValueError(f"Cache file {path} is missing split tensors: {sorted(missing)}")

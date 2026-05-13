import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

from src.data.preprocessing import PreparedArrays
from src.training.metrics import BinaryMetrics
from src.utils.io import dump_json, ensure_dir
from src.utils.logger import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BaselineRunContext:
    run_name: str
    run_dir: Path
    results_dir: Path
    checkpoint_dir: Path
    cache_dir: Path
    cache_enabled: bool
    logger: Any


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=str(SRC_ROOT / "configs" / "default.json"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")


def add_torch_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=None, help="Log training progress every N epochs.")


def resolve_datasets(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.all_datasets:
        return list(config["datasets"])
    if args.dataset:
        return [args.dataset]
    return [config["datasets"][0]]


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_seeds(args: argparse.Namespace, config: dict[str, Any]) -> list[int]:
    return args.seeds if args.seeds is not None else list(config["seeds"])


def get_baseline_config(config: dict[str, Any], baseline: str) -> dict[str, Any]:
    if "baselines" in config:
        return dict(config["baselines"][baseline])

    if baseline == "mlp":
        return {
            "method_name": "ERM-MLP",
            "model": dict(config["model"]),
            "training": dict(config["training"]),
        }

    raise KeyError(f"Missing config['baselines']['{baseline}'].")


def override_training_config(training_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training_config = dict(training_config)
    if getattr(args, "epochs", None) is not None:
        training_config["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        training_config["batch_size"] = args.batch_size
    if getattr(args, "log_every", None) is not None:
        training_config["log_every"] = args.log_every
    return training_config


def make_run_context(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    run_name: str,
    method_name: str,
    datasets: list[str],
    seeds: list[int],
    effective_baseline_config: dict[str, Any],
    extra_config: dict[str, Any] | None = None,
) -> BaselineRunContext:
    cache_config = config.get("cache", {})
    cache_enabled = bool(cache_config.get("enabled", True)) and not args.no_cache
    cache_dir = PROJECT_ROOT / cache_config.get("dir", "outputs/cache/preprocessed")

    output_root = PROJECT_ROOT / config["output_root"]
    checkpoint_root = PROJECT_ROOT / config.get("checkpoint_root", "outputs/checkpoints")
    run_dir = ensure_dir(output_root / "runs" / run_name)
    results_dir = ensure_dir(output_root / "results")
    checkpoint_dir = ensure_dir(checkpoint_root / run_name)
    logger = setup_logger(run_dir / "train.log")

    effective_config = {
        **config,
        "baseline_config": effective_baseline_config,
        "datasets": datasets,
        "seeds": seeds,
        "method": method_name,
        "cache": {
            "enabled": cache_enabled,
            "dir": str(cache_dir),
            "refresh": args.refresh_cache,
        },
    }
    if extra_config:
        effective_config.update(extra_config)
    dump_json(effective_config, run_dir / "config.json")

    logger.info("Log file: %s", run_dir / "train.log")
    logger.info("Running %s on %d dataset(s), seeds=%s", method_name, len(datasets), seeds)

    return BaselineRunContext(
        run_name=run_name,
        run_dir=run_dir,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
        logger=logger,
    )


def as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def split_sizes(arrays: PreparedArrays) -> dict[str, int]:
    return {
        "train_size": len(arrays.y_train),
        "val_size": len(arrays.y_val),
        "id_test_size": len(arrays.y_id_test),
        "ood_test_size": len(arrays.y_ood_test),
    }


def make_result_row(
    *,
    method: str,
    baseline: str,
    dataset_name: str,
    seed: int,
    input_dim: int,
    arrays: PreparedArrays,
    checkpoint_path: Path,
    val_metrics: BinaryMetrics,
    id_metrics: BinaryMetrics,
    ood_metrics: BinaryMetrics,
    best_epoch: int | str = "",
    best_iteration: int | str = "",
    best_val_loss: float | str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "baseline": baseline,
        "dataset": dataset_name,
        "seed": seed,
        "input_dim": input_dim,
        **split_sizes(arrays),
        "best_epoch": best_epoch,
        "best_iteration": best_iteration,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
        "val_accuracy": val_metrics.accuracy,
        "val_balanced_accuracy": val_metrics.balanced_accuracy,
        "val_f1": val_metrics.f1,
        "id_accuracy": id_metrics.accuracy,
        "id_balanced_accuracy": id_metrics.balanced_accuracy,
        "id_f1": id_metrics.f1,
        "ood_accuracy": ood_metrics.accuracy,
        "ood_balanced_accuracy": ood_metrics.balanced_accuracy,
        "ood_f1": ood_metrics.f1,
    }
    row["generalization_gap_accuracy"] = row["id_accuracy"] - row["ood_accuracy"]
    row["generalization_gap_balanced_accuracy"] = (
        row["id_balanced_accuracy"] - row["ood_balanced_accuracy"]
    )
    row["generalization_gap_f1"] = row["id_f1"] - row["ood_f1"]
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, float) and key not in {"seed"}
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["baseline"], row["dataset"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (method, baseline, dataset), group in grouped.items():
        summary: dict[str, Any] = {
            "method": method,
            "baseline": baseline,
            "dataset": dataset,
            "num_runs": len(group),
        }
        for metric in metric_keys:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def format_row(row: dict[str, Any]) -> str:
    suffix = ""
    if row.get("best_epoch") != "":
        suffix = f" best_epoch={row['best_epoch']}"
    elif row.get("best_iteration") != "":
        suffix = f" best_iteration={row['best_iteration']}"
    return (
        f"{row['dataset']} seed={row['seed']} "
        f"val_bacc={row['val_balanced_accuracy']:.4f} "
        f"id_bacc={row['id_balanced_accuracy']:.4f} "
        f"ood_bacc={row['ood_balanced_accuracy']:.4f} "
        f"gap={row['generalization_gap_balanced_accuracy']:.4f}"
        f"{suffix}"
    )


def save_result_tables(
    rows: list[dict[str, Any]],
    *,
    context: BaselineRunContext,
) -> None:
    raw_path = context.results_dir / f"{context.run_name}_raw.csv"
    summary_path = context.results_dir / f"{context.run_name}_summary.csv"
    write_csv(rows, raw_path)
    write_csv(summarize_rows(rows), summary_path)
    context.logger.info("Saved raw results: %s", raw_path)
    context.logger.info("Saved summary: %s", summary_path)

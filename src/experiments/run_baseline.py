import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from logging import Logger
from statistics import mean, stdev

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessed_cache import load_or_prepare_tableshift_arrays
from src.data.preprocessing import PreparedArrays
from src.models.mlp import MLP
from src.training.evaluator import evaluate_binary_classifier
from src.training.trainer import TrainingConfig, make_loader, train_erm
from src.utils.io import dump_json, ensure_dir, load_json
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERM-MLP baseline on TableShift CSV splits.")
    parser.add_argument("--config", type=str, default=str(SRC_ROOT / "configs" / "default.json"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=None, help="Log training progress every N epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    training_config = dict(config["training"])

    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.log_every is not None:
        training_config["log_every"] = args.log_every

    seeds = args.seeds if args.seeds is not None else config["seeds"]
    datasets = resolve_datasets(args, config)
    device = resolve_device(args.device)

    cache_config = config.get("cache", {})
    cache_enabled = bool(cache_config.get("enabled", True)) and not args.no_cache
    cache_dir = PROJECT_ROOT / cache_config.get("dir", "outputs/cache/preprocessed")

    run_name = args.run_name or datetime.now().strftime("erm_mlp_%Y%m%d_%H%M%S")
    output_root = PROJECT_ROOT / config["output_root"]
    checkpoint_root = PROJECT_ROOT / config.get("checkpoint_root", "outputs/checkpoints")
    run_dir = ensure_dir(output_root / "runs" / run_name)
    results_dir = ensure_dir(output_root / "results")
    checkpoint_dir = ensure_dir(checkpoint_root / run_name)
    logger = setup_logger(run_dir / "train.log")

    effective_config = {
        **config,
        "training": training_config,
        "datasets": datasets,
        "seeds": seeds,
        "device": str(device),
        "method": "ERM-MLP",
        "cache": {
            "enabled": cache_enabled,
            "dir": str(cache_dir),
            "refresh": args.refresh_cache,
        },
    }
    dump_json(effective_config, run_dir / "config.json")

    logger.info("Log file: %s", run_dir / "train.log")
    logger.info("Running ERM-MLP on %d dataset(s), seeds=%s, device=%s", len(datasets), seeds, device)
    rows = []
    for dataset_name in datasets:
        logger.info("Preparing %s...", dataset_name)
        data_root = PROJECT_ROOT / config["data_root"]
        arrays, cache_status = load_or_prepare_tableshift_arrays(
            data_root=data_root,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            use_cache=cache_enabled,
            refresh_cache=args.refresh_cache,
        )
        logger.info("Prepared %s: %s", dataset_name, cache_status)

        if args.prepare_cache_only:
            continue

        for seed in seeds:
            logger.info("Start training dataset=%s seed=%s", dataset_name, seed)
            row = run_one_dataset_seed(
                dataset_name=dataset_name,
                seed=seed,
                arrays=arrays,
                config=effective_config,
                training_config=TrainingConfig(**training_config),
                device=device,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
            )
            rows.append(row)
            logger.info(format_row(row))

    if args.prepare_cache_only:
        logger.info("Prepared caches in: %s", cache_dir)
        return

    raw_path = results_dir / f"{run_name}_raw.csv"
    summary_path = results_dir / f"{run_name}_summary.csv"
    write_csv(rows, raw_path)
    write_csv(summarize_rows(rows), summary_path)
    logger.info("Saved raw results: %s", raw_path)
    logger.info("Saved summary: %s", summary_path)


def resolve_datasets(args: argparse.Namespace, config: dict) -> list[str]:
    if args.all_datasets:
        return list(config["datasets"])
    if args.dataset:
        return [args.dataset]
    return [config["datasets"][0]]


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    arrays: PreparedArrays,
    config: dict,
    training_config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: Path,
    logger: Logger | None = None,
) -> dict[str, float | int | str]:
    set_seed(seed)

    train_loader = make_loader(
        arrays.X_train,
        arrays.y_train,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = make_loader(
        arrays.X_val,
        arrays.y_val,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    id_loader = make_loader(
        arrays.X_id_test,
        arrays.y_id_test,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    ood_loader = make_loader(
        arrays.X_ood_test,
        arrays.y_ood_test,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = MLP(
        input_dim=arrays.X_train.shape[1],
        hidden_dims=config["model"]["hidden_dims"],
        dropout=config["model"]["dropout"],
    )
    threshold = config["evaluation"]["threshold"]
    train_result = train_erm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        threshold=threshold,
        logger=logger,
        log_prefix=f"{dataset_name} seed={seed} ",
    )

    val_metrics = evaluate_binary_classifier(model, val_loader, device, threshold=threshold)
    id_metrics = evaluate_binary_classifier(model, id_loader, device, threshold=threshold)
    ood_metrics = evaluate_binary_classifier(model, ood_loader, device, threshold=threshold)
    checkpoint_path = checkpoint_dir / f"{dataset_name}_seed{seed}_erm_mlp.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=arrays.X_train.shape[1],
        config=config,
        training_config=training_config,
        train_result=train_result,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        feature_names=arrays.feature_names,
    )

    return {
        "method": "ERM-MLP",
        "dataset": dataset_name,
        "seed": seed,
        "input_dim": arrays.X_train.shape[1],
        "train_size": len(arrays.y_train),
        "val_size": len(arrays.y_val),
        "id_test_size": len(arrays.y_id_test),
        "ood_test_size": len(arrays.y_ood_test),
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
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
        "generalization_gap_accuracy": id_metrics.accuracy - ood_metrics.accuracy, # ID - OOD gap 也是老师要求的指标之一
        "generalization_gap_balanced_accuracy": id_metrics.balanced_accuracy
        - ood_metrics.balanced_accuracy,
        "generalization_gap_f1": id_metrics.f1 - ood_metrics.f1,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    dataset_name: str,
    seed: int,
    input_dim: int,
    config: dict,
    training_config: TrainingConfig,
    train_result: object,
    val_metrics: object,
    id_metrics: object,
    ood_metrics: object,
    feature_names: list[str],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "method": "ERM-MLP",
            "dataset": dataset_name,
            "seed": seed,
            "input_dim": input_dim,
            "feature_names": feature_names,
            "model_config": config["model"],
            "training_config": training_config.__dict__,
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "best_val_balanced_accuracy": train_result.best_val_balanced_accuracy,
            "metrics": {
                "val": val_metrics.as_dict(),
                "id_test": id_metrics.as_dict(),
                "ood_test": ood_metrics.as_dict(),
            },
            "state_dict": model.cpu().state_dict(),
        },
        path,
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> list[dict]:
    metric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, float) and key not in {"seed"}
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["dataset"]), []).append(row)

    summaries = []
    for (method, dataset), group in grouped.items():
        summary = {
            "method": method,
            "dataset": dataset,
            "num_runs": len(group),
        }
        for metric in metric_keys:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def format_row(row: dict) -> str:
    return (
        f"{row['dataset']} seed={row['seed']} "
        f"val_bacc={row['val_balanced_accuracy']:.4f} "
        f"id_bacc={row['id_balanced_accuracy']:.4f} "
        f"ood_bacc={row['ood_balanced_accuracy']:.4f} "
        f"gap={row['generalization_gap_balanced_accuracy']:.4f} "
        f"best_epoch={row['best_epoch']}"
    )


if __name__ == "__main__":
    main()

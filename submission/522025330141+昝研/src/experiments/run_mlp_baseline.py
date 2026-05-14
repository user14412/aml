import argparse
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessed_cache import load_or_prepare_tableshift_arrays
from src.data.preprocessing import PreparedArrays
from src.experiments.baseline_common import (
    PROJECT_ROOT,
    add_common_args,
    add_torch_training_args,
    format_row,
    get_baseline_config,
    make_result_row,
    make_run_context,
    resolve_datasets,
    resolve_device,
    resolve_seeds,
    save_result_tables,
    override_training_config,
)
from src.models.mlp import MLP
from src.training.evaluator import evaluate_binary_classifier
from src.training.trainer import TrainingConfig, make_loader, train_erm
from src.utils.io import ensure_dir, load_json
from src.utils.seed import set_seed


BASELINE = "mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERM-MLP baseline on TableShift CSV splits.")
    add_common_args(parser)
    add_torch_training_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    baseline_config = get_baseline_config(config, BASELINE)
    model_config = dict(baseline_config["model"])
    training_config = override_training_config(baseline_config["training"], args)

    seeds = resolve_seeds(args, config)
    datasets = resolve_datasets(args, config)
    device = resolve_device(args.device)
    method_name = baseline_config.get("method_name", "ERM-MLP")
    run_name = args.run_name or datetime.now().strftime("erm_mlp_%Y%m%d_%H%M%S")

    context = make_run_context(
        args=args,
        config=config,
        run_name=run_name,
        method_name=method_name,
        datasets=datasets,
        seeds=seeds,
        effective_baseline_config={
            **baseline_config,
            "model": model_config,
            "training": training_config,
        },
        extra_config={"device": str(device)},
    )

    rows = []
    for dataset_name in datasets:
        context.logger.info("Preparing %s...", dataset_name)
        arrays, cache_status = load_or_prepare_tableshift_arrays(
            data_root=PROJECT_ROOT / config["data_root"],
            dataset_name=dataset_name,
            cache_dir=context.cache_dir,
            use_cache=context.cache_enabled,
            refresh_cache=args.refresh_cache,
        )
        context.logger.info("Prepared %s: %s", dataset_name, cache_status)

        if args.prepare_cache_only:
            continue

        for seed in seeds:
            context.logger.info("Start training dataset=%s seed=%s", dataset_name, seed)
            row = run_one_dataset_seed(
                dataset_name=dataset_name,
                seed=seed,
                arrays=arrays,
                model_config=model_config,
                training_config=TrainingConfig(**training_config),
                threshold=config["evaluation"]["threshold"],
                device=device,
                checkpoint_dir=context.checkpoint_dir,
                logger=context.logger,
                method_name=method_name,
            )
            rows.append(row)
            context.logger.info(format_row(row))

    if args.prepare_cache_only:
        context.logger.info("Prepared caches in: %s", context.cache_dir)
        return

    save_result_tables(rows, context=context)


def run_one_dataset_seed(
    dataset_name: str,
    seed: int,
    arrays: PreparedArrays,
    model_config: dict,
    training_config: TrainingConfig,
    threshold: float,
    device: torch.device,
    checkpoint_dir: Path,
    logger: Logger | None = None,
    method_name: str = "ERM-MLP",
) -> dict:
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
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
    )
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
        model_config=model_config,
        training_config=training_config,
        train_result=train_result,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        feature_names=arrays.feature_names,
    )

    return make_result_row(
        method=method_name,
        baseline=BASELINE,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=arrays.X_train.shape[1],
        arrays=arrays,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        best_epoch=train_result.best_epoch,
        best_val_loss=train_result.best_val_loss,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    dataset_name: str,
    seed: int,
    input_dim: int,
    model_config: dict,
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
            "baseline": BASELINE,
            "dataset": dataset_name,
            "seed": seed,
            "input_dim": input_dim,
            "feature_names": feature_names,
            "model_config": model_config,
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


if __name__ == "__main__":
    main()

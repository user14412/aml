import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    override_training_config,
    resolve_datasets,
    resolve_device,
    resolve_seeds,
    save_result_tables,
)
from src.training.metrics import BinaryMetrics, compute_binary_metrics
from src.training.trainer import TrainingConfig, make_loader
from src.utils.io import ensure_dir, load_json
from src.utils.seed import set_seed


BASELINE = "ft_transformer"


@dataclass(frozen=True)
class FTTrainingResult:
    best_epoch: int
    best_val_balanced_accuracy: float
    best_val_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FT-Transformer baseline on TableShift CSV splits.")
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
    method_name = baseline_config.get("method_name", "FT-Transformer")
    run_name = args.run_name or datetime.now().strftime("ft_transformer_%Y%m%d_%H%M%S")

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
    *,
    dataset_name: str,
    seed: int,
    arrays: PreparedArrays,
    model_config: dict[str, Any],
    training_config: TrainingConfig,
    threshold: float,
    device: torch.device,
    checkpoint_dir: Path,
    logger: Any,
    method_name: str,
) -> dict[str, Any]:
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

    input_dim = arrays.X_train.shape[1]
    model = build_ft_transformer(input_dim=input_dim, model_config=model_config)
    train_result = train_ft_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        threshold=threshold,
        logger=logger,
        log_prefix=f"{dataset_name} seed={seed} ",
    )

    val_metrics = evaluate_ft_transformer(model, val_loader, device, threshold=threshold)
    id_metrics = evaluate_ft_transformer(model, id_loader, device, threshold=threshold)
    ood_metrics = evaluate_ft_transformer(model, ood_loader, device, threshold=threshold)
    checkpoint_path = checkpoint_dir / f"{dataset_name}_seed{seed}_ft_transformer.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=input_dim,
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
        input_dim=input_dim,
        arrays=arrays,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        best_epoch=train_result.best_epoch,
        best_val_loss=train_result.best_val_loss,
    )


def build_ft_transformer(input_dim: int, model_config: dict[str, Any]) -> nn.Module:
    try:
        import rtdl_revisiting_models as rtdl_models

        kwargs = rtdl_models.FTTransformer.get_default_kwargs(
            n_blocks=int(model_config.get("n_blocks", 3))
        )
        if "d_token" in model_config:
            kwargs["d_block"] = model_config["d_token"]
        for key in [
            "d_block",
            "attention_n_heads",
            "attention_dropout",
            "ffn_d_hidden",
            "ffn_d_hidden_multiplier",
            "ffn_dropout",
            "residual_dropout",
        ]:
            if key in model_config:
                kwargs[key] = model_config[key]
        return rtdl_models.FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=1,
            **kwargs,
        )
    except ImportError:
        pass

    try:
        import rtdl
    except ImportError as exc:
        raise SystemExit(
            "FT-Transformer dependency is not installed. Install one of:\n"
            "  pip install rtdl_revisiting_models\n"
            "  pip install rtdl"
        ) from exc

    n_blocks = int(model_config.get("n_blocks", 3))
    if model_config.get("constructor", "default") == "baseline":
        d_token = int(model_config.get("d_token", 192))
        ffn_d_hidden = int(model_config.get("ffn_d_hidden", d_token * 4 // 3))
        return rtdl.FTTransformer.make_baseline(
            n_num_features=input_dim,
            cat_cardinalities=[],
            d_token=d_token,
            n_blocks=n_blocks,
            attention_dropout=float(model_config.get("attention_dropout", 0.2)),
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=float(model_config.get("ffn_dropout", 0.1)),
            residual_dropout=float(model_config.get("residual_dropout", 0.0)),
            last_layer_query_idx=model_config.get("last_layer_query_idx", [-1]),
            kv_compression_ratio=model_config.get("kv_compression_ratio"),
            kv_compression_sharing=model_config.get("kv_compression_sharing"),
            d_out=1,
        )

    return rtdl.FTTransformer.make_default(
        n_num_features=input_dim,
        cat_cardinalities=[],
        n_blocks=n_blocks,
        last_layer_query_idx=model_config.get("last_layer_query_idx", [-1]),
        kv_compression_ratio=model_config.get("kv_compression_ratio"),
        kv_compression_sharing=model_config.get("kv_compression_sharing"),
        d_out=1,
    )


def train_ft_transformer(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    threshold: float,
    logger: Any,
    log_prefix: str,
) -> FTTrainingResult:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(train_loader, device, config))
    optimizer = torch.optim.AdamW(
        _parameter_groups(model),
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
        train_loss = 0.0
        train_count = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X, None).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = X.shape[0]
            train_loss += float(loss.item()) * batch_size
            train_count += batch_size

        val_loss = _compute_loss(model, val_loader, criterion, device)
        val_metrics = evaluate_ft_transformer(model, val_loader, device, threshold=threshold)
        mean_train_loss = train_loss / max(train_count, 1)
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

        if logger and _should_log_epoch(epoch, config, improved):
            logger.info(
                "%sepoch=%d/%d train_loss=%.6f val_loss=%.6f "
                "val_acc=%.4f val_bacc=%.4f val_f1=%.4f best_epoch=%d "
                "patience=%d/%d",
                log_prefix,
                epoch,
                config.epochs,
                mean_train_loss,
                val_loss,
                val_metrics.accuracy,
                val_metrics.balanced_accuracy,
                val_metrics.f1,
                best_epoch,
                epochs_without_improvement,
                config.patience,
            )

        if epochs_without_improvement >= config.patience:
            if logger:
                logger.info(
                    "%searly_stop epoch=%d best_epoch=%d best_val_bacc=%.4f "
                    "best_val_loss=%.6f",
                    log_prefix,
                    epoch,
                    best_epoch,
                    best_val_balanced_accuracy,
                    best_val_loss,
                )
            break

    model.load_state_dict(best_state)
    return FTTrainingResult(
        best_epoch=best_epoch,
        best_val_balanced_accuracy=best_val_balanced_accuracy,
        best_val_loss=best_val_loss,
    )


def _parameter_groups(model: nn.Module):
    if hasattr(model, "make_parameter_groups"):
        return model.make_parameter_groups()
    if hasattr(model, "optimization_param_groups"):
        return model.optimization_param_groups()
    return model.parameters()


def _should_log_epoch(epoch: int, config: TrainingConfig, improved: bool) -> bool:
    if improved or epoch == 1 or epoch == config.epochs:
        return True
    return config.log_every > 0 and epoch % config.log_every == 0


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
        logits = model(X, None).squeeze(-1)
        loss = criterion(logits, y)
        batch_size = X.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_ft_transformer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> BinaryMetrics:
    model.eval()
    probabilities = []
    targets = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X, None).squeeze(-1)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())

    import numpy as np

    return compute_binary_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        threshold=threshold,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    dataset_name: str,
    seed: int,
    input_dim: int,
    model_config: dict[str, Any],
    training_config: TrainingConfig,
    train_result: FTTrainingResult,
    val_metrics: BinaryMetrics,
    id_metrics: BinaryMetrics,
    ood_metrics: BinaryMetrics,
    feature_names: list[str],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "method": "FT-Transformer",
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

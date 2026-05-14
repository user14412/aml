import argparse
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessed_cache import load_or_prepare_tableshift_arrays
from src.data.preprocessing import PreparedArrays
from src.experiments.baseline_common import (
    PROJECT_ROOT,
    add_common_args,
    format_row,
    get_baseline_config,
    make_result_row,
    make_run_context,
    resolve_datasets,
    resolve_device,
    resolve_seeds,
    save_result_tables,
)
from src.methods.tta.safe_lc_tta import (
    BinaryLogitOffsetWrapper,
    SafeLCTTAConfig,
    SafeLCTTAResult,
    adapt_safe_lc_binary_classifier,
)
from src.models.mlp import MLP
from src.training.evaluator import evaluate_binary_classifier
from src.training.metrics import BinaryMetrics
from src.training.trainer import make_loader
from src.utils.io import ensure_dir, load_json
from src.utils.seed import set_seed


BASELINE = "mlp_safelc_tta"
DEFAULT_SOURCE_RUN_NAME = "erm_mlp_all6_3seeds_final"
DEFAULT_SAFELC_CONFIG = {
    "batch_size": 1024,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "steps_per_batch": 1,
    "entropy_weight": 0.05,
    "prior_weight": 0.0,
    "anchor_weight": 0.01,
    "confidence_threshold": 0.8,
    "confident_prior_weight": 0.5,
    "min_confident_fraction": 0.5,
    "prior_strength": 0.0,
    "max_prior_shift": 0.0,
    "update_scope": "head_bn",
    "max_batches": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SafeLC-TTA from pretrained ERM-MLP checkpoints on TableShift "
            "CSV splits. Adaptation uses OOD features only."
        )
    )
    add_common_args(parser)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for source/adapted evaluation. Defaults to the MLP training batch size.",
    )
    parser.add_argument(
        "--source-run-name",
        type=str,
        default=DEFAULT_SOURCE_RUN_NAME,
        help="Run name under checkpoint_root containing pretrained ERM-MLP checkpoints.",
    )
    parser.add_argument(
        "--source-checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory containing pretrained ERM-MLP checkpoints. Overrides "
            "--source-run-name when provided."
        ),
    )
    parser.add_argument("--tta-batch-size", type=int, default=None)
    parser.add_argument("--tta-lr", type=float, default=None)
    parser.add_argument("--tta-weight-decay", type=float, default=None)
    parser.add_argument("--tta-steps", type=int, default=None)
    parser.add_argument("--tta-entropy-weight", type=float, default=None)
    parser.add_argument("--tta-prior-weight", type=float, default=None)
    parser.add_argument("--tta-anchor-weight", type=float, default=None)
    parser.add_argument("--tta-confidence-threshold", type=float, default=None)
    parser.add_argument("--tta-confident-prior-weight", type=float, default=None)
    parser.add_argument("--tta-min-confident-fraction", type=float, default=None)
    parser.add_argument("--tta-prior-strength", type=float, default=None)
    parser.add_argument("--tta-max-prior-shift", type=float, default=None)
    parser.add_argument(
        "--tta-update-scope",
        type=str,
        default=None,
        choices=["none", "head", "classifier", "last_layer", "bn", "bn_affine", "head_bn", "all"],
    )
    parser.add_argument("--tta-max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    baseline_config = get_baseline_config(config, "mlp")
    model_config = dict(baseline_config["model"])
    eval_batch_size = args.eval_batch_size or int(baseline_config["training"]["batch_size"])
    tta_config = resolve_tta_config(config, args)
    source_checkpoint_dir = resolve_source_checkpoint_dir(config, args)

    seeds = resolve_seeds(args, config)
    datasets = resolve_datasets(args, config)
    device = resolve_device(args.device)
    method_name = "MLP-SafeLC-TTA"
    run_name = args.run_name or datetime.now().strftime("mlp_safelc_tta_%Y%m%d_%H%M%S")

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
        },
        extra_config={
            "device": str(device),
            "source_checkpoint_dir": str(source_checkpoint_dir),
            "eval_batch_size": eval_batch_size,
            "tta_config": asdict(tta_config),
        },
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
            context.logger.info("Start SafeLC-TTA dataset=%s seed=%s", dataset_name, seed)
            row = run_one_dataset_seed(
                dataset_name=dataset_name,
                seed=seed,
                arrays=arrays,
                fallback_model_config=model_config,
                eval_batch_size=eval_batch_size,
                source_checkpoint_dir=source_checkpoint_dir,
                tta_config=tta_config,
                threshold=config["evaluation"]["threshold"],
                device=device,
                checkpoint_dir=context.checkpoint_dir,
                logger=context.logger,
                method_name=method_name,
            )
            rows.append(row)
            context.logger.info(
                "%s source_ood_bacc=%.4f safelc_ood_bacc=%.4f delta=%.4f "
                "target_prior=%.4f offset=%.4f",
                format_row(row),
                row["source_ood_balanced_accuracy"],
                row["ood_balanced_accuracy"],
                row["ood_delta_balanced_accuracy"],
                row["tta_target_positive_prior"],
                row["tta_logit_offset"],
            )

    if args.prepare_cache_only:
        context.logger.info("Prepared caches in: %s", context.cache_dir)
        return

    save_result_tables(rows, context=context)


def resolve_tta_config(config: dict[str, Any], args: argparse.Namespace) -> SafeLCTTAConfig:
    tta_config = dict(DEFAULT_SAFELC_CONFIG)
    configured = config.get("tta", {})
    if "safe_lc_mlp" in configured:
        tta_config.update(configured["safe_lc_mlp"])

    overrides = {
        "batch_size": args.tta_batch_size,
        "lr": args.tta_lr,
        "weight_decay": args.tta_weight_decay,
        "steps_per_batch": args.tta_steps,
        "entropy_weight": args.tta_entropy_weight,
        "prior_weight": args.tta_prior_weight,
        "anchor_weight": args.tta_anchor_weight,
        "confidence_threshold": args.tta_confidence_threshold,
        "confident_prior_weight": args.tta_confident_prior_weight,
        "min_confident_fraction": args.tta_min_confident_fraction,
        "prior_strength": args.tta_prior_strength,
        "max_prior_shift": args.tta_max_prior_shift,
        "update_scope": args.tta_update_scope,
        "max_batches": args.tta_max_batches,
    }
    for key, value in overrides.items():
        if value is not None:
            tta_config[key] = value

    return SafeLCTTAConfig(**tta_config)


def resolve_source_checkpoint_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.source_checkpoint_dir:
        path = Path(args.source_checkpoint_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    checkpoint_root = PROJECT_ROOT / config.get("checkpoint_root", "outputs/checkpoints")
    return checkpoint_root / args.source_run_name


def run_one_dataset_seed(
    *,
    dataset_name: str,
    seed: int,
    arrays: PreparedArrays,
    fallback_model_config: dict[str, Any],
    eval_batch_size: int,
    source_checkpoint_dir: Path,
    tta_config: SafeLCTTAConfig,
    threshold: float,
    device: torch.device,
    checkpoint_dir: Path,
    logger: Logger | None,
    method_name: str,
) -> dict[str, Any]:
    set_seed(seed)

    val_loader = make_loader(
        arrays.X_val,
        arrays.y_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    id_loader = make_loader(
        arrays.X_id_test,
        arrays.y_id_test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    ood_loader = make_loader(
        arrays.X_ood_test,
        arrays.y_ood_test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    source_checkpoint_path = source_checkpoint_dir / f"{dataset_name}_seed{seed}_erm_mlp.pt"
    checkpoint = load_source_checkpoint(source_checkpoint_path)
    source_model, model_config = build_mlp_from_checkpoint(
        checkpoint=checkpoint,
        arrays=arrays,
        fallback_model_config=fallback_model_config,
    )
    source_model.to(device)
    if logger:
        logger.info(
            "%s seed=%s loaded source checkpoint: %s",
            dataset_name,
            seed,
            source_checkpoint_path,
        )

    val_metrics = evaluate_binary_classifier(source_model, val_loader, device, threshold=threshold)
    id_metrics = evaluate_binary_classifier(source_model, id_loader, device, threshold=threshold)
    source_ood_metrics = evaluate_binary_classifier(source_model, ood_loader, device, threshold=threshold)

    adapted_model = MLP(
        input_dim=arrays.X_train.shape[1],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
    )
    adapted_model.load_state_dict(deepcopy(checkpoint["state_dict"]))
    source_positive_prior = positive_prior(arrays.y_train)
    wrapped_model, tta_result = adapt_safe_lc_binary_classifier(
        adapted_model,
        arrays.X_ood_test,
        source_positive_prior=source_positive_prior,
        config=tta_config,
        device=device,
        logger=logger,
        log_prefix=f"{dataset_name} seed={seed} ",
    )
    adapted_ood_metrics = evaluate_binary_classifier(
        wrapped_model,
        ood_loader,
        device,
        threshold=threshold,
    )

    checkpoint_path = checkpoint_dir / f"{dataset_name}_seed{seed}_mlp_safelc_tta.pt"
    source_metadata = {
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_training_config": dict(checkpoint.get("training_config", {})),
        "source_best_epoch": checkpoint.get("best_epoch", ""),
        "source_best_val_loss": checkpoint.get("best_val_loss", ""),
        "source_best_val_balanced_accuracy": checkpoint.get(
            "best_val_balanced_accuracy",
            "",
        ),
    }
    save_adapted_checkpoint(
        path=checkpoint_path,
        adapted_model=adapted_model,
        wrapped_model=wrapped_model,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=arrays.X_train.shape[1],
        model_config=model_config,
        tta_config=tta_config,
        tta_result=tta_result,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        source_ood_metrics=source_ood_metrics,
        adapted_ood_metrics=adapted_ood_metrics,
        feature_names=arrays.feature_names,
        source_metadata=source_metadata,
    )

    row = make_result_row(
        method=method_name,
        baseline=BASELINE,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=arrays.X_train.shape[1],
        arrays=arrays,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=adapted_ood_metrics,
        best_epoch=checkpoint.get("best_epoch", ""),
        best_val_loss=checkpoint.get("best_val_loss", ""),
    )
    row["source_checkpoint_path"] = str(source_checkpoint_path)
    row.update(source_ood_metrics.as_dict(prefix="source_ood_"))
    row["ood_delta_accuracy"] = adapted_ood_metrics.accuracy - source_ood_metrics.accuracy
    row["ood_delta_balanced_accuracy"] = (
        adapted_ood_metrics.balanced_accuracy - source_ood_metrics.balanced_accuracy
    )
    row["ood_delta_f1"] = adapted_ood_metrics.f1 - source_ood_metrics.f1
    row.update(tta_result.as_dict())
    row["tta_update_scope"] = tta_config.update_scope
    row["tta_lr"] = tta_config.lr
    row["tta_confidence_threshold"] = tta_config.confidence_threshold
    row["tta_entropy_weight"] = tta_config.entropy_weight
    row["tta_prior_weight"] = tta_config.prior_weight
    row["tta_anchor_weight"] = tta_config.anchor_weight
    row["tta_prior_strength"] = tta_config.prior_strength
    row["tta_steps_per_batch"] = tta_config.steps_per_batch
    return row


def load_source_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained ERM-MLP checkpoint: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_mlp_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    arrays: PreparedArrays,
    fallback_model_config: dict[str, Any],
) -> tuple[MLP, dict[str, Any]]:
    model_config = dict(checkpoint.get("model_config", fallback_model_config))
    input_dim = int(checkpoint.get("input_dim", arrays.X_train.shape[1]))
    if input_dim != arrays.X_train.shape[1]:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} does not match prepared data "
            f"input_dim={arrays.X_train.shape[1]}."
        )
    if "state_dict" not in checkpoint:
        raise KeyError("Expected pretrained checkpoint to contain a 'state_dict' entry.")

    model = MLP(
        input_dim=input_dim,
        hidden_dims=list(model_config["hidden_dims"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config


def save_adapted_checkpoint(
    *,
    path: Path,
    adapted_model: torch.nn.Module,
    wrapped_model: BinaryLogitOffsetWrapper,
    dataset_name: str,
    seed: int,
    input_dim: int,
    model_config: dict[str, Any],
    tta_config: SafeLCTTAConfig,
    tta_result: SafeLCTTAResult,
    val_metrics: BinaryMetrics,
    id_metrics: BinaryMetrics,
    source_ood_metrics: BinaryMetrics,
    adapted_ood_metrics: BinaryMetrics,
    feature_names: list[str],
    source_metadata: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "method": "MLP-SafeLC-TTA",
            "baseline": BASELINE,
            "checkpoint_type": "adapted_model_with_logit_offset",
            "dataset": dataset_name,
            "seed": seed,
            "input_dim": input_dim,
            "feature_names": feature_names,
            "model_config": model_config,
            "tta_config": asdict(tta_config),
            "tta_result": tta_result.as_dict(prefix=""),
            "logit_offset": float(wrapped_model.logit_offset.detach().cpu()),
            **source_metadata,
            "metrics": {
                "val": val_metrics.as_dict(),
                "id_test_source": id_metrics.as_dict(),
                "ood_test_source": source_ood_metrics.as_dict(),
                "ood_test_adapted": adapted_ood_metrics.as_dict(),
            },
            "state_dict": _cpu_state_dict(adapted_model),
        },
        path,
    )


def positive_prior(values: np.ndarray | torch.Tensor) -> float:
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return float(array.astype(np.float32).mean())


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


if __name__ == "__main__":
    main()

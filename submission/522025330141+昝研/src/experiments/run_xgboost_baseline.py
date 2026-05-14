import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessed_cache import load_or_prepare_tableshift_arrays
from src.data.preprocessing import PreparedArrays
from src.experiments.baseline_common import (
    PROJECT_ROOT,
    add_common_args,
    as_numpy,
    format_row,
    get_baseline_config,
    make_result_row,
    make_run_context,
    resolve_datasets,
    resolve_seeds,
    save_result_tables,
)
from src.training.metrics import compute_binary_metrics
from src.utils.io import ensure_dir, load_json
from src.utils.seed import set_seed


BASELINE = "xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost baseline on TableShift CSV splits.")
    add_common_args(parser)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--tree-method", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    baseline_config = get_baseline_config(config, BASELINE)
    params = override_xgboost_params(baseline_config.get("params", {}), args)

    seeds = resolve_seeds(args, config)
    datasets = resolve_datasets(args, config)
    method_name = baseline_config.get("method_name", "XGBoost")
    run_name = args.run_name or datetime.now().strftime("xgboost_%Y%m%d_%H%M%S")

    context = make_run_context(
        args=args,
        config=config,
        run_name=run_name,
        method_name=method_name,
        datasets=datasets,
        seeds=seeds,
        effective_baseline_config={**baseline_config, "params": params},
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
                params=params,
                threshold=config["evaluation"]["threshold"],
                checkpoint_dir=context.checkpoint_dir,
                method_name=method_name,
            )
            rows.append(row)
            context.logger.info(format_row(row))

    if args.prepare_cache_only:
        context.logger.info("Prepared caches in: %s", context.cache_dir)
        return

    save_result_tables(rows, context=context)


def override_xgboost_params(params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = dict(params)
    overrides = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_jobs": args.n_jobs,
        "tree_method": args.tree_method,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


def run_one_dataset_seed(
    *,
    dataset_name: str,
    seed: int,
    arrays: PreparedArrays,
    params: dict[str, Any],
    threshold: float,
    checkpoint_dir: Path,
    method_name: str,
) -> dict[str, Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from exc

    set_seed(seed)
    model_params = dict(params)
    model_params.setdefault("random_state", seed)
    model_params.setdefault("objective", "binary:logistic")
    model_params.setdefault("eval_metric", "logloss")
    model_params.setdefault("tree_method", "hist")
    model_params.setdefault("n_jobs", 8)

    if model_params.pop("auto_scale_pos_weight", False):
        y_train = as_numpy(arrays.y_train).astype(int)
        positives = max(int(y_train.sum()), 1)
        negatives = max(int(len(y_train) - positives), 1)
        model_params["scale_pos_weight"] = negatives / positives

    model = XGBClassifier(**model_params)
    X_train = as_numpy(arrays.X_train)
    y_train = as_numpy(arrays.y_train).astype(int)
    X_val = as_numpy(arrays.X_val)
    y_val = as_numpy(arrays.y_val).astype(int)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_metrics = evaluate_xgboost(model, arrays.X_val, arrays.y_val, threshold)
    id_metrics = evaluate_xgboost(model, arrays.X_id_test, arrays.y_id_test, threshold)
    ood_metrics = evaluate_xgboost(model, arrays.X_ood_test, arrays.y_ood_test, threshold)

    checkpoint_path = checkpoint_dir / f"{dataset_name}_seed{seed}_xgboost.json"
    ensure_dir(checkpoint_path.parent)
    model.save_model(checkpoint_path)

    return make_result_row(
        method=method_name,
        baseline=BASELINE,
        dataset_name=dataset_name,
        seed=seed,
        input_dim=as_numpy(arrays.X_train).shape[1],
        arrays=arrays,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        best_iteration=getattr(model, "best_iteration", ""),
    )


def evaluate_xgboost(
    model: Any,
    X: Any,
    y: Any,
    threshold: float,
):
    probabilities = model.predict_proba(as_numpy(X))[:, 1]
    return compute_binary_metrics(as_numpy(y), probabilities, threshold=threshold)


if __name__ == "__main__":
    main()

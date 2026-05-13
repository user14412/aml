import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

DATASETS = [
    "assistments",
    "nhanes_lead",
    "brfss_diabetes",
    "acsfoodstamps",
    "physionet",
    "acsunemployment",
]

ABLATION_DATASETS = [
    "brfss_diabetes",
    "physionet",
    "acsunemployment",
]

FINAL_METHODS = [
    ("ERM-MLP", "erm_mlp_all6_3seeds_final_summary.csv"),
    ("XGBoost", "xgboost_all6_3seeds_final_summary.csv"),
    ("Simple-TTA", "mlp_tta_all6_3seeds_summary.csv"),
    ("SafeGate-TTA", "safelc_gated_entropy_all6_3seeds_summary.csv"),
    ("DPL-TTA", "dpl_all6_3seeds_summary.csv"),
    ("Vanilla-PL-TTA", "vanilla_pl_all6_3seeds_summary.csv"),
]

FINAL_METRICS = [
    "ood_accuracy",
    "ood_balanced_accuracy",
    "ood_f1",
    "generalization_gap_balanced_accuracy",
]

DPL_DIAGNOSTICS = [
    "tta_source_prior",
    "tta_target_prior_raw",
    "tta_target_prior",
    "tta_prior_shift",
    "tta_confident_fraction",
    "tta_num_pseudo_pos",
    "tta_num_pseudo_neg",
    "tta_adapted_or_skipped",
]

ABLATION_VARIANTS = [
    (
        "DPL full",
        {
            dataset: "dpl_all6_3seeds_raw.csv"
            for dataset in ABLATION_DATASETS
        },
    ),
    (
        "w/o prior gate",
        {
            dataset: f"dpl_ablation_no_prior_gate_{dataset}_raw.csv"
            for dataset in ABLATION_DATASETS
        },
    ),
    (
        "w/o anchor",
        {
            dataset: f"dpl_ablation_no_anchor_{dataset}_raw.csv"
            for dataset in ABLATION_DATASETS
        },
    ),
    (
        "vanilla pseudo-label",
        {
            dataset: f"dpl_ablation_vanilla_pseudo_{dataset}_raw.csv"
            for dataset in ABLATION_DATASETS
        },
    ),
]

ABLATION_METRICS = [
    "ood_accuracy",
    "ood_balanced_accuracy",
    "ood_f1",
    "generalization_gap_balanced_accuracy",
    "source_ood_balanced_accuracy",
    "ood_delta_balanced_accuracy",
    "tta_source_prior",
    "tta_target_prior_raw",
    "tta_target_prior",
    "tta_prior_shift",
    "tta_confident_fraction",
    "tta_num_pseudo_pos",
    "tta_num_pseudo_neg",
    "tta_adapted_or_skipped",
]


def main() -> None:
    write_final_comparison()
    write_final_report_table()
    write_dpl_diagnostics()
    write_dpl_ablation()


def write_final_comparison() -> None:
    output_rows: list[dict[str, Any]] = []
    erm_rows = index_by_dataset(
        read_csv(RESULTS_DIR / "erm_mlp_all6_3seeds_final_summary.csv")
    )
    for method, filename in FINAL_METHODS:
        rows_by_dataset = index_by_dataset(read_csv(RESULTS_DIR / filename))
        for dataset in DATASETS:
            source = require_dataset(rows_by_dataset, dataset, filename)
            erm_source = require_dataset(
                erm_rows,
                dataset,
                "erm_mlp_all6_3seeds_final_summary.csv",
            )
            row: dict[str, Any] = {
                "method": method,
                "dataset": dataset,
                "num_runs": source["num_runs"],
                "source_file": filename,
            }
            add_summary_metrics(row, source, FINAL_METRICS)
            row["ood_balanced_accuracy_delta_vs_erm"] = (
                to_float(source["ood_balanced_accuracy_mean"])
                - to_float(erm_source["ood_balanced_accuracy_mean"])
            )
            row["ood_f1_delta_vs_erm"] = (
                to_float(source["ood_f1_mean"])
                - to_float(erm_source["ood_f1_mean"])
            )
            output_rows.append(row)

    write_csv(output_rows, RESULTS_DIR / "final_comparison_summary.csv")


def write_final_report_table() -> None:
    rows_by_method = {
        method: index_by_dataset(read_csv(RESULTS_DIR / filename))
        for method, filename in FINAL_METHODS
    }
    dpl_diagnostics = index_by_dataset(read_csv(RESULTS_DIR / "dpl_all6_3seeds_summary.csv"))
    output_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        erm = require_dataset(
            rows_by_method["ERM-MLP"],
            dataset,
            "erm_mlp_all6_3seeds_final_summary.csv",
        )
        dpl = require_dataset(
            rows_by_method["DPL-TTA"],
            dataset,
            "dpl_all6_3seeds_summary.csv",
        )
        vanilla = require_dataset(
            rows_by_method["Vanilla-PL-TTA"],
            dataset,
            "vanilla_pl_all6_3seeds_summary.csv",
        )
        dpl_diag = require_dataset(
            dpl_diagnostics,
            dataset,
            "dpl_all6_3seeds_summary.csv",
        )
        output_rows.append(
            {
                "dataset": dataset,
                "ERM-MLP bacc": metric_mean(rows_by_method, "ERM-MLP", dataset, "ood_balanced_accuracy"),
                "Simple-TTA bacc": metric_mean(rows_by_method, "Simple-TTA", dataset, "ood_balanced_accuracy"),
                "SafeGate-TTA bacc": metric_mean(rows_by_method, "SafeGate-TTA", dataset, "ood_balanced_accuracy"),
                "DPL-TTA bacc": metric_mean(rows_by_method, "DPL-TTA", dataset, "ood_balanced_accuracy"),
                "Vanilla-PL-TTA bacc": metric_mean(rows_by_method, "Vanilla-PL-TTA", dataset, "ood_balanced_accuracy"),
                "ERM-MLP f1": metric_mean(rows_by_method, "ERM-MLP", dataset, "ood_f1"),
                "Simple-TTA f1": metric_mean(rows_by_method, "Simple-TTA", dataset, "ood_f1"),
                "SafeGate-TTA f1": metric_mean(rows_by_method, "SafeGate-TTA", dataset, "ood_f1"),
                "DPL-TTA f1": metric_mean(rows_by_method, "DPL-TTA", dataset, "ood_f1"),
                "Vanilla-PL-TTA f1": metric_mean(rows_by_method, "Vanilla-PL-TTA", dataset, "ood_f1"),
                "DPL-ERM bacc delta": (
                    to_float(dpl["ood_balanced_accuracy_mean"])
                    - to_float(erm["ood_balanced_accuracy_mean"])
                ),
                "DPL-ERM f1 delta": (
                    to_float(dpl["ood_f1_mean"])
                    - to_float(erm["ood_f1_mean"])
                ),
                "Vanilla-ERM bacc delta": (
                    to_float(vanilla["ood_balanced_accuracy_mean"])
                    - to_float(erm["ood_balanced_accuracy_mean"])
                ),
                "Vanilla-ERM f1 delta": (
                    to_float(vanilla["ood_f1_mean"])
                    - to_float(erm["ood_f1_mean"])
                ),
                "DPL-Vanilla bacc delta": (
                    to_float(dpl["ood_balanced_accuracy_mean"])
                    - to_float(vanilla["ood_balanced_accuracy_mean"])
                ),
                "DPL-Vanilla f1 delta": (
                    to_float(dpl["ood_f1_mean"])
                    - to_float(vanilla["ood_f1_mean"])
                ),
                "DPL adapted_or_skipped": to_float(dpl_diag["tta_adapted_or_skipped_mean"]),
                "DPL confident_fraction": to_float(dpl_diag["tta_confident_fraction_mean"]),
                "DPL prior_shift": to_float(dpl_diag["tta_prior_shift_mean"]),
            }
        )

    write_csv(output_rows, RESULTS_DIR / "final_report_table.csv")


def write_dpl_diagnostics() -> None:
    filename = "dpl_all6_3seeds_summary.csv"
    rows_by_dataset = index_by_dataset(read_csv(RESULTS_DIR / filename))
    output_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source = require_dataset(rows_by_dataset, dataset, filename)
        row: dict[str, Any] = {
            "method": "DPL-TTA",
            "dataset": dataset,
            "num_runs": source["num_runs"],
            "source_file": filename,
        }
        add_summary_metrics(row, source, DPL_DIAGNOSTICS)
        output_rows.append(row)

    write_csv(output_rows, RESULTS_DIR / "dpl_diagnostics_summary.csv")


def write_dpl_ablation() -> None:
    output_rows: list[dict[str, Any]] = []
    for variant, files_by_dataset in ABLATION_VARIANTS:
        for dataset in ABLATION_DATASETS:
            filename = files_by_dataset[dataset]
            rows = [
                row for row in read_csv(RESULTS_DIR / filename)
                if row["dataset"] == dataset
            ]
            if not rows:
                raise ValueError(f"No rows for dataset={dataset} in {filename}.")

            summary = summarize_raw_rows(rows, ABLATION_METRICS)
            output_rows.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "num_runs": len(rows),
                    "source_file": filename,
                    **summary,
                }
            )

    write_csv(output_rows, RESULTS_DIR / "dpl_ablation_summary.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected result file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_by_dataset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["dataset"]: row for row in rows}


def require_dataset(
    rows_by_dataset: dict[str, dict[str, str]],
    dataset: str,
    filename: str,
) -> dict[str, str]:
    if dataset not in rows_by_dataset:
        raise KeyError(f"Missing dataset={dataset} in {filename}.")
    return rows_by_dataset[dataset]


def add_summary_metrics(
    output: dict[str, Any],
    source: dict[str, str],
    metrics: list[str],
) -> None:
    for metric in metrics:
        output[f"{metric}_mean"] = to_float(source[f"{metric}_mean"])
        output[f"{metric}_std"] = to_float(source[f"{metric}_std"])


def metric_mean(
    rows_by_method: dict[str, dict[str, dict[str, str]]],
    method: str,
    dataset: str,
    metric: str,
) -> float:
    return to_float(rows_by_method[method][dataset][f"{metric}_mean"])


def summarize_raw_rows(
    rows: list[dict[str, str]],
    metrics: list[str],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in metrics:
        values = [to_float(row[metric]) for row in rows if row.get(metric, "") != ""]
        if not values:
            continue
        summary[f"{metric}_mean"] = mean(values)
        summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def to_float(value: str | float | int) -> float:
    return float(value)


if __name__ == "__main__":
    main()

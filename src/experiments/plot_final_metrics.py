import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

METHOD_ORDER = [
    "ERM-MLP",
    "XGBoost",
    "Simple-TTA",
    "SafeGate-TTA",
    "DPL-TTA",
    "Vanilla-PL-TTA",
]

METHOD_COLORS = {
    "ERM-MLP": "#4C78A8",
    "XGBoost": "#F58518",
    "Simple-TTA": "#54A24B",
    "SafeGate-TTA": "#B279A2",
    "DPL-TTA": "#E45756",
    "Vanilla-PL-TTA": "#72B7B2",
}


def main() -> None:
    rows = read_csv(RESULTS_DIR / "final_comparison_summary.csv")
    plot_macro_metric(
        rows=rows,
        metric="ood_balanced_accuracy_mean",
        ylabel="Macro OOD balanced accuracy",
        title="Macro OOD Balanced Accuracy by Method",
        output_path=FIGURE_DIR / "macro_ood_balanced_accuracy_by_method.png",
    )
    plot_macro_metric(
        rows=rows,
        metric="ood_f1_mean",
        ylabel="Macro OOD F1",
        title="Macro OOD F1 by Method",
        output_path=FIGURE_DIR / "macro_ood_f1_by_method.png",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_macro_metric(
    *,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    values = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            raise ValueError(f"Missing method rows: {method}")
        values.append(mean(float(row[metric]) for row in method_rows))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    colors = [METHOD_COLORS[method] for method in METHOD_ORDER]
    bars = ax.bar(METHOD_ORDER, values, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(values) - 0.0025, max(values) + 0.0025)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00018,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

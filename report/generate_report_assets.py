from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
RESULTS_DIR = ROOT / "outputs" / "results"
DRAFT_PATH = ROOT / "docs" / "report" / "draft_v1.tex"


METHOD_ORDER = [
    "ERM-MLP",
    "Simple-TTA",
    "SafeGate-TTA",
    "DPL-TTA",
    "Vanilla-PL-TTA",
    "XGBoost",
    "FT-Transformer",
]

DATASET_ORDER = [
    "assistments",
    "nhanes_lead",
    "brfss_diabetes",
    "acsfoodstamps",
    "physionet",
    "acsunemployment",
]

ABLATION_ORDER = [
    "DPL full",
    "w/o prior gate",
    "w/o anchor",
    "vanilla pseudo-label",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main_results_table() -> str:
    rows = read_csv(RESULTS_DIR / "final_comparison_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Macro OOD results averaged over six TableShift datasets. "
            r"The table is generated from \texttt{outputs/results/final\_comparison\_summary.csv}.}"
        ),
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & OOD Acc. & OOD BAcc. & OOD F1 & BAcc. Gap \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        acc = mean([float(row["ood_accuracy_mean"]) for row in method_rows])
        bacc = mean([float(row["ood_balanced_accuracy_mean"]) for row in method_rows])
        f1 = mean([float(row["ood_f1_mean"]) for row in method_rows])
        gap = mean(
            [float(row["generalization_gap_balanced_accuracy_mean"]) for row in method_rows]
        )
        lines.append(
            f"{latex_escape(method)} & {fmt(acc)} & {fmt(bacc)} & {fmt(f1)} & {fmt(gap)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def per_dataset_table() -> str:
    rows = read_csv(RESULTS_DIR / "final_report_table.csv")
    by_dataset = {row["dataset"]: row for row in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Per-dataset OOD balanced accuracy and F1. "
            r"The table is generated from \texttt{outputs/results/final\_report\_table.csv}.}"
        ),
        r"\label{tab:per_dataset}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"Dataset & ERM BAcc. & ERM F1 & DPL BAcc. & DPL F1 & "
            r"Vanilla BAcc. & Vanilla F1 & FT-Trans. BAcc. & FT-Trans. F1 \\"
        ),
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        row = by_dataset[dataset]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(dataset),
                fmt(float(row["ERM-MLP bacc"]), 4),
                fmt(float(row["ERM-MLP f1"]), 4),
                fmt(float(row["DPL-TTA bacc"]), 4),
                fmt(float(row["DPL-TTA f1"]), 4),
                fmt(float(row["Vanilla-PL-TTA bacc"]), 4),
                fmt(float(row["Vanilla-PL-TTA f1"]), 4),
                fmt(float(row["FT-Transformer bacc"]), 4),
                fmt(float(row["FT-Transformer f1"]), 4),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def diagnostics_table() -> str:
    path = RESULTS_DIR / "dpl_diagnostics_summary.csv"
    if not path.exists():
        return ""
    rows = read_csv(path)
    by_dataset = {row["dataset"]: row for row in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{DPL-TTA diagnostics computed without OOD labels. "
            r"The table is generated from \texttt{outputs/results/dpl\_diagnostics\_summary.csv}.}"
        ),
        r"\label{tab:dpl_diagnostics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Adapted Frac. & Conf. Frac. & Source Prior & "
            r"Raw Target Prior & Target Prior & Prior Shift \\"
        ),
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        row = by_dataset[dataset]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(dataset),
                fmt(float(row["tta_adapted_or_skipped_mean"]), 3),
                fmt(float(row["tta_confident_fraction_mean"]), 3),
                fmt(float(row["tta_source_prior_mean"]), 4),
                fmt(float(row["tta_target_prior_raw_mean"]), 4),
                fmt(float(row["tta_target_prior_mean"]), 4),
                fmt(float(row["tta_prior_shift_mean"]), 4),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def ablation_table() -> str:
    path = RESULTS_DIR / "dpl_ablation_summary.csv"
    if not path.exists():
        return ""
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Ablation study averaged over \texttt{brfss\_diabetes}, "
            r"\texttt{physionet}, and \texttt{acsunemployment}. "
            r"The table is generated from \texttt{outputs/results/dpl\_ablation\_summary.csv}.}"
        ),
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Macro OOD BAcc. & Macro OOD F1 & Macro BAcc. Delta vs. Source \\",
        r"\midrule",
    ]
    for variant in ABLATION_ORDER:
        variant_rows = grouped.get(variant, [])
        if not variant_rows:
            continue
        bacc = mean([float(row["ood_balanced_accuracy_mean"]) for row in variant_rows])
        f1 = mean([float(row["ood_f1_mean"]) for row in variant_rows])
        delta = mean([float(row["ood_delta_balanced_accuracy_mean"]) for row in variant_rows])
        lines.append(
            f"{latex_escape(variant)} & {fmt(bacc)} & {fmt(f1)} & {fmt(delta, signed=True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def strip_static_table(body: str, label: str, replacement: str) -> str:
    pattern = re.compile(
        r"\\begin\{table\}\[t\].*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}",
        re.DOTALL,
    )
    body, count = pattern.subn(lambda _: replacement.rstrip(), body, count=1)
    if count != 1:
        raise RuntimeError(f"expected to replace exactly one table with label {label}")
    return body


def replace_figure_suggestions(body: str) -> str:
    figure_block = r"""
\begin{figure}[t]
\centering
\begin{minipage}{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{../outputs/results/figures/macro_ood_f1_by_method.png}\\[-0.5ex]
  \small (a) Macro OOD F1.
\end{minipage}\hfill
\begin{minipage}{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth]{../outputs/results/figures/macro_ood_balanced_accuracy_by_method.png}\\[-0.5ex]
  \small (b) Macro OOD balanced accuracy.
\end{minipage}
\caption{Macro OOD metrics by method. DPL-TTA has the strongest macro OOD F1 among MLP-based TTA variants, while FT-Transformer remains strongest overall.}
\label{fig:macro_ood_metrics}
\end{figure}
""".strip()
    start = body.find("% Suggested Figure 2:")
    end = body.find(r"\subsection{Per-Dataset Analysis and Diagnostics}", start)
    if start == -1 or end == -1:
        raise RuntimeError("could not locate suggested macro figure block")
    return body[:start] + figure_block + "\n\n" + body[end:]


def make_report_tex() -> str:
    draft = DRAFT_PATH.read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", draft, re.DOTALL
    )
    if not abstract_match:
        raise RuntimeError("draft is missing an abstract")
    abstract = abstract_match.group(1).strip()
    body = draft[abstract_match.end() :].lstrip()

    body = strip_static_table(body, "tab:main_results", r"\input{generated/main_results_table.tex}")
    body = replace_figure_suggestions(body)
    body = strip_static_table(body, "tab:per_dataset", r"\input{generated/per_dataset_table.tex}")
    body = strip_static_table(body, "tab:dpl_diagnostics", r"\input{generated/dpl_diagnostics_table.tex}")
    body = strip_static_table(body, "tab:ablation", r"\input{generated/ablation_table.tex}")

    preamble = r"""\documentclass{article}

\usepackage[preprint,nonatbib]{../Styles/neurips_2025}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}

\title{Distribution-Guided Pseudo-Label Test-Time Adaptation for Tabular OOD Generalization}

\author{Anonymous Author(s)}

\begin{document}

\maketitle

\begin{abstract}
"""
    return (
        preamble
        + abstract
        + "\n\\end{abstract}\n\n"
        + body
        + "\n\\end{document}\n"
    )


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write(GENERATED_DIR / "main_results_table.tex", main_results_table())
    write(GENERATED_DIR / "per_dataset_table.tex", per_dataset_table())
    write(GENERATED_DIR / "dpl_diagnostics_table.tex", diagnostics_table())
    write(GENERATED_DIR / "ablation_table.tex", ablation_table())
    write(REPORT_DIR / "dpl_tta_neurips_report.tex", make_report_tex())


if __name__ == "__main__":
    main()

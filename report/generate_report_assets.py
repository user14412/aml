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


def texttt(value: object) -> str:
    return rf"\texttt{{{latex_escape(value)}}}"


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
            r"DPL-TTA achieves the best macro OOD F1 among MLP-based TTA variants, "
            r"while FT-Transformer remains strongest overall.}"
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
            r"\caption{Per-dataset OOD F1 and F1 deltas for the main MLP comparison and the strongest tabular backbone.}"
        ),
        r"\label{tab:per_dataset}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        (
            r"Dataset & ERM F1 & DPL F1 & Vanilla F1 & FT-Trans. F1 & "
            r"DPL-ERM $\Delta$ & DPL-Van. $\Delta$ \\"
        ),
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        row = by_dataset[dataset]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                texttt(dataset),
                fmt(float(row["ERM-MLP f1"]), 4),
                fmt(float(row["DPL-TTA f1"]), 4),
                fmt(float(row["Vanilla-PL-TTA f1"]), 4),
                fmt(float(row["FT-Transformer f1"]), 4),
                fmt(float(row["DPL-ERM f1 delta"]), 4, signed=True),
                fmt(float(row["DPL-Vanilla f1 delta"]), 4, signed=True),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
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
            r"\caption{DPL-TTA adaptation diagnostics computed without OOD labels.}"
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
                texttt(dataset),
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
            r"DPL-TTA has higher macro F1 than vanilla pseudo-labeling on this subset, "
            r"while vanilla pseudo-labeling has higher macro balanced accuracy.}"
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


def replace_between(body: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = body.find(start_marker)
    end = body.find(end_marker, start + len(start_marker))
    if start == -1 or end == -1:
        raise RuntimeError(f"could not replace section between {start_marker} and {end_marker}")
    return body[:start] + replacement.strip() + "\n\n" + body[end:]


def methodology_section() -> str:
    return r"""
\section{Methodology}

\subsection{Problem Setup}

We study binary tabular OOD generalization under a target-label-free test-time adaptation protocol. Let the labeled source training set be
\begin{equation}
\mathcal{D}_{s}
=
\{(x_i^{s}, y_i^{s})\}_{i=1}^{n_s},
\quad y_i^{s} \in \{0,1\},
\label{eq:source_data}
\end{equation}
where $x_i^{s}$ is a tabular input and $y_i^{s}$ is the binary label. At adaptation time, the method receives only unlabeled target-domain OOD features,
\begin{equation}
\mathcal{D}_{t}^{x}
=
\{x_j^{t}\}_{j=1}^{n_t}.
\label{eq:target_features}
\end{equation}
The corresponding OOD labels,
\begin{equation}
\mathcal{D}_{t}^{y}
=
\{y_j^{t}\}_{j=1}^{n_t},
\label{eq:target_labels}
\end{equation}
are not used for training, adaptation, pseudo-label construction, prior estimation, or hyperparameter selection. They are used only once for final evaluation.

We first train a source model $f_{\theta_0}$ on $\mathcal{D}_{s}$ using empirical risk minimization. In this report, the main model family for test-time adaptation is an MLP-based binary classifier. Given an input $x$, the model outputs a logit $z_\theta(x)$ and a predicted positive probability
\begin{equation}
p_{\theta}(y=1 \mid x) = \sigma\!\left(z_{\theta}(x)\right),
\label{eq:positive_probability}
\end{equation}
where $\sigma(\cdot)$ is the sigmoid function. The goal of test-time adaptation is to obtain an adapted model $f_{\theta'}$ using only $\mathcal{D}_{t}^{x}$, while avoiding access to OOD labels.

\subsection{Source-Prior-Aware Target Prior Estimation}

DPL-TTA is based on the observation that pseudo-labeling under OOD shift can be biased by the source label distribution. Therefore, instead of applying confidence thresholding alone, DPL-TTA first estimates a conservative target label prior and uses it to guide pseudo-label selection.

The source positive prior is computed from labeled source training data:
\begin{equation}
\pi_s
=
\frac{1}{n_s}
\sum_{i=1}^{n_s} y_i^{s}.
\label{eq:source_prior}
\end{equation}
This quantity is stored after source training. Because it is computed only from source training labels, it does not violate the target-label-free protocol.

At test time, the source model $f_{\theta_0}$ predicts probabilities on unlabeled OOD features,
\begin{equation}
p_j
=
p_{\theta_0}(y=1 \mid x_j^{t}).
\label{eq:target_probability}
\end{equation}
DPL-TTA selects high-confidence target samples using a confidence threshold $\tau$:
\begin{equation}
\mathcal{C}
=
\left\{
j :
p_j \geq \tau
\ \mathrm{or}\ 
p_j \leq 1-\tau
\right\}.
\label{eq:confident_set}
\end{equation}
In the main configuration, $\tau=0.7$. The raw target prior is estimated as the average predicted positive probability over the high-confidence set,
\begin{equation}
\pi_t^{\mathrm{raw}}
=
\frac{1}{|\mathcal{C}|}
\sum_{j \in \mathcal{C}} p_j .
\label{eq:raw_target_prior}
\end{equation}
Since high-confidence OOD predictions can still inherit source-model bias, DPL-TTA applies shrinkage toward the source prior:
\begin{equation}
\pi_t
=
\beta \pi_t^{\mathrm{raw}}
+
(1-\beta)\pi_s,
\label{eq:shrunk_target_prior}
\end{equation}
where $\beta=0.5$ in the main experiments.

This design is deliberately conservative. DPL-TTA is not prior-free: it intentionally uses $\pi_s$ as a source-domain anchor. At the same time, it remains target-label-free because $\pi_t$ is estimated only from unlabeled target features and source-model predictions. DPL-TTA also includes a prior-shift gate:
\begin{equation}
\left|\pi_t - \pi_s\right| < \delta .
\label{eq:prior_shift_gate}
\end{equation}
If the inequality holds, adaptation is skipped. In the main configuration, $\delta=0.03$.

\subsection{Distribution-Guided Pseudo-Label Selection}

The central difference between DPL-TTA and vanilla pseudo-labeling is how pseudo labels are selected. Vanilla pseudo-labeling uses only confidence thresholds:
\begin{equation}
\hat{y}_j =
\begin{cases}
1, & p_j \geq \tau, \\
0, & p_j \leq 1-\tau.
\end{cases}
\label{eq:threshold_pseudo_label}
\end{equation}
This strategy may produce an imbalanced pseudo-label set whose class composition is determined only by the source model's confidence.

DPL-TTA instead uses the shrinkage target prior $\pi_t$ to control the positive and negative pseudo-label budgets. Let
\begin{equation}
\mathcal{P}
=
\left\{j : p_j \geq \tau\right\},
\qquad
\mathcal{N}
=
\left\{j : p_j \leq 1-\tau\right\}
\label{eq:candidate_sets}
\end{equation}
be the confident positive and negative candidate sets. Let $m$ be the desired number of pseudo-labeled samples, set to the number of confident samples unless capped by a maximum pseudo-sample budget. DPL-TTA computes
\begin{equation}
m_+
=
\operatorname{round}(\pi_t m),
\qquad
m_-
=
m - m_+ .
\label{eq:pseudo_label_budgets}
\end{equation}
The method then selects the $m_+$ most confident positive candidates from $\mathcal{P}$ and the $m_-$ most confident negative candidates from $\mathcal{N}$. To keep the notation simple, we write the resulting pseudo-labeled target set as
\begin{equation}
\hat{\mathcal{D}}_{t}
=
\{(\hat{x}_k^{t}, \hat{y}_k^{t})\}_{k=1}^{m}.
\label{eq:pseudo_labeled_set}
\end{equation}
This distribution-guided selection mechanism makes DPL-TTA more conservative than threshold-only pseudo-labeling. It does not assume that all confident predictions should be used. Instead, it uses the estimated target prior to control the class composition of the pseudo-label set.

\subsection{Conservative Test-Time Update}

After constructing pseudo labels, DPL-TTA performs a small test-time update. The update scope is restricted to the classifier head and BatchNorm affine parameters. All other parameters are frozen by default. This restriction reduces the risk of large destructive updates on tabular data, where the adaptation signal may be noisy.

The adaptation objective combines binary cross-entropy on pseudo labels with an anchor regularization term:
\begin{equation}
\mathcal{L}_{\mathrm{DPL}}(\theta)
=
\frac{1}{m}
\sum_{k=1}^{m}
\mathrm{BCE}\!\left(f_{\theta}(\hat{x}_k^{t}), \hat{y}_k^{t}\right)
+
\lambda
\left\|
\theta_{\mathrm{tr}}
-
\theta_{0,\mathrm{tr}}
\right\|_2^2 .
\label{eq:dpl_loss}
\end{equation}
Here, $\theta_{\mathrm{tr}}$ denotes the trainable subset of parameters and $\lambda$ is the anchor weight. In the main configuration, $\lambda=1.0$. The method uses one adaptation step with learning rate $10^{-4}$. If no pseudo labels are selected, or if the prior-shift gate is not passed, DPL-TTA skips adaptation and evaluates the original source model.
"""


def add_citations_and_rewrite_claims(body: str) -> str:
    replacements = {
        "Tabular data are widely used in high-stakes machine learning applications, including finance, healthcare, industrial inspection, and advertising.": (
            "Tabular data are widely used in high-stakes machine learning applications, including finance, healthcare, industrial inspection, and advertising~\\citep{gardner2023tableshift}."
        ),
        "At the same time, stronger tabular backbones remain important: FT-Transformer is the strongest overall baseline in our experiments. Therefore, we do not claim that DPL-TTA beats all baselines. Instead, our main finding is more modest: DPL-TTA provides a conservative and interpretable improvement over MLP-based TTA variants on macro OOD F1, while its balanced-accuracy gains are small and not consistently superior.": (
            "At the same time, stronger tabular backbones remain important: FT-Transformer is the strongest overall baseline in our experiments. Our goal is not to replace stronger tabular backbones, but to study whether a conservative target-label-free adaptation signal can improve an MLP backbone under OOD shifts. The main finding is therefore modest: DPL-TTA provides a conservative and interpretable improvement over MLP-based TTA variants on macro OOD F1, while its balanced-accuracy gains are small and not consistently superior."
        ),
        "TableShift provides a benchmark for evaluating tabular distribution shift with official train, validation, in-distribution test, and out-of-distribution test splits.": (
            "TableShift provides a benchmark for evaluating tabular distribution shift with official train, validation, in-distribution test, and out-of-distribution test splits~\\citep{gardner2023tableshift}."
        ),
        "Test-time adaptation aims to modify a trained model at deployment time using unlabeled target-domain data. In vision tasks, common TTA strategies include entropy minimization, batch-normalization adaptation, augmentation consistency, and pseudo-labeling.": (
            "Test-time adaptation aims to modify a trained model at deployment time using unlabeled target-domain data. In vision tasks, common TTA strategies include test-time training, entropy minimization, batch-normalization adaptation, augmentation consistency, and pseudo-labeling~\\citep{sun2020testtime,wang2021tent}."
        ),
        "Recent tabular TTA methods explore how to adapt models under these constraints. FTAT, AdapTable, and TabLog represent attempts to design adaptation strategies that are more suitable for tabular data.": (
            "Recent tabular TTA methods explore how to adapt models under these constraints. FTAT~\\citep{zhou2024ftat} combines label-distribution optimization with local consistency weighting for fully test-time adaptation. AdapTable~\\citep{kim2024adaptable} calibrates uncertainty and adjusts predictions under target label-distribution shift. TabLog~\\citep{ren2024tablog} uses logic-rule constraints to guide adaptation on tabular data. PFT$_3$A~\\citep{he2026pft3a} studies a stricter prior-free tabular TTA setting."
        ),
        "This distinction separates DPL-TTA from stricter prior-free adaptation settings such as methods that avoid using source label-prior information at test time.": (
            "This distinction separates DPL-TTA from stricter prior-free adaptation settings such as PFT$_3$A~\\citep{he2026pft3a}, which avoids using source label-prior information at test time."
        ),
        "In our experiments, FT-Transformer is the strongest overall baseline, achieving the best macro OOD balanced accuracy and macro OOD F1.": (
            "In our experiments, FT-Transformer~\\citep{gorishniy2021revisiting} is the strongest overall baseline, achieving the best macro OOD balanced accuracy and macro OOD F1."
        ),
        "We evaluate DPL-TTA on six binary classification datasets from the TableShift benchmark:": (
            "We evaluate DPL-TTA on six binary classification datasets from the TableShift benchmark~\\citep{gardner2023tableshift}:"
        ),
        "The second group contains stronger tabular baselines:": (
            "The second group contains stronger tabular baselines, XGBoost~\\citep{chen2016xgboost} and FT-Transformer~\\citep{gorishniy2021revisiting}:"
        ),
        "However, DPL-TTA does not achieve the best macro OOD balanced accuracy. Vanilla-PL-TTA has a slightly higher macro OOD balanced accuracy than DPL-TTA, and both XGBoost and FT-Transformer outperform DPL-TTA on this metric. Therefore, we do not claim that DPL-TTA significantly improves balanced accuracy. We also do not claim that DPL-TTA beats all baselines. The more accurate interpretation is that DPL-TTA is a conservative MLP-based TTA method that improves macro OOD F1 among MLP-based TTA variants, while stronger tabular backbones remain better overall.": (
            "However, DPL-TTA does not achieve the best macro OOD balanced accuracy. Vanilla-PL-TTA has a slightly higher macro OOD balanced accuracy than DPL-TTA, and both XGBoost and FT-Transformer outperform DPL-TTA on this metric. The more accurate interpretation is that DPL-TTA is a conservative MLP-based TTA method that improves macro OOD F1 among MLP-based TTA variants, while stronger tabular backbones remain better overall."
        ),
        "FT-Transformer achieves the best macro OOD balanced accuracy and macro OOD F1, and XGBoost is also a strong non-neural baseline. Therefore, DPL-TTA should not be interpreted as beating all baselines. Rather, it is best understood as a conservative TTA improvement within the MLP-based adaptation family.": (
            "FT-Transformer achieves the best macro OOD balanced accuracy and macro OOD F1, and XGBoost is also a strong non-neural baseline. Rather than replacing those stronger tabular backbones, DPL-TTA is best understood as a conservative TTA improvement within the MLP-based adaptation family."
        ),
        "One direction is to combine distribution-guided pseudo-labeling with stronger tabular backbones such as FT-Transformer, since the current results suggest that backbone strength remains crucial.": (
            "One direction is to combine distribution-guided pseudo-labeling with stronger tabular backbones such as FT-Transformer~\\citep{gorishniy2021revisiting}, since the current results suggest that backbone strength remains crucial."
        ),
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body


def rewrite_per_dataset_section(body: str) -> str:
    start_marker = r"\subsection{Per-Dataset Analysis and Diagnostics}"
    end_marker = r"\subsection{Ablation Study}"
    replacement = r"""
\subsection{Per-Dataset Analysis and Diagnostics}

Table~\ref{tab:dpl_diagnostics} first reports the adaptation diagnostics, so the per-dataset discussion starts from the adaptation decision rather than the final metric. Table~\ref{tab:per_dataset} then reports the F1-focused performance effects. This ordering reflects the intended logic: DPL-TTA first decides whether and how strongly to adapt from unlabeled OOD features, and only afterward do we evaluate whether that decision changes OOD F1.

On \texttt{assistments}, DPL-TTA skips adaptation because the estimated prior shift is small. The source prior is $0.6939$, the shrinkage target prior is $0.6846$, and the estimated prior shift is only $0.0092$, below the threshold $0.03$. As a result, DPL-TTA matches ERM-MLP on this dataset.

On \texttt{nhanes\_lead}, DPL-TTA adapts in two out of three runs but still produces the same OOD F1 as ERM-MLP. This suggests that adaptation does not always translate into measurable performance improvement, especially when confident pseudo labels provide limited useful gradient signal.

On \texttt{brfss\_diabetes}, \texttt{acsfoodstamps}, and \texttt{physionet}, DPL-TTA improves F1 relative to ERM-MLP. The gains are small, but they are consistent with the macro F1 pattern in Table~\ref{tab:main_results}. On \texttt{acsunemployment}, DPL-TTA slightly decreases F1 relative to ERM-MLP while still outperforming Vanilla-PL-TTA on F1.

These results indicate that DPL-TTA is not uniformly better across datasets or metrics. Its strongest support comes from macro OOD F1 among MLP-based TTA variants. The diagnostics also explain part of its behavior: DPL-TTA adapts conservatively, skips adaptation when estimated prior shift is small, and uses high-confidence target predictions to guide pseudo-label composition.

\input{generated/dpl_diagnostics_table.tex}

\input{generated/per_dataset_table.tex}
"""
    return replace_between(body, start_marker, end_marker, replacement)


def make_report_tex() -> str:
    draft = DRAFT_PATH.read_text(encoding="utf-8")
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", draft, re.DOTALL
    )
    if not abstract_match:
        raise RuntimeError("draft is missing an abstract")
    abstract = abstract_match.group(1).strip()
    body = draft[abstract_match.end() :].lstrip()

    body = add_citations_and_rewrite_claims(body)
    body = replace_between(body, r"\section{Methodology}", r"\section{Experiments}", methodology_section())
    body = strip_static_table(body, "tab:main_results", r"\input{generated/main_results_table.tex}")
    body = replace_figure_suggestions(body)
    body = strip_static_table(body, "tab:per_dataset", r"\input{generated/per_dataset_table.tex}")
    body = strip_static_table(body, "tab:dpl_diagnostics", r"\input{generated/dpl_diagnostics_table.tex}")
    body = rewrite_per_dataset_section(body)
    body = strip_static_table(body, "tab:ablation", r"\input{generated/ablation_table.tex}")

    preamble = r"""\documentclass{article}

\makeatletter
\def\input@path{{../Styles/}}
\makeatother
\usepackage[preprint]{neurips_2025}

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

\author{%
  Zan Yan \\
  Student ID: 522025330141 \\
  School of Computer Science, Nanjing University \\
  \texttt{522025330141@smail.nju.edu.cn}
}

\begin{document}

\maketitle

\begin{abstract}
"""
    return (
        preamble
        + abstract
        + "\n\\end{abstract}\n\n"
        + body
        + "\n\\bibliographystyle{plainnat}\n"
        + "\\bibliography{references}\n\n"
        + "\\end{document}\n"
    )


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write(GENERATED_DIR / "main_results_table.tex", main_results_table())
    write(GENERATED_DIR / "per_dataset_table.tex", per_dataset_table())
    write(GENERATED_DIR / "dpl_diagnostics_table.tex", diagnostics_table())
    write(GENERATED_DIR / "ablation_table.tex", ablation_table())
    report_tex = make_report_tex()
    write(REPORT_DIR / "main.tex", report_tex)
    write(REPORT_DIR / "dpl_tta_neurips_report.tex", report_tex)


if __name__ == "__main__":
    main()

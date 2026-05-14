# Paper Assets Check

Checked date: 2026-05-14

## Checklist

- NeurIPS template is already present locally. Do not download a new template.
- `outputs/results/final_comparison_summary.csv` contains `FT-Transformer`.
- `outputs/results/final_report_table.csv` contains DPL/Vanilla deltas and DPL diagnostics.
- Macro OOD balanced accuracy and macro OOD F1 figures both exist and are non-empty PNG files.
- `report_notes.md` is current with the final CSV/figure assets generated at about `2026-05-14 10:20`, but `report_fact_pack.md` is newer and may contain additional factual support.

## Template Tex Path

- `Styles/neurips_2025.tex`

Related local template/style assets:

- `Styles/neurips_2025.sty`
- `Styles/neurips_2025.pdf`

## Available Figure Paths

- `outputs/results/figures/macro_ood_balanced_accuracy_by_method.png`
  - Metric: macro OOD balanced accuracy by method.
  - Size: 1536 x 768.
- `outputs/results/figures/macro_ood_f1_by_method.png`
  - Metric: macro OOD F1 by method.
  - Size: 1536 x 768.

## Available CSV Paths

Primary paper tables:

- `outputs/results/final_comparison_summary.csv`
  - 42 rows: 7 methods x 6 datasets.
  - Methods present: `ERM-MLP`, `XGBoost`, `FT-Transformer`, `Simple-TTA`, `SafeGate-TTA`, `DPL-TTA`, `Vanilla-PL-TTA`.
  - Contains OOD accuracy, OOD balanced accuracy, OOD F1, balanced-accuracy generalization gap, and deltas vs ERM.
- `outputs/results/final_report_table.csv`
  - 6 rows: one per dataset.
  - Contains per-dataset balanced accuracy and F1 for ERM-MLP, FT-Transformer, Simple-TTA, SafeGate-TTA, DPL-TTA, and Vanilla-PL-TTA.
  - Contains `DPL-ERM bacc delta`, `DPL-ERM f1 delta`, `Vanilla-ERM bacc delta`, `Vanilla-ERM f1 delta`, `DPL-Vanilla bacc delta`, and `DPL-Vanilla f1 delta`.
  - Contains diagnostics: `DPL adapted_or_skipped`, `DPL confident_fraction`, and `DPL prior_shift`.

Supporting result CSVs:

- `outputs/results/dpl_diagnostics_summary.csv`
- `outputs/results/dpl_ablation_summary.csv`
- `outputs/results/dpl_all6_3seeds_summary.csv`
- `outputs/results/dpl_all6_3seeds_raw.csv`
- `outputs/results/vanilla_pl_all6_3seeds_summary.csv`
- `outputs/results/vanilla_pl_all6_3seeds_raw.csv`
- `outputs/results/erm_mlp_all6_3seeds_final_summary.csv`
- `outputs/results/erm_mlp_all6_3seeds_final_raw.csv`
- `outputs/results/xgboost_all6_3seeds_final_summary.csv`
- `outputs/results/xgboost_all6_3seeds_final_raw.csv`
- `outputs/results/ft_transformer_all6_3seeds_e10_bs1024_summary.csv`
- `outputs/results/ft_transformer_all6_3seeds_e10_bs1024_raw.csv`
- `outputs/results/mlp_tta_all6_3seeds_summary.csv`
- `outputs/results/mlp_tta_all6_3seeds_raw.csv`
- `outputs/results/safelc_gated_entropy_minconf50_head_all6_3seeds_summary.csv`
- `outputs/results/safelc_gated_entropy_minconf50_head_all6_3seeds_raw.csv`

Ablation/detail CSVs available if needed:

- `outputs/results/dpl_ablation_no_anchor_acsunemployment_summary.csv`
- `outputs/results/dpl_ablation_no_anchor_acsunemployment_raw.csv`
- `outputs/results/dpl_ablation_no_anchor_brfss_diabetes_summary.csv`
- `outputs/results/dpl_ablation_no_anchor_brfss_diabetes_raw.csv`
- `outputs/results/dpl_ablation_no_anchor_physionet_summary.csv`
- `outputs/results/dpl_ablation_no_anchor_physionet_raw.csv`
- `outputs/results/dpl_ablation_no_prior_gate_acsunemployment_summary.csv`
- `outputs/results/dpl_ablation_no_prior_gate_acsunemployment_raw.csv`
- `outputs/results/dpl_ablation_no_prior_gate_brfss_diabetes_summary.csv`
- `outputs/results/dpl_ablation_no_prior_gate_brfss_diabetes_raw.csv`
- `outputs/results/dpl_ablation_no_prior_gate_physionet_summary.csv`
- `outputs/results/dpl_ablation_no_prior_gate_physionet_raw.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_acsunemployment_summary.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_acsunemployment_raw.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_brfss_diabetes_summary.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_brfss_diabetes_raw.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_physionet_summary.csv`
- `outputs/results/dpl_ablation_vanilla_pseudo_physionet_raw.csv`

## Report Notes Status

- `report_notes.md`
  - Last modified: `2026-05-14 10:21:49`.
  - It references the final figures and final CSVs listed above.
  - It includes the final macro table, per-dataset balanced accuracy, DPL diagnostics, ablation interpretation, and limitations.
  - It was updated after `final_comparison_summary.csv`, `final_report_table.csv`, and the two macro figures.
- `report_fact_pack.md`
  - Last modified: `2026-05-14 10:37:39`.
  - It is newer than `report_notes.md` and appears to be a factual support pack, not a replacement paper note.

Conclusion: use `report_notes.md` as the current report-note source for paper writing, but cross-check implementation/code-location facts against `report_fact_pack.md` because it is newer.

## Suggested Tables For The Paper

- Main macro comparison table from `outputs/results/final_comparison_summary.csv`.
  - Recommended columns: method, macro OOD accuracy, macro OOD balanced accuracy, macro OOD F1, macro balanced-accuracy generalization gap.
  - This table should include ERM-MLP, XGBoost, FT-Transformer, Simple-TTA, SafeGate-TTA, DPL-TTA, and Vanilla-PL-TTA.
- Per-dataset table from `outputs/results/final_report_table.csv`.
  - Recommended columns: dataset, ERM-MLP bacc, FT-Transformer bacc, DPL-TTA bacc, Vanilla-PL-TTA bacc, DPL-ERM bacc delta, DPL-Vanilla bacc delta, DPL adapted_or_skipped, DPL confident_fraction, DPL prior_shift.
- Ablation table from `outputs/results/dpl_ablation_summary.csv`.
  - Recommended use: appendix or compact ablation section, especially for prior gate, anchor, and vanilla pseudo-label comparisons.
- Diagnostics table from `outputs/results/dpl_diagnostics_summary.csv`.
  - Recommended use: appendix or method-analysis section to support adaptation/skip behavior and prior-shift discussion.

## Suggested Figures For The Paper

- Main result figure: `outputs/results/figures/macro_ood_balanced_accuracy_by_method.png`
  - Recommended placement: main experiments section.
- Secondary result figure: `outputs/results/figures/macro_ood_f1_by_method.png`
  - Recommended placement: main experiments section or appendix, depending on space.


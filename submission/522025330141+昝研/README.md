# AMLas TableShift DPL-TTA Submission

Student: Zan Yan  
Student ID: 522025330141  
Affiliation: School of Computer Science, Nanjing University  
Email: 522025330141@smail.nju.edu.cn

## Contents

- `dpl_tta_neurips_report.pdf`: final report for submission.
- `src/`: experiment code for ERM-MLP, XGBoost, FT-Transformer, vanilla pseudo-label TTA, SafeGate-TTA, and DPL-TTA.
- `src/methods/tta/dpl_tta.py`: implementation of the proposed Distribution-Guided Pseudo-Label TTA method.
- `src/experiments/run_mlp_dpl_tta.py`: runnable DPL-TTA experiment entry point.
- `src/experiments/build_final_tables.py`: builds the final comparison, per-dataset, diagnostic, and ablation CSV tables.
- `src/experiments/plot_final_metrics.py`: builds the final report figures.
- `report/`: LaTeX source, BibTeX references, generated tables, and the report-generation script.
- `docs/report/draft_v1.tex`: source draft consumed by `report/generate_report_assets.py`.
- `Styles/`: existing NeurIPS 2025 template files used by the report.
- `outputs/results/`: CSV results and figures used in the report.
- `requirements.txt`: Python dependencies.

The original TableShift dataset files, cached tensors, training logs, and model checkpoints are not included in this clean submission package.

## Environment

Use Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

If running on a CUDA machine, install a PyTorch build that matches the CUDA version on the machine.

## Data

Place the official TableShift CSV files under:

```text
TableShift Dataset/
```

The expected layout is one folder per dataset, for example:

```text
TableShift Dataset/
  assistments/
  nhanes_lead/
  brfss_diabetes/
  acsfoodstamps/
  physionet/
  acsunemployment/
```

The default path is configured in `src/configs/default.json`.

## Main Commands

Run the ERM-MLP source model:

```bash
python src/experiments/run_mlp_baseline.py --all-datasets --seeds 42 3407 2004 --run-name erm_mlp_all6_3seeds_final
```

Run the proposed DPL-TTA method from the ERM-MLP checkpoints:

```bash
python src/experiments/run_mlp_dpl_tta.py --all-datasets --seeds 42 3407 2004 --source-run-name erm_mlp_all6_3seeds_final --run-name dpl_all6_3seeds
```

Run the other baselines:

```bash
python src/experiments/run_xgboost_baseline.py --all-datasets --seeds 42 3407 2004 --run-name xgboost_all6_3seeds_final
python src/experiments/run_ft_transformer_baseline.py --all-datasets --seeds 42 3407 2004 --epochs 10 --batch-size 1024 --run-name ft_transformer_all6_3seeds_e10_bs1024
python src/experiments/run_mlp_tta.py --all-datasets --seeds 42 3407 2004 --source-run-name erm_mlp_all6_3seeds_final --run-name mlp_tta_all6_3seeds
python src/experiments/run_mlp_safelc_tta.py --all-datasets --seeds 42 3407 2004 --source-run-name erm_mlp_all6_3seeds_final --run-name safelc_gated_entropy_all6_3seeds
```

Build the final CSV tables and figures:

```bash
python src/experiments/build_final_tables.py
python src/experiments/plot_final_metrics.py
python report/generate_report_assets.py
```

## Report

The report source is in `report/main.tex` and `report/dpl_tta_neurips_report.tex`.
The final submitted PDF is `dpl_tta_neurips_report.pdf`.

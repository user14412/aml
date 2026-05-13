# AMLas TableShift Experiments

This repository is organized for TableShift OOD tabular experiments. The first implemented method is a clean ERM-MLP baseline with official train/validation/ID-test/OOD-test CSV splits.

## Run

Use the `aml` conda environment:

```powershell
conda run -n aml python src\experiments\run_baseline.py --dataset assistments --seeds 0 --epochs 3
```

Run the full baseline:

```powershell
conda run -n aml python src\experiments\run_baseline.py --all-datasets --seeds 0 1 2
```

Precompute fixed TableShift splits into `.pt` files:

```powershell
conda run -n aml python src\experiments\run_baseline.py --all-datasets --refresh-cache --prepare-cache-only
```

The baseline runner uses cached tensors by default when they exist. Use `--refresh-cache` to rebuild them or `--no-cache` to force CSV preprocessing for a run.

Outputs are written to:

- `outputs/runs/<run-name>/config.json`
- `outputs/results/<run-name>_raw.csv`
- `outputs/results/<run-name>_summary.csv`
- `outputs/cache/preprocessed/<dataset>.pt`

## Structure

- `src/data`: TableShift CSV loading and train-only preprocessing.
- `src/models`: neural network architectures.
- `src/training`: dataloaders, ERM training loop, metrics, evaluation.
- `src/methods`: method-level entry points; TTA methods can be added under `src/methods/tta`.
- `src/experiments`: runnable experiment scripts.

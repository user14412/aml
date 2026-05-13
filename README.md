# AMLas TableShift Experiments

This repository is organized for TableShift OOD tabular experiments with official train/validation/ID-test/OOD-test CSV splits. It currently contains three baselines: ERM-MLP, XGBoost, and FT-Transformer.

## Environment

Create or activate a Python 3.10+ environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU machines, install a CUDA-compatible PyTorch build if the default `pip install torch` does not match the server image.

Do not install the old `rtdl` package unless you know you need it. This code uses `rtdl_revisiting_models` for FT-Transformer; old `rtdl` can pull older Torch versions.

## Smoke Test

Run these commands after copying the repository and data to a new machine:

```bash
python -m py_compile src/experiments/baseline_common.py src/experiments/run_mlp_baseline.py src/experiments/run_xgboost_baseline.py src/experiments/run_ft_transformer_baseline.py src/experiments/run_baseline.py
python src/experiments/run_mlp_baseline.py --dataset nhanes_lead --seeds 0 --epochs 1 --batch-size 4096 --run-name smoke_mlp
python src/experiments/run_xgboost_baseline.py --dataset nhanes_lead --seeds 0 --n-estimators 20 --run-name smoke_xgboost
python src/experiments/run_ft_transformer_baseline.py --dataset nhanes_lead --seeds 0 --epochs 1 --batch-size 512 --run-name smoke_ft_transformer
```

Expected smoke-test outputs:

- `outputs/runs/smoke_*/train.log`
- `outputs/results/smoke_*_raw.csv`
- `outputs/results/smoke_*_summary.csv`
- `outputs/checkpoints/smoke_*/*`

## Run

Run one baseline on one dataset:

```bash
python src/experiments/run_mlp_baseline.py --dataset assistments --seeds 42 --epochs 3
python src/experiments/run_xgboost_baseline.py --dataset assistments --seeds 42
python src/experiments/run_ft_transformer_baseline.py --dataset assistments --seeds 42 --epochs 3
```

The legacy MLP entry point still works:

```bash
python src/experiments/run_baseline.py --dataset assistments --seeds 42 --epochs 3
```

Run full three-seed baselines:

```bash
python src/experiments/run_mlp_baseline.py --all-datasets --seeds 42 3407 2004
python src/experiments/run_xgboost_baseline.py --all-datasets --seeds 42 3407 2004
python src/experiments/run_ft_transformer_baseline.py --all-datasets --seeds 42 3407 2004
```

On a cloud server, it is often better to run seeds as separate jobs, for example:

```bash
python src/experiments/run_mlp_baseline.py --all-datasets --seeds 42 --run-name mlp_seed42
python src/experiments/run_mlp_baseline.py --all-datasets --seeds 3407 --run-name mlp_seed3407
python src/experiments/run_mlp_baseline.py --all-datasets --seeds 2004 --run-name mlp_seed2004
```

Precompute fixed TableShift splits into `.pt` files:

```bash
python src/experiments/run_mlp_baseline.py --all-datasets --refresh-cache --prepare-cache-only
```

The baseline runners use cached tensors by default when they exist. Use `--refresh-cache` to rebuild them or `--no-cache` to force CSV preprocessing for a run.

## Outputs

Outputs are written to:

- `outputs/runs/<run-name>/config.json`
- `outputs/runs/<run-name>/train.log`
- `outputs/results/<run-name>_raw.csv`
- `outputs/results/<run-name>_summary.csv`
- `outputs/checkpoints/<run-name>/<dataset>_seed<seed>_<baseline>.*`
- `outputs/cache/preprocessed/<dataset>.pt`

Follow a running experiment log:

```bash
tail -f outputs/runs/<run-name>/train.log
```

PowerShell equivalent:

```powershell
Get-Content outputs\runs\<run-name>\train.log -Wait
```

## Structure

- `src/data`: TableShift CSV loading and train-only preprocessing.
- `src/models`: neural network architectures.
- `src/training`: dataloaders, ERM training loop, metrics, evaluation.
- `src/experiments`: runnable experiment scripts.

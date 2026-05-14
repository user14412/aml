# Report Fact Pack

## 1. DPL-TTA Code Locations

- Method implementation: `src/methods/tta/dpl_tta.py`
  - `DPLTTAConfig`: line 11
  - `adapt_dpl_binary_classifier`: line 69
  - `build_distribution_guided_pseudo_set`: line 204
  - `configure_trainable_parameters`: line 306
  - `parameter_anchor_loss`: line 331
- Runner: `src/experiments/run_mlp_dpl_tta.py`
  - default DPL config: line 42
  - adaptation call: line 300
  - OOD feature input to adaptation: line 302
  - source prior input: line 303
  - final adapted OOD evaluation: line 309
- Config: `src/configs/default.json`
  - `tta.dpl_mlp`: line 100
- Table-building script: `src/experiments/build_final_tables.py`
- Plot script: `src/experiments/plot_final_metrics.py`

## 2. DPL-TTA Method Flow

Source prior:

- Computed from ID training labels only.
- Formula: `source_prior = mean(y_train)`.
- Code path: `positive_prior(arrays.y_train)` in `src/experiments/run_mlp_dpl_tta.py`.

Target prior:

- The source ERM-MLP predicts probabilities on unlabeled OOD features.
- High-confidence samples are selected with `confidence_threshold = 0.7`.
- Raw target prior is the mean predicted probability over high-confidence samples.
- Shrinkage estimate:

```text
target_prior = beta * target_prior_raw + (1 - beta) * source_prior
beta = 0.5
```

Pseudo labels:

- DPL full uses `pseudo_label_strategy = distribution_guided`.
- Positive candidates: OOD samples with predicted probability `>= confidence_threshold`.
- Negative candidates: OOD samples with predicted probability `<= 1 - confidence_threshold`.
- Desired pseudo-label total is the number of confident samples, capped by `max_pseudo_samples` if set.
- Desired positive count is `round(target_prior * desired_total)`.
- Positive pseudo labels are selected from highest-probability positive candidates.
- Negative pseudo labels are selected from lowest-probability negative candidates.
- Vanilla-PL-TTA uses `pseudo_label_strategy = confidence`, which uses thresholded pseudo labels without target-prior count control.

Updated parameters at test time:

- Default `update_scope = head_bn_affine`.
- Updates the classifier head and BatchNorm affine parameters.
- Does not update all model parameters by default.

Loss:

```text
L = BCEWithLogitsLoss(logits, pseudo_labels)
  + anchor_weight * parameter_anchor_loss
```

- `anchor_weight = 1.0`.
- `parameter_anchor_loss` is mean squared deviation of trainable parameters from their pre-adaptation values.

Skip rule:

- Skip adaptation if `abs(target_prior - source_prior) < prior_shift_threshold`.
- Default `prior_shift_threshold = 0.03`.
- Also skip if no pseudo labels are selected.

## 3. Experimental Protocol

Datasets:

- `assistments`
- `nhanes_lead`
- `brfss_diabetes`
- `acsfoodstamps`
- `physionet`
- `acsunemployment`

Seeds:

- `42`
- `3407`
- `2004`

Metrics:

- OOD accuracy
- OOD balanced accuracy
- OOD F1
- Balanced-accuracy generalization gap

OOD label usage:

- OOD labels are not used for adaptation.
- OOD labels are not used for training.
- OOD labels are not used for hyperparameter selection.
- OOD labels are used only for final evaluation.

Hyperparameters:

- DPL-TTA hyperparameters are fixed across datasets and seeds.
- Main DPL-TTA config:

| Hyperparameter | Value |
|---|---:|
| batch_size | 1024 |
| lr | 0.0001 |
| weight_decay | 0.0 |
| adapt_steps | 1 |
| confidence_threshold | 0.7 |
| beta | 0.5 |
| prior_shift_threshold | 0.03 |
| anchor_weight | 1.0 |
| update_scope | head_bn_affine |
| max_pseudo_samples | null |
| pseudo_label_strategy | distribution_guided |

## 4. Final Main Table

Source: `outputs/results/final_comparison_summary.csv`.

| Method | OOD accuracy | OOD balanced accuracy | OOD F1 | BAcc generalization gap |
|---|---:|---:|---:|---:|
| ERM-MLP | 0.685865 | 0.728499 | 0.440841 | 0.070583 |
| Simple-TTA | 0.704795 | 0.728146 | 0.440222 | 0.070936 |
| SafeGate-TTA | 0.687201 | 0.728740 | 0.440324 | 0.070341 |
| DPL-TTA | 0.697950 | 0.728603 | 0.442267 | 0.070479 |
| Vanilla-PL-TTA | 0.691296 | 0.728996 | 0.440526 | 0.070086 |
| XGBoost | 0.692799 | 0.732953 | 0.443815 | 0.065114 |
| FT-Transformer | 0.691541 | 0.733822 | 0.444965 | 0.068255 |

## 5. Per-Dataset OOD Balanced Accuracy and F1

Source: `outputs/results/final_comparison_summary.csv`.

| Dataset | ERM-MLP BAcc | ERM-MLP F1 | DPL-TTA BAcc | DPL-TTA F1 | Vanilla-PL-TTA BAcc | Vanilla-PL-TTA F1 | XGBoost BAcc | XGBoost F1 | FT-Transformer BAcc | FT-Transformer F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assistments | 0.6337 | 0.6761 | 0.6337 | 0.6761 | 0.6337 | 0.6761 | 0.6347 | 0.6778 | 0.6350 | 0.6724 |
| nhanes_lead | 0.7102 | 0.2586 | 0.7102 | 0.2586 | 0.7102 | 0.2586 | 0.7099 | 0.2582 | 0.7131 | 0.2695 |
| brfss_diabetes | 0.7435 | 0.4910 | 0.7429 | 0.4933 | 0.7434 | 0.4915 | 0.7479 | 0.4976 | 0.7463 | 0.4933 |
| acsfoodstamps | 0.7558 | 0.5655 | 0.7561 | 0.5689 | 0.7557 | 0.5672 | 0.7544 | 0.5720 | 0.7655 | 0.5807 |
| physionet | 0.5948 | 0.1781 | 0.5955 | 0.1824 | 0.5969 | 0.1811 | 0.6129 | 0.1884 | 0.6053 | 0.1823 |
| acsunemployment | 0.9330 | 0.4757 | 0.9332 | 0.4743 | 0.9340 | 0.4686 | 0.9379 | 0.4690 | 0.9377 | 0.4716 |

## 6. DPL Ablation Tables

Source: `outputs/results/dpl_ablation_summary.csv`.

Macro over `brfss_diabetes`, `physionet`, and `acsunemployment`:

| Variant | Macro OOD BAcc | Macro OOD F1 | Macro BAcc delta vs source |
|---|---:|---:|---:|
| DPL full | 0.757199 | 0.383321 | +0.000087 |
| w/o prior gate | 0.757199 | 0.383321 | +0.000087 |
| w/o anchor | 0.757197 | 0.383332 | +0.000085 |
| vanilla pseudo-label | 0.758119 | 0.380404 | +0.001007 |

Per-dataset ablation:

| Variant | Dataset | OOD BAcc | OOD F1 | BAcc delta vs source |
|---|---|---:|---:|---:|
| DPL full | brfss_diabetes | 0.7429 | 0.4933 | -0.0006 |
| DPL full | physionet | 0.5955 | 0.1824 | +0.0007 |
| DPL full | acsunemployment | 0.9332 | 0.4743 | +0.0002 |
| w/o prior gate | brfss_diabetes | 0.7429 | 0.4933 | -0.0006 |
| w/o prior gate | physionet | 0.5955 | 0.1824 | +0.0007 |
| w/o prior gate | acsunemployment | 0.9332 | 0.4743 | +0.0002 |
| w/o anchor | brfss_diabetes | 0.7429 | 0.4933 | -0.0006 |
| w/o anchor | physionet | 0.5955 | 0.1824 | +0.0007 |
| w/o anchor | acsunemployment | 0.9332 | 0.4743 | +0.0002 |
| vanilla pseudo-label | brfss_diabetes | 0.7434 | 0.4915 | -0.0001 |
| vanilla pseudo-label | physionet | 0.5969 | 0.1811 | +0.0021 |
| vanilla pseudo-label | acsunemployment | 0.9340 | 0.4686 | +0.0011 |

## 7. DPL Diagnostics Table

Source: `outputs/results/dpl_diagnostics_summary.csv`.

| Dataset | Adapted fraction | Confidence fraction | Source prior | Target prior raw | Target prior | Prior shift |
|---|---:|---:|---:|---:|---:|---:|
| assistments | 0.000 | 0.831 | 0.6939 | 0.6754 | 0.6846 | 0.0092 |
| nhanes_lead | 0.667 | 0.329 | 0.0246 | 0.2255 | 0.1250 | 0.1005 |
| brfss_diabetes | 1.000 | 0.656 | 0.1247 | 0.3300 | 0.2274 | 0.1027 |
| acsfoodstamps | 1.000 | 0.719 | 0.1901 | 0.3935 | 0.2918 | 0.1017 |
| physionet | 1.000 | 0.388 | 0.0119 | 0.4305 | 0.2212 | 0.2093 |
| acsunemployment | 1.000 | 0.977 | 0.0341 | 0.1372 | 0.0856 | 0.0516 |

## 8. Report Figure Paths

- `outputs/results/figures/macro_ood_balanced_accuracy_by_method.png`
- `outputs/results/figures/macro_ood_f1_by_method.png`

## 9. Result CSV Paths

- `outputs/results/final_comparison_summary.csv`
- `outputs/results/final_report_table.csv`
- `outputs/results/dpl_ablation_summary.csv`
- `outputs/results/dpl_diagnostics_summary.csv`
- `outputs/results/dpl_all6_3seeds_summary.csv`
- `outputs/results/vanilla_pl_all6_3seeds_summary.csv`
- `outputs/results/xgboost_all6_3seeds_final_summary.csv`
- `outputs/results/ft_transformer_all6_3seeds_e10_bs1024_summary.csv`

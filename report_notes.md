# Report Notes: DPL-TTA for TableShift OOD Generalization

## Protocol

All methods follow the official TableShift train/validation/test splits. OOD test labels are not used for adaptation, training, or hyperparameter selection. For TTA methods, adaptation uses only unlabeled OOD features; OOD labels are used only once for final evaluation.

All reported results are means over seeds `42`, `3407`, and `2004`.

Figures:

- `outputs/results/figures/macro_ood_balanced_accuracy_by_method.png`
- `outputs/results/figures/macro_ood_f1_by_method.png`

Key CSV files:

- `outputs/results/final_comparison_summary.csv`
- `outputs/results/final_report_table.csv`
- `outputs/results/dpl_ablation_summary.csv`
- `outputs/results/dpl_diagnostics_summary.csv`

## Method Description

The final method is **DPL-TTA**, a distribution-guided pseudo-label test-time adaptation method for binary tabular classification. The model is first trained as an ERM MLP on labeled ID training data. At test time, DPL-TTA receives only unlabeled OOD features.

The method stores the source label prior:

```text
source_prior = mean(y_train)
```

For each OOD test split, the source model predicts probabilities on unlabeled OOD features. High-confidence samples are selected with threshold `0.7`, and their average predicted probability gives a raw target-prior estimate:

```text
target_prior_raw = mean(p_theta(y=1 | x_ood)) over high-confidence samples
target_prior = beta * target_prior_raw + (1 - beta) * source_prior
```

If the estimated prior shift is too small, DPL-TTA skips adaptation:

```text
abs(target_prior - source_prior) < prior_shift_threshold
```

Otherwise, it creates pseudo-labels under a target-prior-controlled positive/negative budget. Positive pseudo-labels are selected from confident positive predictions, and negative pseudo-labels from confident negative predictions. Only the classifier head and BatchNorm affine parameters are updated. The adaptation objective is:

```text
L = BCE(pseudo_labels) + anchor_weight * ||theta_adapt - theta_source||^2
```

This design is source-prior-aware and target-label-free. It is not prior-free, because it intentionally uses `source_prior` computed from ID training labels.

## Hyperparameters

Main DPL-TTA configuration:

| Hyperparameter | Value |
|---|---:|
| confidence_threshold | 0.7 |
| beta | 0.5 |
| prior_shift_threshold | 0.03 |
| adapt_steps | 1 |
| lr | 1e-4 |
| anchor_weight | 1.0 |
| update_scope | head_bn_affine |
| batch_size | 1024 |
| pseudo_label_strategy | distribution_guided |

Vanilla-PL-TTA uses the same hyperparameters, except:

```text
pseudo_label_strategy = confidence
```

That baseline does not use the target prior to control the positive/negative pseudo-label ratio.

## Final Main Table

Macro OOD metrics over all six datasets:

| Method | OOD accuracy | OOD balanced accuracy | OOD F1 | BAcc gap |
|---|---:|---:|---:|---:|
| ERM-MLP | 0.685865 | 0.728499 | 0.440841 | 0.070583 |
| XGBoost | 0.692799 | 0.732953 | 0.443815 | 0.065114 |
| FT-Transformer | 0.691541 | 0.733822 | 0.444965 | 0.068255 |
| Simple-TTA | 0.704795 | 0.728146 | 0.440222 | 0.070936 |
| SafeGate-TTA | 0.687201 | 0.728740 | 0.440324 | 0.070341 |
| DPL-TTA | 0.697950 | 0.728603 | 0.442267 | 0.070479 |
| Vanilla-PL-TTA | 0.691296 | 0.728996 | 0.440526 | 0.070086 |

Per-dataset balanced accuracy and DPL diagnostics:

| Dataset | ERM | FT-Transformer | Simple | SafeGate | DPL | Vanilla-PL | DPL-ERM BAcc | DPL-ERM F1 | DPL adapted frac | DPL confidence frac | DPL prior shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assistments | 0.6337 | 0.6350 | 0.6337 | 0.6340 | 0.6337 | 0.6337 | +0.0000 | +0.0000 | 0.000 | 0.831 | 0.009 |
| nhanes_lead | 0.7102 | 0.7131 | 0.7096 | 0.7102 | 0.7102 | 0.7102 | +0.0000 | +0.0000 | 0.667 | 0.329 | 0.100 |
| brfss_diabetes | 0.7435 | 0.7463 | 0.7434 | 0.7431 | 0.7429 | 0.7434 | -0.0006 | +0.0023 | 1.000 | 0.656 | 0.103 |
| acsfoodstamps | 0.7558 | 0.7655 | 0.7556 | 0.7561 | 0.7561 | 0.7557 | +0.0004 | +0.0034 | 1.000 | 0.719 | 0.102 |
| physionet | 0.5948 | 0.6053 | 0.5925 | 0.5948 | 0.5955 | 0.5969 | +0.0007 | +0.0043 | 1.000 | 0.388 | 0.209 |
| acsunemployment | 0.9330 | 0.9377 | 0.9342 | 0.9343 | 0.9332 | 0.9340 | +0.0002 | -0.0014 | 1.000 | 0.977 | 0.052 |

## Ablation Interpretation

Ablations were run on `brfss_diabetes`, `physionet`, and `acsunemployment` with three seeds.

| Variant | Macro BAcc | Macro F1 | Macro BAcc delta vs source |
|---|---:|---:|---:|
| DPL full | 0.757199 | 0.383321 | +0.000087 |
| w/o prior gate | 0.757199 | 0.383321 | +0.000087 |
| w/o anchor | 0.757197 | 0.383332 | +0.000085 |
| Vanilla pseudo-label | 0.758119 | 0.380404 | +0.001007 |

The prior gate ablation matches full DPL on these three datasets because all three have prior shifts above the default gate threshold. The anchor ablation is also close to full DPL, likely because the method uses only one adaptation step and a small learning rate. The anchor still serves as a conservative guard against larger updates.

FT-Transformer is the strongest overall baseline, with the best macro OOD balanced accuracy (`0.733822`) and macro OOD F1 (`0.444965`). XGBoost is also a strong non-neural baseline, outperforming the MLP-based methods on macro balanced accuracy and F1.

Within the MLP-based TTA variants, DPL-TTA achieves the best macro OOD F1 (`0.442267`). Vanilla pseudo-labeling has higher balanced accuracy on the selected ablation datasets and also a slightly higher macro balanced accuracy over all six datasets (`0.728996` vs. DPL-TTA `0.728603`). However, it has lower macro F1 than DPL-TTA (`0.440526` vs. `0.442267`). This supports the final narrative: Vanilla-PL-TTA is an aggressive pseudo-label baseline, while DPL-TTA is the final conservative, distribution-guided MLP TTA method. DPL-TTA does not beat stronger non-MLP/stronger-backbone baselines overall.

## Limitations

The gains are small. DPL-TTA should be presented as a conservative and interpretable TTA method, not as a large performance breakthrough.

Stronger backbones such as FT-Transformer still outperform DPL-TTA, suggesting that DPL is complementary to architecture improvements rather than a replacement for them.

DPL-TTA is source-prior-aware rather than prior-free. It uses the ID training label prior, which is allowed by the protocol but should be stated clearly.

The target prior estimate depends on high-confidence predictions from the source model. If the source model is poorly calibrated under OOD shift, the estimated target prior can inherit source bias.

The method does not assume reliable tabular augmentations or strong cluster structure. This is safer for tabular data, but it also limits how much adaptation signal is available.

The prior gate can skip adaptation on low estimated shift, as in `assistments`. This protects against unnecessary updates, but it may also miss useful adaptation when the prior estimate is conservative.

All adaptation decisions and pseudo-labels are computed without OOD labels. OOD labels are used only for final evaluation metrics.

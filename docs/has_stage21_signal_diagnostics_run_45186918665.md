# HAS Stage 21 signal diagnostics — run report

- Workflow run: 45186918665
- URL: https://github.com/inbea3/touch/actions/runs/45186918665
- Artifact: has-signal-diagnostics-45186918665
- Artifact ID: 14823852684
- Artifact SHA-256 (GitHub): effb27fd5abaee32a216bedd5ea8741599eaec8b58b0bf772e2ac8544757fee0
- Workflow: success; runtime 30.2 s
- Input: Stage 19 feature artifact run 45115559336
- Input manifest: all 8 entries passed verification; diagnostic output manifest: all 7 entries passed verification.
- Episodes: 4,192 (Train 2,452 / Validation 880 / Test 860); object groups 21/6/6.

## Findings

### 1. Future target variation is not negligible, but object identity explains only a minority

Descriptive between-object variance fractions:

| Split | Future mean (42) | Future SD (42) | Future correlation (861) |
|---|---:|---:|---:|
| Train | 15.61% | 15.80% | 2.50% |
| Validation | 11.79% | 10.78% | 1.29% |
| Test | 15.84% | 15.80% | 2.10% |

The majority of variation is within object groups. This does not support an object-identity-only predictor as a sufficient solution. Fractions are descriptive, not causal.

### 2. Early last pressure carries substantial matched-channel information about future mean

Mean absolute correlation between each channel's last early value and its future mean:
- Train: 0.614 (median 0.636)
- Validation: 0.675 (median 0.724)

This is consistent with temporal persistence. It explains why the last-value baseline is strong and motivates forecasting *change from the last value* rather than the absolute future state.

### 3. Ridge diagnostics on Validation

Train-only standardization; alpha selected only on Validation.

| Target | Alpha | Ridge MAE | Train-mean MAE | Ridge RMSE | Train-mean RMSE | Uniform-average R² |
|---|---:|---:|---:|---:|---:|---:|
| Future mean | 1000 | 0.02267 | 0.03706 | 0.07936 | 0.10317 | 0.622 |
| Future correlation | 1000 | 0.02090 | 0.01888 | 0.06468 | 0.06205 | -0.247 |

The future-mean task has clear validation signal beyond a global train mean. Correlation prediction does not beat the train-mean baseline and has negative uniform-average R².

Fixed-alpha Test results are exploratory only because this test set was already inspected in prior development:
- Future mean Ridge MAE 0.02073 vs train mean 0.03105.
- Future correlation Ridge MAE 0.01953 vs train mean 0.01821.

Do not treat these as a clean final holdout estimate.

### 4. Target dimensionality diagnostics

Across splits, constant-across-episodes target-coordinate fraction is:
- Future mean: Train 2.38%, Validation 0%, Test 2.38%.
- Future SD: Train 2.38%, Validation 0%, Test 2.38%.
- Future correlation: Train 4.88%, Validation 0.23%, Test 4.99%.

This is distinct from per-episode zero-variance channel counts in the source artifact. Keep the two notions separate.

## Decision

1. Retain future mean as the primary predictive target; it has meaningful validation signal.
2. Reframe the target as future change relative to the last observed pressure, while always retaining persistence as a baseline.
3. Deprioritize future-correlation prediction as a primary task; it currently has little demonstrated episode-specific predictive signal.
4. Do not use object labels as model inputs. Object grouping is for split integrity and descriptive analysis.
5. GNN is not yet justified: the current 44 pressure channels have no validated spatial adjacency graph. A channel graph must be grounded in sensor layout or an independently justified physical relation, not inferred ad hoc from test outcomes.
6. Reptile remains a possible comparison only if the task is redefined as adaptation across object tasks, with enough episodes per task and an evaluation protocol that holds out object tasks. Stage 21 does not itself establish meta-learning benefit.
7. Next experiment should test validation-selected residual/persistence models and window-length sensitivity (e.g. first 5 vs 10 vs 20 frames) without touching the previously inspected Test set for selection. Consider creating a new object-held-out split for final thesis confirmation after model development is frozen.

## Provenance

Artifact files: split_summary.json, object_target_summary.csv, label_target_summary.csv, channel_signal_summary.csv, config.json, run.log, environment.txt, manifest.sha256. All output checksums verified. Preserve immutable artifact.

# HAS Stage 22B — matched window benchmark run report

- Workflow run: 45199549823
- URL: https://github.com/inbea3/touch/actions/runs/45199549823
- Artifact: has-window-benchmark-45199549823
- Artifact ID: 14832070476
- Artifact SHA-256 (GitHub): 3b44f51b51b2006bbf62d7c1cc3e82b8f86a31d31bc7cde134f0f4b4516c199f
- Workflow: success
- Input dataset artifact: Stage 22A run 45193420854
- Input manifest: all 10 entries verified by workflow
- Output manifest: all 13 entries independently verified
- Runtime: 8.63 seconds
- Episodes per window: 4,192; Train/Validation/Test = 2452/880/860; same 880 Validation IDs in paired comparison.

## Validation results (selection-relevant)

| Window | Model | MAE | RMSE | Episode Spearman |
|---|---|---:|---:|---:|
| 5 | Persistence | 0.019816 | 0.083022 | 0.4690 |
| 5 | Residual Ridge | 0.023573 | 0.080599 | 0.7116 |
| 10 | Persistence | 0.019899 | 0.083179 | 0.4702 |
| 10 | Residual Ridge | 0.023396 | 0.080902 | 0.7163 |
| 20 | Persistence | 0.020567 | 0.088628 | 0.4712 |
| 20 | Residual Ridge | 0.023885 | 0.083632 | 0.7125 |

Ridge alpha selected independently on Validation: 100 for all three windows.

## Exploratory Test results (not a clean final holdout)

| Window | Model | MAE | RMSE | Episode Spearman |
|---|---|---:|---:|---:|
| 5 | Persistence | 0.018149 | 0.069900 | 0.4316 |
| 5 | Residual Ridge | 0.021263 | 0.066342 | 0.7116 |
| 10 | Persistence | 0.018392 | 0.071127 | 0.4355 |
| 10 | Residual Ridge | 0.021279 | 0.067813 | 0.7167 |
| 20 | Persistence | 0.018883 | 0.073419 | 0.4336 |
| 20 | Residual Ridge | 0.021741 | 0.069589 | 0.7091 |

## Conclusions

1. Five and ten frames are very close on Validation; twenty frames is consistently worse on MAE/RMSE for both persistence and residual Ridge.
2. Persistence has lower MAE than Ridge at every window. Ridge has lower RMSE and substantially higher episode-wise Spearman, showing a metric trade-off: Ridge improves channel-wise relative ordering and reduces large errors but shifts typical absolute error upward.
3. Five-frame Persistence is the most defensible primary operational baseline when MAE is prioritized. Ten-frame Ridge has marginally lower Validation MAE than five-frame Ridge (0.023396 vs 0.023573), but both are worse than persistence. Five-frame Ridge has slightly lower Validation RMSE than ten-frame Ridge (0.080599 vs 0.080902).
4. The twenty-frame window provides no evidence of benefit and should be dropped from the primary model comparison.
5. Because the fixed Test objects have been examined in prior stages, all Test numbers remain exploratory and cannot support confirmatory generalization claims.

## Methodological issue to resolve before final use

The benchmark script fits a separate Ridge model per window and evaluates each window's own eligible set; paired Validation uses the common 880 IDs. Current extraction found all 4,192 episodes eligible for all windows, so this is equivalent for Validation. For formal paired statistical inference, next calculate per-episode paired differences and object-cluster bootstrap intervals (six Validation objects only, so intervals remain descriptive).

## Decision / next stage

- Lock five frames as the primary observation window; retain ten frames as a secondary sensitivity condition; drop twenty frames from main modelling.
- Retain Persistence as the mandatory MAE baseline.
- Use Ridge as a complementary RMSE/rank-order baseline, not as a uniformly superior predictor.
- Next prepare a leakage-safe cross-object adaptation protocol: define each object group as a task, compare ordinary fine-tuning against Reptile, and include persistence/Ridge references. However, current split has only six Validation and six Test object tasks; meta-learning evaluation must be designed around training tasks and unseen test objects, with no adaptation labels from test objects unless the protocol explicitly defines a support set.

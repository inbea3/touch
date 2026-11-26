# HAS Stage 25 — residual structure diagnostic run report

- Workflow run: 45211031230
- URL: https://github.com/inbea3/touch/actions/runs/45211031230
- Artifact: has-stage25-residual-structure-45211031230
- Artifact ID: 14835813228
- GitHub artifact SHA-256: dedbb23db9be7517f0fbdd4d2dc57b670bb67cfd6f32d4040fc45561793b2521
- Workflow: success; all steps succeeded
- Input: Stage 22A artifact run 45193420854; 5-frame window; 4,192 episodes; 42 targets
- Ridge alpha fixed at 1000 from Stage 24B Validation selection
- All 10 files in the output SHA-256 manifest independently verified.

## Results as emitted by Stage 25 script

These are episode-pooled channel diagnostics (not object-macro metrics); see aggregation caveat below.

| Split | Persistence MAE | Ridge MAE | Persistence RMSE | Ridge RMSE | Ridge improved channels |
|---|---:|---:|---:|---:|---:|
| Validation | 0.019816 | 0.023162 | 0.046617 | 0.042539 | 23/42 |
| Test (exploratory) | 0.018149 | 0.020855 | 0.039410 | 0.035202 | 24/42 |

Ridge therefore has worse pooled MAE but lower pooled RMSE on both splits. Channel effects are heterogeneous. On Validation, channels 19, 42 and 41 (original source channel indices) show the largest Ridge MAE reductions, while several high-error channels, including 8, 5 and 1, worsen substantially. The largest Persistence MAE channels on Validation are source channels 8, 4, 12, 1, 2 and 5.

## Cross-channel residual correlation

Episode-pooled residual correlations are nonzero:
- Validation Persistence mean absolute off-diagonal correlation: 0.0538; maximum absolute pair: 0.6375.
- Validation Ridge mean absolute off-diagonal correlation: 0.0817; maximum absolute pair: 0.7138.
- Test Persistence mean absolute off-diagonal correlation: 0.0620; maximum absolute pair: 0.6830.
- Test Ridge mean absolute off-diagonal correlation: 0.0663; maximum absolute pair: 0.6248.

Some channel pairs recur across splits (e.g. retained indices 6/8 and 11/12), but the strongest pair rankings are not identical. These are error-covariance associations, not physical sensor adjacency.

## Critical interpretation and audit caveats

1. The Stage 25 script computes channel metrics pooled over episodes. It does not first compute each object's channel metric and then average equally across objects. These numbers must not be directly compared with Stage 24B's object-macro metrics. The Stage 25 object_metrics.csv does report object-level aggregate MAE/RMSE, but channel-wise object-macro summaries were not generated.
2. The residual correlation matrices are computed across all episodes within each split, without centering residuals within each object. Thus correlations may partly reflect between-object offsets or differences in episode composition. They do not yet establish within-object cross-channel residual dependence.
3. Correlation of prediction errors does not imply a physically meaningful graph. A GNN requires a defensible adjacency prior, ideally sensor coordinates or an independently specified sensor-neighborhood relation.
4. Stage 25's Test diagnostics remain exploratory. The six Test object groups were inspected in earlier work.
5. The Train residual-target summary reports median channel SD 0.01206 and one channel with SD below 1e-6. This is descriptive only.

## Decision

Do not start GNN training from this result. The observed channel covariance is a useful hypothesis generator, but Stage 25 needs a corrected follow-up:
- compute per-object per-channel errors and object-macro channel metrics;
- compute within-object residual correlation matrices, then aggregate correlation summaries across objects;
- quantify stability across the six Validation objects, with object-level bootstrap/leave-one-object-out sensitivity;
- map retained channel indices back to the physical sensor layout and document the adjacency source before defining any graph.

## Provenance
Artifact includes channel_metrics.csv, object_metrics.csv, four residual correlation matrices, summary.json, config.json, environment.txt, run.log and manifest.sha256. All 10 manifest entries passed SHA-256 verification. No raw pressure arrays were saved.

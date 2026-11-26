# HAS Stage 24B — persistence-residual benchmark run report

- Workflow run: 45208024725
- URL: https://github.com/inbea3/touch/actions/runs/45208024725
- Artifact: has-residual-stage24b-45208024725
- Artifact ID: 14833648997
- Artifact SHA-256 (GitHub): 470d2bce9c78c55130de806589ae7393de10faa724cb76753b5a48021d2d84f2
- Workflow conclusion: success; runtime 108 seconds
- Input: Stage 22A run 45193420854; 5-frame window
- Episodes: 4,192; Train/Validation/Test = 2,452 / 880 / 860; object split 21/6/6
- All 18 files in output SHA-256 manifest verified successfully.
- Three seeds: 20261126, 20261127, 20261128.
- Train-only input and target scaling; Validation-only model selection.

## Validation results — primary evidence

Object-macro metrics:

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence | **0.017386** | 0.063274 |
| Train Mean | 0.032915 | 0.085096 |
| Ridge residual (alpha=1000) | 0.020575 | **0.061011** |
| Residual MLP (mean of 3 seeds) | 0.022549 | 0.065333 |
| Direct MLP (mean of 3 seeds) | 0.023084 | 0.067404 |

Residual MLP individual validation results:
- Seed 20261126: MAE 0.022344; RMSE 0.063909; early stop epoch 47.
- Seed 20261127: MAE 0.022651; RMSE 0.064432; early stop epoch 36.
- Seed 20261128: MAE 0.022653; RMSE 0.067659; early stop epoch 36.

Direct MLP individual validation results:
- Seed 20261126: MAE 0.022806; RMSE 0.067042; early stop epoch 40.
- Seed 20261127: MAE 0.022881; RMSE 0.067472; early stop epoch 45.
- Seed 20261128: MAE 0.023565; RMSE 0.067697; early stop epoch 46.

Ridge alpha was selected as 1000 by Validation object-macro MAE. Persistence remains substantially better on MAE. Ridge improves object-macro RMSE by 0.002263 (about 3.6%) relative to Persistence, but worsens MAE by 0.003190 (about 18.3%). This is a metric trade-off, not an overall win. Both MLP variants underperform Persistence on both primary metrics. Residual MLP is modestly better than direct MLP, but neither learns a useful correction under this setup.

## Exploratory Test results — not for selection

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence | **0.015760** | 0.055764 |
| Train Mean | 0.027763 | 0.069377 |
| Ridge residual (alpha=1000 selected on Validation) | 0.019030 | **0.053442** |
| Residual MLP (mean of 3 seeds) | 0.020410 | 0.056092 |
| Direct MLP (mean of 3 seeds) | 0.020784 | 0.056756 |

Test results are exploratory only because these objects have been inspected in previous stages. They are directionally consistent with the Validation trade-off: Persistence best MAE; Ridge lower RMSE but worse MAE.

## Conclusions

1. Persistence remains the best predictor for absolute-error accuracy. Learned corrections do not improve Validation MAE.
2. Ridge residual at alpha=1000 offers a small RMSE improvement but degrades MAE. It can be reported as a secondary error-profile trade-off, not the primary model.
3. Residual MLP is consistently better than direct MLP on Validation MAE across all three seeds, but the difference is insufficient to beat Persistence or Ridge.
4. The data show strong temporal persistence; the current feature-target formulation offers limited learnable incremental signal beyond the last observation.
5. Do not proceed to a larger MLP, MAML, or GNN based on these results. Next examine channel-level and object-level residual distributions, calibration/shrinkage, and whether the future-mean target is meaningfully different from the last observation. Any new hypothesis and tuning must be preregistered and selected on Validation only.

## Provenance
Artifact retains 6 MLP checkpoints, 6 per-epoch training/validation traces, Ridge alpha selection, object-level metrics, model metrics, exact config, environment, run.log, summary and SHA-256 manifest. All 18 manifest entries verified. No raw frame sequences were saved.

# HAS Stage 20b baseline refinement — run report

- Workflow run: 45184900079
- URL: https://github.com/inbea3/touch/actions/runs/45184900079
- Artifact: has-baseline-refinement-45184900079
- Artifact ID: 14822703786
- Artifact SHA-256 (GitHub): fc6c4c886f62558a6798f08747166f84efeb5c800e26ae446d7037bf7935ab15
- Workflow: success; runtime 33.7 s
- Input: Stage 19 verified feature artifact run 45115559336
- Test objects: can, muesli, softlego, spam, toiletroll, tumbox
- Local SHA-256 verification: all 15 manifest entries OK.

## Test results

### Future pressure mean (42-channel primary)

| Method | MAE | RMSE | Episode Spearman |
|---|---:|---:|---:|
| Stage 20 persistence | 0.01815 | 0.06990 | not reported in Stage 20 summary |
| Residual Ridge (alpha=100) | 0.02126 | 0.06634 | 0.7117 |
| Residual MLP (3-seed mean) | 0.02182 | 0.06639 | 0.6962 |

The residual MLP seed MAEs are 0.021929, 0.021776, 0.021741. Residual Ridge and residual MLP improve RMSE over persistence but worsen MAE. Thus persistence remains the best MAE reference, while learned regressors modestly improve RMSE. This is metric-dependent and not a clear overall win.

### Future inter-channel correlation (861 pairs)

| Method | MAE | RMSE | Episode Spearman |
|---|---:|---:|---:|
| Stage 20 zero matrix | 0.01654 | 0.06865 | not reported in Stage 20 summary |
| Ridge (alpha=1000) | 0.01953 | 0.06553 | 0.0173 |
| Train-template shrinkage (lambda=0.75) | 0.01768 | 0.06549 | 0.0061 |

Both Ridge and shrinkage modestly improve RMSE but worsen MAE relative to the zero matrix. Episode-wise Spearman remains effectively zero, so there is no evidence of useful episode-specific correlation-structure prediction.

## Per-object observations

Future mean residual Ridge object MAE: can 0.01702; muesli 0.02346; softlego 0.01427; spam 0.02313; toiletroll 0.01517; tumbox 0.02273. Performance varies materially across held-out object categories. Muesli, spam and tumbox are more difficult in this metric.

## Interpretation and gate

1. No learned model beats persistence/zero references on MAE.
2. Learned regressors improve RMSE slightly, indicating they reduce large errors but do not improve typical absolute error.
3. Future correlation episode-wise rank agreement is near zero. Do not claim the model predicts object-specific interaction structure.
4. The current HAS episode-level task is dominated by persistence and low-variance targets. The next research step should diagnose whether targets carry object-specific signal beyond object identity/label and whether early windows contain sufficient information.
5. Do not proceed to a large GNN benchmark yet. A graph model is justified only after a clearly specified graph prior and a fair comparison to persistence/zero and regularized linear baselines. Since this HAS feature table does not itself contain a natural spatial adjacency graph, any GNN graph construction must be independently justified; arbitrary channel graphs are not acceptable.

## Provenance

Artifact contains config, metrics, environment, run.log, two Ridge estimators, three residual-MLP checkpoints, three epoch traces, three test prediction arrays and manifest. All checksums passed. Preserve artifact as immutable historical evidence.

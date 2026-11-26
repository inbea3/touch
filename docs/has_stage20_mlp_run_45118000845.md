# HAS Stage 20 MLP baseline — run report

- Workflow run: 45118000845
- URL: https://github.com/inbea3/touch/actions/runs/45118000845
- Artifact: has-mlp-baselines-45118000845
- Artifact ID: 14791604802
- Artifact SHA-256 (GitHub): 7b3c92ca93b419859b6b777bbc3a1a6fb1b9783a298b371fc53785e0ce2acac0
- Workflow status: success
- Runtime: 31.2 seconds
- Input feature artifact: Stage 19 run 45115559336
- Test objects: can, muesli, softlego, spam, toiletroll, tumbox (860 episodes)
- Local verification: every file listed in manifest.sha256 returned OK.

## Main results (episode-weighted Test)

| Task / policy | Baseline | MAE | RMSE |
|---|---|---:|---:|
| Future mean, 42 channels | Last value | 0.01815 | 0.06990 |
| Future mean, 42 channels | MLP seeds 3 mean | 0.02311 | 0.06915 |
| Future mean, all 44 | Last value | 0.01732 | 0.06829 |
| Future mean, all 44 | MLP seeds 3 mean | 0.02313 | 0.06887 |
| Future correlation, 42 channels | Zero matrix | 0.01654 | 0.06865 |
| Future correlation, 42 channels | MLP seeds 3 mean | 0.01985 | 0.06581 |
| Future correlation, all 44 | Zero matrix | 0.01505 | 0.06549 |
| Future correlation, all 44 | MLP seeds 3 mean | 0.02068 | 0.06374 |

The MLP mean values above are descriptive averages across initialization seeds, not independent test replications.

## Seed-level results

### 42-channel future mean
- Seed 20261123: epoch 6; MAE 0.024044; RMSE 0.070416; Spearman 0.7257.
- Seed 20261124: epoch 10; MAE 0.022620; RMSE 0.068335; Spearman 0.7140.
- Seed 20261125: epoch 11; MAE 0.022664; RMSE 0.068693; Spearman 0.7196.
- Last-value reference: MAE 0.018149; RMSE 0.069900.

### 42-channel future correlation
- Seed 20261123: epoch 1; MAE 0.019999; RMSE 0.065803; Spearman 0.0117.
- Seed 20261124: epoch 1; MAE 0.020184; RMSE 0.065944; Spearman 0.0131.
- Seed 20261125: epoch 2; MAE 0.019370; RMSE 0.065675; Spearman 0.0205.
- Zero reference: MAE 0.016539; RMSE 0.068646.

### All-44 future mean
- Seed 20261123: epoch 8; MAE 0.023640; RMSE 0.069993; Spearman 0.6898.
- Seed 20261124: epoch 8; MAE 0.023626; RMSE 0.069346; Spearman 0.7116.
- Seed 20261125: epoch 14; MAE 0.022135; RMSE 0.067257; Spearman 0.7139.
- Last-value reference: MAE 0.017324; RMSE 0.068293.

### All-44 future correlation
- Seed 20261123: epoch 1; MAE 0.021444; RMSE 0.064073; Spearman 0.0147.
- Seed 20261124: epoch 1; MAE 0.020992; RMSE 0.063888; Spearman 0.0155.
- Seed 20261125: epoch 2; MAE 0.019614; RMSE 0.063262; Spearman 0.0023.
- Zero reference: MAE 0.015053; RMSE 0.065489.

## Interpretation / decision

1. The MLP does not beat the persistence (last observed value) baseline on future-mean MAE, under either channel policy. It gives a small RMSE improvement in some seeds, so performance is metric-dependent and not a clear win.
2. For future correlation, MLP lowers RMSE relative to the zero matrix but has worse MAE and near-zero episode-wise Spearman. Validation early stopping occurs at epoch 1–2. This is not sufficient evidence that the model learns useful correlation structure.
3. Retaining all 44 channels does not improve results over excluding channels 18 and 25 in a consistent way. The all-44 mean task has slightly lower RMSE in one seed but MAE remains worse than persistence.
4. The Test set contains only six held-out object categories. Episode count does not make the number of independent object groups large. Confidence intervals are descriptive object-bootstrap intervals only.

## Methodological caveat

The implementation optimizes standardized MSE but reports MAE/RMSE in original units. This is legitimate as a training objective but does not guarantee MAE improvement. For the correlation task, highly concentrated/near-zero targets can make MAE favor a zero predictor. Do not claim the current MLP is a successful learned correlation predictor.

## Artifact inventory

- Four task/policy combinations x three initialization seeds.
- Twelve best checkpoints, twelve training traces, twelve test prediction arrays.
- Four train-only scaler archives.
- metrics.json, config.json, run.log, environment.txt, test_episode_index.csv, manifest.sha256.
- All SHA-256 entries verified.

## Next action

Do not proceed directly to GNN using the current MLP as a strong baseline. First diagnose target scale/distribution and establish a better-specified baseline:
- future mean: persistence plus regularized linear/ridge and residual-learning MLP (predict future minus last observed);
- future correlation: zero baseline plus train-mean, shrinkage/low-rank baseline; evaluate correlation-structure metrics with per-episode centering and report MAE/RMSE plus matrix-level rank/agreement metrics;
- inspect validation curves and output variance, then rerun a pre-registered comparison.
No GNN or meta-learning claims are supported by this Stage 20 run.

# HAS Stage 19 feature construction — run inspection

- Workflow run: 45114665143
- Run URL: https://github.com/inbea3/touch/actions/runs/45114665143
- Artifact: has-pressure-feature-dataset-45114665143
- Artifact ID: 14790908244
- Artifact SHA-256: e9ff0a643ff9ef11178ebacbe43df782fc268a839f72aafa659b627d697c349c
- Dataset: TUM-ICS/Hide-and-Seek
- Resolved dataset revision: 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf
- Split seed: 20261123
- Runtime: 251.14 seconds
- Status: workflow success; feature artifact downloaded and independently inspected

## Constructed arrays

| Array | Shape | Type |
|---|---:|---|
| X_early | 4192 x 176 | float32 |
| Y_future_mean | 4192 x 44 | float32 |
| Y_future_std | 4192 x 44 | float32 |
| Y_future_corr_upper | 4192 x 946 | float32 |
| Y_future_edge_07 | 4192 x 946 | uint8 |
| corr_i / corr_j | 946 each | int64 |

All floating arrays passed finite-value checks. No episodes were rejected. The run processed 1,089,543 frames belonging to the 4,192 object-conditioned episodes. Split episode counts: train 2,452; validation 880; test 860. Mean early window duration was 0.04 seconds; median future duration was approximately 2.29 seconds.

## Critical target diagnostics

Positive-edge prevalence for Pearson r >= 0.7:
- Train: 0.2451%
- Validation: 0.2249%
- Test: 0.2552%

Approximately 45.6%–46.9% of episodes have no positive edges at this threshold. The binary edge task is severely imbalanced. Accuracy must not be used as the primary metric; report PR-AUC/AP, recall, precision, prevalence and suitable baselines. Threshold sensitivity must be selected on Train/Validation only.

Future channel variance is frequently degenerate: per episode, the mean number of channels with zero future variance is 26.86/44 (Train 27.33, Validation 25.83, Test 26.58). Split-level aggregate channel diagnostics show exactly constant channels 18 and 25 across all splits; channel 30 is also exactly constant in Train and Test but not Validation. This indicates channel degeneracy varies by split and must be inspected before modelling. Near-dead channels are common.

## Reproducibility issue found and correction

The initial artifact's SHA-256 manifest did not match only `run.log`, because the script appended the final DONE line after generating the manifest. All other listed files matched. The arrays and summary were successfully inspected, but the artifact should not be treated as having a fully valid all-files checksum manifest.

Fix committed as `22ec153ee897c16817e1d56a42a846834300464f`: finalize `run.log` before computing `manifest.sha256`. Rerun the same workflow to produce a corrected, fully verifiable artifact. Preserve the original artifact as historical evidence; do not overwrite it.

## Research decision

- GO: feature generation pipeline executes and produces finite arrays for all 4,192 episodes.
- HOLD: do not train the binary edge model until the severe positive-edge imbalance, channel degeneracy, and target definition are reviewed.
- Next: rerun to correct the checksum manifest; then perform feature/target diagnostics and design baseline metrics. No model training has occurred.

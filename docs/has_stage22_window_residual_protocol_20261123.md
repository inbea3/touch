# HAS Stage 22 — matched observation-window and residual prediction protocol

Input source: TUM-ICS/Hide-and-Seek at immutable revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf; existing frozen object-group split (21 train / 6 validation / 6 test objects). This stage re-extracts from source frames because the prior Stage 19 artifact only contains five-frame features and cannot validly reconstruct 10- or 20-frame inputs or their corresponding future targets.

## A. Matched windows

For each eligible episode, compare observation windows of 5, 10 and 20 frames. For each window w:
- Input: mean, population SD, timestamp-based slope and last value from the first w frames; concatenate 4 blocks and exclude globally constant channels 18 and 25 (168 dimensions).
- Target: per-channel mean pressure over all frames strictly after frame w (42 dimensions).
- Require at least five future frames. Eligibility is reported separately for each window; comparisons use the intersection of episode IDs across windows as a paired sensitivity analysis, as well as each window's full eligible set.
- No raw sequences are persisted.

## B. Prediction models

For each window, compare:
1. Persistence: last observed channel value.
2. Train-mean predictor.
3. Ridge regression on future-mean residual (target minus last observed value), alpha grid {0.1,1,10,100,1000}, selected by Validation standardized MSE.
4. Residual MLP only if Ridge and window diagnostics support additional nonlinear modelling; three seeds, early stopping on Validation.

All scaling and model fitting use Train only. Validation selects alpha/epoch. Previously inspected Test set is exploratory only and is never used to select windows, models, thresholds or hyperparameters.

## C. Metrics

Report MAE, RMSE, per-episode channel Spearman, per-object metrics, object-macro MAE/RMSE, and paired episode-level comparisons across windows. Also report each window's eligible episode count and common-intersection count.

## D. Interpretation limits

This is a sensitivity analysis on the existing object split, not a new independent final holdout. The Test set has been inspected in earlier stages. No claims of confirmatory generalization are permitted. The primary question is whether longer observation windows improve future-mean prediction beyond persistence, not whether an architecture wins on one metric.

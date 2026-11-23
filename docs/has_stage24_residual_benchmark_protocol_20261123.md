# HAS Stage 24B — persistence-residual benchmark protocol

## Research question
Can learned corrections to the last-observed pressure vector improve future-mean prediction on held-out object groups?

## Frozen data and split
Use Stage 22A run 45193420854, 5-frame window. The 4,192 episodes retain the fixed 21/6/6 object-disjoint train/validation/test split. Test is exploratory only because it has been inspected during prior development.

## Target and baselines
For each episode, define persistence prediction P as the last 42 pressure channels in the 168-D feature vector (indices 126:168). Define residual R = Y_future_mean - P. Models predict a 42-D correction Δ and final prediction is P + Δ.

Compare:
1. Persistence (zero correction).
2. Train-mean constant baseline.
3. Multi-output Ridge regression on standardized 168-D inputs to standardized residuals; select alpha from {0.01, 0.1, 1, 10, 100, 1000} using Validation object-macro MAE only.
4. Residual MLP (168-128-64-42) trained on residual target.
5. Direct MLP with same architecture and training procedure trained on Y_future_mean, as a diagnostic comparator.

All scalers fit on Train only. MLP uses AdamW, lr=1e-3, weight decay=1e-3, batch 32, up to 400 epochs, early stopping patience 35 using Validation object-macro MAE. Three fixed seeds: 20261126–20261128. No Test-based selection.

## Metrics
Primary: object-macro MAE; secondary: object-macro RMSE. Report per-object metrics and all seed-level results. Do not treat episodes as independent inferential units.

## Provenance
Save config, model weights, per-epoch training/validation traces, model_metrics.csv, object_metrics.csv, summary, environment, run.log and SHA-256 manifest. Do not overwrite earlier artifacts or save raw pressure sequences.

## Decision rule
A residual method is promising only if Validation object-macro MAE improves consistently over Persistence across seeds and is not driven by a single object. If no method beats Persistence, report that result and reconsider the target/feature formulation rather than escalating model complexity. Test remains descriptive only.

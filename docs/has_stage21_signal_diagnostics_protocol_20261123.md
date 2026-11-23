# HAS Stage 21 — target signal and object-specificity diagnostics

Input: immutable Stage 19 artifact run 45115559336 (dataset revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf), 4,192 episodes, fixed 21/6/6 object-disjoint split.

Purpose: determine whether the weak Stage 20/20b predictive results reflect low target variability, object-level target shifts, or insufficient early-window information. This is a diagnostic stage only; no GNN or meta-learning model is trained.

## Analyses

1. Split-wise target distribution and degeneracy for 42-channel future mean, 42-channel future SD, and 861 future correlation pairs.
2. Object-wise target means and within-object variability for every split.
3. Descriptive between-object variance fraction (ICC-like decomposition) computed separately by split and target. This is descriptive and not causal.
4. Label/weight-variant target summaries, descriptive only; labels are never used as model inputs.
5. Per-channel future mean/SD, zero/degeneracy indicators and matched-channel early-last to future-mean association.
6. Validation-only Ridge diagnostic with Train-only feature/target scaling and alpha grid {0.1, 1, 10, 100, 1000}; compare with Train-mean predictor on Validation.
7. Report fixed-alpha Test metrics only as exploratory, because the fixed Test objects have already been inspected in prior model development. No Test-based selection.

## Outputs

split_summary.json; object_target_summary.csv; label_target_summary.csv; channel_signal_summary.csv; config.json; run.log; environment.txt; manifest.sha256.

## Decision gates

- If object-level target shifts are substantial but within-object variability is high, object identity alone is insufficient and task framing must be reconsidered.
- If matched-channel persistence dominates, consider a change-focused/residual target and report it as such.
- If correlation targets are near-zero/degenerate and episode-wise association remains negligible, do not prioritize correlation GNN.
- A GNN requires a justified channel graph prior independent of target/test outcomes; this dataset contains no spatial adjacency in the current pressure-only feature table.
- Reptile is only warranted if the task can be expressed as adaptation to held-out object tasks with enough episodes per task; this diagnostic alone does not establish that condition.

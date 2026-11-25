# TouchScale contact-conditioned early-to-late pad-structure audit

Date: 2026-11-25
Branch: thesis-few-shot-structure-20261125

## Objective
Re-evaluate within-episode pad-level spatial-structure persistence after removing inactive frames from the window-selection rule.

## Data and sample
Reuse the same deterministic 300-Episode sample selected by `audit_touchscale_signal_300.choose_sample` from the configured 15 shards. This preserves sample comparability with the fixed-clock audit. It is a scene-proportional sample, not the separately prepared task-family quota sample.

## Contact-conditioned windows
1. For each hand and timestamp, aggregate nonnegative normal-force values over taxels within each of the 15 native pads.
2. Define an active contact frame as the sum across all 15 pad loads being greater than 3 N, consistent with the dataset contact-fraction convention.
3. Sort active frame indices chronologically.
4. Select the first 30% of active frames as the early window and the last 30% as the late window, with at least 20 active frames per window.
5. Require at least 40 active frames and strictly non-overlapping windows. Records failing this rule are reported as insufficient active frames; they are not silently dropped.
6. Apply log1p to pad loads. Use a shared early/late mask of pads with nonzero variance in both windows. Require at least four common variable pads.

## Outcomes
- Early/late upper-triangle pad-correlation edge Pearson correlation and cosine similarity.
- Persistence baseline edge MAE and RMSE.
- Mean pad-load cosine similarity.
- Top-3 loaded-pad Jaccard overlap.
- Contact-frame coverage, selected frame ranges, and valid-record counts.

## Interpretation
This remains a within-Episode temporal persistence audit. It is not cross-object generalization, not a learned predictor, and not few-shot adaptation. The pad aggregation is a coarse interpretable descriptor, not a validated physical adjacency graph.

## Comparability
Compare the results with the fixed-clock 30%/30% audit on the same deterministic sample. Report both denominators and failure statuses. Any change in metrics can reflect the changed temporal sampling rule; it does not by itself establish improved prediction.

## Decision gate
Only proceed to a learned predictor after checking whether the contact-conditioned metric is measurable across enough episodes and whether heterogeneity across actors/scenes remains. Any subsequent model evaluation must split by independent episodes and prevent actor/object/task leakage.

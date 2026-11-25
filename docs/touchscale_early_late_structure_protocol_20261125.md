# TouchScale early-to-late pad-level structure feasibility protocol

Date: 2026-11-25
Branch: thesis-few-shot-structure-20261125
Status: implementation and execution gate

## Objective

Test whether a coarse tactile spatial-structure descriptor is temporally persistent within an episode before committing to a learned prediction target or few-shot adaptation model.

## Operational definition

- Preserve the 15 native tactile pad groups in each hand.
- For each pad and timestamp, aggregate its nonnegative taxel normal-force values into one pad-load trace.
- Apply log1p to pad-load traces.
- Divide each hand recording chronologically into first 30% and final 30% windows (at least 20 frames each).
- In each window, compute the Pearson correlation matrix among pad-load traces, using pads with nonzero variance in both windows.
- Compare upper-triangle edge vectors between early and late windows.

## Primary diagnostics

1. Early/late edge-vector Pearson correlation.
2. Cosine similarity of early/late edge vectors.
3. Persistence-baseline edge MAE and RMSE (predict late structure with early structure).
4. Cosine similarity of 15-dimensional mean pad-load vectors.
5. Jaccard overlap of top-3 loaded pads.
6. Per-hand valid-record counts and task/scene/actor stratification.

## Interpretation limits

- This is a within-episode temporal persistence audit, not cross-object transfer and not few-shot adaptation.
- Pad-level aggregation is a coarse, interpretable structure proxy; it is not a validated physical adjacency graph.
- The 300-episode sample is deterministic under the existing 15-shard scene-proportional sampler; it is not the manually prepared task-family quota sample.
- No learned model is trained in this stage.
- The same episode is not split into support/query examples. Any future adaptation experiment must define independent episodes and a grouping policy first.

## Decision gate

Proceed to learned early-to-late structure prediction only if:
- enough hands yield valid early/late structure vectors;
- persistence is not trivially perfect or wholly absent;
- results are not driven by a small number of scenes/actors;
- variability supports a meaningful query target.

If within-episode persistence is weak, revise the structural descriptor before attempting Reptile. Do not claim adaptation based on this audit alone.

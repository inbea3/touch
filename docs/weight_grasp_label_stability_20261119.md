# STAG Weight: repeated-grasp future-correlation label stability audit

**Date:** 2026-11-19  
**Input:** user-provided `weights_lite` pressure-only archive  
**Reference pipeline:** selected 96 sensor nodes and pressure transform from Reptile selected-validation run 44502406385  
**Status:** exploratory label-quality audit only; no model training.

## 1. Question

Can the six repeated grasp intervals for the same physical object provide sufficiently stable future sensor-pair Pearson correlation targets to support recording-level Support/Query episodes?

This audit tests whether correlation structures estimated from disjoint groups of grasp intervals agree. It does not test predictive model performance.

## 2. Frozen preprocessing used for this diagnostic

- Exclude `empty_hand`.
- Use the six main grasp intervals `graspIndex=0..5`; `graspIndex=6` is a trailing incomplete segment with zero `graspValid` frames.
- Use only `graspValid=1` pressure frames.
- Sort each grasp interval by timestamp.
- Use the exact 96 sensor locations selected in run 44502406385 (sensor seed 20261116); verified by regenerating the seed-based selection from the 548 active STAG sensor positions.
- Use the existing transform `clip((pressure - 500) / 150, 0, 1)`.
- Within each grasp, treat the first five valid frames as early input and the remaining valid frames as future target frames, matching the original temporal target definition as closely as the weight metadata permits.
- Compute Pearson correlation matrices over the 96 sensors (4,560 undirected edges), omitting diagonal entries and non-finite edges.

The weight README defines `graspValid` as the valid grasp segment between the fourth and sixth seconds of the timed grasp procedure. This differs from classification's `hasValidLabel`; it is not silently relabelled as the same field.

## 3. Available sample size

Across 26 physical objects:
- 6 main grasp intervals per object = 156 grasp intervals.
- Valid frames per grasp interval range from 11 to 15.
- After removing the first five valid frames, future target samples per grasp range from 6 to 10.
- A few intervals therefore have fewer than eight future frames, making single-grasp Pearson estimates especially noisy.

For each object, enumerate the ten unique 3-grasp-versus-3-grasp partitions (the complementary duplicate is omitted by requiring grasp 0 in the first group). Each side pools the future frames from its three complete grasp intervals. This yields 260 group comparisons nested within 26 objects.

## 4. Results

### 4.1 Individual grasp-to-grasp agreement

Comparing all 15 pairs of single-grasp future correlation matrices within each object (390 nested comparisons):
- mean Spearman correlation across the 4,560 edges: 0.0026; median 0.0024;
- mean Jaccard overlap of positive edges at (r\geq0.7): 0.0112; median 0.0096.

These are descriptive repeated-pair summaries, not 390 independent observations.

### 4.2 Pooled 3-grasp versus 3-grasp agreement

For each object, average the ten split comparisons first; then summarize the 26 object-level means. Bootstrap intervals resample the 26 objects (5,000 repetitions; seed 20261119).

| Metric | Object-level mean | Object-level median | 95% object bootstrap interval |
|---|---:|---:|---:|
| Edge-correlation Spearman | 0.0504 | 0.0425 | [0.0366, 0.0660] |
| Positive-edge Jaccard, (r\geq0.5) | 0.0690 | 0.0623 | [0.0568, 0.0822] |
| Positive-edge Jaccard, (r\geq0.7) | 0.0406 | 0.0361 | [0.0283, 0.0541] |
| Positive-edge Jaccard, (r\geq0.9) | 0.0401 | 0.0167 | [0.0153, 0.0747] |

The Jaccard statistic is undefined when both partitions have no positive edges; such splits are omitted for that threshold. For (r\geq0.9), only 216 of 260 comparisons had a non-empty union.

### 4.3 Full valid-grasp-window sensitivity

As a diagnostic only, repeat the 3-vs-3 grouping using all valid frames from each grasp (including the first five, so this is not the causal future-target protocol). Agreement remains low:
- mean edge Spearman across 260 comparisons: 0.0559;
- mean (r\geq0.7) positive-edge Jaccard: 0.0411.

Thus the low agreement is not explained solely by discarding the first five input frames.

## 5. Interpretation

The six repeated grasps provide more within-object episodes, but the estimated sensor-pair correlation structures are not stable across grasp groups under this definition. Even after pooling three grasps per side, edge-ranking agreement remains weak and positive-edge overlap is low.

This is a substantive limitation for using the weight archive as a straightforward extension of the current object-specific future co-activation target. It may reflect short windows, real grasp-to-grasp variation, contact-state differences, or some combination; this audit cannot identify the cause.

**Decision:** do not yet feed Weight into Reptile/ANIL training as if it supplied six exchangeable recordings per object. Do not claim that it solves the low meta-task count. Retain it as a potentially useful repeated-grasp variability dataset, but first investigate whether a scientifically justified, pre-specified aggregation or a different target is defensible.

## 6. Recommended next checks

1. Examine the distribution of per-grasp sensor variance and effective contact channels, still using pressure only.
2. Compare correlation estimates using robust/shrinkage estimators (e.g. Ledoit-Wolf) as estimator sensitivity, not as a way to tune against validation performance.
3. Test a pre-specified object-level aggregate target from all six grasps, with Support/Query split by complete grasp intervals; report that this changes the target from within-recording future prediction to cross-grasp object-level relation estimation.
4. If no estimator/aggregation produces stable object-level targets, exclude Weight from the main co-activation prediction experiment and report it only as a negative feasibility result.
5. Do not use frame-level random splits and do not treat 156 grasp intervals as 156 independent objects.

## 7. Boundaries

- No image/video or non-pressure modality was accessed.
- No object identity was added; there remain 26 physical objects.
- No model was trained and no MAE/RMSE predictive performance is reported.
- All resampling summaries use the object as the outer statistical unit.

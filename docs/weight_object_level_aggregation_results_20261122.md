# Stage 16: STAG Weight cross-grasp object-level aggregation

**Date:** 2026-11-22  
**Status:** completed exploratory reliability analysis; no model training  
**Protocol:** `docs/weight_object_level_aggregation_protocol_20261122.md`  
**Input archive SHA-256:** `37b1c5fcb246383cdaed552ce07e35690883545cdfdbfc118b48c1cbd2ee3cf3`

## 1. Question and scope

This stage tested whether six repeated pressure-only grasp intervals for each physical object can be aggregated into a reproducible object-level sensor-relation target. This is a new estimand, distinct from the original within-recording future-correlation target. No image, video, pose, or other modality was used. No predictive model was trained.

The analysis used 26 physical objects, six complete grasp intervals per object (156 total), the fixed 96/548 sensor subset (seed 20261116), the transform `clip((pressure-500)/150,0,1)`, and 4,560 undirected sensor edges. The primary target used valid frames after the first five frames in each grasp. All-valid-frame results are secondary and non-future.

## 2. Methods

Four aggregation methods were compared:
1. Equal-grasp Fisher-z mean of six grasp-level Pearson matrices.
2. Equal-grasp Fisher-z median.
3. Future-frame-count-weighted Fisher-z mean.
4. Pooled-frame Pearson correlation (reference).

For each object, ten unique 3-grasp-vs-3-grasp partitions were enumerated. Each method was fitted independently to both halves. The ten partition comparisons were averaged within object before inference. Bootstrap resampling used the 26 physical objects (5,000 repetitions; seed 20261122). Thus the 260 partitions, 156 grasps, 4,560 edges, and frames are nested measurements, not independent inferential units.

## 3. Primary results: future windows

| Aggregation | Edge Spearman mean (95% object bootstrap CI) | Mean absolute edge difference (95% CI) | Positive-edge Jaccard, r >= 0.7 (95% CI) |
|---|---:|---:|---:|
| Equal-grasp Fisher mean | 0.00688 [0.00322, 0.01076] | 0.25599 [0.24601, 0.26804] | 0.00254 [0.00011, 0.00686] |
| Equal-grasp Fisher median | 0.00394 [-0.00033, 0.00800] | 0.28620 [0.27690, 0.29790] | 0.00257 [0.00023, 0.00659] |
| Frame-weighted Fisher mean | 0.00696 [0.00332, 0.01071] | 0.25554 [0.24553, 0.26773] | 0.00404 [0.00000, 0.01173] |
| Pooled-frame Pearson | 0.05043 [0.03603, 0.06646] | 0.33943 [0.32518, 0.35358] | 0.04059 [0.02801, 0.05408] |

The Fisher aggregation methods produced lower absolute edge differences, but substantially worse edge-rank agreement and positive-edge overlap. This is consistent with averaging/shrinkage toward zero; lower absolute difference alone is not evidence of a more faithful or reproducible relation structure.

Paired object-level differences versus pooled-frame Pearson:
- Equal-grasp Fisher mean: Spearman -0.04355 (95% CI [-0.05990, -0.02958]); Jaccard at r >= 0.7: -0.03805 ([-0.05172, -0.02528]).
- Equal-grasp Fisher median: Spearman -0.04649 ([-0.06280, -0.03219]); Jaccard at r >= 0.7: -0.03802 ([-0.05157, -0.02575]).
- Frame-weighted Fisher mean: Spearman -0.04347 ([-0.05961, -0.02949]); Jaccard at r >= 0.7: -0.03656 ([-0.05160, -0.02240]).

## 4. Secondary non-future sensitivity

Using all valid frames (including the first five, therefore not a future-only target) did not change the conclusion. Edge Spearman was 0.01112 for equal-grasp Fisher mean, 0.00801 for Fisher median, 0.01118 for frame-weighted Fisher mean, and 0.05592 for pooled-frame Pearson. The Fisher aggregation Jaccard at r >= 0.7 was zero.

## 5. Leave-one-grasp-out and heterogeneity

For each object, each of six five-of-six Fisher-mean matrices was compared with its all-six counterpart:
- Mean Spearman: 0.90581 (95% object bootstrap CI [0.90511, 0.90656]).
- Mean absolute edge difference: 0.05760 (95% CI [0.05537, 0.06026]).
- Jaccard at r >= 0.5: 0.32368 (95% CI [0.27655, 0.36850]).

This high Spearman is an influence diagnostic only: each leave-one-out estimate shares five of six grasp matrices with the all-six reference. It is not an independent reliability or generalization test. At r >= 0.7, the summary was available for only 13 objects because many matrices had empty positive-edge unions.

Across six grasp-level Fisher-z matrices, the median across-object median edge SD was 0.36268 (object bootstrap 95% CI for the mean of object medians [0.36388, 0.39834]). This indicates appreciable grasp-level edge variation under the chosen target definition.

## 6. Decision

The current six-grasp Weight data do **not** support a reproducible object-level sensor-relation target under the tested equal-grasp Fisher aggregation. The Fisher methods performed worse than pooled-frame Pearson on edge ranking and positive-edge overlap. The high leave-one-grasp-out similarity is explained by substantial data overlap and must not be treated as validation.

**Do not add Weight to the main Reptile/ANIL training set and do not claim that it increases the number of independent object tasks.** Retain it as a repeated-grasp variability / target-feasibility result. Any future object-level target requires a new scientific definition and independent object-level evaluation.

## 7. Reproducibility and retained outputs

Run bundle: `weight_object_aggregation_20261122_run02.zip` (local artifact; SHA-256 `0cc4ad67c467fd28109a3b29649cc006fbc98deaebf5d4e7e84c64eb6596fbb2`). It contains 2,080 partition rows, 156 leave-one-grasp-out rows, object summaries, derived grasp-level correlation matrices, configuration, input hashes, environment, execution log, and file manifest. It does not contain raw pressure frames.

No training trace applies because no model was trained. The run log and complete diagnostic outputs were retained instead.

# STAG Weight pressure-only variance and shrinkage audit

**Date:** 2026-11-22  
**Run:** `weight_variance_shrinkage_20261122_run01`  
**Input SHA-256:** `37b1c5fcb246383cdaed552ce07e35690883545cdfdbfc118b48c1cbd2ee3cf3`  
**Scope:** pressure-only; no model training; no video or other modality.

## Integrity and data inventory

The uploaded `weights_lite` ZIP passed integrity testing. It contains `metadata.mat` and `readme.html`. Pressure shape is 11,682 × 32 × 32, dtype int16, raw range 510–654, with no NaN/Inf. There are 27 recording IDs (including empty_hand), 26 physical objects, and 2,301 `graspValid` frames.

The 548 active sensor positions were reconstructed from non-zero temporal standard deviation. The fixed 96-node subset was reproduced using `default_rng(20261116).choice(548, 96, replace=False)`, sorted. Exact coordinates and indices are retained in the run bundle.

## Part A — variance and coverage

There are 156 complete object/grasp intervals (26 objects × graspIndex 0–5); incomplete graspIndex 6 was excluded. Each interval has 11–15 valid frames and 6–10 future frames after excluding the first five valid frames.

- Median future-frame count per object: 10 (IQR 9.125–10; object-bootstrap 95% CI for mean of object medians [8.85, 9.71]).
- Median fraction of selected sensors with non-zero future variance: 1.00.
- Median fraction with variance > 1e-6: 1.00.
- Median effective sensor count: 74.49/96 (mean of object-level medians 71.56; object-bootstrap 95% CI [67.84, 74.55]).
- Individual grasp effective sensor count range: 12.23–82.46.

**Threshold caveat:** under `clip((raw-500)/150,0,1)`, all selected future-window values exceed 0.05. Therefore the predeclared >0.01, >0.02, and >0.05 activity fractions all equal 1.00 and are uninformative for contact coverage in this archive. They must not be interpreted as contact labels.

## Part B — covariance-estimator sensitivity

Each object has ten unique 3-grasp-vs-3-grasp partitions. Each side pools the future frames of three complete grasp intervals. The primary window uses frames after the first five valid frames; the full-valid-window calculation is secondary and non-future. Sample Pearson, Ledoit–Wolf, and OAS were evaluated over 4,560 undirected edges. There are 260 nested partitions per estimator/window and 1,560 rows overall. No estimator warnings or failures occurred.

Metrics were averaged over the ten partitions within each object; bootstrap resampling uses the 26 physical objects (5,000 repetitions, seed 20261122).

| Estimator | Edge Spearman (95% object bootstrap CI) | Mean absolute edge difference (95% CI) | Positive-edge Jaccard r≥0.7 (95% CI) |
|---|---:|---:|---:|
| Pearson | 0.05043 [0.03647, 0.06671] | 0.33943 [0.32513, 0.35398] | 0.04059 [0.02821, 0.05399] |
| Ledoit–Wolf | 0.05122 [0.03729, 0.06748] | 0.26786 [0.25141, 0.28397] | 0.04051 [0.02600, 0.05603] |
| OAS | 0.05142 [0.03699, 0.06817] | 0.23260 [0.22086, 0.24445] | 0.04100 [0.02594, 0.05701] |

Paired object-bootstrap differences versus Pearson:
- Ledoit–Wolf Spearman +0.00078 [0.00019, 0.00137]; Jaccard r≥0.7 −0.00009 [−0.00405, 0.00400].
- OAS Spearman +0.00098 [0.00005, 0.00191]; Jaccard r≥0.7 +0.00040 [−0.00551, 0.00710].

Shrinkage reduces mean absolute edge differences, but it does not materially improve rank agreement or r≥0.7 positive-edge overlap. It also reduces Jaccard at r≥0.5 (Pearson 0.06896; Ledoit–Wolf 0.05717; OAS 0.05335). Since shrinkage pulls estimates toward zero, a smaller absolute difference alone is not evidence of a more faithful physical target.

Median future-window shrinkage intensity was 0.121 for Ledoit–Wolf and 0.184 for OAS. Shrinkage improves numerical conditioning, but the cross-grasp agreement remains weak. The full-valid-window sensitivity also remained weak (edge Spearman: Pearson 0.05592, Ledoit–Wolf 0.05647, OAS 0.05667).

## Decision

1. Most selected channels show non-zero variance, but the future windows remain only 6–10 frames for 96 variables.
2. Ledoit–Wolf and OAS improve numerical regularization and reduce absolute correlation differences, but do not materially improve rank agreement or high-threshold positive-edge overlap.
3. The weak cross-grasp agreement is **not resolved** by these standard shrinkage estimators.
4. Do not treat the six Weight grasp intervals as exchangeable Recording-level Support/Query samples for Reptile/ANIL. Do not add Weight to the main meta-learning training set or claim that it increases independent object-task count.
5. Retain this as a repeated-grasp variability / label-feasibility result. A future pooled object-level target would be a different estimand and requires a separate frozen protocol.

## Statistical boundaries

The physical object is the inferential unit (n=26). Grasp intervals, partitions, edges, and frames are nested observations. This is a measurement/label-quality audit, not confirmatory model-performance evidence. No model was trained; therefore a model training trace is not applicable. Execution logs, configuration, hashes, and diagnostic tables are retained in the run bundle.

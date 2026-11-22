# Stage 16 protocol: object-level cross-grasp relation aggregation

**Protocol freeze:** 2026-11-22  
**Dataset:** STAG Weight, pressure-only `weights_lite`  
**Parent audit:** `docs/weight_variance_shrinkage_audit_protocol_20261122.md` and its completed run report  
**Purpose:** exploratory estimand/reliability analysis only; no predictive model training.

## 1. Research question

Can six repeated grasp intervals for one physical object be aggregated into a reproducible object-level sensor-relation target, despite low agreement between individual grasp-level future Pearson matrices?

This is a new estimand. It is not the original within-recording future-correlation prediction target used in the classification recordings. Any positive result must be described as cross-grasp object-level relation estimation, not as evidence that the original target is stable.

## 2. Data and preprocessing frozen before analysis

- Use the exact Weight archive already audited; verify its SHA-256 against the run manifest before execution.
- Pressure-only: use pressure array and metadata fields needed for object, grasp index, valid frames and timestamp. No video, images, pose, or other modalities.
- Exclude `empty_hand`; include only complete `graspIndex=0..5`; exclude incomplete index 6.
- Use the same 96/548 sensor subset, selected with seed 20261116, and the same transform `clip((raw_pressure - 500)/150, 0, 1)`.
- Sort valid frames by timestamp. For the primary analysis, remove the first five valid frames within each grasp and use remaining future frames. The all-valid-frame analysis is secondary and must be labelled non-future.
- Each object has six grasp-level matrices; each grasp is a repeated measurement, not an independent object. The physical object is the outer statistical unit (n=26).
- Use all 4,560 undirected sensor edges and the same finite edge mask within each paired comparison.

## 3. Pre-specified aggregation estimators

For each object and each grasp, compute the Pearson correlation matrix across its future frames. Convert correlations to Fisher z after clipping to [-0.999999, 0.999999]. Construct object-level relation estimates by:

1. **Equal-grasp Fisher mean (primary):** arithmetic mean of the six grasp-level Fisher-z matrices, then inverse Fisher transform. Each grasp receives equal weight.
2. **Equal-grasp Fisher median (robust sensitivity):** edge-wise median of the six Fisher-z matrices, then inverse Fisher transform.
3. **Frame-count-weighted Fisher mean (sensitivity):** Fisher-z average weighted by each grasp's number of future frames.
4. **Pooled-frame Pearson (reference):** concatenate all six future windows and compute a single Pearson correlation matrix. This is included to compare with the pooled-frame approach already examined in Stage 15, not to select a preferred target.

Do not tune clipping, weights, sensor subset, or estimator based on observed agreement.

## 4. Reliability evaluation

### 4.1 Split-half cross-grasp reproducibility

For each object, enumerate the ten unique 3-grasp-vs-3-grasp partitions (complementary duplicates removed by requiring grasp 0 in group A). Apply each aggregation method independently to each side. Compare the two side estimates using:
- edge Spearman and Pearson correlation;
- mean absolute edge difference;
- positive-edge Jaccard at r >= 0.5, 0.7, and 0.9;
- finite edge count and empty-union counts.

This yields 260 nested partition comparisons per aggregation method and window protocol. First average the ten partitions within each object. Bootstrap only the 26 object-level summaries (5,000 repetitions, seed 20261122). Do not treat partitions, edges, frames, or grasps as independent inferential units.

### 4.2 Leave-one-grasp-out influence

For each object, calculate the six leave-one-grasp-out object-level Fisher-mean matrices. Quantify how much the estimate changes when one grasp is omitted, using edge Spearman, mean absolute difference, and thresholded Jaccard against the all-six estimate. Summarize first within object and then across objects. This is an influence diagnostic, not an independent validation set.

### 4.3 Heterogeneity diagnostics

For each object, report across-grasp dispersion of edge-wise Fisher-z estimates (median edge SD and IQR across edges), plus the distribution of per-edge between-grasp variance. Do not interpret edge count as sample size. Compare object-level summaries descriptively.

### 4.4 Non-future sensitivity

Repeat the same calculations using all valid frames in each grasp. Clearly label this as non-future sensitivity and do not mix it into the primary result.

## 5. Interpretation and stopping rules

- If equal-grasp aggregation improves split-half reproducibility over the pooled-frame Pearson reference across objects, it may be retained as a candidate object-level target for a separately designed experiment.
- If all methods remain weak, conclude that the present six-grasp archive does not support a reproducible object-level relation target under this operational definition. Retain it as repeated-grasp variability evidence.
- A lower mean absolute difference without improved rank agreement/edge overlap is not sufficient evidence of a stable target.
- Leave-one-grasp-out influence is not an independent test and cannot establish generalization to new objects.
- Do not train Reptile, ANIL, MLP, or GNN in this stage. Do not reuse the current Meta-validation objects for model selection.
- Do not claim causal or physical ground-truth sensor synergy.

## 6. Reproducibility artifacts

Create a unique run directory and retain:
- `config.json`, `environment.txt`, `input_manifest.json`, `sensor_selection.json`;
- `grasp_level_correlations.npz` (compressed correlation matrices only; no raw pressure frames);
- `partition_metrics.csv`, `object_summary.csv`, `leave_one_grasp_out.csv`, `heterogeneity_by_object.csv`;
- `bootstrap_summary.json`, `summary.md`, timestamped `run.log`, and `manifest.sha256`;
- script and unit tests in the repository.

Preserve intermediate outputs; never overwrite a previous run. If execution fails, mark the run incomplete and retain logs. Raw pressure arrays must not be committed to the repository or public CI artifacts.

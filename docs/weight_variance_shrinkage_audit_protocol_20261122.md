# Stage 15 protocol: Weight pressure variance, coverage, and covariance-estimator sensitivity

**Protocol date:** 2026-11-22  
**Dataset:** STAG Weight, pressure-only `weights_lite`  
**Status:** protocol freeze; execution requires the original `weights_lite.zip` (or `metadata.mat`) to be present in the runner.  
**Parent audit:** `docs/weight_grasp_label_stability_20261119.md`

## 1. Purpose and decision question

Determine whether the weak repeated-grasp agreement reported in the parent audit is mainly associated with (a) sparse/uneven sensor activation and short effective windows, (b) instability of the sample covariance/Pearson estimator at 6–10 future frames per grasp, or (c) genuine grasp-to-grasp variation that remains after estimator sensitivity checks.

This is a label-quality and measurement-sensitivity audit only. It is not a predictive-model experiment and must not be used to select a model against the four-object Meta-validation set.

## 2. Frozen data boundary

- Use only the pressure matrix and metadata fields needed to identify object, grasp interval, timestamps, and `graspValid`.
- Do not load, inspect, derive, or use images, video, hand pose, or any non-pressure modality.
- Exclude `empty_hand`; use complete `graspIndex=0..5`; exclude incomplete tail `graspIndex=6`.
- Keep the existing 548 active physical sensor locations and the exact 96-node subset from sensor seed 20261116 for direct comparability with the parent audit. Do not reselect nodes based on results.
- Keep the original transform `clip((pressure - 500) / 150, 0, 1)`. Also report raw-pressure distribution diagnostics before transformation; do not refit a global normalization from all splits.
- Within each grasp interval, sort by timestamp; first five valid frames remain the early-input period and subsequent valid frames are the future target window. The primary estimator analysis uses only future frames. The full-window variant is sensitivity-only and must be labelled non-future.
- A grasp interval, not a frame or sensor edge, is the repeated-measure unit. A physical object is the outer statistical unit (n=26).

## 3. Part A — pressure variance and effective coverage

For each object × grasp interval × selected sensor, calculate:
- number of valid frames and number of future frames;
- raw and transformed mean, standard deviation, variance, minimum, maximum;
- fraction of frames above the predeclared contact/activity threshold (transformed pressure > 0.02);
- fraction of sensors with non-zero variance, and fraction with variance above 1e-6;
- effective sensor count using the participation ratio: `(sum(v))^2 / sum(v^2)`, where `v` is the non-negative per-sensor variance vector; return NA if all variances are zero;
- sensor coverage and variance summaries separately for each graspIndex and pooled by object.

Thresholds are descriptive diagnostics, not labels. Include sensitivity summaries at transformed pressure > 0.01 and > 0.05; do not choose a threshold based on correlation agreement.

Summarize first at object level, then across the 26 objects. Report medians, IQRs, ranges, and object-bootstrap 95% percentile intervals (5,000 resamples, seed 20261122). Do not treat the 156 grasp intervals as independent objects.

## 4. Part B — covariance and correlation estimator sensitivity

On each grasp's future pressure matrix (n=6–10 frames, p=96 sensors), estimate:
1. sample covariance / Pearson correlation (reference estimator);
2. Ledoit–Wolf shrinkage covariance, converted to correlation using its diagonal;
3. Oracle Approximating Shrinkage (OAS), converted to correlation using its diagonal.

Use the same finite sensor set for all three estimators within a grasp. Record estimator warnings/failures, covariance condition number, shrinkage intensity (where defined), and the number of finite off-diagonal edges. Do not silently replace failed estimates.

For every estimator, repeat the existing ten unique 3-grasp-versus-3-grasp partitions per object. For each side, concatenate only the future frames from the three complete grasp intervals, estimate the covariance/correlation on the pooled frames, and compare the two sides using:
- Spearman correlation over the same finite upper-triangle edges;
- Pearson correlation over edge values as a secondary descriptor;
- mean absolute edge-correlation difference;
- positive-edge Jaccard at r >= 0.5, 0.7, and 0.9 (record undefined empty-union cases explicitly).

Primary comparison is whether Ledoit–Wolf/OAS improve the *within-object cross-partition agreement* over sample Pearson, not whether they maximize agreement on a pooled full-data target. Summarize ten splits within each object first; bootstrap the 26 object-level summaries. Do not use individual edges, frames, or the 260 nested partitions as independent inferential units.

Also report the full-valid-window variant as a clearly secondary, non-future sensitivity check. Do not tune shrinkage intensity manually or select an estimator separately for each object.

## 5. Part C — interpretation and stopping rules

- If variance/coverage is poor and all estimators remain unstable, classify Weight as unsuitable for the current future-correlation target; retain it as a repeated-grasp variability/feasibility result.
- If shrinkage improves cross-partition agreement consistently across objects, report it as estimator sensitivity only. It does not by itself establish that the resulting target is the same construct as the classification Recording target.
- If agreement remains low despite adequate coverage and shrinkage, interpret this as evidence that grasp-to-grasp relation variation and/or unmodelled acquisition variation remain substantial. Do not claim a causal mechanism.
- Do not add Weight to Reptile/ANIL training in this stage. Any later use requires a separate, frozen target definition and object-level validation protocol.
- Do not run another broad hyperparameter search on the existing Meta-validation objects.

## 6. Required provenance and retained artifacts

Each execution must create a unique run directory, for example `results/data_expansion/weight_variance_shrinkage_YYYYMMDD_run01/`, and retain:

- `config.json`: every threshold, estimator, sensor list/seed, window rule, bootstrap count/seed, and command-line argument;
- `environment.txt`: Python and package versions, OS, CPU information where available;
- `input_manifest.json`: input filenames, byte sizes, SHA-256 hashes, ZIP integrity result, metadata schema and counts; never include raw pressure frames;
- `sensor_selection.json`: exact selected physical sensor coordinates/indices and a hash;
- `variance_by_grasp.csv`, `coverage_by_object.csv`, `estimator_diagnostics.csv`;
- `partition_metrics.csv`: one row per object × split × estimator × window protocol;
- `object_summary.csv` and `bootstrap_summary.json`;
- `training_trace.csv` is not applicable because no model is trained; instead retain `run.log` with timestamped stage starts, counts, warnings, exceptions, and completion status;
- `summary.md`: concise results, limitations, and decision;
- `manifest.sha256`: hashes of every output file except the manifest itself.

Write intermediate CSV/JSON outputs incrementally and atomically where possible. On interruption, preserve completed diagnostics and mark the run `incomplete`; do not overwrite an earlier run directory. Do not upload original pressure arrays or raw frames to the repository or public CI artifacts.

## 7. Reproduction

The audit runner should accept explicit `--metadata`, `--output-dir`, and `--bootstrap-reps` arguments; default bootstrap seed is 20261122. It must validate required fields, sensor-mask compatibility, grasp counts, timestamp order, and future-window minimum before computing metrics. A dry-run/schema-only mode must be available.

No numerical result is considered produced until the script has run against the actual Weight metadata and emitted the provenance files above.

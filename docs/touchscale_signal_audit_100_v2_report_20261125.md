# TouchScale 100-Episode Raw Tactile Signal Audit v2

Date: 2026-11-25  
Branch: `thesis-few-shot-structure-20261125`  
Workflow: [Run 45598661542](https://github.com/inbea3/touch/actions/runs/45598661542)  
Artifact: `touchscale-signal-audit-100-v2`

## 1. Execution integrity

- 100/100 selected HDF5 episodes found after correcting one malformed episode UUID in the original sample manifest.
- The corrected record is `88430eb1-984f-4a5f-926e-7893708db835`; the prior manifest contained the invalid value `88430eb1-984f-4a5f-9260-244a51b4badc`.
- 100/100 episode JSON metadata files found.
- 200/200 hand-level signal records processed.
- Every hand record contains 15 pads and 880 taxels; finite-value fraction is 1.0.

## 2. Timestamp and sampling-rate audit

The first audit estimated rate as the reciprocal of the median inter-sample interval. This is not a reliable estimate when intervals are irregular: median `dt` coefficient of variation is about 0.299.

Use the effective rate:
`(N - 1) / (t_last - t_first)`

Across 200 hand records:
- Median effective rate: 56.91 Hz.
- Median metadata `glove_rate_hz`: 57.0 Hz.
- Median absolute effective-rate error: 0.099 Hz.
- Mean absolute effective-rate error: 2.35 Hz.
- 43/200 hand records differ by more than 2 Hz and require further review.
- All 200 hand timestamp sequences are monotonic.
- No inter-sample gap exceeds 0.25 s; maximum observed gap is 0.0778 s.

Decision: preserve original timestamps for all future processing. Do not infer a single global sampling rate or use reciprocal median `dt` as the nominal rate. Use timestamp-aware resampling only when a downstream model requires a common grid.

## 3. Contact-fraction cross-check

The official dataset card documents per-taxel calibrated normal force in newtons and defines the metadata contact fraction using summed normal force greater than 3 N. The v2 script reproduces this definition by summing the 880 taxel values per sample and thresholding at 3 N.

Across 200 hand records:
- Pearson correlation between recomputed and metadata contact fractions: 0.9978.
- Mean absolute difference: 0.0405 (4.05 percentage points).
- Median recomputed contact fraction: 0.4226.
- Median contact fraction at 1 N: 0.5077.
- Median contact fraction at 5 N: 0.3759.

The very high correlation supports consistency in relative contact ranking, but the absolute differences are non-negligible. Use the recomputed raw-signal measure for analysis and retain the metadata value as an independent QC cross-check. Investigate per-episode differences before applying strict contact eligibility cutoffs.

## 4. Within-episode spatial structure repeatability

Exploratory split-half correlation of upper-triangular spatial correlation-matrix entries:
- Left hand median: 0.4575.
- Right hand median: 0.3591.

This measures within-episode temporal repeatability only. It is not cross-object generalization, adaptation performance, or evidence that Reptile/GNN is effective.

## 5. Corrected conclusions

1. The 3 N contact threshold is supported by the official dataset documentation; it is not an arbitrary uncalibrated threshold.
2. The apparent large sampling-rate mismatch from reciprocal median `dt` was substantially caused by irregular inter-sample intervals. Effective-rate estimates align much better with metadata for most records.
3. The 100-episode signal pipeline is operational and the corrected manifest has 100/100 files.
4. A minority of hand records still have >2 Hz effective-rate discrepancy; keep these flagged for follow-up.
5. Contact-fraction recomputation is highly correlated with metadata but has an average absolute discrepancy of about 4 percentage points.
6. The split-half structure statistic is exploratory and cannot yet justify the thesis prediction target.

## 6. Next gate

The 300-episode stratified audit is being run across 15 shards and all 22 scene types. It will repeat the timestamp, contact-fraction, threshold-sensitivity, and split-half diagnostics at larger scale. No model training is authorized until the expanded audit confirms signal validity and a defensible task/episode split is defined.

## Sources

- TouchScale dataset card: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale
- 100-episode v2 workflow: https://github.com/inbea3/touch/actions/runs/45598661542

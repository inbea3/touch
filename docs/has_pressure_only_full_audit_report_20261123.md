# HAS pressure-only external audit — Full run report

- Workflow run: 45108789983
- Run URL: https://github.com/inbea3/touch/actions/runs/45108789983
- Artifact: has-pressure-only-audit-45108789983
- Artifact ID: 14788847778
- Artifact SHA-256: 8fddc1b16f486c495314586a6fc8c3631aa569b43cf978a2c03ba1ec2921e3d2
- Workflow commit: 0f05ea11bceb9a19528fcea42e9e86258eeb1693
- Dataset: TUM-ICS/Hide-and-Seek
- Dataset revision: Hugging Face default at run time (not pinned)
- Run completed: 2026-11-23T04:02:45Z
- Runtime: 401.2 seconds
- Status: success; full split streaming audit, no model training

## 1. Data volume

| Split | Frames | Episodes |
|---|---:|---:|
| train | 777,169 | 2,991 |
| validation | 221,195 | 860 |
| test | 121,752 | 463 |
| Total | 1,120,116 | 4,314 |

There are 33 distinct physical object categories represented by the label mapping. Episode length: median 235 frames; mean 259.5 frames; range 64–850 frames. Episode duration median 2.34 s; mean 2.586 s; range 0.63–8.49 s. Frame timestamps were strictly increasing in every episode, with no duplicate timestamps; observed adjacent interval was approximately 0.01 s.

## 2. Critical split leakage for object-generalization

All 33 physical object categories occur in all three official splits (train, validation and test). The object split audit therefore flags 33/33 categories as cross-split.

**Decision:** official train/validation/test splits are episode-level splits, not object-disjoint splits. They must not be used as evidence of unseen-object generalization. Before any cross-object claim, construct a new object-grouped split manifest, grouping every weight variant of the same physical object category together. Keep the official split only for reproducing the dataset's episode-level benchmark, with that limitation explicit.

## 3. Label consistency and time quality

- Inconsistent-label episodes: 0/4,314.
- Episodes with non-increasing timestamps: 0.
- Episodes with duplicate timestamps: 0.
- Bad 44-D force frames: 0.
- All audited force vectors had finite values and expected dimension 44.

The class label is constant within each episode under this audit. This is suitable for episode-level object/weight metadata, but does not establish within-episode temporal label transitions.

## 4. 44-channel quality

- Two channels are exactly constant zero over all 1,120,116 frames: channels 18 and 25 (zero-based indices).
- 13/44 channels have zero fraction >= 0.99.
- Mean zero fraction across channels: 0.89854; median: 0.93952; minimum: 0.64168.
- Channel 8 has the largest mean and standard deviation (mean 0.12777; SD 0.32952); channel 17 is almost always zero (zero fraction 0.999961), and channel 30 is effectively always zero (0.999999).

**Decision:** raw 44-D force data are finite and structurally usable, but highly sparse. Remove exactly constant channels for primary modelling, and predefine sensitivity checks for near-dead channels. Do not infer that all 44 channels are equally informative.

## 5. Research suitability

The dataset has a promising episode volume and 33 physical object categories, with clean timestamps and within-episode label consistency. However, official splits do not isolate physical object categories. A custom object-grouped split is mandatory for the thesis's cross-object generalization question.

This audit only inspected the approved fields `obs_skin_force_left`, `timestamp`, `episode_index`, and `classification_label`. It did not inspect other modalities, train a model, or establish that HAS's 44-channel tactile skin is physically equivalent to STAG's 548 pressure sensors.

## 6. Go / No-Go

- **GO** for a separate HAS pressure-only feasibility and cross-object benchmark, after custom object-grouped split construction.
- **NO-GO** for using official splits to claim unseen-object generalization.
- **NO-GO** for directly merging HAS channels with STAG channels or pooling the datasets without a domain-specific model and separate evaluation.
- **Next required experiment:** build object-grouped train/validation/test split manifest over 33 object categories, grouping weight variants by base object; inspect per-split class coverage and episode counts; then evaluate simple 44-channel baselines before any GNN/meta-learning.

## 7. Reproducibility

The GitHub Actions Artifact contains config, environment, run.log, episode_inventory.csv, episode_object_labels.csv, object_split_audit.csv, channel_quality.csv, label_consistency.csv, summary.json, summary.md and manifest.sha256. Raw force vectors were not written to the artifact.

Artifact digest: `sha256:8fddc1b16f486c495314586a6fc8c3631aa569b43cf978a2c03ba1ec2921e3d2`.

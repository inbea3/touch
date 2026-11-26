# HAS Stage 19 — Pressure feature and target specification (v1)

Date: 2026-11-23
Dataset: TUM-ICS/Hide-and-Seek, pressure-only field `obs_skin_force_left` (44 channels).
Locked object split: seed 20261123, 21/6/6 object groups; see `has_object_grouped_split_report_20261123.md`.

## 1. Purpose and scope

This stage creates episode-level derived features and future targets for a separate HAS benchmark. It does not merge HAS channels with STAG sensors and does not train a model. Only the pressure vector, timestamp, episode index and classification label are read. Point clouds, poses, RGB/video, distance/proximity and all other modalities are prohibited.

## 2. Split mapping

Train objects (21): bag, bluecube, bowl, chip, cleanser, cube, measuringcup, mustard, noodle, pitcher, ppbox, pringles, redcup, tape, teabox, toast, tomatosoup, udon, weinglass, wood, youcook.

Validation objects (6): cubemarsbox, foam, maxwell, papercup, ricecake, sugar.

Test objects (6): can, muesli, softlego, spam, toiletroll, tumbox.

All weight variants of a base object remain in its assigned split. The 122 negative episodes remain excluded from this object-conditioned task.

## 3. Episode segmentation and temporal protocol

An episode is keyed by (official source split, episode_index), matching the audited inventory. Sort frames by timestamp. Require exactly 44 finite pressure values and strictly increasing timestamps (the full audit found these properties for all frames).

- Early input: first 5 frames of an episode.
- Future target: all frames after the first 5, within the same episode.
- Episodes with fewer than 10 frames are rejected by the feature builder (the audit's observed minimum was 64, so this is a defensive check).
- Record the actual early-window duration and future duration. Five frames are approximately 50 ms only if the measured timestamp spacing supports it; do not describe as fixed 50 ms without verifying.

The five-frame cutoff is a frozen first-pass protocol, not an optimized choice. Any alternative windows must be registered as separate sensitivity experiments and selected using training/validation only.

## 4. Input features

For each channel independently, calculate from the first five frames:
1. mean;
2. standard deviation;
3. least-squares temporal slope against actual timestamps (centered/scaled time in seconds);
4. last observed value.

This yields 4 x 44 = 176 input features. No normalization is fitted in this stage. Feature scaling parameters must later be fitted on Train objects only and applied unchanged to Validation/Test.

## 5. Future prediction targets

Two complementary targets are retained, corresponding to the project's "both" target decision:

### T1 — Future channel state (44-dimensional regression)
For each channel, calculate future-window mean and standard deviation over all frames after the first five. These are separate 44-dimensional targets. Primary continuous target for the first baseline is future mean; future SD is a secondary target.

### T2 — Future inter-channel co-activation (44 x 44 correlation)
Compute Pearson correlation between each pair of the 44 channels using future-window frames only. Retain the 946 unique off-diagonal upper-triangle correlations. Also retain the auxiliary binary edge label I(r >= 0.7). The binary target is highly likely to be imbalanced and must be reported with prevalence, AP/PR-AUC and AUROC; accuracy is not primary. The 0.7 threshold is inherited as an exploratory threshold and must be sensitivity-checked (e.g. 0.5/0.7/0.9) using Train/Validation only before any final test evaluation.

These targets describe within-episode future statistical summaries. They are not frame-by-frame forecasting, physical contact ground truth, or causal sensor interactions.

## 6. Data quality and target diagnostics

For each episode save: object group, assigned split, official source split, episode index, label/weight variant, frame count, early/future duration, future-frame count, per-channel future variance and zero fraction, constant-channel count, finite target count, and positive-edge prevalence.

Global summaries must be computed separately by Train/Validation/Test. Report channel sparsity and target degeneracy. Exactly constant channels 18 and 25 from the earlier full-dataset audit are excluded from the primary target/model input candidate set; however, keep the original 44-D audit outputs and include all 44 channels in the first feature artifact so this exclusion can be implemented transparently in a later modelling config. Near-dead channels are not automatically removed; assess their utility on Train/Validation.

## 7. Required outputs and provenance

- `config.json`: dataset, revision, split seed and object assignment, first-five rule, feature/target schema, threshold.
- `run.log`: timestamped progress, rejected episodes, completion/failure.
- `episode_features_targets.npz`: derived features and targets only; no raw time-series.
- `episode_index.csv`: episode metadata and array row index.
- `channel_diagnostics.csv`: split-specific channel diagnostics.
- `summary.json` and `summary.md`: counts, target distributions and quality checks.
- `environment.txt`, `manifest.sha256`: environment and hashes.

The Hugging Face dataset revision must be pinned before a final thesis-grade training run. This first feature-construction run may use the current default revision only as an explicitly provisional artifact; record the resolved revision/commit if available. Do not overwrite prior run artifacts.

## 8. Analysis gates

Before model training:
1. Verify 4,192 object-conditioned episodes are present and no object crosses splits.
2. Verify each episode has 5 early and at least 5 future frames.
3. Check future target finite values and variance degeneracy.
4. Report split-wise positive-edge prevalence at r >= 0.7.
5. Confirm no target/feature computation uses future frames for inputs or first five frames for future targets.
6. Review channel diagnostics and decide whether channels 18/25 are excluded from the primary model input/target; retain an all-channel sensitivity version.
7. Only then freeze baseline model protocol.

No model is trained in Stage 19.

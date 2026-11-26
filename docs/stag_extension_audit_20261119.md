# STAG pressure-only extension audit: classification, blind-folded, weights, hand poses

**Audit date:** 2026-11-19  
**Scope:** metadata-only audit of four user-provided official STAG lite ZIPs. No video/image data were opened or used.  
**Status:** structural audit completed; no model training or performance experiment has been run on the extension datasets.

## 1. Executive findings

1. The three extension archives are readable and contain only `metadata.mat` plus README/note files. They contain no JPEG/video files.
2. All four archives use `int16` pressure arrays of shape `N × 32 × 32`. The observed variable-pressure pixel mask contains exactly 548 positions in each archive, and the four masks are identical. The remaining 476 positions are constant across frames. This agrees with the STAG README statement that 548 pixels correspond to physical glove sensors.
3. Blind-folded and weight datasets have exactly the same 26 non-empty object names as classification, plus `empty_hand`. Their object identities can therefore be joined by exact object name after verifying the mapping.
4. The extension datasets add repeated recordings / grasp episodes for existing objects, not new physical objects. The independent object-task count remains 26.
5. The blind-folded archive is explicitly a held-out test condition in its metadata: all 6,686 frames have `splitId=1`. Each of its 105 recordings has exactly five `hasValidLabel` frames (525 total). It should not be added to meta-training, and five valid frames are insufficient for a stable future sensor-pair correlation target under the current protocol.
6. The weight archive has 27 recordings (26 objects plus empty hand), one per object, and `graspIndex` segments. Indices 0–5 contain six main grasp intervals per recording; index 6 is a short trailing segment with zero `graspValid` frames. Across the full archive, 2,301 frames are marked `graspValid` (2,214 across physical objects; 87 empty-hand). Each main grasp index has only 14–15 valid frames per object, so single-grasp correlation estimates may be noisy. A grouped multi-grasp Support/Query design may be viable after label-stability tests.
7. The hand-pose archive labels `G1`–`G7` grasp types, not physical object identities. Its README note explicitly says each recording contains multiple different grasps and there is no one-to-one mapping between recording and object/pose class. It cannot increase object-level meta-task count.

## 2. Archive inventory and provenance

| Archive | ZIP bytes | SHA-256 |
|---|---:|---|
| classification_lite.zip | 38,397,353 | `ce0f27ae87e8ce2ec405f756dfe278b0b8011ef12e9de93e468b7ef31ad76c58` |
| blindfolded_lite.zip | 1,668,001 | `e9ef881085a722d51d7765f6b76aa9e26318f6a0a9d59e10cddd3a6ea14f0a40` |
| weights_lite.zip | 2,993,759 | `37b1c5fcb246383cdaed552ce07e35690883545cdfdbfc118b48c1cbd2ee3cf3` |
| handposes_lite.zip | 5,911,911 | `6babe1d8da0add1a0b8d539739b8473e5e59f925d675ea6a3c5dae1fdfb6ef84` |

The three extension ZIPs contain only:
- blind-folded: `metadata.mat`, `readme.html`;
- weights: `metadata.mat`, `readme.html`;
- hand poses: `metadata.mat`, `readme.html`, `note.txt`.

The note in hand poses and all three READMEs were read. They state non-commercial use only; retain attribution and consult the official STAG license before redistribution or any commercial use. The ZIPs are not copied into the Git repository.

## 3. Structural comparison

| Dataset | Frames | Recordings | Physical objects | Empty-hand | Validity field |
|---|---:|---:|---:|---:|---|
| Classification | 135,187 | 82 | 26 | 4 recordings | `hasValidLabel`, `isGrasp`, `isTransition` |
| Blind-folded | 6,686 | 105 | 26 | 1 recording | `hasValidLabel` |
| Weight | 11,682 | 27 | 26 | 1 recording | `graspValid`, `graspIndex`, `weights` |
| Hand poses | 24,037 | 7 | Not an object dataset | Pose labels include empty hand | `isGrasp` |

Classification has three recordings for each of the 26 physical objects plus four empty-hand recordings. Blind-folded has four recordings for each physical object plus one empty-hand recording. Weight has one continuous recording for each physical object plus one empty-hand recording.

### 3.1 Sensor layout and timestamps

For all four archives:
- pressure dtype: `int16`;
- pressure shape: `N × 32 × 32`;
- observed variable-pressure locations: 548;
- constant locations: 476;
- exact equality of the 548-position masks across all four archives;
- timestamps are non-decreasing within every recording.

Observed pressure value ranges:
- classification: 446–1000;
- blind-folded: 509–991;
- hand poses: 510–992;
- weight: 510–654.

The differing value ranges are a reason to retain train-only scaling and inspect calibration / baseline behavior; they are not a reason to concatenate raw frames blindly.

## 4. Dataset-specific audit

### 4.1 Blind-folded classification

- 6,686 pressure frames;
- 105 recordings: 26 × 4 object recordings + 1 empty-hand recording;
- 5 `hasValidLabel` frames per recording, 525 total;
- all frames have `splitId=1` (official test split);
- one batch/session in the archive;
- object names exactly match the 26 non-empty classification object names;
- README says this dataset tests generalization from visually aware training to blind-folded conditions.

**Decision:** do not use for meta-training or ordinary train/validation mixing. Preserve it as a held-out condition-shift evaluation resource. However, the five valid frames per recording are too few to estimate a stable future pressure-channel correlation matrix using the current STAG protocol. It may support descriptive pressure-distribution / object-recognition domain-shift analysis, but that is outside the current thesis target unless a separately justified protocol is developed.

### 4.2 Weight prediction

- 11,682 pressure frames;
- 27 recordings: one per 26 physical objects plus empty hand;
- 6 main grasp intervals per recording, `graspIndex=0..5`;
- trailing `graspIndex=6` has 128 frames total and zero `graspValid` frames;
- 2,301 `graspValid` frames total: 2,214 physical-object frames and 87 empty-hand frames;
- each of the six main grasp indices has 375–389 valid frames across the 27 recordings, corresponding to a median of 14–15 valid frames per object;
- README defines `graspValid` as the valid grasp segment between the fourth and sixth seconds of the timed grasp procedure;
- exact 26-object name match with classification.

**Decision:** this is the most promising same-hardware extension for within-object repeated-grasp analysis. Treat each valid grasp interval (indices 0–5) as a repeated episode nested within its object. Do not treat six grasps as six independent meta-tasks. Before modeling, test whether the future sensor-pair correlation labels are stable when estimated from 14–15 frames. If not, aggregate pre-specified groups of independent grasp intervals within Support and Query (e.g. three grasp intervals per side), then assess label stability. Do not choose this aggregation based on validation performance.

### 4.3 Hand poses

- 24,037 pressure frames;
- 7 continuous recordings;
- labels: `empty_hand`, `G1`–`G7`;
- 7,697 frames have `isGrasp=1`;
- split counts: 17,072 train and 6,965 test;
- one batch/session;
- README note explicitly states each recording contains multiple different grasps and there is no one-to-one recording-to-class mapping.

**Decision:** not suitable for object-level meta-learning task expansion. Retain as an optional pressure-only grasp-pose / grasp-state covariate analysis, only if it directly supports a defined confounding-factor question. Do not count G1–G7 as physical objects.

## 5. Implications for the thesis

- Independent physical object tasks remain 26. The original 18/4/4 object split is not enlarged by these archives.
- Weight adds six repeated grasp episodes per object, which can improve within-object Support/Query sampling and help study recording-level variability.
- Blind-folded adds four recordings per object in a distinct visual-awareness condition, but its official test-only split and five valid frames per recording make it unsuitable for current meta-training.
- Hand poses adds pressure observations under grasp-pose labels, not object tasks.
- The exact shared sensor mask supports reusing the STAG physical sensor index map. It does not by itself establish that raw calibration or pressure distributions are interchangeable; use train-only scaling and preserve condition indicators.

## 6. Recommended next experiments (not yet run)

1. **Weight label-stability audit:** calculate the future pressure-pair Pearson correlation matrices for each valid grasp interval, quantify stability across six repeated grasps within object, and compare 1-grasp versus 3-grasp pooled estimates. Use only pressure values and the existing fixed sensor subset / valid-sensor map.
2. **Recording-level split feasibility:** within each object, assign complete grasp intervals to Support and Query; never split frames from the same grasp across both. Ensure at least two independent grasp intervals per side. If the current early-five-frame / future-correlation target is too noisy, explicitly report the failure and revise the task only through a separately frozen protocol.
3. **Blind-folded condition-shift descriptive audit:** summarize per-recording valid-frame counts, pressure distributions and object coverage. Do not use its frames as training examples; do not claim correlation-target validation with five frames.
4. **Hand-pose confounding audit (optional):** only if needed, test whether pressure-derived features differ by G1–G7 on the official `isGrasp` subset. This is an auxiliary analysis, not an object-task expansion.
5. **Do not rerun Reptile/ANIL yet.** First establish label reliability and a leakage-safe object/episode split. The independent object count remains 26, so these STAG extensions alone do not resolve the meta-learning task-count limitation.

## 7. Explicit boundaries

- No video, RGB, images, depth, or other modalities were read or used.
- No external dataset was merged.
- No model was trained and no predictive performance result is claimed in this audit.
- No new independent physical object tasks were added.
- All raw ZIP files remain user-provided local inputs and are not committed to the repository.

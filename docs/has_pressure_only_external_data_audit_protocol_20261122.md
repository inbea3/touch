# Stage 17 protocol: external HAS pressure-only data audit

**Protocol freeze:** 2026-11-22  
**Dataset:** TUM-ICS Hide-and-Seek (HAS), Hugging Face `TUM-ICS/Hide-and-Seek`  
**Official code:** `tactile-hide-and-seek/Tactile-Hide-and-Seek`  
**Purpose:** metadata/schema/label feasibility audit before any model experiment.

## 1. Why this dataset and what it is not

The public dataset card describes 1,120,116 frame-wise examples, approximately 4.3k trajectories, 33 physical objects, 61 object-weight categories, and CC BY 4.0 licensing. The public schema includes a 44-dimensional `obs_skin_force_left` field, alongside other tactile and non-tactile fields.

HAS is a robot-mounted tactile-skin dataset, not the same sensor hardware or acquisition domain as human-hand STAG. It must be treated as a separate external dataset. Do not concatenate its 44 channels with STAG's 548/96 sensor coordinates, and do not claim hardware-level equivalence.

## 2. Strict pressure-only boundary

The audit may read only:
- `obs_skin_force_left` (44-dimensional skin-force vector);
- `timestamp`;
- `episode_index`;
- `classification_label` and the public label mapping, solely to identify object/weight labels and group repeated weight variants by physical object.

Do not load or use `obs_point_cloud_skin_contact`, `mask_point_cloud_skin_contact`, `obs_pose_left`, `obs_skin_prox_left`, `obs_skin_dist_left`, `obs_dami_force_left`, `obs_dami_prox_left`, `obs_ft_left`, or any other sensor/modality field. No video is part of the selected audit pipeline.

Exclude the `negative` class from object-level tasks. For each combined label (e.g. `bag_medium`), use the mapping to recover the physical object name and weight class. All weight variants of the same physical object must be grouped into the same outer object fold; never treat object-weight labels as independent physical objects.

## 3. First-pass audit questions

1. Verify the dataset repository revision, license metadata, split names, file inventory, and data hashes.
2. Verify exact feature schema and whether `obs_skin_force_left` is consistently a finite 44-vector.
3. Check timestamp monotonicity within each `episode_index`, duplicate timestamps, missing values, and episode lengths.
4. Verify that each episode maps to exactly one combined classification label; report any label changes within an episode.
5. Reconstruct physical object and weight class from the official mapping. Count distinct physical objects, object-weight variants, episodes, and frames by official split.
6. Check whether physical objects or object-weight labels occur across multiple official splits. Do not assume official splits are object-disjoint.
7. Report per-channel variance, zero-variance frequency, range, missingness, and effective channel participation using only `obs_skin_force_left`.
8. Check whether each object has enough independent episodes and enough within-episode frames to define an early-input/future-target sensor-pair correlation task. Do not randomly split frames across Support/Query.
9. Check whether grasp/push phases can be distinguished from available metadata without using pose, video, or other modalities. If phase cannot be reliably identified from allowed metadata, do not claim a grasp-only target; either analyze the full episode interaction signal as a separate construct or stop.

## 4. Candidate target — not yet approved

Only if the schema audit passes, consider a dataset-specific future force-channel relation target:
- group frames by complete `episode_index`;
- use timestamp order;
- define early Support frames and future Query frames using a predeclared temporal rule after inspecting frame-rate and episode-length distributions;
- compute continuous Pearson correlation across the 44 left skin-force channels on future frames;
- report the 946 undirected channel pairs;
- estimate label stability across independent episodes of the same physical object before any model training.

This is a separate HAS-specific estimand. It is not numerically or physically interchangeable with the STAG 96/548 sensor target. Use a separate scaler fitted only on the training object folds and a separate graph/channel geometry. Do not infer physical spatial adjacency unless the dataset provides validated sensor coordinates.

## 5. Statistical design if the audit passes

- Outer statistical unit: physical object (not object-weight class, frame, window, channel pair, or episode).
- Group all weight variants of the same object into one outer fold.
- Prefer object-level 5-fold CV if at least 25 usable non-negative physical objects have sufficient repeated episodes; otherwise use a predeclared leave-object-out or repeated grouped holdout design and report the reduced task count.
- Keep all episodes of a physical object in one outer fold.
- Fit preprocessing/scaling and any target-selection rule on training objects only.
- Report object-level metrics and object bootstrap intervals. Channel pairs are repeated measurements, not independent samples.
- Compare simple baselines before meta-learning: object-independent constant, pooled MLP zero-shot, and Support-matched MLP adaptation. Only then consider Reptile/ANIL, with identical episode schedule and compute budget.
- Do not use the official frame-level split as proof of object-level generalization.

## 6. Stop/go rules

Stop before model training if:
- object labels cannot be mapped unambiguously;
- episode labels vary within an episode without an explainable segmentation;
- object identity is not stable across weight variants;
- too few physical objects have at least two independent usable episodes;
- episode duration is insufficient for a predeclared early/future window;
- phase/target semantics cannot be defined without prohibited modalities.

If the audit passes, freeze a separate HAS target protocol before training. Keep all HAS results separate from STAG results and describe them as external-domain replication/transfer evidence, not pooled-sensor validation.

## 7. Reproducibility artifacts

Retain:
- dataset revision, license text/URL, file inventory and hashes;
- `config.json`, `environment.txt`, `run.log`;
- `episode_inventory.csv`, `object_split_audit.csv`, `channel_quality.csv`, `label_consistency.csv`;
- `summary.json`, `summary.md`, and `manifest.sha256`.

Do not retain unrelated modalities in the derived working dataset. If Parquet shards contain other columns, select only the four allowed fields before materializing a local analysis table. Never upload the source dataset or raw force arrays to the project repository.

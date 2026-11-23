# HAS Stage 27A — sensor-channel provenance and physical-layout audit

Date: 2026-11-24

## Objective
Determine whether the 42 retained dimensions in the Stage 22A pressure-only feature matrix can be assigned to physical tactile-skin cells and whether a defensible physical adjacency graph can be reconstructed from public dataset documentation and official baseline code.

## Sources inspected

### 1. Official Hugging Face dataset card
Dataset: `TUM-ICS/Hide-and-Seek` (CC-BY-4.0).
The card documents:
- `obs_skin_force_left`: length 44, described as left-hand tactile skin force readings.
- `obs_skin_prox_left` and `obs_skin_dist_left`: each length 44.
- `obs_point_cloud_skin_contact`: shape 88 × 6; described as tactile contact point cloud from robot skin, with 88 points corresponding to tactile skin cells across both robot hands. Each point contains spatial and sensor-derived channels.
- `mask_point_cloud_skin_contact`: length 88.
The public card does not specify the ordering of the 44 force-vector entries, a per-index cell identifier, a left-hand-only index-to-coordinate table, or a mapping between the 44-element left force vector and the 88 point-cloud rows.

Source: https://huggingface.co/datasets/TUM-ICS/Hide-and-Seek

### 2. Official project page and paper metadata
The project page describes tactile-only bimanual object search/retrieval and the HAS dataset, but the public page content inspected does not provide a machine-readable 44-channel index-to-cell coordinate map.
Paper metadata: Fu et al., “Tactile Hide and Seek: Bimanual Object Blind Search and Retrieval via Tactile-Only Feedback,” ICRA 2026, DOI 10.1109/ICRA57385.2026.11696735.
Source: https://tactile-hide-and-seek.github.io/

### 3. Official public code
Repository: https://github.com/tactile-hide-and-seek/Tactile-Hide-and-Seek
Inspected files:
- `has_dataset/hf_loader.py`: declares the 44-element left-hand force/proximity/distance vectors and 88×6 point cloud shapes.
- `has_dataset/window_dataset.py`: groups frame rows by episode and creates temporal windows; it does not define a sensor-index-to-coordinate map.
- `models/pointcloud_encoder.py`: treats the 88 points as a set, applying pointwise MLP and max/mean pooling; it does not expose a fixed row-to-left-force-index mapping.
- `models/dual_head_classifier.py`: concatenates the 44-element left proximity, force and distance vectors as skin features; it does not supply per-channel geometry.
The inspected public code therefore confirms data shapes and model use, but does not establish physical ordering or adjacency for the 44 force-vector entries.

## Stage 22A index semantics — important correction
The Stage 22A extraction code starts from a 44-element `obs_skin_force_left` vector and excludes zero-based indices 18 and 25. The retained 42 dimensions are then used consistently in the four feature blocks (mean, population SD, time slope, last value) and the 42-D future-mean target.

Thus:
- “retained channel 0..41” means position in the reduced 42-D vector;
- “source index 0..43 excluding 18 and 25” means original array position in the published 44-element vector;
- neither identifier is currently demonstrated to be a physical sensor-cell ID or a coordinate.
- The Stage 22A script hard-codes exclusion of indices 18 and 25; the reviewed extraction code does not document a hardware-level reason for these exclusions. Their exclusion rationale must be recovered or the sensitivity to retaining them must be evaluated before final claims.

## Evidence status

| Question | Status |
|---|---|
| Is the left force vector length 44? | Confirmed by official dataset card and loader |
| Is the point cloud 88×6? | Confirmed by official dataset card and loader |
| Do 88 points represent cells across both hands? | Confirmed by dataset card |
| Does public metadata define ordering of the 44 left force entries? | Not found in inspected public sources |
| Is there a documented mapping from 44 force entries to 88 point-cloud rows? | Not found |
| Is there a documented coordinate per force-vector index? | Not found |
| Can a physically grounded adjacency graph be built now? | No |
| Is exclusion of original indices 18 and 25 justified in the Stage 22A script? | Not documented in inspected script |

“Not found” means not present in the sources and files inspected for this audit; it is not proof that the authors do not possess such metadata privately.

## Decision
1. Do not assign anatomical names, coordinates, or physical neighbors to any of the 42 retained dimensions.
2. Do not construct or describe the current correlation pairs as physical sensor connections.
3. Before a physically grounded GNN, obtain an official index-to-cell map or sensor-layout metadata from the dataset authors.
4. Independently investigate why source indices 18 and 25 were excluded. Preserve the current 42-channel branch as a frozen historical experiment; any revised 44-channel or alternative feature set must be a new run with a new artifact.
5. If the authors cannot provide the mapping, a correlation-derived graph may only be tested as an explicitly exploratory statistical graph, estimated from Train objects only, with graph construction frozen before Validation evaluation and compared against non-graph baselines under the same split.

## Suggested author clarification request
Ask the dataset authors for:
- the ordering convention for `obs_skin_force_left[0:44]`;
- the corresponding sensor-cell IDs and 3D/2D coordinates in a named frame;
- the exact relation, if any, between these 44 entries and rows in `obs_point_cloud_skin_contact[0:88]`;
- whether channels 18 and 25 are invalid, unused, duplicated, or otherwise special;
- whether a sensor-layout/URDF/TF or calibration file can be cited and redistributed under the dataset license.

No author contact was sent as part of this audit.

## Reproducibility
This is a source/code provenance audit, not a model experiment. It retains the exact source URLs, inspected code paths, decisions, and unresolved questions. No data or model artifact was modified.

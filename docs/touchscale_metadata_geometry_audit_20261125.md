# TouchScale Metadata and Geometry Audit — 2026-11-25

## Status

- Metadata access: PASS.
- Episode table integrity: PASS.
- Hand geometry integrity: PASS.
- Object-level few-shot task construction: NOT YET PASSED.
- Raw tactile shard download/training: NOT performed.

Workflow: [GitHub Actions run 45588023843](https://github.com/inbea3/touch/actions/runs/45588023843)  
Branch: `thesis-few-shot-structure-20261125`  
Artifact: `touchscale-metadata-audit` (downloaded and inspected).

## 1. Episode metadata

| Audit item | Result |
|---|---:|
| Episode rows | 15,324 |
| Unique episode IDs | 15,324 |
| Duplicate episode IDs | 0 |
| Task IDs | 953 |
| Participants | 24 |
| Scene types | 22 |
| Median episode duration | 21.87 s |
| Duration range | 5.30–118.43 s |
| Episodes with empty `items` arrays | 4,045 (26.4%) |
| Episodes with one item label | 131 |
| Episodes with multiple item labels | 11,148 (72.7%) |
| Distinct item labels | 880 |
| Distinct exact item bundles | 821 |

Important: `items` is an array of objects involved in an episode, not a canonical primary-object ID. It often contains the manipulated object plus tools, containers, surfaces, materials, or props. Therefore 880 distinct labels and their frequency counts are not equivalent to 880 independent object classes or physical instances.

The 193-row mesh registry is partial. A normalized exact-name comparison found only 66 overlaps between its object names and the 880 episode item labels. The data card also states mesh names do not always match episode `items`.

## 2. Repetition is not yet a valid K-shot count

Individual item-label frequency screens show many repeated mentions, but they cannot yet establish few-shot object eligibility:
- Some labels are common background/support items (e.g. anti-slip mats and trays).
- The same label may refer to different physical instances.
- Multiple items occur in the same episode.
- Repeated episodes can share actor, task, scene, date, or collection context.
- No canonical physical instance ID is supplied for every episode item.

Consequently, no 1/2/5/10-shot object eligibility count is approved at this stage. Any raw label frequency threshold is descriptive only.

## 3. Tactile geometry

Both official geometry files were decoded and validated.

| Geometry check | Left | Right |
|---|---:|---:|
| Taxels | 880 | 880 |
| Regions/pads | 15 | 15 |
| Coordinate columns | 9 | 9 |
| Finite XYZ coordinates | Yes | Yes |
| Duplicate XYZ rows | 0 | 0 |
| Region cell total | 880 | 880 |

Each taxel provides position `(x,y,z)` and two local direction vectors. The official note says points are in a flat palmar layout and ordered by region then row-major within each region, matching `tactile_<id>` arrays.

This is sufficient to construct a geometry-aware graph, subject to validating pad boundaries and cross-region distances. It does not justify flattening all pads into a naive 2-D kNN graph without anatomical constraints.

## 4. Contact and timing

- Median left contact fraction: 0.2978.
- Median right contact fraction: 0.3302.
- Contact fractions vary substantially across episodes.
- Per-episode tactile sampling rates vary; the raw timestamps must be used rather than assuming a fixed frame rate.
- Median episode duration: 21.87 s.

A valid future-window association target still requires inspection of raw tactile arrays for a small, carefully selected episode sample. Metadata contact fractions alone do not define stable contact intervals or guarantee enough samples for a future Pearson matrix.

## 5. Data integrity hashes

The audit script records SHA-256 hashes for:
- `episodes.parquet`
- `objects.parquet`
- `glove_geometry_left.json`
- `glove_geometry_right.json`

The exact values are retained in `audit_summary.json` in the workflow artifact.

## 6. Decision

**GO for a small semantic/episode audit; NO-GO for model training at present.**

The sensor geometry is usable. The unresolved bottleneck is the experimental unit: which object in an episode is the target object, and how can the same object category/instance be matched across independent episodes without mixing props or unrelated objects?

## 7. Immediate next action

Conduct a stratified manual review of approximately 200 episodes, sampled across scenes, task IDs, participants, item-count groups, and repeated item labels.

For each episode, annotate:
1. primary manipulated object (if unambiguous);
2. secondary/support objects;
3. whether the primary object label is reusable across episodes;
4. whether the episode contains a single continuous interaction or multiple distinct object interactions;
5. whether the primary hand and contact fractions support a consistent single-hand analysis.

Then estimate:
- the number of reusable primary-object labels;
- episodes per primary-object label;
- number of distinct participants/tasks/dates per label;
- candidate 1-/2-shot support-query feasibility;
- inter-reviewer agreement if a second reviewer is available.

Only after this audit should the team decide whether to keep object-level adaptation or narrow the task to task-family/interaction-level adaptation. Do not download raw shards or start TouchScale training before that decision.

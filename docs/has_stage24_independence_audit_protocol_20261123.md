# HAS Stage 24A — trajectory-level independence audit

## Purpose
Audit the Stage 23 Reptile evaluation split at the granularity supported by the published HAS dataset schema. The upstream schema identifies `episode_index` as an episode/trajectory identifier. It does not publish a separate recording-session identifier in the frame-level columns.

## Frozen inputs
- Stage 22A window dataset artifact: run 45193420854.
- Stage 23 Reptile artifact: run 45201237382 (5-frame primary).
- Trajectory key: `(official_split, episode_index)`.

## Checks
1. 4,192 unique trajectory keys map one-to-one to one object group and one grouped split.
2. Physical object groups remain disjoint across train, validation and test.
3. For every Stage 23 evaluation row, Support count equals requested 1/5/10 shots.
4. Support and Query trajectory keys are unique and disjoint.
5. Every Support/Query key maps to the same object and grouped split as the evaluation row.
6. For a given split/object/seed/repeat/shot, Reptile and pooled MLP use identical Support and Query keys.

## Scope and limitations
This establishes trajectory-level separation, not higher-level recording-session independence. No recording-session ID is exposed by the published frame schema, so no such claim is made. Episode-level separation does not remove possible temporal or collection-condition correlations across trajectories. Test results remain exploratory because the fixed test objects were inspected during prior development.

## Outputs
- `summary.json`
- `trajectory_index_audit.csv`
- `support_query_audit.csv`
- `config.json`
- `run.log`
- `environment.txt`
- `manifest.sha256`

The audit performs no model training and does not alter existing artifacts.

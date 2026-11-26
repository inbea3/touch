# TouchScale LOTO / LOAO split feasibility results（2026-11-26）

## Run

- Workflow: [TouchScale LOTO and LOAO split feasibility](https://github.com/inbea3/touch/actions/runs/45795358966)
- Run ID: `45795358966`
- Result: success
- Artifact: `touchscale-split-feasibility`, ID `15104861509`
- Script: `scripts/build_touchscale_split_feasibility.py`
- Workflow: `.github/workflows/touchscale-split-feasibility.yml`
- Data access: metadata-only; no HDF5 shard or video downloaded.

## Metadata census

- 15,324 unique Episodes; 920 task templates keyed by `(scene, task_name)`; 24 actors.
- 0 invalid rows for required grouping fields.
- Actor count by task template: 862 templates have 1 actor, 56 have 2 actors, 2 have 3 actors.
- At least two actors each have at least 2 / 3 / 6 Episodes in 50 / 44 / 28 task templates respectively.

## Generated LOTO manifests

LOTO holds out the entire task template. Actors may appear in other training task templates, so this is **not** unseen-participant evaluation.

| Shot | Eligible held-out task templates | Manifest rows (3 seeds each) |
|---|---:|---:|
| 1-shot | 821 | 2,463 |
| 2-shot | 807 | 2,421 |
| 5-shot | 791 | 2,373 |

## Generated LOAO manifests

LOAO holds out all Episodes from one actor. A test actor-task unit is eligible only if the held-out actor has at least shot+1 Episodes and another actor has at least 2 Episodes for the same task template. Support/Query are sampled without replacement from complete Episode IDs.

| Shot | Eligible held-out actor-task units | Manifest rows (3 seeds each) | Actors with eligible units |
|---|---:|---:|---:|
| 1-shot | 101 | 303 | 20 |
| 2-shot | 95 | 285 | 19 |
| 5-shot | 78 | 234 | 16 |

Four actors have no eligible shared task-template unit under the current rule: actor_05, actor_10, actor_21, actor_24. Coverage is imbalanced; several actor holdouts contribute only one or two eligible task templates. The LOAO manifest is therefore a feasibility inventory, not a balanced evaluation result.

## Interpretation

- The metadata supports a potentially useful **new-participant adaptation on previously observed interaction task templates** evaluation.
- The metadata also supports a large **new task-template adaptation** candidate pool, but LOTO does not isolate participants.
- Neither split establishes unseen physical object-instance adaptation. The dataset does not provide a reliable episode-level physical object instance ID for every involved item.
- Candidate split feasibility does not establish signal quality. The selected Episodes still require HDF5 signal QC.
- No model has been trained in this workflow.

## Artifact contents

- `summary.json`
- `task_template_census.csv`
- `actor_task_coverage.csv`
- `loto_support_query_manifest.csv`
- `loao_support_query_manifest.csv`
- `loao_actor_summary.csv`
- `invalid_metadata_rows.csv`

The artifact contains metadata identifiers and split manifests, not raw pressure arrays.

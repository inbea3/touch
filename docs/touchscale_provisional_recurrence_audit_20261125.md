# TouchScale provisional recurrence audit — 2026-11-25

## Scope

This audit joins the 100-row metadata sample to the model-assisted desk coding V1 workbook and checks repeated provisional target labels and repeated actor-task records. It is a descriptive audit of this sample only. It is not human-verified annotation and is not a dataset-wide object census.

## Results

- Sample size: 100 episodes.
- Provisional target labels recurring in more than one row: 15.
- Maximum recurrence for any provisional target label in this sample: 3 episodes.
- Repeated actor-task groups (same actor_id + task_id + task_name with >1 episode): 6.
- No provisional label in this sample has enough rows to establish a 5-shot Support set plus at least one Query episode (which would require at least 6 independent episodes); in fact, no label appears more than 3 times.
- The 100-row sample does not establish the full TouchScale dataset's K-shot feasibility.

## Repeated provisional target labels

| Provisional label | Episodes | Actors | Task IDs | Dates | Shards |
|---|---:|---:|---:|---:|---:|
| test tube | 3 | 2 | 2 | 2 | 3 |
| rubber stopper | 3 | 1 | 2 | 1 | 3 |
| marker cap | 2 | 2 | 2 | 2 | 2 |
| packing tape | 2 | 2 | 2 | 2 | 2 |
| book page | 2 | 1 | 1 | 1 | 2 |
| bubble mailer | 2 | 1 | 2 | 1 | 2 |
| cucumber | 2 | 1 | 1 | 1 | 2 |
| desk drawer | 2 | 1 | 2 | 2 | 2 |
| drink bottle cap / bottle | 2 | 1 | 1 | 1 | 2 |
| hook-and-loop cable strap | 2 | 1 | 2 | 1 | 2 |
| large hex nut | 2 | 1 | 1 | 1 | 2 |
| pillowcase | 2 | 1 | 2 | 2 | 2 |
| tea-corner pickup area | 2 | 1 | 1 | 1 | 2 |
| two cleaning cloths | 2 | 1 | 2 | 1 | 2 |
| white tray with props | 2 | 1 | 1 | 1 | 2 |

These are task-based provisional labels. Repeated names do not prove repeated physical instances, and some labels denote components, multi-object sets, or even scene/setup tasks. The recurrence table must not be interpreted as few-shot task eligibility.

## Repeated actor-task groups

Six actor/task combinations have multiple episodes in this 100-row sample. Keep all episodes from the same actor-task group together during any provisional split. This is only a first leakage screen; session-level and near-duplicate task checks are still required.

## Decision

1. The sample does not provide evidence for 5-shot object-level adaptation. It is too small and labels are provisional.
2. Do not infer 1-/2-shot eligibility from the recurrence table either; target identity and episode independence are not yet adjudicated.
3. Complete blind second annotation and adjudication before counting target recurrence.
4. If the blind review confirms only a small number of repeated target categories, run a dataset-wide metadata mapping only after a defensible primary-object normalization policy is agreed.
5. Inspect raw tactile signals for selected candidate episodes only after the object/episode unit is fixed.

## Artifact

A spreadsheet containing repeated provisional labels, repeated actor-task groups, and interpretation limits is generated as `touchscale_provisional_recurrence_audit.xlsx`.

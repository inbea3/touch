# TouchScale 100-Episode Desk Coding v1 — 2026-11-25

## Status and limitation

This is a model-assisted desk coding pass over the 100-episode stratified metadata sample using task names, instructions, and item lists. It is **not human annotation, not adjudicated ground truth, and not a validated object identity registry**. All labels are provisional hypotheses for reviewer confirmation. No raw tactile shards or video were inspected.

Workbook: `touchscale_100_episode_desk_coding_v1.xlsx` (generated locally; the accompanying review workbook contains the full 100 rows and editable review fields).

## Sample

- 100 episodes
- 73 task IDs
- 15 participants
- 36 collection dates
- 22 scenes

## Provisional target-level coding

| Target level | Count |
|---|---:|
| Whole object | 44 |
| Multi-object set | 18 |
| Component | 17 |
| Material/substance | 13 |
| Environment/surface | 8 |

## Provisional task-scope coding

| Task scope | Count |
|---|---:|
| Single object | 25 |
| Tool-mediated | 24 |
| Component action | 22 |
| Multi-object transfer/sort | 13 |
| Scene setup/reset | 9 |
| Other / underspecified | 7 |

## Provisional episode-quality triage

| Triage | Count |
|---|---:|
| Usable candidate for further signal review | 61 |
| Needs raw-signal/contact review | 29 |
| Exclude from object-level target pool pending task redesign | 10 |

These are desk-review triage counts only. In particular, “usable candidate” does not mean the episode passes contact quality, independent support/query, or reusable object identity checks.

## Main interpretation

1. Only 25/100 episodes look like straightforward single-object tasks under the current codebook.
2. Tool-mediated and component-action episodes are common. The annotation unit must distinguish the task target from the object physically grasped and from tools/fixtures.
3. Multi-object transfer/sort and setup/reset tasks should not be forced into a single-object target label.
4. Repeated task names (e.g. duplicated records) are not automatically independent support/query observations. Actor, date, scene, shard, and task context must be checked.
5. A category label recurring across episodes does not prove that the same physical object instance recurs.

## Decision

**Continue with object-level feasibility analysis only as a conditional path. Do not train yet.**

The next review should prioritize:
- all repeated task/label groups;
- all low-confidence or multi-object cases;
- a second independent reviewer on at least 20–30 rows;
- adjudication of target-level and task-scope disagreements;
- raw signal inspection only after target labels are stable.

If reviewer agreement is poor or reusable target identity is sparse, revise the adaptation unit to task-family/interaction-family rather than asserting cross-object few-shot adaptation.

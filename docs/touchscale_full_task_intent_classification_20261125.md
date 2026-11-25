# TouchScale Full Task-Intent Classification — 2026-11-25

## Scope and source

- Source: `metadata/episode_metadata_review.csv` from successful metadata-only audit artifact.
- Records classified: 15,324 episodes.
- Classification unit: 920 distinct `task_name` templates (with scene retained).
- Method: deterministic/model-assisted rule coding from `task_name` and full `instruction`; 37 residual templates were manually assigned to a broad interaction family after inspecting their titles and instructions.
- No raw tactile shards were downloaded for this task-intent pass.

## 1. Task-family counts

| Family | Episodes | Distinct task templates |
|---|---:|---:|
| Transfer / pouring / dispensing | 3,547 | 205 |
| Insertion / assembly / component operation | 2,973 | 155 |
| Scene setup / organization / reset | 2,425 | 187 |
| Material processing / deformation | 2,180 | 113 |
| Single-object grasp / transport / pose manipulation | 1,495 | 95 |
| Surface / tool cleaning | 1,358 | 86 |
| Inspection / trial / comparison | 1,346 | 79 |
| **Total** | **15,324** | **920** |

This classification is task-intent, not object identity. It should not be described as human-verified annotation or ground truth.

## 2. Correct experimental unit warning

The metadata's `task_id` is not globally unique: one numeric task_id maps to multiple task names. Group by `scene + task_name` for this task-template audit.

Across 920 scene/task templates:
- Median episodes per template: 18.
- 821 templates have at least 2 episodes; 807 have at least 3; 791 have at least 6.
- But 862/920 templates have episodes from only one actor; 56 have two actors; only 2 have three actors.
- Thus only 58 templates have at least 2 distinct actors; only 2 have at least 3 distinct actors; none have at least 6 distinct actors.

Episode count is not equivalent to independent few-shot count. Repeated trials within a single actor/task are correlated and must not be split naively between Support and Query. These data do not support a general participant-independent 5-shot claim at the task-template level.

## 3. Primary-object decision

The task descriptions often specify a relation, not a single object:
- Transfer tasks require source, destination/container, and transferred material.
- Assembly tasks require component, parent object, and mating part.
- Cleaning tasks require tool and contacted surface.
- Scene setup tasks involve several objects and spatial arrangement.
- Inspection tasks require a tested object and the tool/procedure.
- Material processing tasks involve material/object plus tool or support surface.

Therefore the correct annotation unit is initially an **interaction tuple**, not a single free-text object label:
`(primary manipulated entity, counterpart/source/destination, tool, contact surface, task phase)`.

Do not infer physical instance identity from a repeated noun or task template. The metadata does not provide a canonical physical object-instance ID for every episode.

## 4. Research route decision

**NO-GO for claiming TouchScale currently validates few-shot adaptation to unseen physical objects.**

**GO for continuing a task/interaction-structure audit.** The next defensible candidate is few-shot adaptation of tactile interaction structure across repeated task templates or interaction families, but the actor concentration is a major limitation. A participant-held-out or actor-disjoint Support/Query evaluation is feasible only for the small subset with sufficient distinct actors; the number of eligible tasks must be stated.

The thesis can retain the broad research question of adapting tactile structure, but must not silently relabel task adaptation as unseen-object adaptation. The estimand needs a formal decision after signal-level feasibility checks.

## 5. Immediate next actions

1. Inspect raw tactile arrays for a small stratified set of task families to verify timestamps, contact windows, usable force channels, and target construction.
2. Build an interaction-tuple annotation protocol for a small sample, including ambiguous multi-object tasks.
3. Quantify actor-disjoint eligible task counts and report the small eligible subset transparently.
4. Compare within-actor episode adaptation only as a separate, weaker setting; do not mix it with actor-disjoint generalization.
5. Freeze the target and Support/Query protocol before model training.

## 6. Deliverables generated

- `touchscale_task_catalog_920_v1.xlsx`: all 920 task templates, family, interaction scope, episode/actor/date counts, and example instruction.
- `touchscale_all_15324_task_classification_v1.csv`: all 15,324 episodes with task-family labels and source metadata fields.

These local artifacts are delivered in the current conversation; the repository report documents their derivation and limitations.

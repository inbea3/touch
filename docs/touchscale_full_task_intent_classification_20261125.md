# TouchScale Full Task-Intent Classification — Corrected V1 (2026-11-25)

## Scope and source
- Source: successful metadata-only audit artifact, `episode_metadata_review.csv`.
- Classified all 15,324 episode rows and 920 distinct task-name templates.
- Classification unit: scene + task_name. Numeric `task_id` values collide across different task names in the inspected export; do not use task_id alone as the grouping key.
- Method: rule-assisted coding from task title and full instruction; 37 residual templates were manually assigned to broad interaction families after direct inspection. The earlier false-positive rule that treated the noun phrase “test tube” as an inspection action was corrected.
- This is task-intent classification, not human-verified object annotation or canonical object-instance identification.

## 1. Corrected family counts

| Family | Episodes | Task templates |
|---|---:|---:|
| Transfer / pouring / dispensing | 4,127 | 235 |
| Insertion / assembly / component operation | 3,276 | 172 |
| Scene setup / organization / reset | 2,425 | 187 |
| Material processing / deformation | 2,180 | 113 |
| Single-object grasp / transport / pose manipulation | 1,683 | 109 |
| Surface / tool cleaning | 1,429 | 89 |
| Inspection / trial / comparison | 204 | 15 |
| **Total** | **15,324** | **920** |

## 2. Task repetition is not independent few-shot support

Across 920 scene/task templates:
- Median episode count per template: 18.
- 821 templates have at least 2 episodes; 807 have at least 3; 791 have at least 6.
- Actor distribution: 862 templates have 1 actor, 56 have 2 actors, and 2 have 3 actors.
- Thus only 58 templates have at least 2 distinct actors, 2 have at least 3 distinct actors, and none have at least 6 distinct actors.

Repeated episodes for the same actor and task may be correlated. The large episode counts do not justify treating the repeated recordings as independent support/query shots. A participant-independent 5-shot task-template evaluation is not supported by this release metadata.

## 3. Correct experimental unit

Most TouchScale tasks are relational:
- Transfer: source + destination/container + transferred material.
- Assembly: component + parent object + mating part.
- Cleaning: tool + contacted surface.
- Setup: multiple objects + spatial arrangement.
- Inspection: tested object + tool/procedure.
- Material processing: material/object + tool or support surface.

The initial target representation should therefore be an **interaction tuple**, not a single object label:
`(primary manipulated entity, counterpart/source/destination, tool, contact surface, task phase)`.

The metadata does not provide a canonical physical object-instance ID for every episode. Repeated nouns or task templates cannot be treated as proof of the same physical instance.

## 4. Research decision

- **NO-GO:** claim that the current metadata establishes few-shot adaptation to unseen physical objects.
- **GO:** continue auditing tactile interaction-structure adaptation.
- Keep within-actor repeated-episode adaptation separate from actor-disjoint generalization.
- Do not silently rename task adaptation as unseen-object adaptation.
- Do not start model training until the interaction target and leakage-safe split are fixed.

## 5. Next signal-level feasibility pilot

A single shard `data/train-00174-of-00430.tar` was selected because it contains all seven task families, 12 actors, and a median episode quality score of 0.5826. Seven high-quality episodes with nontrivial tactile contact were selected, one per family.

The dataset card reports WebDataset shards of approximately 2.5 GB each. The pilot downloads only this one shard, extracts only the selected episodes' left/right tactile NPZ files and episode HDF5 metadata, and does not extract or retain video files. The artifact contains aggregate signal diagnostics only.

The pilot checks:
- per-hand timestamp monotonicity and sampling rate;
- tactile pad count, taxel count and array shapes;
- finite values and force distribution;
- force-threshold contact fractions versus metadata;
- active contact segments;
- preliminary split-half stability of sensor-pair correlations, only when enough active frames and variable taxels exist.

This is a feasibility diagnostic, not model training and not a final estimator validation.

## 6. Deliverables
- `touchscale_task_catalog_920_v1.xlsx`: 920 templates with family, interaction scope, episode/actor/date counts and example instruction.
- `touchscale_all_15324_task_classification_v1.csv`: all 15,324 episode rows with family labels.
- `scripts/audit_touchscale_signal_pilot.py` and `.github/workflows/touchscale-signal-pilot.yml`: selected raw tactile signal pilot.

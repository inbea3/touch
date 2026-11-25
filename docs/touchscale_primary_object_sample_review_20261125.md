# TouchScale primary-object sample review — 2026-11-25

## Scope and limits

A 100-episode metadata sample was inspected using task names, item arrays, and instruction text. This is a **desk review**, not an independent human annotation study. The provisional target labels in the accompanying workbook are hypotheses to guide review; they are not canonical object IDs and must not be used as ground-truth labels.

No raw tactile shard or video was downloaded for this review.

## Sample composition

| Measure | Result |
|---|---:|
| Episode records | 100 |
| Distinct task_id values | 73 |
| Distinct actors | 15 |
| Collection dates | 36 |
| Scenes | 22 |
| Empty item arrays | 17 |
| One item label | 6 |
| Two item labels | 17 |
| Three item labels | 27 |
| Four item labels | 25 |
| Five or more item labels | 8 |

The sample contains 10 repeated actor-task pairs, covering 22 records. Twelve task IDs occur across multiple actors; 13 task IDs span multiple collection dates. This is enough to show that episode-level random splitting could create dependence/leakage. It does not, by itself, prove leakage in a particular split.

## Primary-object ambiguity observed

The task list contains several qualitatively different cases:

1. **Relatively clear single-object interactions**: e.g. peel a potato, turn a cucumber, open a bubble mailer, rotate a cube, open a drawer.
2. **Object-component interactions**: e.g. unscrew a bottle cap, pull a zipper, press a stopper into a test tube. The target may be the whole object or the component, depending on the research unit.
3. **Tool-mediated interactions**: e.g. use tweezers to place a test tube, use a peeler on a potato, use a wrench on a nut. The task target and the object physically grasped by the hand are not necessarily the same.
4. **Multi-object transfer/sorting tasks**: e.g. pour between vessels, sort cloths, transfer samples, arrange food. A single target-object label would be misleading without a task-specific rule.
5. **Scene setup/reset tasks**: e.g. set up a shared drinking-water station or restore a workstation. These are not suitable as object-level adaptation tasks without a narrower subtask definition.

The `items` list often contains both target and non-target objects (tools, surfaces, trays, containers, fixtures, props). Therefore, selecting the first item in the list as the target is invalid.

## Independence findings

The 100-row sample has multiple repeated actor-task combinations and repeated task IDs across actors/dates. A valid support/query split must be grouped by the actual intended generalization unit:
- unseen object instance/category for object adaptation;
- separate episodes, not windows from the same episode;
- avoid shared recording/session or near-duplicate task demonstrations across support and query;
- report participant overlap and, where possible, evaluate participant-held-out sensitivity.

The current metadata has no universal canonical physical-instance ID for every item in every episode. Thus, true cross-instance or cross-category object adaptation cannot be established from `items` alone.

## Decision

**Do not start TouchScale object-level few-shot training yet.**

The current evidence supports:
- sensor geometry and metadata are technically accessible;
- task instructions frequently identify a plausible manipulated object;
- primary-object identity is ambiguous for multi-object, component, tool-mediated, and setup tasks;
- the 100-row sample is useful for designing an annotation protocol, but it is not a validated object-episode eligibility census.

## Next gate

1. Finalize an annotation codebook defining target object, manipulated component, tool, support object, and environment.
2. Have a reviewer annotate the 100 episodes; if feasible, independently double-annotate at least 20–30 and calculate agreement.
3. Build a candidate primary-object table, preserving ambiguity and annotator confidence.
4. Audit repeated target labels across actors, dates, task IDs, and recording shards.
5. Only then calculate 1-/2-/5-shot feasibility using independent episodes and freeze Support/Query splits.

If the labels cannot support a defensible object-level unit, revise the research task to interaction/task-family structure adaptation rather than forcing an object-level claim.

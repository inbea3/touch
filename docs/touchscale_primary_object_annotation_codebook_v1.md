# TouchScale primary-object annotation codebook

Version: 1.0
Date: 2026-11-25

## Unit of annotation

One row = one TouchScale episode. Annotate what the instruction and episode metadata identify as the primary object of the tactile interaction, not every object present in the scene.

The codebook distinguishes the task's target from the item the hand happens to grasp. For example, in 'use tweezers to place a test tube', the test tube is the task target; tweezers are the tool. For a component operation such as unscrewing a cap, record both the manipulated component (cap) and parent object (bottle), and retain the component-level ambiguity.

## Required fields

| Field | Allowed values / instruction |
|---|---|
| primary_target_label | Exact normalized name of the target object/component; do not invent a canonical ID |
| target_level | whole_object / component / material_or_substance / multi_object_set / environment_or_surface / unclear |
| target_role | manipulated_target / tool / support_or_fixture / container_or_destination / background_or_surface |
| secondary_objects | Other items involved, with roles |
| target_confidence | high / medium / low |
| task_scope | single_object / component_action / tool_mediated / multi_object_transfer_or_sort / scene_setup_or_reset / other |
| reusable_identity | likely_reusable_label / category_only / physical_instance_unknown / not_applicable |
| episode_quality | usable_candidate / needs_raw_signal_check / exclude_from_target_pool |
| exclusion_reason | empty_or_missing_target / setup_reset / multiple_targets_unresolved / ambiguous_item_mapping / other |
| annotator_note | Short evidence from task name/instruction |

## Decision rules

1. Use the task name and instruction together. The first entry in the items array is not automatically the target.
2. Name the smallest object/component explicitly acted on when the task is a component operation, while also recording its parent object. Example: zipper pull (component) and jacket (parent object).
3. For tool-mediated tasks, separate target from tool. Example: potato = target; peeler = tool.
4. For transfer/pouring/sorting, do not force a single target if both source and destination objects are essential. Use multi_object_set and list source/destination roles.
5. For station setup, workstation restoration, or broad arrangement, mark environment_or_surface or unclear unless the instruction identifies a discrete manipulated target and a bounded subtask.
6. If the instruction describes multiple target objects (e.g. two cloths sorted by color), retain the set rather than selecting one arbitrarily.
7. Item labels may describe multiple physical instances or generic object categories. Do not equate label recurrence with physical-instance identity.
8. A reusable label is not proof that the same physical object instance recurs. Record category-level recurrence separately from instance identity.
9. Episode quality and few-shot eligibility are separate. A clear target can still be unusable if contact signal is insufficient or if it overlaps the support/query recording context.
10. Never create K-shot eligibility from item frequency alone. Require independent episodes and an explicit grouping policy.

## Inter-annotator quality check

- Independently annotate at least 20–30 of the 100 episodes.
- Report exact agreement for target level, task scope, and target label after normalization.
- For categorical labels, report Cohen's kappa (or a suitable multi-rater statistic if more than two annotators).
- Resolve disagreements by documenting a rule change; do not silently overwrite one annotator's label.
- Keep the original annotations and adjudicated value in separate columns.

## Eligibility gate

An episode enters a candidate few-shot pool only if:
- primary target is clear enough for the declared task;
- target identity can be matched at the declared level (category or physical instance);
- the raw tactile segment passes signal/contact checks;
- support and query episodes are independent under the predeclared actor/task/date/shard grouping;
- no episode windows from the same recording leak across support and query.

The eventual thesis must state whether adaptation is across object categories, object instances, or task families. These are different generalization claims and must not be conflated.
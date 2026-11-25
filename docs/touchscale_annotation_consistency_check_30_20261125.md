# TouchScale annotation consistency check — 30 episodes (2026-11-25)

## Purpose and status

A second model-assisted coding pass was performed on the same 30 stratified episodes selected for blind review. The second pass re-read the original task names, item arrays, and instructions without consulting the first-pass labels during coding.

**This is not independent human double annotation.** Both passes were produced by the same assistant/model workflow. The metrics below are internal consistency diagnostics only and must not be reported as thesis inter-annotator reliability.

## Results

| Measure | Result |
|---|---:|
| Episodes re-read | 30 |
| Target-level exact agreement | 27/30 (90.0%) |
| Task-scope exact agreement | 29/30 (96.7%) |
| Exact primary-label string agreement | 15/30 (50.0%) |
| Cohen's kappa — target level | 0.858 |
| Cohen's kappa — task scope | 0.958 |

Exact label string agreement is low partly because equivalent descriptions can differ in granularity or wording; it also flags that the normalization policy needs to be explicit.

## Three target-level disagreements requiring adjudication

1. Sample row 9 — first pass called it a `multi_object_set`; second pass called it `environment_or_surface`. The instruction describes arranging a multi-object drink-pickup station. Recommended adjudication: `environment_or_surface` / `scene_setup_or_reset`, exclude from object-level target pool.
2. Sample row 32 — first pass called the target `material_or_substance`; second pass called it `multi_object_set`. The instruction is transfer of dried beans from bowl to jar using a measuring cup. Recommended adjudication: `multi_object_set` because source and destination vessels define the interaction; beans are the transferred substance and measuring cup is the tool.
3. Sample row 37 — first pass called it `environment_or_surface`; second pass called it `whole_object`. The instruction includes tightening a screw-top bottle cap and placing the bottle in a convenient position. Recommended adjudication: mixed setup task; exclude from the core object-level pool unless the task is narrowed to the cap-tightening sub-action.

## Additional recurring issue

Object component vs parent-object granularity must be standardized. For cap/zipper/plunger/nozzle tasks, retain both:
- `primary_target_component`
- `parent_object`

Do not merge component and whole-object tasks into one category without an explicit estimand.

## Decision

- The codebook is sufficiently clear for provisional screening but needs an explicit rule for scene setup and source/destination transfer tasks.
- No TouchScale few-shot eligibility is approved yet.
- A real second human reviewer is still needed for at least 20–30 blind episodes if the thesis will report inter-rater agreement.
- After human review, adjudicate labels, then perform a dataset-wide object normalization and episode-independence audit.
- Do not start model training until the target unit is frozen and raw tactile signal feasibility is checked.

## Artifact

`touchscale_ai_second_pass_consistency_30.xlsx` contains the row-level comparison and summary metrics. It is explicitly labeled as model-assisted internal consistency, not human inter-rater reliability.

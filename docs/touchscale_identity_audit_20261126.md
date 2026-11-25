# TouchScale 300 Episode identity audit — 2026-11-26

## Scope
Audited the completed deterministic 300-Episode sample using `sample_selection.csv` from workflow run 45613561287. This is a metadata-only audit; no video was inspected and no object identity was inferred from imagery.

## Sample findings
- 300 unique Episodes; 209 distinct `task_id` values; 296 distinct `task_name` values.
- 22 scenes, 20 actors, 52 collection dates.
- `items` is empty in 74/300 rows; 226 rows contain one or more free-text item labels.
- 50 task IDs map to multiple task names; 45 task IDs occur in multiple scenes in this sample.
- Only four exact task names repeat (eight Episodes total); only one repeated task name spans multiple actors.

The sample contains clear examples of task-ID collisions: e.g. task_id 2 appears with unrelated task names and scenes. This conflicts with the current Hugging Face dataset card's statement that the same task maps to the same task_id. The complete `episodes.parquet` must be checked before deciding whether this is a sampling/export issue or a dataset metadata inconsistency.

## Official schema findings
The official TouchScale card currently describes 15,324 Episodes, 929 tasks, 22 scenes and 905+ involved objects. It also documents `metadata/objects.parquet` with 193 mesh objects and stable `object_id`, `object_name`, `mesh_path`, and `capture_id` fields.

However:
- The documented Episode metadata exposes `items` as a free-text list of involved objects; it does not document an episode-level `object_id` or physical object-instance ID.
- The card explicitly warns that mesh `object_name` does not always match free-text Episode `items`.
- The 193 mesh IDs describe reconstructed object meshes; they are not documented as physical-instance identifiers for every Episode.

Therefore, exact text matching between `items` and mesh names is only a candidate association, not proof of the object instance contacted in an Episode.

## Decision
**The 300-Episode sample is not sufficient to construct a defensible object-disjoint few-shot adaptation split.** Do not randomly split Episodes and call it new-object adaptation. Do not use `task_id` as a global grouping key until the full metadata discrepancy is resolved.

Dynamic tactile representation work can continue. Object-level few-shot claims remain on hold pending a full metadata audit and an auditable mapping from Episodes to object identities. If no physical object-instance mapping exists, the research claim should be reframed as task-family/condition adaptation, not new-object adaptation.

## Next execution
Added `scripts/audit_touchscale_full_metadata_identity.py`, which downloads only `metadata/episodes.parquet` and `metadata/objects.parquet` and produces global task-ID collision reports, repeated-task reports, item-to-mesh-name candidate matches, and an Episode metadata manifest. It requires an accepted Hugging Face dataset agreement and `HF_TOKEN`.

Added workflow `.github/workflows/touchscale-full-metadata-audit.yml`. Run it manually from GitHub Actions (workflow_dispatch) on branch `thesis-few-shot-structure-20261125`, or execute the script on the research server.

## Source
Official dataset card: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale

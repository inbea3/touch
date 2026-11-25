# Revised Thesis Route — Few-Shot Cross-Object Tactile Structure Adaptation

- Decision date: 2026-11-25
- Status: route selected; TouchScale audit is the immediate next task.
- This document records a deliberate research-route revision. It does not erase prior protocols or results.
- Repository branch: thesis-few-shot-structure-20261125
- Supersedes the *research emphasis* of Scheme A for future work; historical STAG protocols and results remain immutable evidence.

## 1. Final research question

For an object not observed during meta-training, can a model use a small number of independent early tactile interaction episodes to adapt its prediction of that object's subsequent spatial co-activation structure?

The primary scientific question is few-shot cross-object adaptation. Spatial topology is a candidate inductive bias to be tested, not an assumed contribution. Meta-learning is a candidate adaptation strategy, not a required winner.

## 2. Scope and claim boundaries

### In scope
- Tactile pressure/force time series and sensor spatial geometry.
- Object-level train/validation/test separation.
- Support/query separation by independent interaction episode.
- Prediction of a future within-episode sensor-pair statistical association structure, provided the dataset supports a defensible contact-window definition.
- Comparison of zero-shot prediction, ordinary fine-tuning, and selected parameter-efficient/meta-learning alternatives.
- Physical sensor topology as an ablation.

### Out of scope for the core thesis
- RGB/video or vision-language inputs.
- Claims about robot grasp success, manipulation policy improvement, or closed-loop control without robot deployment experiments.
- Pooling raw channels or metrics across STAG, TouchScale, RCT, or HAS.
- Treating correlation edges as causal links or ground-truth physical contacts.
- Requiring MAML, a complex dynamic GNN, or every recent architecture to be implemented.

## 3. Data strategy

### Primary development/evidence dataset: STAG/HumanGrasp
Retain the existing pressure-only audit, 96-sensor repeated object-grouped CV, label stability work, G2-R2/G2-R3 artifacts, and Reptile/ANIL negative or neutral evidence.
- STAG has 26 non-empty objects and 3 recordings per object (78 recordings), with 548 valid physical sensors.
- Existing repeated object-grouped CV supports a zero-shot MLP improvement over a constant baseline on the fixed 96-sensor subset.
- Existing ordinary Support adaptation did not show stable improvement; Reptile did not outperform ordinary MLP in recorded validation experiments.
- Existing objects have already participated in exploratory model selection and repeated CV. Do not describe any previously reused object set as an untouched confirmatory test.
- Because there are only three recordings per object, STAG is not sufficient for a broad 5/10-shot study. Use it for historical replication, protocol development, and carefully qualified 1/2-shot analyses.

### Main public-data candidate: TouchScale
TouchScale is the first public dataset to audit, not yet an approved training dataset.
- First inspect metadata and sensor geometry only; do not download the full raw dataset.
- Verify the actually accessible release/version, license, schema, object/item identity, episode boundaries, timestamps, taxel geometry, missingness, and usable contact intervals.
- Count independent episodes per normalized object identity, participant, hand, task, and recording session.
- Establish whether repeated episodes for the same object are genuinely independent and whether object identity can be mapped without ambiguous free-text grouping.
- Use one hand initially only if the metadata and geometry support a consistent, reproducible spatial layout.
- Do not assume that every episode is single-object or that every frame is a valid contact frame.
- Do not assume the advertised total object count equals the number of usable few-shot tasks.

### Optional external validation: RCT
Keep RCT as optional, separate robot-collected tactile validation only if a compatible sensor-level target can be defined. Its DIGIT image sensor representation is not directly interchangeable with STAG pressure cells or TouchScale taxels. It is not a graduation-critical dependency.

### Frozen branch: HAS
Keep HAS immutable and separate. No pooling, no cross-dataset transfer claim, and no new HAS training in the core thesis without a separately approved common estimand and protocol.

## 4. Target and episode protocol

Candidate target (must pass dataset-specific audit):
- Input: an early valid-contact window of sensor pressure/force observations.
- Target: the sensor-pair association matrix calculated over a later, non-overlapping valid-contact window within the same interaction episode.
- For STAG, retain the previously specified future Pearson correlation target as historical protocol.
- For TouchScale, first determine whether its normal-force taxel signals and episode timing support a meaningful future-window correlation target. Do not transfer STAG's frame counts or thresholds mechanically.
- Correlation is a statistical association, not causality or a physical contact graph.
- Consider robust alternatives only as a preregistered sensitivity analysis after the primary target is fixed.

Few-shot unit:
- One task = one normalized object identity.
- Support = K independent episodes for that object.
- Query = different independent episodes of that same object.
- Outer meta-test objects must be absent from meta-training and model selection.
- No splitting frames, sensor pairs, or windows from one episode across Support and Query.
- Participants/sessions should be accounted for; run a participant-held-out sensitivity analysis if the metadata permit.
- If an object lacks enough independent episodes, it is ineligible for that K-shot evaluation and must be counted transparently.

## 5. Immediate TouchScale audit gates

Gate A — metadata availability and integrity:
- Retrieve only metadata tables and geometry files.
- Record exact release revision, file sizes, schema, and SHA-256 where files can be downloaded.
- Confirm license and access restrictions.

Gate B — identity and episode integrity:
- Build a reproducible object identity normalization table with original raw labels retained.
- Manually review ambiguous/high-frequency labels.
- Detect multi-object episodes, duplicate episodes, repeated frames, and session/participant overlap.
- Produce episode counts per object and the number of objects eligible for 1-, 2-, 5-, and 10-shot.

Gate C — geometry:
- Verify taxel count, coordinate order, units, coordinate frame, missing/duplicate coordinates, pad boundaries, and left/right hand mapping.
- Compare candidate physical adjacency constructions (within-pad geometry and cross-pad hand geometry); do not assume a single flat kNN graph is physically valid.
- Save graph construction code, edge lists, and hashes only after geometry validation.

Gate D — target feasibility:
- Inspect time intervals, sample rates, force channels, missingness, and contact segmentation.
- Test whether early and future windows can be separated without overlap and with sufficient valid samples.
- Quantify target reliability across window choices on development objects only.
- If valid future windows or object identity are inadequate, stop and report the dataset as unsuitable for this target; do not force the task.

## 6. Minimal experimental matrix after audit

Stage 1 — task validity and zero-shot:
1. Constant baseline.
2. Node-MLP zero-shot.
3. Spatial-GNN zero-shot (only after geometry passes).

Stage 2 — adaptation:
4. Ordinary MLP fine-tuning.
5. Frozen encoder + lightweight head/adapter adaptation.
6. Reptile only if the task pool and fair baseline protocol are adequate.

Stage 3 — topology ablation:
7. Physical-topology GNN vs Node-MLP.
8. Degree/connectedness-controlled rewired graph vs physical graph, if a physically defensible graph exists.

Do not run a broad model grid initially. Start with a small, predeclared pilot, inspect failure modes on development objects, then freeze a confirmatory protocol.

K-shot schedule:
- STAG: at most 1-/2-shot exploratory evaluation under the existing three-recording constraint.
- TouchScale: 1/2/5-shot only if enough independent episodes per object; 10-shot is optional and requires adequate task counts.
- Report the number of eligible objects at each K. Never compare K-shot means over silently changing object populations without also reporting a common-object analysis.

## 7. Baselines, fairness, and statistics

- Zero-shot and adapted versions use the same encoder, input features, target definition, preprocessing, and outer object split.
- Fine-tuning and meta-learning receive the same K support episodes and comparable gradient/update budgets; report any unavoidable exposure differences.
- Hyperparameters are selected only using training/validation objects, never outer test queries.
- Primary inferential unit: object. Episode-level results are nested within objects; sensor pairs and frames are not independent samples.
- Report per-object MAE, RMSE, Spearman, paired adaptation gain, number of improved objects, object-bootstrap confidence intervals, and adaptation time.
- Report both the common-object comparison and eligible-object counts for each K.
- A meta-learning claim requires consistent improvement over ordinary fine-tuning, not merely over zero-shot.
- A topology claim requires physical graph improvement over both Node-MLP and suitable rewired controls, with object-level uncertainty and practical effect size.
- If an added method fails to show reliable incremental value, report the negative/inconclusive result and remove it from claimed contributions.

## 8. Retained and frozen experiments

Retain unchanged:
- STAG data audit and label-stability work.
- Stage 8 96-sensor repeated object-grouped CV as historical primary predictive evidence.
- G2-R2 full-548 results as a distinct exploratory stream.
- G2-R3 reconstruction protocol, implementation, traces, checkpoints, graph edge lists, and outputs; verify current run state and artifact integrity before any continuation or migration.
- Reptile and ANIL protocols/results as negative/neutral evidence.
- HAS as a separate frozen supplementary branch.

Do not:
- overwrite or rerun completed G2-R3 configurations;
- claim G2-R3 is complete based on a stale progress snapshot;
- combine 96-node and 548-node metrics;
- claim that GNN or meta-learning works before object-level evidence supports it;
- use historical STAG test objects as untouched final confirmation.

## 9. Thesis contribution candidates (conditional on evidence)

1. A task protocol for few-shot adaptation of future tactile spatial co-activation structure across unseen objects, with episode-level leakage controls.
2. An empirical analysis of whether small numbers of independent tactile episodes improve object-specific structure prediction over zero-shot and ordinary fine-tuning.
3. A controlled assessment of whether physical sensor topology provides incremental value for this adaptation task.

These are candidate contributions, not pre-asserted discoveries. Do not claim novelty until a focused related-work review confirms the exact gap.

## 10. Ordered execution plan

1. Preserve and verify the current G2-R3 run; do not interrupt it during route revision.
2. Complete TouchScale metadata/geometry audit only.
3. Produce an object-episode eligibility table and a go/no-go decision for each K.
4. Freeze the TouchScale target and Support/Query protocol.
5. Implement a small zero-shot and ordinary fine-tuning pilot.
6. Evaluate whether adaptation helps; only then add Reptile or lightweight adaptation.
7. Add GNN only if geometry and pilot results justify it.
8. Perform object-level confirmatory evaluation and statistical synthesis.
9. Write the thesis with claims limited to observed evidence.


### TouchScale public-card pre-audit (2026-11-25)
- Public dataset card reports 15,324 episodes, 929 tasks, 22 scenes, 24 participants, 880 taxels per hand, and 35–90 Hz tactile sampling. Row-level metadata contains 953 distinct task_id values; this discrepancy with the card's 929-task figure remains unresolved and must be reported rather than silently reconciled. The currently hosted release is 100 hours; the remaining 400 hours must not be assumed available.
- Episode metadata includes free-text `items`, task/scene/instruction, actor, primary hand, duration, tactile contact fractions and shard. The object mesh table contains 193 reconstructed objects and does not provide a complete canonical identity map for all episode item labels; the card explicitly notes that mesh object names do not always match episode `items`.
- Episodes may involve multiple objects; naive item-label counts are not valid object-task counts. The audit must retain raw labels and manually review ambiguous mappings.
- Metadata/geometry files were accessed after dataset terms acceptance and a read-only Hugging Face token was configured as repository secret HF_TOKEN; the token itself is not stored in the repository or chat.
- Added `scripts/audit_touchscale_metadata.py`, `requirements-touchscale-audit.txt`, and manual workflow `.github/workflows/touchscale-metadata-audit.yml`. The workflow downloads metadata and geometry only, not raw episode shards.


### TouchScale metadata/geometry audit completed (2026-11-25)
- Workflow run 45588023843 succeeded; metadata-only artifact was downloaded and inspected.
- 15,324 episode rows and unique episode IDs; 953 task IDs; 24 actors; 22 scenes; 880 distinct individual item labels; 821 exact item bundles.
- 4,045 episodes (26.4%) have empty item arrays; 11,148 (72.7%) contain multiple item labels. `items` lists all objects involved, often including props/tools/surfaces, and is not a canonical primary-object ID.
- Only 66 of the 193 mesh object names match an episode item label under normalized exact-name matching. The mesh registry is partial.
- Both hand geometries decoded: 880 taxels, 15 regions each, 9 values per taxel (XYZ plus two local direction vectors), all XYZ finite, no duplicate XYZ rows. Official geometry note specifies a flat palmar layout and region/row-major order.
- Median episode duration 21.87 s. Median contact fractions: left 0.2978, right 0.3302. Sampling rates vary; use timestamps.
- Decision: metadata and geometry gates pass; object-identity/primary-manipuland and independent Support/Query gates remain open. Do not infer K-shot eligibility from raw item-label frequencies. Do not start TouchScale model training yet.
- Next: stratified manual review of ~200 episodes to identify primary manipulated object, support props, repeated labels, task/actor/date overlap, and valid single-hand contact intervals.
- Full report: [docs/touchscale_metadata_geometry_audit_20261125.md](touchscale_metadata_geometry_audit_20261125.md).



### TouchScale primary-object sample review (2026-11-25)
- Inspected a 100-episode metadata sample using task names, item arrays and instructions. This is a desk review, not independent human annotation; provisional target hypotheses are not ground truth.
- Sample composition: 73 distinct task IDs, 15 actors, 36 dates, 22 scenes; 17 empty item arrays; item-array size distribution: 6 with one label, 17 with two, 27 with three, 25 with four, and 8 with five or more.
- Ten repeated actor-task pairs cover 22 records; 12 task IDs occur across multiple actors and 13 across multiple dates. Random episode-level splitting is therefore not acceptable without group controls.
- Task types include clear single-object actions, component actions (zipper/cap), tool-mediated tasks, multi-object transfers/sorting, and scene setup/reset. The first item in items must not be treated as the target object.
- Decision: no TouchScale object-level few-shot training yet. First define the annotation codebook, annotate the 100 sample (double-annotate a subset if possible), preserve ambiguous cases, and build the repeated-target/actor/date/shard table.
- Review workbook: touchscale_primary_object_review_100.xlsx (provided separately). Full report: docs/touchscale_primary_object_sample_review_20261125.md.
- Annotation rules are frozen in docs/touchscale_primary_object_annotation_codebook_v1.md; double-annotate at least 20–30 episodes before treating the labels as reliable.


### 100-episode provisional desk coding (2026-11-25)
- A model-assisted desk coding pass was completed from task names, instructions, and item lists. It is explicitly provisional and not human-verified ground truth.
- Provisional target levels: 44 whole-object, 18 multi-object-set, 17 component, 13 material/substance, 8 environment/surface.
- Provisional task scopes: 25 single-object, 24 tool-mediated, 22 component-action, 13 multi-object transfer/sort, 9 setup/reset, 7 other/underspecified.
- Triage: 61 usable candidates for later raw-signal review, 29 requiring signal/contact review, 10 excluded from object-level pool pending task redesign. These are not K-shot eligibility counts.
- A blinded 30-episode double-review workbook has been prepared to enable independent reviewer agreement analysis. No inter-rater agreement exists yet because independent human annotations have not been returned.
- Next gate: obtain independent review/adjudication of the 30-row blind sample (and then review the full 100 if the codebook changes), finalize target identity unit, and only then compute eligible few-shot tasks.
- Desk coding report: [TouchScale 100-episode desk coding v1](touchscale_100_episode_desk_coding_v1_20261125.md).


### Full TouchScale task-intent classification (2026-11-25)
- Classified all 15,324 episode metadata rows into seven broad task-intent families across 920 scene/task templates. The 37 residual templates were manually assigned to a broad family after inspection; this is still task-intent coding, not human-verified object annotation.
- Family counts: transfer/pouring/dispensing 3,547 episodes/205 templates; insertion/assembly/component operation 2,973/155; scene setup/organization/reset 2,425/187; material processing/deformation 2,180/113; single-object grasp/transport/pose manipulation 1,495/95; surface/tool cleaning 1,358/86; inspection/trial/comparison 1,346/79.
- `task_id` is not globally unique; use scene + task_name for task-template audit.
- Of 920 templates, 862 have only one actor, 56 have two actors, and 2 have three actors. Thus only 58 templates have >=2 actors, 2 have >=3 actors, and none have >=6 actors. Median episode count is 18, but these are mostly repeated within-actor episodes and are not independent shots.
- Decision: no-go for claiming current TouchScale metadata supports participant-independent few-shot adaptation to unseen physical objects. Continue with interaction-structure task audit; distinguish within-actor episode adaptation from actor-disjoint generalization. Do not silently relabel task adaptation as unseen-object adaptation.
- Next: inspect a stratified raw-signal sample, define interaction-tuple targets (manipulated entity, counterpart/source/destination, tool, contact surface, task phase), and quantify actor-disjoint eligible tasks before training.
- Full report: [TouchScale full task-intent classification](touchscale_full_task_intent_classification_20261125.md).


### Corrected full task-family classification (2026-11-25)
- Re-coded all 15,324 episodes across 920 task templates, correcting false positives where “test tube” (object noun) was mistaken for a testing action.
- Corrected counts: transfer/pouring/dispensing 4,127 episodes/235 templates; insertion/assembly/component operation 3,276/172; scene setup/organization/reset 2,425/187; material processing/deformation 2,180/113; single-object grasp/transport/pose manipulation 1,683/109; surface/tool cleaning 1,429/89; inspection/trial/comparison 204/15.
- Task classification is still task-intent coding, not human-verified object identity. The interaction tuple is the candidate target unit.
- A seven-episode, seven-family signal-feasibility pilot has been prepared against one shard (00174/00430); it extracts only selected tactile NPZ and HDF5 files, not video, and reports diagnostics only. It does not train a model.

## 11. Current status (2026-11-25)

- Research direction: adapt to unseen objects; target spatial co-activation structure; public TouchScale data audit is the first data gate.
- Metadata/geometry audit: passed at the structural level (15,324 episodes; 880 taxels per hand; 15 pads per hand). This does not establish object identity or few-shot task feasibility.
- Full task-intent classification: 15,324 episodes mapped to 7 broad interaction families across 920 scene/task templates. These labels are task-intent labels, not verified primary-object IDs.
- 100-episode raw HDF5 audit v2: completed successfully; 100/100 episode files and JSON metadata found after correcting one mistyped UUID; 200/200 hand records processed.
- Signal integrity: all 200 hand timestamp sequences monotonic; no gaps >0.25 s; maximum gap 0.0778 s; 15 pads and 880 taxels per hand; finite-value fraction 1.0.
- Rate calibration: reciprocal median inter-sample interval was biased by irregular intervals (median interval CV ≈0.299). Use effective rate `(N-1)/(t_last-t_first)`; median effective rate 56.91 Hz vs metadata median 57.0 Hz; median absolute error 0.099 Hz. 43/200 hand records still differ by >2 Hz and remain flagged.
- Contact QC: official dataset documentation confirms per-taxel calibrated normal force in N and defines contact fraction as summed normal force >3 N. Recomputed-vs-metadata contact fraction correlation is 0.9978, but MAE is 0.0405 (4.05 percentage points); inspect outliers before strict contact filtering.
- Within-episode split-half spatial-structure correlation median: left 0.458, right 0.359. This is exploratory temporal repeatability only, not cross-object generalization or adaptation evidence.
- One missing episode in the first 100-row manifest was a malformed UUID; the corrected UUID `88430eb1-984f-4a5f-926e-7893708db835` is present. The corrected manifest has 100/100 files.
- 300-episode stratified raw-signal audit: workflow run [45600096911](https://github.com/inbea3/touch/actions/runs/45600096911) was launched across 15 shards and all 22 scene types; final results are pending.
- No TouchScale model training has started. Object identity, interaction-tuple annotation, actor-disjoint Support/Query feasibility, and a valid future-structure prediction target remain open gates.
- Historical G2-R3 execution is separate from the TouchScale signal audit; verify its current workflow/process state independently before making any completion or termination claim.

### Latest signal audit report
- [TouchScale 100-episode raw tactile signal audit v2](touchscale_signal_audit_100_v2_report_20261125.md)
- Official dataset card: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale

### Next decision gate
1. Complete the 300-episode audit.
2. Identify rate/contact outliers and validate timestamp-aware resampling.
3. Compare spatial-structure repeatability across scene, participant, task and contact strata.
4. Only then freeze an interaction-structure prediction target and actor-/task-disjoint Support/Query protocol.
5. Run baseline models before Reptile/ANIL; no claim of meta-learning benefit without a fair ordinary-fine-tuning comparison.

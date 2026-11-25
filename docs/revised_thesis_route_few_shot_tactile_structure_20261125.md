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
- Public dataset card confirms 15,324 episodes, 929 tasks, 22 scenes, 24 participants, 880 taxels per hand, and 35–90 Hz tactile sampling. The currently hosted release is 100 hours; the remaining 400 hours must not be assumed available.
- Episode metadata includes free-text `items`, task/scene/instruction, actor, primary hand, duration, tactile contact fractions and shard. The object mesh table contains 193 reconstructed objects and does not provide a complete canonical identity map for all episode item labels; the card explicitly notes that mesh object names do not always match episode `items`.
- Episodes may involve multiple objects; naive item-label counts are not valid object-task counts. The audit must retain raw labels and manually review ambiguous mappings.
- Metadata/geometry files are behind the dataset access gate. No row-level parquet or geometry inspection has yet occurred. The researcher must accept the dataset terms and provide a read-only Hugging Face token via the repository secret `HF_TOKEN`; never paste the token into chat or commit it.
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

## 11. Current status (2026-11-25)

- Research direction selected by the researcher: adapt to unseen objects; target spatial co-activation structure; audit TouchScale first; first experiment priority is whether few-shot adaptation improves structure prediction.
- Literature review is exploratory; exact novelty is not established.
- Public metadata and geometry have not yet been inspected at record level in this project execution.
- No TouchScale raw data have been downloaded or trained on.
- The audit has not passed any data-dependent gate yet.
- Current G2-R3 execution status must be checked from its live process/workflow artifacts before reporting completion or stopping it.

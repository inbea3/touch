# Thesis final research route — Scheme A (STAG primary, HAS frozen supplementary branch)

- Decision date: 2026-11-24
- Decision: Scheme A confirmed.
- Authoritative research scope: STAG/HumanGrasp pressure-only.
- HAS status: independent supplementary/external feasibility branch; frozen for the core thesis and excluded from the primary pooled analysis.
- No cross-dataset pooling, no transfer-learning claim, and no use of HAS metrics as STAG replication/validation unless a separately preregistered common estimand and protocol are approved.

## 1. Final thesis question

Can early pressure observations from STAG/HumanGrasp predict the subsequent within-recording statistical co-activation structure of tactile sensors across unseen objects, and do spatial graph structure or few-shot meta-learning provide reliable incremental value over simpler baselines?

Primary estimand: for each recording, Pearson correlation between each unordered physical sensor pair over subsequent valid-contact frames, conditional on the audited valid-contact protocol. Auxiliary edge label r >= 0.7 is exploratory and not causal ground truth.

## 2. Data boundary

Primary dataset:
- STAG/HumanGrasp pressure-only.
- 135,187 frames, 82 source recordings; after excluding four empty-hand recordings, 26 non-empty objects and 78 recordings (three recordings/object); 548 valid physical sensors.
- No synchronized video, visual labels, pose, or multimodal inputs.

HAS:
- TUM-ICS Hide-and-Seek, 44-element left-hand force vector and separate 88x6 tactile point-cloud field.
- Its experiments predict future pressure mean/correlation using a different sensor system and a different estimand.
- Keep HAS artifacts and negative results immutable as a supplementary research branch. Do not combine HAS episodes, channels, model metrics, or splits with STAG.
- Do not describe HAS as an external test set for STAG. A future cross-dataset transfer study would require a new common estimand, mapping/normalization protocol, and separately registered evaluation; it is outside the current thesis plan.

## 3. Evidence retained for the STAG thesis

### A. Data and target reliability
Retain STAG data audit, valid-contact/recording metadata audit, correlation-label definition, moving-block bootstrap stability analysis, and repeated-grasp/Weight aggregation feasibility analysis. The Weight aggregation result is a negative feasibility finding; do not merge it into the primary task.

### B. Main 96-sensor predictive evidence
Retain the fixed 96/548 sensor-subset baseline and Stage 8 repeated object-grouped 5-fold x 5-repeat CV as the strongest existing predictive evidence:
- MLP zero-shot MAE 0.11716 (object-bootstrap 95% CI 0.11188–0.12304).
- Constant baseline MAE 0.13305 (95% CI 0.12674–0.13985).
- Paired improvement 0.01589 (95% CI 0.01306–0.01872), same direction in 26/26 objects.
- Ordinary Support adaptation does not show stable improvement.
These are exploratory estimates over the available 26 objects, not independent population-level confirmation. Do not claim full-548 performance from this stream.

### C. Full-548 spatial graph stream
Retain G2-R2 as a distinct full-548 experiment, not merged numerically with the 96-node stream:
- 15/15 folds reported complete.
- Node-MLP MAE 0.120538; k6 MAE 0.120172 (absolute difference 0.000353, approximately 0.3%).
- The observed mean gain is small; convergence was not established in the available blueprint; k4/k8 and randomized-topology evidence require full audit.
- G2-R3 was previously recorded as started with 8/225 runs in an older log snapshot. That snapshot is stale and is not proof of current state. Recover the latest workflow status, logs, checkpoints, manifests and result report before deciding whether it is complete or needs continuation. Never rerun blindly or overwrite existing artifacts.
- A spatial-graph claim is allowed only if the complete R3 initialization and degree-preserving randomized-topology controls support it at object level. Otherwise report the graph result as inconclusive or a small exploratory effect.

### D. Meta-learning stream
Retain Reptile and first-order ANIL as negative/neutral evidence:
- Reptile did not outperform ordinary MLP fine-tuning on Validation MAE in the recorded STAG experiments; adaptation diagnostics did not show reliable benefit.
- ANIL/Query-aware first-order comparison did not outperform ordinary pooled MLP; the available validation sample is only four objects.
- Do not claim meta-learning effectiveness. Do not start second-order MAML or another broad hyperparameter search on the same historical validation objects.
- Keep the algorithmic protocols, training traces, checkpoints, support/query IDs, seed logs and paired object metrics as reproducibility evidence.

## 4. Frozen or excluded lines

1. HAS stages 17–27A: freeze as a separate supplementary branch. No additional HAS model training for the current thesis; preserve protocols, reports, workflow artifacts and logs.
2. Cross-dataset STAG-HAS pooling or transfer: excluded from current thesis.
3. Video, RGB, pose, multimodal inputs: excluded.
4. Weight repeated-grasp aggregation as a replacement target: excluded from primary task; retain negative feasibility report.
5. Second-order MAML and broad Reptile/ANIL tuning on reused validation objects: frozen.
6. Physical GNN on HAS 42 channels: not supported because public channel-to-cell coordinates and the reason for excluding source indices 18 and 25 have not been established.
7. Any claim that correlation edges are causal or physical contacts: excluded.

## 5. Ordered remaining work

### Priority 0 — recover and audit, no new model first
- Recover latest G2-R3 workflow run state; identify whether the 225 model runs completed, failed, or were cancelled.
- Verify artifact completeness, SHA-256 manifests, per-run traces/checkpoints, graph edge lists, fold/object metrics and source-code commit.
- Reconcile G2-R2/R3 summaries against raw fold outputs; audit split leakage, preprocessing, validation-only selection and convergence.
- Preserve all old runs; any rerun gets a new run ID and output directory.

### Priority 1 — finish STAG primary-task validity analyses
- Input-frame continuity and actual timestamp-span sensitivity (the current five valid frames are not necessarily five consecutive frames; 18/78 recordings were previously noted to lack five consecutive earliest valid frames).
- Keep the same object-level split protocol and report actual time gaps; no test-driven selection.
- Audit the 96-node subset selection and clearly report that it is a subset, not all 548 sensors.

### Priority 2 — decide the graph claim
- If G2-R3 supports a robust real spatial topology advantage over Node-MLP and degree-preserving rewired graphs, retain GNN as a qualified incremental contribution.
- If not, report the graph investigation as a negative/inconclusive result and remove GNN from the thesis title/claimed contribution. Do not force a GNN.
- Do not combine graph and meta-learning architectures unless each separately demonstrates incremental value and there is sufficient time/data for a predeclared interaction experiment.

### Priority 3 — final statistical synthesis
- Object is the inferential unit. Report per-object metrics, paired differences, number of objects improved, object-bootstrap intervals and effect sizes.
- Distinguish 96-node and full-548 streams, exploratory fixed splits and repeated grouped CV.
- Existing 26 objects have already participated in model development/validation. State that no untouched independent STAG object cohort is available unless new data are acquired. Do not label historical Test as confirmatory.
- Keep raw pressure arrays out of public repository/CI artifacts. Preserve configs, code commit, input hashes, checkpoints, training traces, run.log, environment and manifests.

## 6. Thesis structure (recommended final)

1. Introduction and research questions.
2. STAG/HumanGrasp pressure-only data, hierarchy and statistical co-activation target.
3. Target reliability, temporal stability and object-level evaluation protocol.
4. Cross-object prediction baselines and repeated grouped validation.
5. Spatial structure / graph-model comparison (include only audited G2 results; clearly separate 96-node and full-548 streams).
6. Few-shot adaptation and meta-learning comparison (Reptile/ANIL negative or neutral findings reported honestly).
7. Integrated discussion: predictive increment, graph evidence, adaptation evidence, uncertainty and limitations.
8. Conclusion and future work.

HAS may appear only in a short supplementary-data exploration subsection or appendix explaining why it was not merged into the primary STAG study. Its results must be labelled as a distinct robot tactile task and must not support the main STAG claims.

## 7. Claim language

Allowed:
- On the available 26 STAG objects and fixed 96-sensor subset, early pressure features showed an object-grouped cross-validation predictive improvement over a constant baseline.
- Reptile/ANIL did not show a reliable Validation MAE advantage over ordinary MLP under the tested protocols.
- The full-548 graph stream showed a small preliminary k6-vs-Node-MLP difference; robustness remains conditional on complete R3 audit.

Not allowed:
- Meta-learning has been proven superior.
- GNN has been proven superior before R3 audit.
- The model has been confirmed on unseen populations of objects.
- HAS validates STAG performance.
- Statistical correlations establish causal/physical sensor interactions.

## 8. Completion rule

The thesis is ready for final writing when: G2-R3 is recovered and audited or explicitly classified as unavailable/inconclusive; continuity sensitivity is completed or explicitly documented as a limitation; all retained metrics are traceable to immutable artifacts; and title/abstract/contributions match the actual evidence. No new HAS experiments are required for Scheme A.

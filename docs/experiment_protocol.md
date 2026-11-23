# Experiment Protocol（阶段 A 冻结版）

- Version: A-1.0
- Freeze date: 2026-11-23
- Scope: STAG/HumanGrasp pressure-only
- Repository branch: `integration/old-research-assets`

## 1. Universal rules

1. No synchronized video, visual labels, pose, or multimodal inputs.
2. The primary estimand is within-recording future sensor-pair Pearson correlation, conditional on valid contact frames.
3. (r\geq0.7) is an auxiliary threshold label only; it is not causal ground truth.
4. Keep recordings from one object in the same outer fold.
5. Support and Query are separated by complete Recording IDs, never by random edge rows.
6. The object is the inferential unit. Edges, frames, episodes, repeated folds and grasps are nested observations.
7. All scalers, feature selection, threshold selection and hyperparameter choices are fitted on training/validation data only.
8. Keep all runs immutable: unique run directory, config, seed, code commit, input hashes, logs, traces, metrics and manifest. Never overwrite prior run outputs.
9. Retain useful training traces and diagnostic logs for replay/audit. Raw pressure arrays must not be committed to the repository or public CI artifacts.
10. Clearly distinguish 96-node experiments from full 548-node G2 experiments; do not compare them as if they had identical input spaces.

## 2. Data and estimand

### 2.1 Audited data
135,187 frames; 32×32 pressure array; 82 source recordings. After excluding four empty-hand recordings: 26 non-empty objects, 78 recordings, three recordings per object; 548 valid physical sensors.

### 2.2 Main target
For each recording, sort frames by timestamp. Use the earliest five `hasValidLabel=True` valid contact frames to construct the early input. The regression target is the Pearson correlation of each undirected sensor pair computed on subsequent valid contact frames. The auxiliary binary label is (I(r_{ij}\geq0.7)).

The transform used in the established 96-node baseline is `clip((pressure-500)/150, 0, 1)`. Do not silently transfer this transform or the 96-node feature schema to a separate G2 full-548 experiment; use the exact preprocessing documented by that experiment and audit it before comparison.

### 2.3 Input continuity
Existing results report that 18/78 recordings do not have five consecutive earliest valid frames. This task is therefore not automatically a fixed-duration continuous-window forecast. Record actual timestamps and frame gaps. Any continuity/time-span sensitivity analysis must preserve the same object-level split.

## 3. Historical object-level split and its limitation

Historical split seed 20261118:
- Meta-train: 18 objects / 54 recordings.
- Meta-validation: 4 objects / 12 recordings.
- Meta-test: 4 objects / 12 recordings.

One object is one meta-task; Support uses one or two recordings; remaining recording(s) form Query. The split passed recording-level leakage checks. However, historical Meta-test objects were used in Stage 7 exploratory reporting and all 26 objects appeared in Stage 8 repeated CV. They are not untouched confirmatory data. Any later reuse must be labelled exploratory. Confirmatory claims require new independent objects/data or explicit limitation.

## 4. Evaluation and inference

### 4.1 Regression
- Primary: object-aggregated MAE (lower is better).
- Secondary: RMSE and Spearman.
- First aggregate within recording/episode and then within object according to the experiment's predeclared aggregation rule.

### 4.2 Auxiliary classification
Report AUROC, Average Precision/PR-AUC, Precision, Recall, F1 and positive prevalence. Because the established positive prevalence is about 0.567%, accuracy is not a primary metric. Threshold calibration/selection is validation-only.

### 4.3 Paired comparison
For each object (m), define (Delta_m=MAE_{baseline,m}-MAE_{candidate,m}). Positive values favour the candidate. Report mean paired difference, 95% object-bootstrap interval, number of objects improved, and per-object distribution. Bootstrap resamples objects, not edges or episodes.

## 5. Experiment-specific frozen protocols

### E01 — Baseline replication / continuity sensitivity
Purpose: retain the known 96-node baseline as the reference and quantify sensitivity to actual input-frame continuity/time span.
- Use the documented 96-node subset, seed 20261116, and the existing repeated-object CV split seeds 20261119–20261123 where appropriate.
- Preserve the primary 5-valid-frame feature construction as the reference.
- Predefine continuity strata / alternative input-window definitions before looking at outcomes.
- Fit preprocessing independently inside each outer training fold.
- Report object-level MAE/RMSE/Spearman and paired changes from the reference.
- Do not tune on historical Meta-test.

### E02 — G2-R2 evidence audit
Purpose: verify the full-548 graph experiment before any new run.
- Retrieve the exact G2-R2 report, source commit, 15 fold manifests, feature/preprocessing definitions, graph construction, training curves, checkpoints, and object-level metrics.
- Verify the reported 15/15 fold completion and k6 primary candidate, with k4/k8 as neighborhood ablations.
- Reconcile Node-MLP baseline, model initialization, split IDs, metric aggregation, and whether validation was nested within each fold.
- Do not infer missing metrics from memory or from the 96-node experiment.

### E03 — G2-R3 initialization robustness and topology random control
Purpose: test whether k6 graph gains persist across initialization and exceed a degree-preserving randomized topology.
Frozen scope from the current project decision:
- Compare repeated Node-MLP and k6 graph initializations under the same fold/data/features/training budget.
- Compare the selected k6 topology against degree-preserving randomly rewired k6 controls.
- Use 100–150 epochs with internal validation; exact epoch count, patience, optimizer, seeds, rewiring count, and fold-level selection must be filled from the G2-R2 implementation/report before launch.
- Primary analysis: paired object-level k6 vs Node-MLP and k6 vs randomized-k6 effects, with object-level uncertainty.
- Report initialization variability separately from object variability.
- Do not select a favourable seed or graph realization post hoc.
- Do not combine Reptile with this run. Establish spatial-graph robustness first.
- Keep all per-seed training traces, validation curves, best checkpoints, graph edge lists, rewiring seeds and object-level metrics.

Execution gate: do not start E03 until E02 audit resolves exact data split, graph constructor, metric and training config. If a field cannot be recovered, record it as a protocol deviation and obtain explicit approval before proceeding.

### E04 — Reptile replication and interpretation
Purpose: retain existing Reptile evidence and prevent overclaim.
- 200 meta-iterations × 3 seeds; meta-batch 4; inner steps 10; inner LR 0.001; outer step 0.1; adaptation 5 steps; adaptation LR 0.001; batch 1024; weight decay 0.001.
- Fair primary comparator is Support-matched MLP replaying the same task order and Support Recording IDs; pooled ordinary MLP is secondary because its Recording access schedule differs.
- Report zero-shot and adapted MAE/RMSE/Spearman, adaptation-step curves, per-object paired differences and limitations.
- Existing result: Reptile did not outperform Support-matched MLP in MAE on four historical validation objects. Treat as exploratory (n=4), not confirmatory.

### E05 — Meta-learning extension decision
Do not automatically implement MAML/ANIL. First explain a testable mechanism and define equal architecture, Support/Query IDs, optimizer/update budget, initialization and object-level evaluation. Only run if a validation-only plan is specified and the result can answer a distinct research question.

### E06 — Weight repeated-grasp target audit
Existing result only; no predictive training. Six-grasp Fisher aggregation did not improve split-half edge ranking/Jaccard over pooled-frame Pearson. Keep as negative feasibility evidence; do not merge Weight into the main task or count grasps as new objects.

## 6. Reproducibility and logging requirements

Every run must preserve:
- `config.json`, environment/package versions, input data hashes and file manifest;
- Git commit SHA and working-tree state;
- fold/object/recording manifests and Support/Query Recording IDs;
- all random seeds, graph construction and random-rewiring seeds;
- epoch/iteration training loss and internal validation metrics;
- best checkpoint and final checkpoint, with hashes;
- per-object, per-recording and per-episode metrics at the level needed for audit;
- paired object effects, bootstrap seed and bootstrap outputs;
- wall-clock runtime, warnings, errors, failed/retried jobs and a timestamped run log.

Do not store raw pressure arrays in public artifacts. Keep derived summaries and non-sensitive model artifacts only where repository policy permits.

## 7. Deviations and reporting

Any change to data, preprocessing, split, target, graph topology, loss, optimizer, epochs, stopping rule, metric or inference unit must be logged before execution. If a deviation is made after observing validation outcomes, mark it exploratory and do not present it as pre-registered. Negative and null results must be retained and reported.

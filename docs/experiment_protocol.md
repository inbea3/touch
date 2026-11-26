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
Purpose: preserve the full-548 graph experiment as a distinct experimental stream and audit its evidence.
- Dataset scope: 548 sensors, 26 objects, 78 recordings, 149,878 unordered sensor pairs per recording.
- Outer split seeds: 20261119, 20261120, 20261121; five object-grouped folds per seed; 15 folds total.
- Input: earliest five valid pressure frames; target: subsequent valid-frame Pearson correlations.
- Transform: clip((pressure - 500)/150, 0, 1).
- Node features: 10 dimensions — mean, standard deviation, minimum, maximum, range, slope, last value, first value, normalized x and y.
- Recorded metrics (MAE / RMSE / Spearman): Node-MLP 0.120538 / 0.157582 / 0.344821; k4 0.120255 / 0.157204 / 0.343845; k6 0.120172 / 0.157116 / 0.344356; k8 0.120209 / 0.157171 / 0.343664.
- k6 vs Node-MLP: MAE difference -0.000353 (k6 lower), approximately 0.3%; 22/26 objects favour k6; Spearman difference +0.000085. Treat as a small preliminary increment, not conclusive evidence.
- Reported real k6 vs degree-preserving rewired k6 MAE difference: +0.000655 in favour of real k6; 6/26 objects favour the rewired graph.
- Training ran approximately 45–50 epochs; convergence was not established. Full-node k4/k8 sensitivity is not yet considered fully audited.
- Before any new R3 run, retrieve the exact G2-R2 report, 15 fold manifests, feature/preprocessing code, graph constructor, validation selection procedure, traces/checkpoints, and per-object metrics. Resolve any mismatch between summary tables and detailed outputs.

### E03 — G2-R3 initialization robustness and topology random control
Purpose: determine whether the observed k6 increment is robust to initialization and exceeds a degree/connectedness-preserving randomized topology.
- This run was already started on 2026-11-22 07:48:14 UTC. The latest available log snapshot at retrieval reported status running and 8/225 model runs complete; do not assume that snapshot is the current completion state. First recover the latest run log, progress checkpoint and manifest before taking action.
- Design: 5 object-grouped folds × 3 split seeds (20261119–20261121) × 3 initialization seeds (20261123–20261125).
- Conditions per fold/seed: Node-MLP, real spatial k6, and three connected degree-preserving rewired k6 graphs.
- Rewire seeds: 2026112201, 2026112202, 2026112203.
- Training: 20% internal validation; max 100 epochs; patience 15; min_delta 1e-5; AdamW learning rate 0.0005; weight decay 0.001; validation pairs 8192; evaluation chunk 32768.
- The total planned model runs are 225. Preserve every initialization and graph realization; do not select the best random seed/rewiring realization post hoc.
- Primary comparisons: paired object-level real-k6 vs Node-MLP; real-k6 vs the distribution of rewired-k6 controls. Report initialization variance separately from object-level uncertainty.
- Retain per-run training/validation traces, best checkpoints, edge lists, rewire seeds, object-level metrics, progress checkpoints and timestamped logs.
- Do not combine Reptile with this run. Establish graph/topology robustness first.
- Source hashes recorded in the launch log: run_g2r3_robustness.py 56c8cc…bf897f; run_full548_gnn_comparison.py d03f7b…3d86; run_g2r2_nested_cv.py af6735…55fd9.

Execution gate: recover the latest R3 progress and outputs first. If the run completed, audit all 225 expected runs and compare results to the frozen protocol. If it is still running, resume/monitor the existing run rather than launching a duplicate. If it failed, preserve the failed run and logs before any controlled restart.

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

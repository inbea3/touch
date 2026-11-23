# G2-R3 Reconstruction Protocol — 2026-11-24

Status: SPECIFICATION FROZEN; implementation and execution pending.
Experiment ID: G2-R3-RECONSTRUCTION-20261124
Dataset boundary: STAG/HumanGrasp pressure-only, full 548 sensor nodes.
This is a new reconstruction, not a recovery or continuation of the unverified 2026-11-22 run.

## 1. Why reconstruction is required
The Stage-A blueprint records an earlier G2-R3 design and a stale 8/225 progress snapshot, but the current accessible repository branch and Actions history do not contain the original G2-R3 workflow, scripts, run URL, checkpoint or artifacts. The prior run's completion state and source code therefore cannot be verified. No claim is made that the historical run completed or failed.

## 2. Frozen design recovered from the Stage-A blueprint
- Dataset: STAG/HumanGrasp pressure-only.
- 26 non-empty objects, 78 recordings, 3 recordings per object.
- 548 valid physical pressure sensors; 149,878 undirected sensor pairs per recording.
- Each recording input: earliest 5 valid pressure frames.
- Target: Pearson correlation for each undirected sensor pair over subsequent valid frames.
- Pressure transform: clip((pressure - 500) / 150, 0, 1).
- Node features: 10 dimensions — mean, standard deviation, minimum, maximum, range, temporal slope, last value, first value, normalized x and y coordinates.
- Outer evaluation: 5 object-grouped folds × 3 split seeds (20261119, 20261120, 20261121).
- Initialization seeds: 20261123, 20261124, 20261125.
- Topology conditions: Node-MLP; true spatial k=6 graph; three degree-preserving and connectedness-preserving rewired k=6 graphs (rewire seeds 2026112201, 2026112202, 2026112203).
- Total: 5 folds × 3 split seeds × 3 initialization seeds × 5 topology/model conditions = 225 fits.
- Per-fit training policy: 20% of training objects reserved as internal validation; max 100 epochs; patience 15; min_delta 1e-5; AdamW learning rate 0.0005; weight decay 0.001; 8,192 validation pairs; evaluation chunk 32,768.
- Report MAE as primary; RMSE and Spearman as secondary. Unit of inference is object.

## 3. Important reconstruction boundary
The Stage-A blueprint does not contain the exact G2-R2/R3 model source, model-layer dimensions, graph edge list, rewiring implementation, pair decoder, normalization code, batch sampling schedule, or exact early-stopping/checkpoint selection code. The old source-file hashes alone cannot reconstruct source bytes. Therefore this protocol separates design elements directly recoverable from the frozen blueprint from implementation details that must be explicitly specified and unit-tested in the new reconstruction before full execution.
The reconstruction must not claim bitwise or exact source reproduction of the historical G2-R3. Any implementation choice not documented in the blueprint must be listed in a reconstruction deviation register and held constant across all topology conditions. If the exact G2-R2 implementation or archived source can later be recovered, compare it before running the full grid.

## 4. Data construction and leakage controls
- Rebuild per-recording node features and future Pearson targets from the original metadata.mat.
- Sort frames by timestamp within recording. Use the earliest five hasValidLabel frames as input; target frames are valid frames strictly after the fifth input frame.
- Do not silently require consecutive frame IDs; preserve actual timestamp span and contiguity metadata for each recording.
- Fit any learned scaler using training objects only inside each outer fold. Never fit on validation or held-out objects.
- Construct graph geometry from the official 32×32 sensor mask coordinates, not from target correlations.
- Generate the true k=6 graph once from physical sensor coordinates. Generate three randomized graphs with the same node set, degree sequence and connectedness. Store edge lists and SHA-256.
- For each fold, all recordings of a given object must be assigned together. Internal validation objects are selected only from the outer training objects. Held-out outer fold is never used for early stopping or model selection.
- Use identical recording/fold assignments, target pairs, preprocessing, training budget and evaluation pairs for all five conditions and all three initialization seeds.
- If the original graph coordinate order or exact 548-node mask cannot be verified from the source MAT and existing project code, stop before training and report the blocker.

## 5. Implementation decisions that must be frozen before the 225-fit run
Create docs/g2r3_reconstruction_deviations.md and record: exact node encoder and pair decoder dimensions/activations; permutation-invariant undirected pair representation; message-passing operator and normalization; exact degree-preserving connected rewiring algorithm and acceptance criteria; training pair sampling strategy; handling of constant/near-constant sensor series; metric aggregation from recording to object; deterministic behavior and software versions.
These choices must be fixed before looking at outer-fold results. Use a small synthetic smoke test and unit tests for graph degree, connectedness, symmetry, no self-loops, node count, edge count, split isolation, target symmetry, and deterministic seeds. The smoke test is not counted as one of 225 fits.

## 6. Required outputs and retained training traces
For every fit retain config.json, source commit SHA, input metadata SHA-256 and sensor-mask SHA-256; split manifest; graph edge list and graph SHA-256; epoch-level train/validation loss trace; best epoch, early-stop reason and checkpoint; per-recording and per-object MAE/RMSE/Spearman; environment.txt, run.log and manifest.sha256. At run level retain a 225-row status table. Resume only missing/failed configurations; never overwrite completed fits. Do not upload raw pressure arrays.

## 7. Primary analysis and decision rule
- Primary paired contrasts: true k6 vs Node-MLP; true k6 vs each rewired graph and vs mean rewired performance.
- Aggregate metrics at object level; compute paired object-level differences and object-bootstrap 95% intervals.
- Report number of objects improved, mean/median paired difference, effect size and per-object heterogeneity.
- Assess consistency across split seeds and initialization seeds. Do not treat the 225 fits as 225 independent observations.
- Claim a spatial-topology increment only if true k6 improves over Node-MLP and rewired controls with consistent object-level direction and uncertainty excluding a predeclared practically negligible effect. Otherwise report no reliable topology increment.
- Historical G2-R2 summary (Node-MLP MAE 0.120538; k6 MAE 0.120172) is context only and is not pooled with reconstruction outputs until raw artifacts and protocol are verified.

## 8. Execution gates
Gate 1: source-data and 548-node mask available; input hash recorded.
Gate 2: node coordinate order and graph construction verified.
Gate 3: all five topology conditions pass graph and symmetry unit tests.
Gate 4: one-fit smoke test completes and produces logs/checkpoint/manifest.
Gate 5: freeze deviation register and commit SHA.
Gate 6: launch 225 fits with resumable per-fit directories and artifact retention.
Gate 7: audit all 225 statuses and independently recompute aggregate metrics from per-object outputs.

## 9. Current status
Protocol reconstructed from the Stage-A blueprint. Full implementation has not yet been written or run. No new training was launched in this protocol-freezing step. Required next dependency: original pressure-only metadata.mat available to the execution environment, plus confirmation of the canonical 548-sensor coordinate ordering. The raw licensed dataset must not be committed to GitHub.
# HAS Stage 23 — cross-object Reptile adaptation protocol (frozen)

Date: 2026-11-23
Dataset: TUM-ICS/Hide-and-Seek, immutable revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf.
Feature artifact: Stage 22A run 45193420854. Primary window: 5 frames. Sensitivity: 10 frames.
Fixed object-disjoint split: 21 train objects / 6 validation objects / 6 test objects; 2,452 / 880 / 860 episodes. Episode counts per object are unequal (65–255), so meta-task sampling is uniform over objects, never proportional to episode count.

## Research question
Does a Reptile initialization learned across training object tasks improve few-shot adaptation to a held-out object, compared with an ordinary pooled MLP trained on the same training objects and adapted with the same support episodes and gradient-step budget?

## Task and sample definition
- Meta-task: one object_group.
- Sample: one pressure episode row (168-D input; 42-D future-mean target).
- Support/query split is within each object, by episode row; no episode may appear in both.
- Primary shots: 1, 5 and 10 support episodes. Query uses all remaining episodes for that object.
- Meta-train tasks: 21 train objects; Meta-validation: 6 validation objects; Meta-test: 6 test objects.
- 5-frame window is primary; 10-frame window is a secondary sensitivity run. 20-frame window is excluded based on Stage 22B.

## Prediction target and model
Primary target is 42-channel future pressure mean. Model: MLP 168 -> 128 ReLU -> 64 ReLU -> 42. Inputs and targets are standardized using Train episodes only. Train target mean is the constant baseline; last observed pressure (last 42 features) is persistence baseline. No object name/label is a model input.

## Reptile algorithm
At each meta-iteration, sample B=4 distinct training object tasks uniformly. For each task, sample a Support mini-batch of 8 episodes (with replacement only if needed), initialize task parameters from θ, and perform K=5 Adam inner updates (batch size 8, lr=1e-3). Average task parameter displacements equally:
θ <- θ + ε * mean_i(φ_i - θ), with ε=0.05.
The Reptile outer update uses Support-only task adaptation; Query is not used in outer updates. Query is used only for validation and final evaluation. This is first-order Reptile, not MAML.

## Fair ordinary baseline
Train the same MLP on pooled Meta-train episodes. Match Reptile's total number of parameter-gradient updates exactly: each meta-iteration performs B*K inner gradient updates, so pooled MLP performs B*K ordinary minibatch updates per meta-iteration, with batch size 8 and Adam lr=1e-3. Same three seeds and Train-only scalers. Record total updates and wall time. At evaluation, adapt ordinary and Reptile initializations on the exact same support episode IDs, using the same Adam optimizer settings and 1, 5, or 10 support episodes; report zero-shot and adapted performance.

## Validation and selection
Run 300 meta-iterations, B=4, K=5, ε=0.05, 3 seeds (20261123–20261125). Use Validation objects only to assess stability; no hyperparameter search on Test. Report validation results for every shot/model/seed. Do not choose a seed based on Test. If a follow-up tuning run is needed, preregister it separately and keep Test closed.

## Evaluation
For each of the 6 Test objects, repeat support sampling with 5 fixed support seeds; query episodes are the complement. Use identical support IDs for ordinary MLP and Reptile. Report:
- episode-weighted MAE/RMSE/Spearman;
- object-macro MAE/RMSE/Spearman (object is inferential unit);
- per-object paired differences: ordinary adapted MAE minus Reptile adapted MAE;
- support adaptation gain relative to each initialization's zero-shot result;
- persistence and train-mean references.
Use descriptive object bootstrap intervals only (6 test objects; very low precision). The existing Test object set has been inspected in prior stages; all Test results remain exploratory and cannot be described as untouched confirmatory evidence.

## Provenance and retained training traces
Save model checkpoints (Reptile initialization and pooled MLP per seed), per-iteration Reptile train loss/parameter displacement, pooled baseline loss, validation metrics, per-object/per-shot predictions and metrics, support/query episode IDs, exact configs, seed values, environment, run.log and SHA-256 manifest. Do not save raw pressure frame sequences. Never overwrite earlier artifacts.

## Decision rule
Reptile is considered to show a promising adaptation benefit only if it reduces object-macro MAE versus ordinary MLP fine-tuning consistently across seeds and shots, with positive paired differences across a majority of held-out objects. Any benefit limited to one shot or a few objects is conditional. If it does not outperform ordinary fine-tuning, report the negative result; do not automatically escalate to MAML. GNN is a separate later experiment and requires a physically justified graph prior.

# First-order ANIL protocol: pressure-only cross-Recording adaptation

**Protocol freeze:** 2026-11-19  
**Status:** pre-registered exploratory experiment; no Meta-test evaluation  
**Dataset:** STAG/HumanGrasp, pressure-only  
**Object split:** existing fixed 18 Meta-train / 4 Meta-validation / 4 historical Meta-test  
**Feature pipeline:** exact 96-sensor subset and train-only scaler from selected Reptile run 44502406385

## 1. Research question

Does a meta-initialized shared representation, trained using Query Recording loss after task-specific Support-head adaptation, improve few-shot cross-Recording prediction compared with ordinary pooled training, Reptile, and a Support-only control?

The primary outcome is object-level MAE. RMSE and Spearman are secondary descriptive metrics.

## 2. Task and episode definition

- One object is one meta-task.
- Each object contains exactly three distinct Recordings.
- Each task episode samples one or two Support Recordings (1-shot or 2-shot); the remaining Recording(s) are Query.
- Support and Query are disjoint at Recording level.
- Meta-training samples objects uniformly without replacement within each meta-batch.
- 1-shot and 2-shot are sampled with equal probability during meta-training.
- The exhaustive Meta-validation evaluation enumerates all three possible Support subsets for each shot count and each of the four validation objects.
- Sensor edges are prediction examples nested within a Recording; they are never the inferential unit.

## 3. First-order ANIL algorithm

The model is the same 31-input, 64/32-hidden-unit ReLU MLP used by the Reptile experiments. The final linear layer is the task-specific regression head. The preceding layers form the shared representation.

For each sampled meta-task i:

1. Copy the current meta-initialization θ to task parameters φ_i.
2. Update only the final regression head using Support Recording minibatches for K=9 Adam steps, inner learning rate α=0.001 and weight decay 0.001.
3. Compute the Query loss on all Query Recording edges, using the equal mean of per-Recording MSEs.
4. Compute Query gradients at the adapted task parameters.
5. Transfer those gradients directly to θ and average equally across the four tasks in the meta-batch. This is the first-order approximation: no Hessian/Jacobian through the inner head updates is computed.
6. Apply one Adam outer update to θ, outer learning rate β=0.001 and weight decay 0.001.

The outer objective is Query-aware. Query data are not used in inner-loop adaptation. Every task contributes one equally weighted Query gradient to the meta-batch.

## 4. Frozen training configuration

| Parameter | Frozen value |
|---|---:|
| Meta-iterations | 200 per seed |
| Meta-batch size | 4 objects |
| Inner head steps | 9 per task |
| Inner learning rate | 0.001 |
| Outer learning rate | 0.001 |
| Support minibatch size | 1024 |
| Inner / outer weight decay | 0.001 |
| Evaluation head adaptation | 5 Adam steps |
| Evaluation learning rate | 0.001 |
| Seeds | 20261116, 20261117, 20261118 |
| Sensor subset | 96 / 548 |
| Sensor selection seed | 20261116 |
| Bootstrap repetitions | 5000 |
| Device | CPU |

No hyperparameter selection is performed using Meta-validation or Meta-test in this experiment. These settings are fixed before execution.

## 5. Compute and baseline accounting

For each seed, ANIL performs:
- 200 × 4 × 9 = 7,200 Support head-gradient updates;
- 200 × 4 = 800 full-Query gradient evaluations;
- 200 outer optimizer updates;
- 8,000 total Support/Query backward passes.

The Support-matched ANIL-schedule control starts from the same seed initialization and processes exactly the same object / Support Recording episode schedule, with 9 full-network Support updates per episode (7,200 updates). It does not use Query gradients. This isolates the contribution of Query-aware meta-updating, although it has 800 fewer backward passes than ANIL because it has no Query evaluation.

Two frozen reference models are reused from selected-validation run 44502406385:
- Reptile checkpoint (200 meta-iterations, 3 seeds; 8,000 inner updates per seed);
- ordinary pooled MLP checkpoint (8,000 gradient updates per seed).

Their initialization hashes are verified from the source training traces. The references use the same architecture, sensor subset, feature scaler, train objects and seed-specific initialization. Their original task / Recording schedules differ from ANIL's newly sampled schedule; this difference is explicitly retained as a limitation rather than described as strict schedule matching.

## 6. Validation comparison

All four model initializations are evaluated on identical exhaustive validation Support/Query episodes:

- ANIL;
- Support-matched MLP trained on ANIL's exact task/Support schedule;
- Reptile reference;
- ordinary pooled MLP reference.

For each model, report zero-shot and five-step final-head-only adaptation at learning rate 0.001. The same head-only adaptation operator and Support Recording set are used for every model. This isolates the value of the initialization / representation under the same adaptation capacity. Previous full-network Reptile adaptation diagnostics remain separately reported and are not mixed into this head-only comparison.

Expected row counts:
- 24 exhaustive episodes per seed;
- 36 Query-Recording evaluations per seed, per model and policy;
- 864 episode-query-method-policy rows over three seeds;
- 192 object-seed aggregates;
- 64 object aggregates;
- 16 method-policy-shot summary rows.

## 7. Statistical analysis

- Aggregate Query Recording metrics within object × seed × shot × method × policy.
- Average the three seeds within each object.
- Give each of the four validation objects equal weight.
- Report object-level paired differences and 5,000-resample percentile bootstrap intervals.
- A positive 'baseline MAE − ANIL MAE' favours ANIL.
- A positive 'ANIL adapted MAE − ANIL zero-shot MAE' favours zero-shot.
- No edge-, frame-, Recording-, episode- or seed-level significance tests are permitted.
- With only four validation objects, intervals are descriptive uncertainty summaries, not confirmatory inference.

## 8. Evaluation boundary and limitations

The fixed validation objects were already used in earlier model selection and diagnostics. Therefore this ANIL experiment is exploratory even though its configuration is frozen. The four historical Meta-test objects must not be loaded, inspected, used for early stopping, hyperparameter selection or reported as a final untouched test.

The experiment cannot establish population-wide superiority or confirmatory significance. A thesis-level confirmatory claim requires new independent objects / recordings or an external dataset. The study remains pressure-only and uses 96 of 548 sensors; no video, multimodal, full-array or causal-mechanism claim is made.

## 9. Decision rules

1. ANIL is considered to show a promising exploratory signal only if its head-only adapted MAE is lower than both the Reptile reference and the ordinary pooled MLP on the same object-level validation comparison, with direction consistency reported object by object.
2. If ANIL does not beat the ordinary pooled baseline, do not add more ANIL hyperparameter searches on this validation set.
3. If ANIL beats the Support-matched ANIL-schedule control but not ordinary pooled MLP, interpret that only as evidence that Query-aware outer gradients differ from Support-only sequential training—not as overall predictive superiority.
4. Regardless of outcome, report all four objects, all seeds, the full pre-specified metrics and the small-n limitation.

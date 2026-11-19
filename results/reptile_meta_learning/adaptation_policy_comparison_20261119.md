# Reptile conservative adaptation-policy comparison

**Date:** 2026-11-19  
**Workflow:** [GitHub Actions run 44509113507](https://github.com/inbea3/touch/actions/runs/44509113507)  
**Artifact:** `reptile-adaptation-policy-comparison-44509113507-1`  
**Artifact SHA-256:** `41910cb0e841553b77f6b507f79429a71611e2bc0bf7da09b5b1647fb3198a97`  
**Retention:** through 2026-12-19  
**Scope:** STAG/HumanGrasp pressure-only; 96 selected sensors; fixed 18/4/4 object split; no video or multimodal input.

## 1. Research question and pre-specified design

The previous diagnostic showed that Reptile's MAE did not improve across 0/1/2/5/10/20 full-batch adaptation steps. The next controlled question was whether a more conservative adaptation policy could reveal a useful Reptile initialization.

No model was retrained. The experiment reused the exact three-seed checkpoints and train-only scaler from the selected 200-meta-iteration validation run.

### Compared methods

- Reptile initialization;
- Support-matched sequential MLP;
- Ordinary pooled MLP.

### Pre-specified adaptation policies

| Policy | Adam steps | Learning rate |
|---|---:|---:|
| Zero-shot | 0 | — |
| One-step standard | 1 | 0.001 |
| Two-step standard | 2 | 0.001 |
| One-step conservative | 1 | 0.0001 |
| Two-step conservative | 2 | 0.0001 |

All adaptation uses the same full-Support loss: equal mean of per-Recording MSEs over all Support edges. Weight decay is fixed at 0.001. Each method receives the same Support/Query episode definitions.

### Evaluation and aggregation

- Seeds: 20261116, 20261117, 20261118.
- Meta-train objects: 18.
- Meta-validation objects: 4 (IDs 6, 8, 10, 18).
- Exhaustive validation episodes: 24 per seed (six per object; three 1-shot and three 2-shot).
- Query-recording evaluations: 36 per seed.
- Output: 1,620 episode-level query evaluations, 360 object-seed aggregates, 120 object-level aggregates, 30 method-policy-shot summaries and 44 paired comparison summaries.
- Metrics: MAE (primary), RMSE and Spearman (secondary).
- Aggregation order: query recordings / episodes within object and seed, then seeds within object, then equally weighted object-level summaries.
- Meta-test was not loaded or evaluated.

The four validation objects had already been used in previous model selection and diagnostics. Results and object-bootstrap intervals are exploratory, not confirmatory. The four objects—not seeds, Episodes, recordings, edges or frames—are the inferential unit.

## 2. Full results

Values are means across the four validation objects after averaging the three seeds within each object. Lower MAE/RMSE is better; higher Spearman is better.

### 1-shot

| Method | Policy | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| Reptile | Zero-shot | 0.123225 | 0.160992 | 0.345731 |
| Reptile | 1 step, lr=0.001 | 0.123755 | 0.161494 | 0.344700 |
| Reptile | 2 steps, lr=0.001 | 0.124702 | 0.162622 | 0.343990 |
| Reptile | 1 step, lr=0.0001 | 0.123215 | 0.160962 | 0.345818 |
| Reptile | 2 steps, lr=0.0001 | 0.123216 | 0.160947 | 0.345879 |
| Support-matched MLP | Zero-shot | 0.123537 | 0.163436 | 0.325867 |
| Support-matched MLP | 1 step, lr=0.001 | 0.123074 | 0.162538 | 0.328670 |
| Support-matched MLP | 2 steps, lr=0.001 | 0.122853 | 0.161964 | 0.330492 |
| Support-matched MLP | 1 step, lr=0.0001 | 0.123476 | 0.163327 | 0.326197 |
| Support-matched MLP | 2 steps, lr=0.0001 | 0.123417 | 0.163222 | 0.326517 |
| Ordinary MLP | Zero-shot | 0.122340 | 0.160213 | 0.341859 |
| Ordinary MLP | 1 step, lr=0.001 | 0.122395 | 0.160053 | 0.342052 |
| Ordinary MLP | 2 steps, lr=0.001 | 0.122793 | 0.160330 | 0.341434 |
| Ordinary MLP | 1 step, lr=0.0001 | 0.122323 | 0.160169 | 0.341957 |
| Ordinary MLP | 2 steps, lr=0.0001 | 0.122310 | 0.160131 | 0.342043 |

### 2-shot

| Method | Policy | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| Reptile | Zero-shot | 0.123225 | 0.160992 | 0.345731 |
| Reptile | 1 step, lr=0.001 | 0.123347 | 0.161113 | 0.345822 |
| Reptile | 2 steps, lr=0.001 | 0.124009 | 0.161668 | 0.347383 |
| Reptile | 1 step, lr=0.0001 | 0.123183 | 0.160937 | 0.345863 |
| Reptile | 2 steps, lr=0.0001 | 0.123153 | 0.160895 | 0.346016 |
| Support-matched MLP | Zero-shot | 0.123537 | 0.163436 | 0.325867 |
| Support-matched MLP | 1 step, lr=0.001 | 0.122788 | 0.162176 | 0.328956 |
| Support-matched MLP | 2 steps, lr=0.001 | 0.122384 | 0.161335 | 0.331286 |
| Support-matched MLP | 1 step, lr=0.0001 | 0.123448 | 0.163292 | 0.326217 |
| Support-matched MLP | 2 steps, lr=0.0001 | 0.123362 | 0.163152 | 0.326573 |
| Ordinary MLP | Zero-shot | 0.122340 | 0.160213 | 0.341859 |
| Ordinary MLP | 1 step, lr=0.001 | 0.122314 | 0.159848 | 0.342007 |
| Ordinary MLP | 2 steps, lr=0.001 | 0.122584 | 0.159897 | 0.341589 |
| Ordinary MLP | 1 step, lr=0.0001 | 0.122317 | 0.160151 | 0.341953 |
| Ordinary MLP | 2 steps, lr=0.0001 | 0.122298 | 0.160095 | 0.342036 |

## 3. Main findings

### 3.1 Lowering the Reptile adaptation learning rate prevents most deterioration, but does not create a meaningful gain

Relative to Reptile zero-shot MAE:

| Shot | Policy | Adapted minus zero-shot MAE |
|---|---|---:|
| 1-shot | 1 step, lr=0.001 | +0.000530 |
| 1-shot | 2 steps, lr=0.001 | +0.001477 |
| 1-shot | 1 step, lr=0.0001 | -0.000010 |
| 1-shot | 2 steps, lr=0.0001 | -0.000009 |
| 2-shot | 1 step, lr=0.001 | +0.000122 |
| 2-shot | 2 steps, lr=0.001 | +0.000784 |
| 2-shot | 1 step, lr=0.0001 | -0.000043 |
| 2-shot | 2 steps, lr=0.0001 | -0.000072 |

The conservative learning rate makes adaptation almost neutral. Its apparent MAE improvements are extremely small (roughly 0.00001–0.00007) and should not be interpreted as a useful adaptation gain. Object-level paired bootstrap intervals for these conservative-policy differences include zero.

Standard 0.001 adaptation is consistently worse on average, especially two steps in 1-shot.

### 3.2 Ordinary pooled MLP remains the strongest MAE baseline

Ordinary MLP's zero-shot MAE is 0.122340 for both shot settings. Its best observed policy is two steps at lr=0.0001:

- 1-shot: 0.122310;
- 2-shot: 0.122298.

These differences from zero-shot are negligible. Ordinary MLP is relatively insensitive to the tested adaptation policies and has lower MAE than Reptile under every matched policy in this validation sample.

At zero-shot, ordinary MLP's MAE is lower than Reptile by 0.000885 for both shot settings. Under the conservative learning-rate policies, the gap remains approximately 0.00085–0.00091.

### 3.3 Support-matched MLP benefits from standard short adaptation more than Reptile

Support-matched MLP achieves its lowest MAE at two steps with lr=0.001:

- 1-shot: 0.122853, a reduction of 0.000684 from its zero-shot value;
- 2-shot: 0.122384, a reduction of 0.001153.

Under this policy it also outperforms Reptile by 0.001849 MAE in 1-shot and 0.001625 in 2-shot. The object-bootstrap intervals for these paired differences do not include zero, but n=4 objects and prior validation reuse mean these are descriptive signals only—not confirmatory significance evidence.

The low-learning-rate policy removes most of Support-matched MLP's adaptation benefit as well. This suggests that the three initializations respond differently to Support gradients; it does not establish that Support-matched MLP is a generally superior meta-learning algorithm.

### 3.4 Ranking and absolute-error metrics tell slightly different stories

Reptile has the highest mean Spearman among the three models in most settings (about 0.344–0.347), but it does not have the lowest MAE or RMSE. This indicates that ranking agreement and absolute calibration are not interchangeable. The primary MAE objective should remain explicit, with Spearman reported as a secondary measure.

### 3.5 Object-level behavior remains heterogeneous

For example, Object 6 deteriorates substantially under Reptile's standard two-step policy (1-shot MAE 0.153532 versus 0.151291 zero-shot; 2-shot 0.155151 versus 0.151291). Object 10 improves under 2-shot standard two-step adaptation (0.114214 versus 0.116679 zero-shot). Object 8 changes only slightly, while Object 18 deteriorates.

This reinforces the need to report object-level results and avoid assuming a single adaptation policy will benefit every new object.

## 4. Interpretation and limits

The controlled comparison answers the immediate question:

**Conservative adaptation does not reveal a meaningful Reptile advantage.** A tenfold smaller inner learning rate prevents much of the MAE degradation, but it mostly leaves Reptile at its zero-shot performance. The ordinary pooled MLP remains better on MAE, while Support-matched MLP can benefit from two standard short updates.

This does not prove that Reptile is intrinsically ineffective. The evidence is limited by:
- only 18 distinct Meta-train objects and four Meta-validation objects;
- repeated use of the same validation object pool in earlier selection and diagnostics;
- correlated Support/Query permutations over the same three recordings;
- no untouched independent object-level confirmation set.

Bootstrap intervals resample the four objects and are included as descriptive uncertainty summaries only. Do not turn them into confirmatory p-values or claim population-wide significance.

## 5. Recommended research decision

1. **Do not claim a Reptile improvement.** The current evidence does not support that statement.
2. **Do not continue broad Reptile learning-rate or step-count searches on the same four validation objects.** The pre-specified conservative policy comparison has already answered the narrow question.
3. **Keep Ordinary MLP as the strongest current MAE reference.** Keep Support-matched MLP as the important control for task-sequential Support exposure.
4. **If another meta-learning method is explored, formulate it around the actual cross-Recording objective.** A query-aware outer objective (e.g. a carefully budget-matched first-order ANIL/MAML-style experiment) is a more principled next research question than further reducing Reptile's test-time learning rate. It must be treated as exploratory on the current split and must not use Meta-test for design or tuning.
5. **For a confirmatory thesis claim, obtain new independent objects or an external recording set.** The current 18/4/4 object pool cannot supply an untouched final test after the accumulated model-selection and diagnostic work.

## 6. Reproducibility

- Runner: `scripts/compare_reptile_adaptation_policies.py`
- Tests: `tests/test_reptile_adaptation_policy_comparison.py`
- Workflow: `.github/workflows/reptile-adaptation-policy-comparison.yml`
- Outputs: five CSVs plus `adaptation_policy_summary.json`.
- Artifact SHA-256: `41910cb0e841553b77f6b507f79429a71611e2bc0bf7da09b5b1647fb3198a97`.
- The artifact contains aggregate and episode-level diagnostic outputs and data-audit metadata, not raw STAG pressure frames.

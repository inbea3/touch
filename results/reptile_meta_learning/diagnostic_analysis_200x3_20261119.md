# Reptile 200×3 validation: diagnostic analysis

**Date:** 2026-11-19  
**Evidence:** GitHub Actions run `44502406385`, Artifact `reptile-selected-validation-44502406385-1`  
**Scope:** STAG/HumanGrasp pressure-only; fixed 18/4/4 object split; no video. All conclusions are exploratory because the object pool has already been used in earlier project stages.

## 1. Main finding

Reptile did not improve MAE over the support-matched sequential MLP baseline. More importantly, Support adaptation increased Reptile's own MAE relative to its zero-shot initialization:

| Method | 1-shot MAE | 2-shot MAE |
|---|---:|---:|
| Reptile zero-shot | 0.12323 | 0.12323 |
| Reptile Support-adapted | 0.12587 | 0.12454 |
| Support-matched MLP zero-shot | 0.12354 | 0.12354 |
| Support-matched MLP Support-adapted | 0.12332 | 0.12273 |
| Ordinary MLP zero-shot | 0.12234 | 0.12234 |
| Ordinary MLP Support-adapted | 0.12443 | 0.12356 |

This indicates that the current adaptation procedure is not reliably improving generalization. It does not establish that Reptile as an algorithm is unsuitable.

## 2. Highest-priority protocol mismatch: inner-loop train/evaluation mismatch

The code and frozen configuration show:
- Reptile meta-training uses 10 Adam inner updates per task, learning rate 0.001.
- Each inner update samples a total batch of 1,024 edge rows across Support Recordings (equal per-Recording MSE weighting; 1-shot uses 1,024 rows from one Recording; 2-shot uses 512 from each of two Recordings).
- Validation-time adaptation uses only 5 Adam updates, but computes the loss on every edge in every Support Recording (`full_support_loss`).

Thus the meta-initialization is optimized for a 10-step stochastic-minibatch adaptation operator, while reported validation uses a 5-step full-batch operator. Both update count and gradient estimator differ. This is a concrete methodological mismatch and the first issue to isolate. It is a plausible explanation, not yet a proven cause.

## 3. Adaptation may overfit or be noisy

Ordinary pooled MLP adaptation also worsened mean MAE:
- 1-shot: 0.12234 → 0.12443
- 2-shot: 0.12234 → 0.12356

Reptile worsened more:
- 1-shot: 0.12323 → 0.12587
- 2-shot: 0.12323 → 0.12454

The support data comprise only one or two Recordings per object. Although each Recording yields 4,560 sensor-pair rows, these rows are highly dependent through shared sensors and the same underlying pressure time series; they must not be treated as 4,560 independent observations. Full-batch adaptation over correlated edge rows can still overfit the specific Support Recording(s), and Adam steps may move the model away from a useful cross-object initialization.

The pattern is not universal across objects, which reinforces that pooled MAE alone hides object-specific adaptation behavior.

## 4. Object-level heterogeneity and influential object

For Reptile adaptation minus Reptile zero-shot MAE:
- Object 6: +0.00813 (1-shot), +0.00761 (2-shot), clearly worse after adaptation.
- Object 8: -0.00137 (1-shot), -0.00135 (2-shot), improved.
- Object 10: +0.00053 (1-shot), -0.00218 (2-shot), mixed.
- Object 18: +0.00329 (1-shot), +0.00118 (2-shot), worse.

Object 6 is particularly influential. Against the Support-matched MLP, Reptile's object-level MAE disadvantage is 0.00644 for 1-shot and 0.00570 for 2-shot on object 6. Removing object 6 descriptively reduces the mean disadvantage from 0.00255 to 0.00125 (1-shot), and from 0.00181 to 0.00051 (2-shot). This leave-one-object-out calculation is diagnostic only; it must not be used to change the primary reported metric or cherry-pick a preferred subset.

The current artifact does not contain per-object target-distribution summaries, per-task query losses, task-gradient cosine similarities or per-object adaptation trajectories on Meta-train. Therefore task heterogeneity is a plausible mechanism, but it has not yet been directly established.

## 5. What the training trace does and does not show

Across the three seeds, mean support minibatch MSE in the first 50 iterations versus last 50 iterations decreased:
- Seed 20261116: 0.02444 → 0.02265
- Seed 20261117: 0.03349 → 0.02426
- Seed 20261118: 0.02964 → 0.02340

The recorded norm of the meta-batch mean task displacement also decreased (first-50 means about 0.286–0.352; last-50 means about 0.189–0.197). This suggests smaller average inner-loop task displacements later in training and a reduction in sampled Support loss. It does **not** prove meta-generalization or convergence to an optimal initialization. The logged displacement is the norm of the current meta-batch mean task displacement, not cumulative distance travelled by the initialization. The trace does not log task-gradient alignment or held-out Query performance.

Task sampling itself was reasonably balanced: each of the 18 Meta-train objects was sampled about 33–58 times per seed, and 1-shot/2-shot episodes were close to 50/50. The number of independent training tasks nevertheless remains 18 objects, not 800 episodes.

## 6. Statistical caution

There are only four Meta-validation objects. Three random seeds improve assessment of optimization randomness, but they do not increase the number of independent objects from four to twelve. The object-bootstrap intervals are descriptive and fragile at n=4. All 26 objects have also influenced earlier project decisions, so these are exploratory rather than confirmatory results. Do not use Meta-test for tuning.

## 7. Diagnostic priorities before another full training run

1. **Align adaptation operators.** First compare validation adaptation using the same 10 inner steps and 1,024-row per-Recording balanced minibatch procedure used during Reptile meta-training. Also retain the current 5-step/full-Support result as a secondary protocol, clearly labelled. Do not silently replace the pre-registered primary metric; treat this as a protocol diagnostic.
2. **Run an adaptation-step curve** (0, 1, 2, 5, 10, 20 steps) with fixed support episodes and identical optimizer/loss rules across Reptile, Support-matched MLP and pooled MLP. Use only the existing Meta-validation objects and label all results exploratory.
3. **Measure per-object behavior.** Save per-object, per-seed MAE change from zero-shot to adapted, plus Support loss and Query loss. Report object 6, but retain all four objects in the primary aggregate.
4. **Measure task compatibility.** During Reptile training, log per-task parameter displacement norms and pairwise cosine similarities between task displacement vectors; summarize variation by object and shot. Add train-task target summaries without treating edges as independent.
5. **Only then decide on a limited sensitivity analysis or ANIL.** Do not jump to MAML or expand the model before resolving the adaptation mismatch.

## 8. Current conclusion

The most defensible interpretation is: the current Reptile configuration has not demonstrated an advantage; its test-time Support adaptation can worsen MAE; and the mismatch between meta-training and evaluation inner loops is a concrete design issue that should be isolated before drawing algorithm-level conclusions. Small numbers of object tasks and object-specific behavior are additional limitations, but need further diagnostics.

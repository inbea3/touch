# Reptile 200×3 validation: completed diagnostic analysis

**Date:** 2026-11-19  
**Base validation run:** [GitHub Actions 44502406385](https://github.com/inbea3/touch/actions/runs/44502406385)  
**Diagnostic run:** [GitHub Actions 44506987275](https://github.com/inbea3/touch/actions/runs/44506987275)  
**Diagnostic artifact:** `reptile-diagnostics-44506987275-1`, SHA-256 `3a0709587348ddd83cd1a8fa222c9bba6bc1a89fd022e57bf37009001e6849bb`, retained through 2026-12-19.  
**Scope:** STAG/HumanGrasp pressure-only; 96 selected sensors; fixed 18/4/4 object split; no video or multimodal input. All validation findings are exploratory because all 26 objects have already participated in earlier research stages.

## 1. Diagnostic design

The diagnostic reuses the exact three-seed checkpoints and train-only scaler from the 200-iteration validation run. It does not retrain or tune the models.

- **A — Adaptation-step curve:** Evaluate 0, 1, 2, 5, 10 and 20 full-Support Adam steps on the same exhaustive Support/Query Recording permutations for every model.
- **B — Adaptation-operator alignment:** Compare 10 full-Support steps with 10 Reptile-training-style stochastic minibatch steps (per-Recording equal weighting, total batch size 1,024).
- **C — Task update-direction compatibility:** At each trained Reptile initialization, compute 10-step inner-loop parameter displacements for every 1-shot and 2-shot Support combination for each of the 18 Meta-train objects. Average combinations within object, then compute the 153 pairwise cosine similarities between object-level displacement directions.

The inferential unit remains the object. Repeated Support/Query permutations, seeds, pairwise task comparisons, sensor edges and frames are not independent object-level replicates.

## 2. A — Adaptation-step curve

Mean MAE across the four Meta-validation objects (n=4 objects; each object is first averaged across the three seeds). Lower is better.

### 1-shot

| Method | 0 steps | 1 | 2 | 5 | 10 | 20 | 10-step train-style minibatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reptile | 0.12323 | 0.12376 | 0.12470 | 0.12587 | 0.12687 | 0.12795 | 0.12622 |
| Support-matched MLP | 0.12354 | 0.12307 | 0.12285 | 0.12332 | 0.12513 | 0.12664 | 0.12491 |
| Ordinary pooled MLP | 0.12234 | 0.12240 | 0.12279 | 0.12443 | 0.12542 | 0.12642 | 0.12525 |

### 2-shot

| Method | 0 steps | 1 | 2 | 5 | 10 | 20 | 10-step train-style minibatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reptile | 0.12323 | 0.12335 | 0.12401 | 0.12454 | 0.12394 | 0.12379 | 0.12397 |
| Support-matched MLP | 0.12354 | 0.12279 | 0.12238 | 0.12273 | 0.12426 | 0.12382 | 0.12396 |
| Ordinary pooled MLP | 0.12234 | 0.12231 | 0.12258 | 0.12356 | 0.12347 | 0.12338 | 0.12339 |

### Interpretation

1. **Reptile does not show an MAE gain from adaptation in this validation sample.** For 1-shot, full-batch MAE worsens monotonically as steps increase. For 2-shot, it rises through five steps and partially recovers at 10–20 steps, but remains above its zero-shot MAE.
2. **The effect is not unique to Reptile.** Ordinary pooled MLP also deteriorates after a few steps. The Support-matched MLP improves slightly at 1–2 steps, then worsens at larger step counts. This supports an over-adaptation / Support-specific fitting explanation, although it does not by itself prove statistical overfitting.
3. **The best adaptation horizon is short and model-dependent.** The Support-matched MLP's lowest MAE occurs at two steps for both shot counts; ordinary MLP is best at zero/one step. Reptile is best at zero steps in both settings.
4. Reptile's Spearman correlation generally rises with more steps (1-shot 0.3457 at zero steps to 0.3632 at 20; 2-shot 0.3457 to 0.3767), while MAE/RMSE do not improve. This suggests ranking and absolute-error calibration behave differently; the metrics must be reported separately.

## 3. B — Does train/evaluation adaptation mismatch explain the result?

There is a real protocol mismatch in the original run:
- Reptile meta-training: 10 Adam updates, each based on randomly sampled per-Recording minibatches (total batch size 1,024).
- Original validation: 5 Adam updates using all Support edges.

The diagnostic compares 10-step full-batch and 10-step training-style minibatch adaptation from the same checkpoints and on the same episodes.

| Shot | Method | Full-batch 10-step MAE | Train-style minibatch 10-step MAE | Difference (mini − full) |
|---:|---|---:|---:|---:|
| 1 | Reptile | 0.12687 | 0.12622 | -0.00065 |
| 1 | Support-matched MLP | 0.12513 | 0.12491 | -0.00022 |
| 1 | Ordinary pooled MLP | 0.12542 | 0.12525 | -0.00017 |
| 2 | Reptile | 0.12394 | 0.12397 | +0.00004 |
| 2 | Support-matched MLP | 0.12426 | 0.12396 | -0.00029 |
| 2 | Ordinary pooled MLP | 0.12347 | 0.12339 | -0.00009 |

**Conclusion:** the batch-construction mismatch is genuine, but aligning it does not materially rescue Reptile's MAE. It improves 1-shot Reptile by only 0.00065 and has virtually no effect for 2-shot. At 10 training-style minibatch steps, Reptile remains worse than its zero-shot MAE (0.12323) in both shot settings. Therefore the mismatch is a protocol limitation to correct in future experiments, but it is not sufficient to explain the current lack of Reptile gain.

The difference between 5 and 10 inner steps also does not provide a simple explanation: more steps worsen 1-shot, while 2-shot partially recovers but still does not beat zero-shot.

## 4. C — Task update-direction compatibility

At each trained Reptile initialization, the mean 10-step parameter displacement was computed per Meta-train object, averaging over all three Support combinations for each shot count. The 18 object-level vectors produce 153 pairwise cosine similarities per seed.

| Shot | Mean cosine (across seeds) | Median cosine | Negative-pair fraction | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 0.2544 | 0.2688 | 8.9% | Mostly positive, modest alignment |
| 2 | 0.2085 | 0.2299 | 13.9% | Weaker alignment; more opposing pairs |

The pattern is stable across the three seeds:
- 1-shot mean cosine: 0.2517, 0.2508, 0.2607.
- 2-shot mean cosine: 0.2030, 0.2078, 0.2148.

**Conclusion:** the results do not support a claim of widespread, severe task-direction conflict. Most object-pair directions are positively aligned, although alignment is only modest and 2-shot is less compatible than 1-shot. This is a descriptive geometric diagnostic, not an inferential test: the 153 pairwise comparisons share objects and are dependent. Task heterogeneity may still matter, but it is not established as the dominant failure mechanism by these results.

## 5. Object-specific adaptation

Reptile's MAE change (adapted minus zero-shot), averaged across seeds:

| Object | 1-shot, 5 steps | 2-shot, 5 steps | Pattern |
|---:|---:|---:|---|
| 6 | +0.00813 | +0.00761 | Strong deterioration in both settings |
| 8 | -0.00137 | -0.00135 | Improvement in both settings |
| 10 | +0.00053 | -0.00218 | 2-shot improves |
| 18 | +0.00329 | +0.00118 | Deterioration in both settings |

Object 6 is an influential difficult case, but it is not the only source of the overall result. Object 8 improves, and object 10 benefits from 2-shot adaptation. This supports object-dependent adaptation behavior and motivates reporting per-object results rather than only pooled means.

## 6. Updated diagnosis

The completed diagnostics revise the initial hypotheses:

1. **Most directly supported: excessive / poorly targeted Support adaptation.** Reptile's MAE does not improve at any tested adaptation horizon; the other models also show deterioration after a small number of steps. The optimal step count is model- and shot-dependent.
2. **Confirmed but not dominant: train/evaluation adaptation-operator mismatch.** It exists in the original protocol, but the controlled 10-step comparison shows only small MAE changes when the minibatch operator is aligned.
3. **Possible but not dominant based on current evidence: task heterogeneity.** Update directions are moderately aligned overall, with a minority of negative pairs. There is no evidence here for severe global gradient conflict.
4. **Structural limitation: only 18 independent Meta-train objects.** The 800 episodes per seed do not create 800 independent tasks; the number of distinct training objects remains 18.

A further conceptual point is that the current Reptile outer update is based on Support-trained parameter displacement, while the scientific objective is low Query-recording error. Reptile does not directly optimize Query loss. This is not an implementation bug—standard Reptile is first-order—but it may limit alignment between the meta-training signal and the cross-Recording generalization target in this dataset.

## 7. Recommended next step

Do not immediately run another 200×3 training or a 27-configuration search. The current diagnostics indicate that simply changing the test-time adaptation operator is unlikely to solve the problem.

The next controlled experiment should be a **pre-specified validation-only adaptation-policy comparison** using the existing checkpoints:
- zero-shot (primary reference);
- one-step adaptation;
- two-step adaptation;
- optional small learning-rate reduction (pre-specified, not selected post hoc);
- identical Support recordings and Query recordings across all methods.

If the objective remains MAE, select adaptation policy using only Meta-validation and report the selection as exploratory. Do not use the four historical Meta-test objects for tuning. If the thesis needs a confirmatory claim, obtain new independent objects/recordings or explicitly state that the current dataset cannot provide an untouched object-level test.

No MAML/ANIL implementation is recommended until this simpler adaptation-policy issue is resolved.

## 8. Reproducibility

- Diagnostic script: `scripts/diagnose_reptile_adaptation.py`
- Workflow: `.github/workflows/reptile-diagnostics.yml`
- Diagnostic outputs: `adaptation_episode_metrics.csv`, `adaptation_object_seed_metrics.csv`, `adaptation_object_metrics.csv`, `task_direction_object_metrics.csv`, `task_direction_pairwise_cosines.csv`, `diagnostic_summary.json`.
- Artifact SHA-256: `3a0709587348ddd83cd1a8fa222c9bba6bc1a89fd022e57bf37009001e6849bb`.
- All inputs remain pressure-only. Meta-test was not evaluated. The artifact contains diagnostics and audit metadata, not raw STAG frames.

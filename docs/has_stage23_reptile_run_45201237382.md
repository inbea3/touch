# HAS Stage 23 — Reptile cross-object adaptation run report

- Workflow run: 45201237382
- URL: https://github.com/inbea3/touch/actions/runs/45201237382
- Artifact: has-reptile-stage23-45201237382-w5
- Artifact ID: 14832555618
- Artifact SHA-256 (GitHub): e981ef30c04a866e7fa89b83361c68cb20b9f3d87a1bb07f1da353e4d62c790a
- Workflow conclusion: success; runtime 208.6 s
- Input: Stage 22A run 45193420854, 5-frame dataset
- Output manifest: all 18 files independently verified.
- Three seeds; 300 Reptile meta-iterations per seed; 6,000 gradient updates per model/seed.
- Objects: 21 train / 6 validation / 6 test; episodes 2452 / 880 / 860.
- Support repeats: 5; shots: 1, 5, 10; adaptation steps: 10.

## Validation results (primary selection evidence)

Object-macro and episode-average metrics coincide here because all episode rows are averaged within object before equal-object aggregation, and each object has equal repeat/seed count.

| Shots | Model | MAE | RMSE | Spearman |
|---:|---|---:|---:|---:|
| 1 | Persistence | **0.01739** | 0.06327 | 0.4584 |
| 1 | Pooled MLP adapted | 0.02422 | 0.07341 | 0.5272 |
| 1 | Reptile adapted | 0.02760 | 0.08316 | 0.5815 |
| 5 | Persistence | **0.01739** | 0.06326 | 0.4577 |
| 5 | Pooled MLP adapted | 0.02373 | 0.07307 | 0.5001 |
| 5 | Reptile adapted | 0.02587 | 0.07811 | 0.5826 |
| 10 | Persistence | **0.01737** | 0.06312 | 0.4577 |
| 10 | Pooled MLP adapted | 0.02313 | 0.07141 | 0.5098 |
| 10 | Reptile adapted | 0.02494 | 0.07522 | 0.5879 |

Reptile adapted MAE is worse than pooled MLP adaptation for all three shots. Object-level paired differences (pooled adapted MAE minus Reptile adapted MAE) are negative on average:
- 1-shot: -0.003387; Reptile better on 2/6 validation objects.
- 5-shot: -0.002141; Reptile better on 1/6.
- 10-shot: -0.001814; Reptile better on 2/6.

Reptile increases per-episode channel Spearman relative to pooled adaptation, but this does not compensate for its worse MAE/RMSE. It is not evidence of an overall adaptation advantage.

## Exploratory Test results (not confirmatory)

| Shots | Model | MAE | RMSE | Spearman |
|---:|---|---:|---:|---:|
| 1 | Persistence | **0.01577** | **0.05581** | 0.4180 |
| 1 | Pooled MLP adapted | 0.02125 | 0.05996 | 0.5189 |
| 1 | Reptile adapted | 0.02110 | 0.06450 | 0.5870 |
| 5 | Persistence | **0.01577** | **0.05589** | 0.4182 |
| 5 | Pooled MLP adapted | 0.02045 | **0.05803** | 0.4960 |
| 5 | Reptile adapted | 0.01959 | 0.06121 | 0.5707 |
| 10 | Persistence | **0.01579** | 0.05593 | 0.4180 |
| 10 | Pooled MLP adapted | 0.01979 | **0.05656** | 0.4965 |
| 10 | Reptile adapted | 0.01913 | 0.06025 | 0.5743 |

Test paired MAE differences favor Reptile modestly for 5/6 objects at all shots; mean gains are 0.000147, 0.000852, 0.000656 for 1/5/10 shots. However, this fixed Test set has been inspected in prior stages, and the direction conflicts with Validation. Treat as exploratory heterogeneity, not proof of meta-learning benefit.

## Interpretation

1. The strongest absolute-error predictor remains the simple persistence baseline. Both neural initializations plus adaptation are worse on MAE than persistence.
2. Reptile is not supported by Validation: it loses to update-matched pooled MLP fine-tuning on MAE and RMSE at 1/5/10 shots.
3. Test's small Reptile MAE advantage is exploratory and conflicts with Validation; it must not be used to claim success or select the method.
4. Reptile yields higher per-episode channel Spearman than pooled MLP, suggesting it changes channel-wise ordering, but this is a secondary metric and does not establish practical predictive superiority.
5. Do not proceed to expensive Reptile tuning or MAML yet. Diagnose task/sample independence, support sampling, model calibration, persistence-relative residual targets, and the high within-object episode dependence first.
6. The published HAS frame schema documents `episode_index` as the episode/trajectory identifier. The Stage 23 Support/Query split is therefore interpretable as trajectory-level separation when keyed by `(official_split, episode_index)`. The published frame schema does not expose a higher-level recording-session ID, so independence across collection sessions cannot be asserted. Stage 24 audits trajectory-key uniqueness, object/split consistency, and Support/Query disjointness.

## Provenance

Artifact contains three Reptile checkpoints, three pooled MLP checkpoints, per-iteration Reptile support-loss/parameter training traces, pooled training traces, episode metrics, object-macro metrics, paired comparisons, exact config, environment, run.log and manifest.sha256. All output checksums verified. The raw pressure sequences were not saved.

# Reptile selected-config validation: 200 iterations × 3 seeds

**Run date:** 2026-11-19  
**GitHub Actions run:** [44502406385](https://github.com/inbea3/touch/actions/runs/44502406385)  
**Source commit:** `e3fad2171241cb91409268dc014376ce59d7922d)  
**Artifact:** `reptile-selected-validation-44502406385-1` (SHA-256 `68602f212260b93ef7b4d1fcc2026aaee010455c3d8484de813a2ca765a72f36`; retained by GitHub until 2026-12-19)

## Execution and data provenance

- GitHub Actions job completed successfully; all nine job steps passed, including data download, audit, 200-iteration training, output verification and artifact upload.
- Official STAG/HumanGrasp pressure-only archive was downloaded and audited. `metadata.mat` SHA-256: `19062ed3415df6e21ae055c5149315a5350652a27d59ab55fed74e395f2f1378`.
- The audit found 135,187 frames, 32×32 pressure maps, 82 source recordings and 27 object IDs. The modelling manifest uses 26 non-empty objects and 78 recordings (18 Meta-train objects/54 recordings; 4 Meta-validation objects/12 recordings; 4 historical Meta-test objects/12 recordings). The Meta-test split was not evaluated.
- Fixed sensor subset: 96 of 548 physical sensor pixels; 31-dimensional feature representation. Pressure-only throughout; no video or multimodal input.
- The run is explicitly exploratory because all 26 objects were involved in earlier research stages. There is no untouched confirmatory test set.

## Frozen configuration

| Parameter | Value |
|---|---:|
| Meta-iterations | 200 per seed |
| Meta-batch size | 4 |
| Inner steps | 10 |
| Inner learning rate | 0.001 |
| Reptile outer step | 0.1 |
| Adaptation steps | 5 |
| Adaptation learning rate | 0.001 |
| Batch size | 1024 |
| Weight decay | 0.001 |
| Seeds | 20261116, 20261117, 20261118 |
| Bootstrap repetitions | 5,000 |

For each seed, Reptile, Support-matched MLP and pooled ordinary MLP each received 8,000 gradient updates. All three methods shared the same architecture and seed-specific initial parameter hash. The Support-matched MLP replayed the exact Reptile task order and Support Recording IDs, with matching per-Recording loss weighting and batch size. The pooled ordinary MLP is a secondary benchmark: it samples from all Meta-train recordings and therefore does not have the same task/Recording access schedule.

## Meta-validation results

The primary error metric is MAE. Values below are means across the four validation objects, after averaging each object's results over the three random seeds. RMSE and Spearman are secondary descriptive metrics.

| Shots | Method after Support adaptation | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---:|---|---:|---:|---:|
| 1 | Reptile | 0.12587 | 0.16390 | 0.34905 |
| 1 | Support-matched MLP | 0.12332 | 0.16170 | 0.33307 |
| 1 | Ordinary pooled MLP | 0.12443 | 0.16187 | 0.33995 |
| 2 | Reptile | 0.12454 | 0.16176 | 0.35975 |
| 2 | Support-matched MLP | 0.12273 | 0.16068 | 0.33688 |
| 2 | Ordinary pooled MLP | 0.12356 | 0.16053 | 0.34244 |

Reptile zero-shot MAE was 0.12323 for both shot settings. Thus, in this run, Support adaptation did not improve Reptile's MAE.

### Paired object-level MAE differences

Define `Δ = baseline MAE − Reptile MAE`; positive values favour Reptile. The intervals below are 5,000-resample percentile bootstrap intervals over the **four validation objects**, after averaging across seeds.

| Shots | Baseline | Δ | 95% object-bootstrap interval | Objects favouring Reptile |
|---:|---|---:|---:|---:|
| 1 | Ordinary pooled MLP | -0.00144 | [-0.00427, 0.00074] | 2/4 |
| 1 | Support-matched MLP | -0.00255 | [-0.00527, -0.00033] | 1/4 |
| 2 | Ordinary pooled MLP | -0.00098 | [-0.00317, 0.00059] | 1/4 |
| 2 | Support-matched MLP | -0.00181 | [-0.00447, -0.00009] | 1/4 |

Negative Δ means the baseline has lower MAE. The descriptive object-bootstrap intervals favour the Support-matched MLP in this particular four-object validation sample; intervals against the pooled MLP include zero. These are not confirmatory hypothesis tests: the independent inferential unit is the object and n=4 is very small. Do not treat the three seeds, episodes, recordings, frames or sensor edges as independent object-level replicates.

Reptile has somewhat higher mean Spearman correlation than the two adapted baselines in this run, while having worse MAE/RMSE. This metric-specific pattern should be reported rather than collapsed into a single claim of superiority.

## Interpretation and next steps

1. This 200-iteration, three-seed run completed and passed the workflow's output assertions. It establishes a reproducible exploratory comparison, not evidence that Reptile improves few-shot adaptation.
2. The primary fair comparator is Support-matched MLP, not the pooled MLP. It matches architecture, initialization, task sequence, Support recordings, update count and minibatch/loss weighting.
3. Reptile did not improve MAE over the Support-matched baseline for either shot count in this validation sample. Its adaptation also did not improve its own zero-shot MAE.
4. Do not tune on Meta-test. Before any further parameter search, inspect per-object and per-seed variation, verify the training/validation feature pipeline and examine whether the Reptile update direction is producing useful meta-initialization changes.
5. If further tuning is justified, define a limited, pre-specified validation-only sensitivity plan and continue to label results exploratory. Given prior use of all 26 objects, a confirmatory claim requires genuinely new independent objects/data or an explicit limitation in the thesis.

## Reproducibility files

The GitHub Actions artifact contains: `config.json`, `run_summary.json`, `provenance.json`, `episode_metrics.csv`, `object_seed_metrics.csv`, `object_metrics.csv`, `task_sampling.csv`, three training traces, the train-only feature scaler, nine model checkpoints and the data audit outputs. Raw STAG pressure frames are not included in the artifact.

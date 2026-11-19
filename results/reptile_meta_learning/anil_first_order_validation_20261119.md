# First-order ANIL exploratory validation results

**Date:** 2026-11-19  
**Protocol:** docs/anil_meta_learning_protocol.md  
**Workflow:** [GitHub Actions run 44511962054](https://github.com/inbea3/touch/actions/runs/44511962054)  
**Artifact:** anil-meta-learning-44511962054-1  
**Artifact SHA-256:** 635bec289f39fb2986795e31b07a104407e2ab23dd18b5129d3f55c014151b78  
**Retention:** through 2026-12-19  
**Scope:** STAG/HumanGrasp pressure-only; 96 selected sensors; fixed 18/4/4 split; no video or multimodal input.

## 1. Execution status

The frozen experiment completed successfully. Unit tests, compilation, reference checkpoint verification, official dataset download/audit, ANIL training, output-count assertions and artifact upload all passed.

The run reused the exact three-seed Reptile and ordinary pooled MLP checkpoints and the exact train-only scaler from selected-validation run 44502406385. Reference initialization hashes were checked against their original training traces. Meta-test objects were not loaded or evaluated.

The ANIL run produced:
- 3 seeds;
- 200 meta-iterations per seed;
- 800 task episodes per seed;
- 7,200 Support head-gradient updates and 800 Query-gradient evaluations per seed;
- 200 outer optimizer updates per seed;
- 8,000 total Support/Query backward passes per seed;
- 864 episode-query-method-policy rows;
- 192 object-seed aggregates;
- 64 object aggregates;
- 16 method-policy-shot summary rows;
- 56 paired object-difference rows.

## 2. Frozen method and evaluation

ANIL adapts only the final regression head on Support Recording minibatches for nine Adam steps (lr=0.001). The outer objective is the equal mean of full-Recording Query MSEs across the four tasks in each meta-batch. First-order Query gradients are transferred to the initialization without differentiating through the inner updates.

Evaluation enumerates all 1-shot and 2-shot Support combinations for the four Meta-validation objects (IDs 6, 8, 10, 18). All methods receive the same Support/Query split and are evaluated zero-shot and after five steps of final-head-only adaptation at lr=0.001.

Methods:
- ANIL;
- Support-matched MLP trained on ANIL's exact object / Support Recording schedule (7,200 full-network Support updates, no Query outer gradients);
- frozen Reptile reference;
- frozen ordinary pooled MLP reference.

The primary metric is MAE. RMSE and Spearman are secondary.

## 3. Main validation results

Means are calculated by first aggregating Query Recordings within object and seed, then averaging the three seeds within each object, and finally giving the four objects equal weight. Lower MAE/RMSE is better; higher Spearman is better.

### 1-shot

| Method | Evaluation policy | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---|---:|---:|---:|
| ANIL | Zero-shot | 0.123058 | 0.161575 | 0.331403 |
| ANIL | 5-step head-only | 0.123608 | 0.161538 | 0.332178 |
| Support-matched ANIL schedule | Zero-shot | 0.123304 | 0.162215 | 0.319277 |
| Support-matched ANIL schedule | 5-step head-only | 0.123081 | 0.161705 | 0.319351 |
| Reptile reference | Zero-shot | 0.123225 | 0.160992 | 0.345731 |
| Reptile reference | 5-step head-only | 0.123674 | 0.161234 | 0.345793 |
| Ordinary pooled MLP | Zero-shot | 0.122340 | 0.160213 | 0.341859 |
| Ordinary pooled MLP | 5-step head-only | 0.122418 | 0.160034 | 0.341838 |

### 2-shot

| Method | Evaluation policy | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---|---:|---:|---:|
| ANIL | Zero-shot | 0.123058 | 0.161575 | 0.331403 |
| ANIL | 5-step head-only | 0.123342 | 0.161162 | 0.332488 |
| Support-matched ANIL schedule | Zero-shot | 0.123304 | 0.162215 | 0.319277 |
| Support-matched ANIL schedule | 5-step head-only | 0.123182 | 0.161641 | 0.319386 |
| Reptile reference | Zero-shot | 0.123225 | 0.160992 | 0.345731 |
| Reptile reference | 5-step head-only | 0.123441 | 0.161054 | 0.345951 |
| Ordinary pooled MLP | Zero-shot | 0.122340 | 0.160213 | 0.341859 |
| Ordinary pooled MLP | 5-step head-only | 0.122327 | 0.159852 | 0.341864 |

The zero-shot numbers are identical across shot labels because no Support information is used; they are repeated to keep the evaluation table aligned.

## 4. Paired object-level comparisons

The paired difference is defined as baseline MAE minus ANIL MAE; positive values favour ANIL. Each interval resamples the four validation objects and is descriptive only.

| Shots | Policy | Baseline | Mean difference | 95% object-bootstrap interval |
|---:|---|---|---:|---:|
| 1 | Zero-shot | Support-matched ANIL schedule | +0.000246 | [-0.000112, 0.000463] |
| 1 | Zero-shot | Reptile | +0.000167 | [-0.000738, 0.001072] |
| 1 | Zero-shot | Ordinary pooled MLP | -0.000718 | [-0.001199, 0.000067] |
| 1 | 5-step head-only | Support-matched ANIL schedule | -0.000527 | [-0.001405, -0.000013] |
| 1 | 5-step head-only | Reptile | +0.000066 | [-0.000447, 0.000718] |
| 1 | 5-step head-only | Ordinary pooled MLP | -0.001190 | [-0.001904, -0.000426] |
| 2 | Zero-shot | Support-matched ANIL schedule | +0.000246 | [-0.000112, 0.000463] |
| 2 | Zero-shot | Reptile | +0.000167 | [-0.000738, 0.001072] |
| 2 | Zero-shot | Ordinary pooled MLP | -0.000718 | [-0.001199, 0.000067] |
| 2 | 5-step head-only | Support-matched ANIL schedule | -0.000160 | [-0.001141, 0.000812] |
| 2 | 5-step head-only | Reptile | +0.000099 | [-0.000271, 0.000660] |
| 2 | 5-step head-only | Ordinary pooled MLP | -0.001015 | [-0.001869, 0.000144] |

ANIL's own head-adaptation effect is adapted MAE minus zero-shot MAE; positive values favour zero-shot:
- 1-shot: +0.000550; 95% object-bootstrap interval [0.000081, 0.001278].
- 2-shot: +0.000284; interval [-0.000680, 0.001106].

The 1-shot result suggests that five head updates can overfit the single Support Recording. The 2-shot average is also worse, but its object-bootstrap interval includes zero. These small-n intervals are not confirmatory tests.

## 5. Interpretation

### 5.1 Query-aware ANIL is slightly better than the Support-only schedule control before adaptation, but the gain is small

Zero-shot ANIL MAE is 0.123058, compared with 0.123304 for the MLP trained sequentially on the exact same task and Support Recording schedule. The paired difference is +0.000246 in favour of ANIL, but its object-bootstrap interval includes zero. This is a weak exploratory signal that Query-aware outer gradients change the learned initialization; it is not evidence of a reliable performance improvement.

After five head-only adaptation steps, the 1-shot Support-matched control has lower MAE than ANIL. The 2-shot comparison is close and uncertain.

### 5.2 ANIL does not beat the ordinary pooled MLP

Ordinary pooled MLP has the lowest MAE in both shot settings:
- zero-shot MAE: 0.122340;
- 1-shot head-only MAE: 0.122418;
- 2-shot head-only MAE: 0.122327.

ANIL's corresponding MAE is 0.123058 zero-shot, 0.123608 for 1-shot head-only and 0.123342 for 2-shot head-only. The mean paired differences favour ordinary MLP. Thus the pre-specified decision rule for a promising ANIL result is not met.

### 5.3 ANIL head adaptation does not provide a reliable benefit

Five head-only steps worsen ANIL's average MAE by 0.000550 in 1-shot and 0.000284 in 2-shot. The pattern is consistent with Support-specific fitting on very few Recordings. A head-only adaptation restriction does not automatically prevent overfitting.

### 5.4 Reptile and ANIL are close on MAE, but Reptile retains higher ranking correlation

Zero-shot ANIL MAE is 0.000167 lower than Reptile on average, but the object-level interval is wide and crosses zero. After head-only adaptation, the two methods remain close. Reptile has higher Spearman (about 0.346 versus ANIL's 0.331–0.332), while ordinary MLP remains better on absolute error.

Ranking agreement and absolute error should therefore be reported separately.

### 5.5 The ANIL optimization itself completed and reduced Query loss

Across the three seeds, the mean task Query loss in the first ten meta-iterations was approximately 0.0311, 0.0778 and 0.0568. In the final ten iterations it was approximately 0.0249, 0.0257 and 0.0259. This indicates that the Query-aware outer optimization ran and reduced its training Query objective. It does not imply improved generalization to the four validation objects.

## 6. Research decision

The frozen ANIL experiment does not justify a claim that ANIL improves few-shot cross-Recording prediction. It provides a limited positive signal relative to a Support-only schedule control, but it does not beat ordinary pooled MLP and does not show a stable benefit from head-only adaptation.

Recommended decision:
1. Do not proceed directly to full second-order MAML or another broad hyperparameter search on the same four validation objects.
2. Retain ANIL as a meaningful Query-aware meta-learning comparison and report the negative/limited result transparently.
3. Keep ordinary pooled MLP as the strongest current MAE reference; keep Reptile and Support-matched controls to explain what the meta-learning objective changes.
4. If the thesis needs a confirmatory meta-learning contribution, collect new independent object tasks / recordings or obtain an external pressure-only dataset. The existing 18/4/4 pool has already been used in prior research decisions and cannot support an untouched final test.
5. If no new data can be obtained, frame the thesis as an exploratory study of cross-object few-shot adaptation, with the central finding that Query-aware meta-learning changes the initialization but has not yet demonstrated a reliable predictive advantage on the available object pool.

## 7. Limitations

- Only four Meta-validation objects; all were previously used in model selection and diagnostics.
- Object-level bootstrap intervals have n=4 and are descriptive, not confirmatory significance tests.
- The first-order ANIL approximation omits second-order derivatives through the inner head updates.
- The Support-matched ANIL-schedule control has 7,200 Support updates, whereas ANIL uses 7,200 Support updates plus 800 Query gradient evaluations.
- Reptile and ordinary checkpoints are reused from the selected 200-iteration × 3-seed run; their original task / Recording schedules differ from ANIL's newly sampled schedule.
- Only 96 of 548 sensors are used; no full-array performance claim is made.
- Pressure-only throughout; no video, multimodal, causal or physical-mechanism claim is made.
- Meta-test objects were not loaded or evaluated.

## 8. Reproducibility

- Protocol: docs/anil_meta_learning_protocol.md
- Runner: scripts/run_anil_meta_learning.py
- Tests: tests/test_anil_meta_learning.py
- Workflow: .github/workflows/anil-meta-learning.yml
- Workflow run: 44511962054
- Artifact: anil-meta-learning-44511962054-1
- Artifact SHA-256: 635bec289f39fb2986795e31b07a104407e2ab23dd18b5129d3f55c014151b78
- Artifact includes model checkpoints, task schedule, training trace, episode/object metrics, paired comparisons, summary and data-audit outputs. Raw STAG pressure frames are not included.

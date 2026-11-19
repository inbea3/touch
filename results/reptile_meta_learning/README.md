# Stage 9 Reptile implementation and smoke-test status

## Verified execution

- All 17 Reptile implementation and sensitivity-grid unit tests ran successfully in GitHub Actions run `44493081143`.
- A real-data pressure-only smoke test completed successfully in GitHub Actions run `44491723400`. It downloaded the official lite archive, extracted and audited `metadata.mat`, built the 96-sensor features, trained Reptile and the pooled MLP baseline, evaluated the four Meta-validation objects, and verified that all expected output files were created.
- Smoke settings were intentionally minimal: one seed, two meta-iterations, meta-batch 2, two inner steps, and one adaptation step. These outputs are pipeline diagnostics only; they are not a model-performance result and must not be cited as evidence for or against Reptile.

## Fair-comparison audit

Both methods use the same MLP architecture, seeded initialization, Meta-train object pool, and gradient-update count. Their task/recording access schedules are not identical: Reptile samples task Episodes and performs multiple inner updates on the 1/2 Support Recordings, while the pooled MLP samples one object/Recording per gradient step across all three Meta-train Recordings. The implementation records this difference and does not claim strict data-exposure equivalence.

## Current experiment

The 27-configuration Meta-validation sensitivity grid completed successfully in GitHub Actions run `44493070529`. Aggregate-only results are saved in [`sensitivity_grid_summary.json`](sensitivity_grid_summary.json) and [`sensitivity_grid_summary.csv`](sensitivity_grid_summary.csv).
- inner steps: 1, 5, 10
- inner learning rate: 1e-4, 5e-4, 1e-3
- Reptile outer step: 0.01, 0.05, 0.1
- meta-batch: 4
- three fixed random seeds per configuration
- 50 meta-iterations per seed

The selected configuration was K=10, inner LR=0.001, outer step=0.1, with equal-shot mean Reptile MAE 0.128356. For this selected cell, ordinary MLP's equal-shot mean MAE was 0.124438. Across all 27 configurations, Reptile's adapted MAE was higher than ordinary MLP in both 1-shot and 2-shot settings. This is an exploratory negative result, not a universal rejection of meta-learning.

A longer 200-meta-iteration run of the selected configuration is now queued to check whether the result is merely due to short meta-training. Meta-test is excluded from tuning and remains exploratory because all 26 objects were used in earlier stages.

## Data handling and citation

Raw pressure data, extracted features, scalers, checkpoints, and detailed per-Episode/per-object outputs remain in the temporary execution environment. They are not committed to this public repository or uploaded as Actions artifacts.

Subramanian Sundaram, Petr Kellnhofer, Yunzhu Li, Jun-Yan Zhu, Antonio Torralba, and Wojciech Matusik. “Learning the signatures of the human grasp using a scalable tactile glove.” *Nature*, 2019. DOI: 10.1038/s41586-019-1234-z.

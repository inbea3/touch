# Stage 9 Reptile pilot artifacts

Real STAG/HumanGrasp pressure-only run, 2026-11-19. No synchronized video or multimodal data were used.

Tracked in this directory:
- `pilot_summary.json`: configuration, validation MAE, object-bootstrap intervals and interpretation.
- `pilot_object_metrics.csv`: object-level metrics averaged across the three random seeds.
- `pilot_config.json` and `pilot_provenance.json`: run configuration and data/code provenance.

The complete run bundle (episode-level metrics, task sampling, training traces, train-only feature scaler and model checkpoints) is retained separately with the session artifact package. It is not duplicated in Git history as binary model files.

Configuration: 50 meta-iterations, meta-batch 4, five inner steps, inner learning rate 0.0005, outer step 0.05, adaptation steps 5, batch size 1024, and seeds 20261116/20261117/20261118. Both methods received exactly 1,000 gradient updates per seed.

Only the four fixed Meta-validation objects were evaluated. The old Meta-test objects were not run. Since all 26 objects were used in prior stages, this validation is exploratory, not confirmatory. Reptile did not outperform the ordinary MLP fine-tuning baseline in this first configuration; do not interpret this single pilot as a universal rejection of Reptile.

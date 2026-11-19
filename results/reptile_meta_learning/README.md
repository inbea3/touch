# Stage 9 Reptile pilot artifacts

Real STAG/HumanGrasp pressure-only run, 2026-11-19. No synchronized video or multimodal data were used.

- `pilot_summary.json`: configuration, validation MAE, object-bootstrap intervals and interpretation.
- `pilot_episode_metrics.csv`: query-recording metrics for each seed and each 1-shot/2-shot episode.
- `pilot_object_metrics.csv`: object-level metrics averaged across the three seeds.
- `pilot_task_sampling.csv`: Reptile task, Support and Query recording IDs; Query is marked as not used for gradients.
- `pilot_reptile_training_trace.csv`: per-meta-iteration support loss and task parameter displacement.
- `pilot_ordinary_training_trace.csv`: pooled MLP minibatch loss and sampled object/recording IDs.
- `pilot_feature_scaler.npz`: train-only fitted feature scaler.
- `reptile_seed_*.pt`, `ordinary_mlp_seed_*.pt`: trained model checkpoints.
- `pilot_config.json`, `pilot_provenance.json`: configuration and SHA-256 provenance.

Configuration: 50 meta-iterations, meta-batch 4, five inner steps, inner learning rate 0.0005, outer step 0.05, adaptation steps 5, batch size 1024, and seeds 20261116/20261117/20261118. Both methods received exactly 1,000 gradient updates per seed.

Only the four fixed Meta-validation objects were evaluated. The old Meta-test objects were not run. Since all 26 objects were used in prior stages, this validation is exploratory, not confirmatory. Reptile did not outperform the ordinary MLP fine-tuning baseline in this first configuration; do not interpret this single pilot as a universal rejection of Reptile.

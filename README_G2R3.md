# G2-R3 reproducibility package

## Aim
1. Test whether the G2-R2 full-548-node Spatial-GNN k=6 MAE increment is robust to parameter initialization.
2. Test whether real spatial k=6 topology outperforms three independent connected degree-preserving rewired k=6 graphs.

## Fixed study boundaries
- STAG `classification_lite.zip` / `metadata.mat`; pressure-only.
- 548 sensors, 26 non-empty objects, 78 recordings.
- Earliest five valid pressure frames as input; future valid-frame Pearson correlation as the continuous pair target.
- Three fixed object-grouped outer-CV split seeds: 20261119, 20261120, 20261121; five folds each.
- Three initialization seeds: 20261123, 20261124, 20261125.
- Three degree-preserving connected k=6 rewiring seeds: 2026112201, 2026112202, 2026112203.
- Maximum 100 epochs, patience 15; Node-MLP inner-validation object-macro MAE selects one common epoch for the matched model conditions within each fold/initialization.
- Same initialization, scaler, training-recording order and sampled sensor-pair indices within each matched comparison.

## Retained traces and recovery
- `run_manifest.json`: command arguments, data/source SHA-256, environment and protocol.
- `run.log`: complete stdout/stderr from the run.
- `split_manifest.json`: object/recording membership for outer train, inner train, inner validation and outer test.
- `inner_validation_trace.csv`: epoch-by-epoch training MSE and inner-validation object-macro MAE.
- `refit_epoch_trace.csv`: epoch-by-epoch training MSE for each model condition.
- `initial_state.pt`: matched initial parameter state for each fold/initialization.
- `model_checkpoint.pt`: final model, scaler, optimizer state, RNG state and graph hash.
- `test_metrics.json`: recording-level and object-level held-out metrics.
- `progress_checkpoint.json`: atomically updated progress for safe restart.
- `graph_adjacencies.npz` and `graph_audit.json`: exact adjacency matrices, SHA-256, edge counts, degree-sequence and connectivity checks.

The script is restartable at model-condition granularity: completed conditions with both metrics and checkpoint files are reused. If a trace exists for an interrupted, incomplete condition, it is renamed with an `interrupted` suffix before retry so the old trace is not overwritten.

The current execution environment is CPU-only. The short smoke run is an execution check only and is not scientific evidence.

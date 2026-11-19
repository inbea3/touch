# G2: Full-548-node STAG pressure-only spatial graph experiment

**Status:** primary k=6 experiment completed; 3 repeats of 5-fold object-grouped CV (15 outer folds).  
**Task:** predict continuous future sensor-pair Pearson correlations from the earliest five valid pressure frames.  
**Data:** 135,187 × 32 × 32 pressure tensor; 548 frozen active sensors; 26 objects; 78 non-empty recordings; 149,878 unordered sensor pairs per recording.

## Protocol

- Input normalization: `clip((pressure - 500) / 150, 0, 1)`.
- Node features: eight early-window statistics (mean, standard deviation, minimum, maximum, range, time slope, last frame, first frame) plus normalized x/y coordinates.
- Target: Pearson correlation over all subsequent `hasValidLabel=True` pressure frames.
- All 149,878 target pairs were finite for all 78 recordings.
- Outer split seeds: 20261119, 20261120, 20261121; five folds per seed; each object held out once per seed.
- Training: 5 epochs, AdamW (lr=0.0005, weight decay=0.001), 4,096 uniformly sampled valid pairs per training recording/epoch. Pair indices were identical across model conditions.
- Evaluation: all unordered pairs, chunked inference. Feature scaler fitted only on training recordings within each fold.
- Metrics: average recordings within object, then equal-weight the 26 objects.

## Matched model conditions

All three conditions share the same NodeRelNet architecture and initialization; only adjacency changes.

1. Node-MLP: identity adjacency (no cross-node message passing).
2. Spatial-GNN k=6: physical tie-inclusive Euclidean k-NN graph, union symmetrization, self-loops and symmetric degree normalization.
3. Rewired-GNN k=6: connected double-edge swaps preserving the exact degree sequence and edge count.

The physical k=6 graph has 2,226 undirected edges, one connected component, and degree min/median/max of 6/8/11. The rewired control also has 2,226 edges and the same degree sequence; 21,638 of 22,260 target swaps succeeded.

## Results

Mean metrics across the three split repeats:

| Model | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| Node-MLP | 0.122504 | 0.159556 | 0.324799 |
| Spatial-GNN k=6 | **0.122150** | **0.159044** | **0.324883** |
| Rewired-GNN k=6 | 0.123158 | 0.160570 | 0.317289 |

Paired object-level MAE difference is defined as comparison model minus Node-MLP. Averaging each object's paired difference across the three split seeds, then bootstrapping 26 objects (50,000 resamples):

- Spatial-GNN k=6: -0.000353; descriptive 95% object-bootstrap interval [-0.000525, -0.000168]; lower MAE for 22/26 objects.
- Rewired-GNN k=6: +0.000655; interval [+0.000373, +0.000990]; lower MAE for 6/26 objects.

Spatial-GNN's MAE is about 0.29% lower than Node-MLP, while Spearman is effectively unchanged (+0.000085). The real spatial graph has lower MAE than Node-MLP in all three split repeats; the degree-preserving rewired graph is worse.

## Interpretation and limitations

This is preliminary evidence that the full physical sensor topology may provide a small incremental benefit over Node-MLP, and that the real spatial layout matters beyond degree sequence alone. It is not evidence of decisive GNN superiority: the gain is small, Spearman barely changes, and training was fixed at five epochs with loss still decreasing at epoch five. Only k=6 was tested in this multi-repeat full-548 comparison; k=4/k=8 sensitivity remains outstanding. The three repeats reuse the same 26 objects, so the object-bootstrap interval is descriptive, not independent external replication. The active sensor mask is the unsupervised full-archive mask frozen in G0 and should be disclosed as such.

## Next

1. Run full-548 k=4 and k=8 sensitivity under the same split seeds.
2. Increase training budget and select epochs using training-only inner validation.
3. Test multiple independent degree-preserving rewiring realizations.
4. Only if the physical-graph gain remains stable, proceed to ANIL/Reptile integration.

The executable script and complete fold-level outputs are included in the G2 experiment package. The raw metadata file is not committed; its SHA-256 is recorded in the result artifact.

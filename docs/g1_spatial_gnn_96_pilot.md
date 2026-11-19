# G1 Pilot: Edge-MLP vs Node-MLP vs Spatial-GNN (96 sensors)

**Status:** completed exploratory pilot; one repeat of 5-fold object-grouped CV.  
**Data:** STAG pressure-only, 26 non-empty objects, 78 recordings.  
**Node subset:** legacy random 96/548 sensor subset (seed 20261116).  
**Target:** continuous Pearson correlation over all valid future-contact frames after the earliest five valid input frames.

## 1. Purpose

This pilot asks whether explicit physical-neighbor message passing improves future sensor-pair correlation prediction beyond a node-wise encoder and symmetric pair decoder. It is deliberately limited to the same 96 sensors used by the existing Edge-MLP baseline so the first architecture comparison shares the same candidate pairs.

## 2. Matched architecture comparison

- **Edge-MLP:** edge-level representation (31 features per pair), MLP hidden widths 64/32.
- **Node-MLP:** 10 node features (8 early pressure statistics plus normalized grid coordinates), node encoder, two message-mixing blocks, symmetric pair decoder.
- **Spatial-GNN:** exactly the same trainable architecture and initial weights as Node-MLP; only the adjacency changes from identity to a physical k-NN graph (k=4/6/8). This isolates message passing more directly than comparing unrelated network designs.

For Node-MLP, the aggregation matrix is identity. For Spatial-GNN, the graph is Euclidean k-NN with all kth-distance ties included, union symmetrization, self-loops, and symmetric degree normalization. The pair decoder predicts all 4,560 unordered pairs, not just spatial graph edges.

All models used 10 epochs, AdamW (learning rate 0.0005, weight decay 0.001), one update per training Recording per epoch, the same model seed and shuffled Recording order. Edge/node scalers were fitted only on training objects in each fold.

## 3. Evaluation design

One repeat of 5-fold grouped cross-validation was run over all 26 non-empty objects. All three recordings of an object stayed in the same fold; each object was held out exactly once.

| Fold | Held-out object IDs |
|---:|---|
| 1 | 5, 8, 16, 22, 23, 25 |
| 2 | 7, 9, 10, 15, 19 |
| 3 | 3, 12, 17, 18, 20 |
| 4 | 4, 11, 13, 14, 26 |
| 5 | 1, 2, 6, 21, 24 |

Metrics were first averaged over the three held-out recordings of each object, then averaged over 26 objects. Sensor pairs were not treated as independent inferential units.

## 4. Results

| Model | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| Edge-MLP | 0.117187 | 0.153845 | 0.347046 |
| Node-MLP | **0.115275** | **0.151562** | **0.390053** |
| Spatial-GNN k=4 | 0.115431 | 0.151795 | 0.383465 |
| Spatial-GNN k=6 | 0.115683 | 0.151959 | 0.382081 |
| Spatial-GNN k=8 | 0.115715 | 0.151857 | 0.381235 |

### Paired object-level comparison against Node-MLP

Difference is MAE(GNN) - MAE(Node-MLP); negative values favor GNN.

| Graph | Mean paired difference | Exploratory object-bootstrap 95% CI | Objects GNN better |
|---|---:|---:|---:|
| k=4 | +0.000156 | [-0.000181, +0.000520] | 11 / 26 |
| k=6 | +0.000408 | [+0.000037, +0.000807] | 11 / 26 |
| k=8 | +0.000440 | [+0.000015, +0.000895] | 11 / 26 |

## 5. Interpretation

1. Node-MLP improves on Edge-MLP in this pilot, suggesting that a node-wise representation followed by a symmetric pair decoder may be useful.
2. Explicit spatial message passing does **not** show an incremental gain over the matched Node-MLP control on the legacy random 96-node subset. k=4 is very close; k=6 and k=8 are slightly worse on average.
3. This does not justify abandoning GNN. G0 showed that the random 96-node subset produces a much coarser spatial graph than the full 548-sensor layout. This pilot therefore tests message passing on a sparse, subsampled graph, not the full physical sensor topology.
4. The small MAE differences and single CV repeat are exploratory. The bootstrap intervals are descriptive, not confirmatory evidence.

## 6. Decision and next experiment

- Keep Node-MLP as a mandatory matched control.
- Do not claim GNN is better based on this pilot.
- Proceed to a full-548-node pilot, where the physical graph preserves 1-2 pixel neighborhoods.
- Then add a degree-preserving rewired graph control and repeat outer object-grouped CV with multiple split seeds.
- Only after these tests should we decide whether GNN merits integration with ANIL/Reptile.

Per-object metrics and the executable pilot script are included in the downloadable experiment package. This pilot used no video or image data.

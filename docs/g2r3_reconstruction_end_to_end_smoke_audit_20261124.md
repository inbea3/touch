# G2-R3 End-to-End Smoke Audit — 2026-11-24

## Status
A local end-to-end smoke harness completed on the uploaded `classification_lite.zip` metadata. This is an independently executed smoke harness following the committed pipeline specification; it is **not** a completed run of the committed CLI's `--smoke-only` path.

## Data and graph
- 78 non-empty recordings, 26 objects, 548 active pressure sensors.
- Five earliest valid-label frames form input features; subsequent valid-label frames form the future correlation target.
- Spatial graph: 548 nodes, 1,966 undirected edges, connected; degree range 6–10.
- Outer split for this smoke: 16 training objects, 4 validation objects, 6 test objects.

## One-epoch smoke fits
Both Node-MLP and Spatial-GNN k6 completed one training epoch on CPU. Training used 256 randomly sampled valid sensor pairs per training recording; validation and exploratory test metrics were computed across all valid pairs.

| Condition | Train loss | Validation object-macro MAE | Exploratory test object-macro MAE |
|---|---:|---:|---:|
| Node-MLP | 0.017117 | 0.142822 | 0.132318 |
| Spatial-GNN k6 | 0.015135 | 0.140598 | 0.130851 |

Runtime: 18.02 seconds (CPU).

## Interpretation boundary
These values are smoke-test diagnostics only. One epoch, one split, one initialization, and a reduced pair sample are insufficient for model comparison or a thesis claim. The exploratory test values must not be reported as final test performance. The 3 rewired-GNN conditions were module-tested earlier but were not trained in this smoke harness.

## Next gates
1. Execute the committed script's own `--smoke-only` CLI path after the source-level output-directory fix.
2. Run repository unit tests and verify script compilation.
3. Only after successful gates, launch the prespecified 225-fit grid.

## Additional topology and five-condition module checks
- Three connected degree-preserving rewired graphs were generated from the corrected 1,966-edge spatial graph.
- Successful connected edge swaps: 19,226 / 19,215 / 19,186 for seeds 2026112201 / 2026112202 / 2026112203.
- Rewired edge-list SHA-256:
  - 2026112201: `d26d64dcae978a47e374b2a538c73708ca7d91ab219436675d17863ed92a45b1`
  - 2026112202: `e46d9b65f8ce52c989e422a1881aabd8b2e5554959e8738b6c111e462a5f179d`
  - 2026112203: `882a8953759549a99b451895a548a4a203244e1171a448e8b481571eafca4fb1`
- All five conditions passed an additional forward/backward module check. Loss and gradients were finite; pair-order maximum absolute prediction difference was 0.0 for every condition.
- These checks validate graph/model mechanics, not comparative predictive performance.

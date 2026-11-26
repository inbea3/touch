# G2-R3 Reconstruction Module Smoke Audit — 2026-11-24

## Status
Module-level data, graph, forward/backward and pair-symmetry smoke checks passed locally against the uploaded classification_lite.zip. The full script has not yet completed an end-to-end fold fit or the 225-fit grid.

## Verified input
- 78 non-empty recordings across 26 objects.
- Each recording produced 548-node features and 149,878 valid upper-triangle target pairs in this extraction check.
- Five earliest valid-label frames were used as input; later valid-label frames formed the future correlation target.
- Each target was finite for all 149,878 pairs in the inspected 78 recordings under the current data extraction. This is dataset-specific and does not imply future datasets will have no constant channels.

## Graph checks
- Spatial graph: 548 nodes, 1,966 undirected edges, degree range 6–10, connected.
- Three rewired graphs: 1,966 edges each; all connected and preserve the original degree sequence.
- Rewiring seeds: 2026112201, 2026112202, 2026112203.
- NetworkX 3.6.1 API correction: `connected_double_edge_swap` does not accept `max_tries`; implementation now uses the supported signature.

## Model module checks
For Node-MLP, Spatial-GNN k6 and all three rewired GNN conditions:
- 512-pair forward output shape correct.
- Smooth-L1 loss finite.
- All model parameter gradients finite after backward.
- Pair decoder invariance verified: max absolute difference between prediction(i,j) and prediction(j,i) = 0.0.

These checks were run in the local analysis environment using the uploaded metadata and a synthetic 512-pair batch drawn from the first recording's valid targets. They are module-level checks, not a full epoch or held-out validation result.

## Corrected preliminary graph statistics
An earlier preliminary note stated 1,925 edges and degree range 6–11. Direct recomputation corrected this to 1,966 edges and degree range 6–10. Use the corrected statistics in all future records.

## Remaining gates
1. Run repository unit tests through the new GitHub Actions workflow.
2. Execute the complete script on the uploaded metadata with `--smoke-only`, retaining a best checkpoint, epoch trace, and per-object metrics.
3. Review runtime/memory and resumability.
4. Only then launch the 225-fit grid.

No model-performance conclusion is drawn from this module smoke audit.

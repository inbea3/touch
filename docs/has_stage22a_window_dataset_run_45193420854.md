# HAS Stage 22A — matched observation-window dataset build

- Workflow run: 45193420854
- URL: https://github.com/inbea3/touch/actions/runs/45193420854
- Artifact: has-window-dataset-45193420854
- Artifact ID: 14827790165
- Artifact SHA-256: 222f6f779c50fc0cbdd2283c27803f4acfb038b6f7d53926bbcca41e66ae64d7
- Workflow conclusion: success
- Runtime: 371.4 seconds
- Dataset: TUM-ICS/Hide-and-Seek, immutable revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf
- Source frames processed: 1,089,543
- Episodes: 4,192 for each window
- All 10 files listed in manifest.sha256 verified OK.

## Dataset dimensions

| Observation window | Episodes | Input | Target | Train / Val / Test |
|---|---:|---:|---:|---|
| 5 frames | 4,192 | 168 | 42 | 2452 / 880 / 860 |
| 10 frames | 4,192 | 168 | 42 | 2452 / 880 / 860 |
| 20 frames | 4,192 | 168 | 42 | 2452 / 880 / 860 |

Each split retains 21/6/6 disjoint object groups. The same episode IDs are eligible in all three windows, enabling paired comparison. Target for each window is the mean of all frames strictly after that observation window; at least five future frames are required. Raw frame sequences were not persisted.

## Interpretation

The extraction completed with matched episode counts, dimensions and split composition. This validates data availability and comparability only; it does not establish which window or predictor performs best. That requires Stage 22B benchmark.

## Provenance

Artifact includes window_5/10/20.npz and corresponding episode indexes, config.json, summary.json, run.log, environment.txt and manifest.sha256. Preserve this artifact unchanged.

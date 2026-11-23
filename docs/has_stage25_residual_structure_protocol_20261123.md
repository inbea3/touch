# HAS Stage 25 — channel and object residual structure diagnostics

## Objective
Diagnose where the 5-frame Persistence and Stage 24B Ridge-residual predictions fail. This stage is descriptive and does not train a new neural architecture or search for a graph.

## Frozen inputs
- Stage 22A dataset artifact run 45193420854.
- 5-frame input, 168 features, 42 future-mean targets.
- Fixed object split: 21 Train / 6 Validation / 6 Test objects.
- Persistence prediction is the final 42-feature block (X[:,126:168]).
- Ridge residual model is refit on Train only with alpha fixed at 1000, selected in Stage 24B using Validation object-macro MAE. No retuning in Stage 25.
- Original channel indices are mapped after excluding source channels 18 and 25.

## Diagnostics
1. Per-channel Persistence and Ridge residual bias, MAE, RMSE and residual standard deviation.
2. Per-object macro metrics and average signed bias.
3. Cross-channel residual error correlation matrices for Persistence and Ridge, with eigenvalue spectra and strongest absolute channel pairs.
4. Count of channels where Ridge improves/worsens MAE relative to Persistence.
5. Train residual target mean and standard deviation summaries.

## Interpretation
- Validation is primary; Test is exploratory only.
- Error covariance is not a physical sensor adjacency graph. Correlated errors may arise from common target dynamics, shared preprocessing or object effects.
- Do not claim GNN suitability from residual correlation alone. A graph prior requires physical sensor layout or independently justified sensor relationships.
- Objects, not episodes, are the relevant comparison units; six Validation objects make inferential claims weak.

## Provenance
Save config, channel/object CSVs, NumPy correlation matrices, eigenvalue summaries, run.log, environment and SHA-256 manifest. Do not overwrite prior artifacts or save raw pressure sequences.

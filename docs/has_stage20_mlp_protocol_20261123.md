# HAS Stage 20 — MLP baseline protocol (frozen before run)

Date: 2026-11-23
Input artifact: Stage 19 feature artifact run 45115559336; dataset revision 4fd8e7e91b0b3ccca8306cc9f07657af19f73ddf; object split seed 20261123.

## Research questions

1. Can the first-five-frame pressure summary predict the future-window per-channel mean on unseen object categories?
2. Can the same input predict future inter-channel Pearson correlation structure on unseen object categories?
3. Does excluding globally constant channels 18 and 25 improve or worsen performance relative to retaining all 44 channels?

## Frozen data split

Train 2,452 episodes / 21 objects; Validation 880 / 6 objects; Test 860 / 6 objects. Objects are disjoint. Train is used to fit normalization and model parameters; Validation is used for early stopping; Test is evaluated only after checkpoint selection. No hyperparameter or threshold selection uses Test.

## Tasks and channel policies

- Primary 42-channel policy: exclude channels 18 and 25 from all four input feature blocks and from target channels/pairs. Input 168-D; future mean 42-D; future correlation 861-D.
- Sensitivity all-44 policy: retain all channels. Input 176-D; future mean 44-D; future correlation 946-D.
- Primary regression tasks: future-window channel mean and future-window upper-triangle Pearson correlations. Future SD and binary edge classification are deferred; the latter is severely imbalanced and needs a separate protocol.

## Model

MLP: input -> 256 ReLU -> Dropout(0.2) -> 128 ReLU -> Dropout(0.2) -> output.
Optimizer AdamW, learning rate 0.001, weight decay 0.001, batch 128, max 200 epochs, early-stopping patience 20 on Validation standardized MSE. Seeds 20261123, 20261124, 20261125.

X and Y standardization mean/std are fitted only on Train. A zero/near-zero standard deviation is replaced by 1. Each task and channel policy has its own scaler and checkpoint.

## Reference baselines

- Future mean: train-set constant mean and persistence (last observed value from the early input).
- Future correlation: zero-correlation prediction and train-set constant mean.
These baselines are evaluated on the same held-out Test episodes.

## Metrics and inferential unit

Report episode-weighted MAE/RMSE and mean episode-wise Spearman. Also report per-object metrics and object-macro MAE/RMSE/Spearman. Bootstrap the six held-out objects (2,000 resamples) for a descriptive 95% interval on object-macro MAE. With only six Test objects, uncertainty is wide and results are exploratory, not confirmatory.

## Required artifacts

Best checkpoint per task/policy/seed; epoch-level train/validation traces; train-only scalers; test predictions; test episode-to-object mapping; metrics.json; config.json; run.log; environment.txt; manifest.sha256. Never overwrite previous run artifacts.

## Interpretation restrictions

This is episode-level prediction across six held-out object categories, not frame-level forecasting, causal interaction inference, or proof of generalization to arbitrary unseen objects. Do not select the best initialization seed based on Test. Report all three seeds. Compare channel policies and reference baselines without changing the locked object split.

# TouchScale contact-structure sensitivity audit

Use the same deterministic 300-episode sample from the existing 15-shard sampler.
For each hand, aggregate nonnegative taxel normal force within each of the 15 pads.
Test contact thresholds of 1N, 3N, and 5N and active-frame window fractions of 20%, 30%, and 40%.
Within each condition, sort frames satisfying sum of all pad loads > threshold chronologically. Use the earliest fraction and latest fraction of active frames, requiring at least 20 frames per window and no overlap.
Apply log1p, a shared early/late variable-pad mask, and calculate edge Pearson correlation, edge cosine, edge MAE/RMSE, pad-load cosine, and top-3 pad Jaccard.
Report valid denominators and failure counts for every condition. This is a sensitivity audit, not model training, cross-object generalization, or few-shot adaptation.

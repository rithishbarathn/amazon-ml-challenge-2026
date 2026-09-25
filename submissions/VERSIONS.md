# Submission versions

Holdout = 100,000 training Source 1 entities from hash bucket 9 (never used to
train the encoder or the matcher), US + India only. France has no labels, so
it is only measured by the leaderboard.

| Version | Code | Holdout macro F0.5 | Leaderboard | Cand/S1 (test) | Changes |
|---|---|---|---|---|---|
| baseline | pre-commit | 0.9727 | - | 10.61 | Encoder k-NN blocking (KEEP_A=1, KEEP_B=10), XGBoost 2000 trees, threshold 0.675 |
| v3 | pre-commit | 0.9751 | 0.964 | 6.55 | Zero-padded numbers ("0069" == "69"), postcode fix, dotted legal forms ("S.A.S." -> sas), KEEP_B=5, early stopping |
| v4 | see v5 commit (without address-IDF features) | 0.9832 | - | 6.55 | Number-difference features (num_only1/2, num_min_diff), IDF-weighted name-difference features, all ~660k matcher-bucket S1 for training, threshold 0.725 |

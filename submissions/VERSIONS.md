# Submission versions

Holdout = 100,000 training Source 1 entities from hash bucket 9 (never used to
train the encoder or the matcher), US + India only. France has no labels, so
it is only measured by the leaderboard.

| Version | Code | Holdout macro F0.5 | Leaderboard | Cand/S1 (test) | Changes |
|---|---|---|---|---|---|
| baseline | pre-commit | 0.9727 | - | 10.61 | Encoder k-NN blocking (KEEP_A=1, KEEP_B=10), XGBoost 2000 trees, threshold 0.675 |
| v3 | pre-commit | 0.9751 | 0.964 | 6.55 | Zero-padded numbers ("0069" == "69"), postcode fix, dotted legal forms ("S.A.S." -> sas), KEEP_B=5, early stopping |
| v4 | 8b18ab7 minus addr_idf_* features | 0.9832 (India 0.9807, US 0.9848) | - | 6.55 | Number-difference features (num_only1/2, num_min_diff), IDF-weighted name-difference features, all ~660k matcher-bucket S1 for training, threshold 0.725 |
| v5 | commit 8b18ab7 | 0.9834 (India 0.9810, US 0.9850) | - | 6.55 | + IDF-weighted address-word differences (addr_idf_*), threshold 0.75; France matched pairs 837k (v4: 833k, v3: 854k) |

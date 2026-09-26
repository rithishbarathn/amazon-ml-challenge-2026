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
| v7 | b7c7589 | 0.9848 (India 0.9826, US 0.9862) | - | 8.21 | + address-only k-NN channel (`pipeline.block addr`, KEEP_C=1): every target also proposes its nearest S1 by address vector alone, catching DBA / domain-name records at the owner's address. Holdout blocking recall 97.74% -> 98.34%. Threshold 0.725 |
| v6 | 6bb4099 | 0.9847 (India 0.9822, US 0.9864) | 0.97509 (rank 249) | 7.61 | v7 with KEEP_B=3 (fewer candidates) and a bigger XGBoost (depth 10, lr 0.03; early-stopped at 2875 trees). Same score as v7 with ~7% fewer candidates. Threshold 0.75 |

## Leaderboard probe: France contribution

`output_v6/variants/matching_results_unseen_empty.tsv` (v6 with every France S1
left empty) scored **0.842**. France is 14.98% of test S1; an empty France row
only scores 1 for true singletons (~5.5%, as in US / India). Solving
0.842 = 0.850 * U + 0.150 * 0.055 and 0.97509 = 0.850 * U + 0.150 * F gives
**US + India ~ 0.981** and **France ~ 0.944** on the public leaderboard
(France range 0.92-0.97 for a 3-8% singleton rate).

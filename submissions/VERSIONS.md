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

## Experiment (not submitted): v8 multilingual embeddings

`pipeline.mlemb` embeds raw names / addresses with intfloat/multilingual-e5-small
(MIT, 118M params), PCA to 128-d, and adds ml_name_cos / ml_addr_cos /
ml_cross_cos features (opt-in via `ER_USE_MLEMB=1`).
Holdout 0.9850 vs v6 0.9847 (India 0.9826, US 0.9866) - the model barely uses
them. Cross-country check (train on US only, score India as unseen country):
0.9392 without -> 0.9346 with, i.e. they *hurt* transfer to an unseen country,
so they were not used for the France-containing test set.

Other offline checks on the same US -> India setup: self-training on confident
pseudo-labels +0.5 on the unseen country (not used: small overall gain and it
trains on unlabelled test records); shallower trees and dropping length /
postcode features all transferred worse than the submitted configuration.

## Experiment (not submitted): cross-encoder (branch `cross-encoder`)

`pipeline.crossenc` fine-tunes multilingual-e5-small as a pair classifier on
raw "name · address" pairs (buckets 0-5 only) and adds its probability as an
XGBoost feature (opt-in `ER_USE_CE=1`). Unseen-country check with a US-only
cross-encoder (300k pairs) and US-only matcher: US 0.9857 -> 0.9878 (+0.21),
India (unseen) 0.9376 -> 0.9331 (-0.45); cross-encoder alone on India 0.8813.
Not used: it hurts transfer to a country without training labels (France).
Encoder-feature ablation (same setup): raw 0.9392, within-country percentiles
0.9360, no encoder features 0.8636 -> encoder features carry the transfer.

## v9 (branch `encoder-v2`)

| Version | Code | Holdout macro F0.5 | Leaderboard | Cand/S1 (test) | Changes |
|---|---|---|---|---|---|
| v9 | 2f6ecb4 | 0.9867 (India 0.9849, US 0.9879) | - | 7.57 | Encoder trained on all 1.25M bucket 0-5 S1 (was 200K): owner@1 0.9703 -> 0.9741, blocking recall 97.24% -> 97.61% at the same 4.84 cand/S1. Feature groups A (number decoys) + B (name structure). Stage-2 relational XGBoost on out-of-fold stage-1 probabilities (folds by S1). Expected-F0.5 decision rule (floor 0.4, extra 0.3). France-US top-1 cosine gap -0.059 -> -0.018. |

Offline evidence per component (all in `submissions/experiments.csv`):
- US-only model scored on India (unseen-country check): base 0.9352 -> +A+B 0.9461 -> +stage-2 0.9501.
- Stage-2 leakage check: inner +0.06, bucket-9 holdout +0.08 (consistent).
- Rejected: hard-negative encoder batches (no gain), group C acronym/domain (unseen India -0.35), XGBoost random search (10 trials, none beat depth 8 / lr 0.05).
| v10 | 0794fe7 | 0.9873 (India 0.9858, US 0.9884) | - | 7.57 | v9 + sibling features (similarity to the S1's other confident candidates, out-of-fold) + 3-seed stage-2 average; expected-F0.5 rule (floor 0.2, extra 0.05). Sibling experiment: holdout 0.9866 -> 0.9872, unseen India 0.9449 -> 0.9479. |
| v11 | (this commit) | 0.9878 (India 0.9868, US 0.9884) | - | 9.43 | v10 with the second search widened (KEEP_B 3 -> 7): holdout blocking recall 98.01% -> 98.78% at 8.35 cand/S1. Low-memory two-pass prediction. |

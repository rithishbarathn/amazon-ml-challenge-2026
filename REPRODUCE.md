# Reproducing our Amazon ML Challenge 2026 results

This guide explains what is on each branch, how to rebuild the submission
files from the raw data, and where every result is recorded. It assumes you
have the competition dataset (it is **not** in git).

---

## 1. Where things stand

| Version | Branch / commit | Holdout macro F0.5 (US + India) | Leaderboard | Test candidates / S1 |
|---|---|---|---|---|
| v3 | pre-commit | 0.9751 | 0.9643 (rank 426) | 6.55 |
| v6 | `main` (`6bb4099`) | 0.9847 | **0.97509 (rank 249)** — best submitted | 7.61 |
| **v9** | **`encoder-v2`** (`2f6ecb4`) | **0.9867** (India 0.9849, US 0.9879) | not yet submitted | 7.57 |

Leaderboard probe: v6 with every France record left empty scored 0.842, which
puts **US + India ≈ 0.981** and **France ≈ 0.944** on the public leaderboard.
France (15% of test, no training labels) is the weakest part.

The *holdout* is 100,000 training Source 1 entities (hash bucket 9) that no
model ever trains on. It only covers US and India, so leaderboard scores have
come out about 1 point lower than holdout scores so far.

Full history: `submissions/VERSIONS.md`. One row per experiment:
`submissions/experiments.csv`.

---

## 2. Branches

### `main` — v6, our best leaderboard submission (0.97509)
The production pipeline as submitted:
- **Prep** — transliteration to ASCII, lower-casing, abbreviation expansion.
- **Encoder** — hashed character n-gram + word bags → two sparse
  `EmbeddingBag` layers, trained with a contrastive loss on matched pairs
  (200K Source 1 entities).
- **Blocking** — per-country exact k-NN on GPU in three channels:
  Source 2/3 → nearest Source 1 (`rank_a`), Source 1 → nearest targets
  (`rank_b`), and an address-only search (`rank_c`, catches trade names /
  web-domain names at the same address). ~7.6 candidates per Source 1.
- **Features** — encoder cosines, ranks / margins, RapidFuzz name and address
  similarities, zero-padding-safe house-number / postcode checks,
  number-difference features, IDF-weighted name / address word differences.
- **Matcher** — XGBoost (depth 10, lr 0.03) + one-owner rule (each Source 2/3
  record goes to at most one Source 1) + a tuned probability threshold.
- Also contains the multilingual-embedding experiment (`pipeline/mlemb.py`),
  **switched off** (`ER_USE_MLEMB=1` enables it): holdout +0.0003 but it hurt
  an unseen country, so it is not used.

### `cross-encoder` — experiments that did not make it (not for submission)
`main` + `pipeline/crossenc.py`: fine-tunes multilingual-e5-small (MIT) to
read both records of a pair and output a match probability, added as an
XGBoost feature (opt-in `ER_USE_CE=1`). US-only test: US +0.21, **India as an
unseen country −0.45**, so it was dropped. Kept for the methodology write-up.

### `encoder-v2` — v9, best offline model (recommended next submission)
`cross-encoder` + four improvements, each checked on the holdout **and** on an
unseen-country test (train on US only, score India):
1. **Encoder trained on all 1.25M** bucket 0–5 Source 1 entities (was 200K):
   blocking recall 97.24% → 97.61% at the same 4.84 candidates per S1.
2. **Feature group A — number decoys**: truncated / split house numbers
   (102 vs 10, 133 vs 33, "3 05" vs 305), first-number match, number counts.
3. **Feature group B — name structure**: edit distance on core names,
   first / last word match, word-count difference, one name contained in the
   other. Biggest transfer gain: unseen India 0.9352 → 0.9441 on its own.
4. **Stage-2 relational matcher**: stage-1 XGBoost → out-of-fold
   probabilities (3 folds split **by Source 1**, never by pair, to avoid
   leakage) → features relative to competing candidates (rank within its
   Source 1, margin to the best competing owner of the same target, counts
   above 0.5) → stage-2 XGBoost. Holdout +0.08, unseen India +0.40.

Also on this branch: `pipeline/experiments.py` (experiment harness) and
`pipeline/explog.py` (writes `submissions/experiments.csv`).

Tried and rejected on `encoder-v2` (details in `experiments.csv`):
hard-negative encoder batches (no gain), feature group C acronym / domain
names (unseen India −0.35), XGBoost random search (10 trials, none beat
depth 8 / lr 0.05).

---

## 3. Requirements

- **GPU**: NVIDIA with ≥ 6 GB and CUDA (we used an RTX 4050 Laptop, 6 GB).
- **RAM**: 24 GB recommended (peaks around 13–16 GB).
- **Disk**: ~30 GB free (the `work/` folder ends up ~15 GB).
- **Python** 3.11+ (we ran 3.14 on Windows 11).

```bash
git clone https://github.com/rithishbarathn/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
git checkout encoder-v2          # or main for v6
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Put the competition data here (or set `ER_DATA_DIR` to its location):
```
dataset/student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/student_resource/dataset/test/test_source{1,2,3}.tsv
dataset/student_resource/utils/validate_submission.py
```
Share the dataset only through a channel the competition rules allow.

Optional folders (environment variables, read by `src/pipeline/common.py`):
`ER_WORK_DIR` (intermediate files, default `work/`) and `ER_OUTPUT_DIR`
(submission files, default `output/`). Use a separate `ER_WORK_DIR` to keep
two versions side by side.

---

## 4. Rebuild v9 (branch `encoder-v2`)

Run from `src/`, **one command at a time** — never two GPU steps at once
(6 GB GPUs run out of memory). Times are from our laptop.

```bash
cd src
python -m pipeline.prep                 # ~3 min   raw TSV -> normalised parquet
python -m pipeline.encoder train        # ~9 min   encoder on all 1.25M S1
python -m pipeline.encoder eval v9      # ~2 min   optional: owner@1 / target@5
python -m pipeline.encoder embed        # ~9 min   vectors for every record
python -m pipeline.block train          # ~35-55 min  candidate search (train)
python -m pipeline.block test           # ~20-35 min  candidate search (test)
python -m pipeline.block addr train     # ~20-45 min  address-only channel
python -m pipeline.block addr test      # ~10-20 min
python -m pipeline.block eval           # ~3 min   optional: recall vs candidates
python -m pipeline.matcher train        # ~15 min  stage-1, OOF folds, stage-2, rule
python -m pipeline.matcher predict      # ~22 min  writes output/ + runs validator
```
Total ≈ 3–4 hours. Feature groups A+B and stage-2 are on by default on this
branch (`ER_FEATURE_GROUPS=A,B`, `STAGE2 = True` in `matcher.py`).

Expected `matcher train` summary (end of its log):
```
BEST {"mode": "expected", "floor": 0.4, "extra": 0.3} -> 0.9867
  India: ... macro F0.5 0.9849
  US:    ... macro F0.5 0.9879
```

## 5. Rebuild v6 (branch `main`)

Same commands as above, without `encoder eval` (the command does not exist on
`main`). On `main` the encoder trains on 200K Source 1 entities and the
matcher is single-stage. Expected holdout: 0.9847.

---

## 6. Outputs and submitting

`matcher predict` writes:
- `output/matching_results.tsv` — **the file uploaded to the leaderboard**
- `output/candidate_pairs.tsv` — the candidate set the model scored (part of
  the final zip; smaller sets rank higher in the final review)
- `output/variants/matching_results_unseen_{strict,empty}.tsv` — France
  probes (France matches only if p ≥ 0.9 / no France matches)

It runs the official validator automatically; to run it by hand:
```bash
python dataset/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/student_resource/dataset/test
```

Limits: 5 leaderboard submissions per day; deadline **27 Sep 2026, 11:59 PM IST**.
Final ranking uses the private leaderboard.

---

## 7. Running experiments

All experiment runs append one row to `submissions/experiments.csv`
(inner F0.5, US, India transfer, France proxy, candidate recall, #features).
They reuse the features cached by `matcher train` in `work/features/`.

```bash
python -m pipeline.experiments prepare          # compute feature groups (CPU)
python -m pipeline.experiments groups           # base / A / B / C / combined
python -m pipeline.experiments stage2 A,B       # stage-2 + bucket-9 leakage check
python -m pipeline.experiments stage2_transfer A,B
python -m pipeline.experiments tune A,B 16      # XGBoost random search
```

Encoder variants: `ER_ENC_PATH` (checkpoint file), `ER_ENC_DIM`,
`ER_ENC_EPOCHS`, `ER_ENC_HARD=1` (hard-negative batches), `ER_ENC_QUERIES`
(0 = all).

Multilingual / cross-encoder experiments need the MIT-licensed model once:
```python
from huggingface_hub import snapshot_download
snapshot_download("intfloat/multilingual-e5-small", local_dir="work/e5-small",
                  allow_patterns=["config.json", "model.safetensors", "tokenizer*",
                                  "sentencepiece*", "special_tokens_map.json"])
```
then `python -m pipeline.mlemb train|test` (`ER_USE_MLEMB=1`) or
`python -m pipeline.crossenc train|score` (`ER_USE_CE=1`).

---

## 8. Lessons so far

- Feature engineering beat model tuning: number / name-structure features
  gave +0.8 and +1.1 (unseen country); a bigger XGBoost and a random search
  gave nothing.
- Always check an unseen country (train US, score India). Several changes
  that helped the holdout made an unseen country worse — that is France.
- Stage-2 features must come from out-of-fold predictions split by Source 1,
  or they leak labels (confirm any gain on bucket 9, not just bucket 8).
- Rules: MIT / Apache 2.0 models ≤ 8B parameters, no external data or APIs.

## 9. Troubleshooting

- **CUDA out of memory** — another GPU job is running; run steps one by one.
- **Windows paging-file errors** in `encoder` — keep `WORKERS = 4`.
- **Candidates / features "out of sync"** in `experiments.py` — re-run
  `matcher train` so `work/features/` matches the current candidates.

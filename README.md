# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business record, find all Source 2 / Source 3 records that
describe the same real-world business (US, India and — test only — France).

## Pipeline

```
raw TSVs ─► prep ─► encoder (train + embed) ─► block ─► features ─► matcher ─► output/
```

| Stage | Module | What it does |
|---|---|---|
| 1. Prep | `pipeline.prep` | Unicode → ASCII transliteration (`unidecode`), lower-casing, abbreviation expansion (Pvt→private, St→street, TX→texas…), cached as parquet. |
| 2. Encoder | `pipeline.encoder` | Hashed char n-gram + word bags of name and address → two sparse `EmbeddingBag` layers, trained on GPU with a contrastive (InfoNCE) loss on labelled Source 1 ↔ Source 2/3 pairs. Every record is embedded to a 192-d vector. |
| 3. Blocking | `pipeline.block` | Per-country (open set, no hard-coded labels) exact k-NN on GPU in both directions: each Source 2/3 record proposes its closest Source 1 owners, and each Source 1 keeps its closest targets. The union of the top ranks is the candidate set. |
| 4. Features | `pipeline.features` | Encoder cosines, rank / margin features, RapidFuzz similarities on normalised, legal-form-stripped and transliteration-squashed names and on addresses, house-number / postcode agreement, candidate counts. No country features. |
| 5. Matcher | `pipeline.matcher` | XGBoost (GPU). Decision rule: each Source 2/3 record is assigned only to its highest-probability Source 1 candidate (Source 1 is deduplicated, so a record has at most one owner) if the probability clears a threshold tuned for macro F0.5 on held-out training entities. |

Training Source 1 entities are split deterministically by a hash of their id:
buckets 0–5 train the encoder, 6–8 train the matcher, 9 is the holdout used
for every reported number.

## Reproduce

Requirements: Python 3.11+, an NVIDIA GPU with CUDA (≥ 6 GB), ~16 GB free RAM.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# the dataset is expected at dataset/student_resource/dataset/{train,test}
# (or set ER_DATA_DIR)
cd src
python -m pipeline.prep
python -m pipeline.encoder train
python -m pipeline.encoder embed
python -m pipeline.block train
python -m pipeline.block test
python -m pipeline.block eval        # optional: blocking recall vs. size
python -m pipeline.matcher train
python -m pipeline.matcher predict   # writes output/*.tsv and runs the validator
```

Outputs: `output/matching_results.tsv` (leaderboard file) and
`output/candidate_pairs.tsv` (the exact candidate set scored by the matcher).

`src/*.py` outside `pipeline/` are the earlier pilot experiments
(TF-IDF blocking on 2,000-query samples) kept for reference.

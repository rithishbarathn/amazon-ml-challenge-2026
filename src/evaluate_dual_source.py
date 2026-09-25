
import glob
import os

import pandas as pd


TRAIN_DIR = "dataset/train"
CANDIDATE_DIR = "output/dual_source_pilot"

GROUND_TRUTH_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"
SOURCE1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
SOURCE2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
SOURCE3_PATH = f"{TRAIN_DIR}/train_source3.tsv"

COUNTRY = "US"
QUERY_LIMIT = 2000
TARGET_LIMIT = 500000
READ_CHUNK_SIZE = 50000


def load_country_ids(path, country, limit):
    ids = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
    ):
        selected = chunk.loc[
            chunk["country"] == country,
            "entity_id",
        ]

        remaining = limit - len(ids)

        ids.update(selected.head(remaining))

        if len(ids) >= limit:
            break

    return ids


print("Loading pilot query IDs...", flush=True)

query_ids = load_country_ids(
    SOURCE1_PATH,
    COUNTRY,
    QUERY_LIMIT,
)

print(f"Queries: {len(query_ids):,}")


print("Loading Source 2 eligible target IDs...", flush=True)

source2_ids = load_country_ids(
    SOURCE2_PATH,
    COUNTRY,
    TARGET_LIMIT,
)

print(f"Source 2 targets: {len(source2_ids):,}")


print("Loading Source 3 eligible target IDs...", flush=True)

source3_ids = load_country_ids(
    SOURCE3_PATH,
    COUNTRY,
    TARGET_LIMIT,
)

print(f"Source 3 targets: {len(source3_ids):,}")

eligible_target_ids = source2_ids | source3_ids


print("Loading ground truth...", flush=True)

truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

truth = truth[
    truth["source1_entity_id"].isin(query_ids)
]

all_true_pairs = set()
eligible_true_pairs = set()

for row in truth.itertuples(index=False):

    source_id = row.source1_entity_id

    for target_id in row.matched_entity_ids.split(","):

        target_id = target_id.strip()

        if not target_id:
            continue

        pair = (source_id, target_id)

        all_true_pairs.add(pair)

        if target_id in eligible_target_ids:
            eligible_true_pairs.add(pair)


print("Loading saved candidate chunks...", flush=True)

candidate_files = sorted(
    glob.glob(
        os.path.join(
            CANDIDATE_DIR,
            "source*_chunk_*.tsv",
        )
    )
)

if len(candidate_files) != 10:
    raise ValueError(
        f"Expected 10 candidate files, found "
        f"{len(candidate_files)}. Check the output directory."
    )

retrieved_pairs = set()
total_rows = 0

for path in candidate_files:

    candidates = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    total_rows += len(candidates)

    retrieved_pairs.update(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    print(
        f"Loaded {os.path.basename(path)}: "
        f"{len(candidates):,} pairs",
        flush=True,
    )


found_pairs = eligible_true_pairs & retrieved_pairs

recall = (
    len(found_pairs) / len(eligible_true_pairs)
    if eligible_true_pairs
    else 0.0
)


print("\nDUAL-SOURCE PILOT RECALL RESULTS")

print(f"Queries evaluated: {len(query_ids):,}")
print(f"All genuine pairs: {len(all_true_pairs):,}")
print(
    f"Eligible genuine pairs in pilot targets: "
    f"{len(eligible_true_pairs):,}"
)
print(f"Retrieved eligible genuine pairs: {len(found_pairs):,}")
print(
    f"Missed eligible genuine pairs: "
    f"{len(eligible_true_pairs - found_pairs):,}"
)
print(f"Conditional candidate recall: {recall:.4%}")

print(f"Candidate rows across chunks: {total_rows:,}")
print(f"Unique candidate pairs: {len(retrieved_pairs):,}")

print("\nDUAL-SOURCE EVALUATION COMPLETED")

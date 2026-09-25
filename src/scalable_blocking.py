
import gc
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import normalize_name, normalize_address


TRAIN_DIR = "dataset/train"

COUNTRY = "US"

QUERY_LIMIT = 2000
TARGET_LIMIT = 500000

TOP_K_NAME = 20
TOP_K_ADDRESS = 20

BATCH_SIZE = 5


def load_country_sample(path, country, limit):

    parts = []
    collected = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=50000
    ):

        selected = chunk[chunk["country"] == country]

        remaining = limit - collected

        selected = selected.head(remaining)

        if not selected.empty:
            parts.append(selected)
            collected += len(selected)

        if collected >= limit:
            break

    return pd.concat(parts, ignore_index=True)


def top_candidates(row, ids, k):

    if row.nnz == 0:
        return set()

    # Select only the highest-scoring nonzero entries.
    count = min(k, row.nnz)

    positions = np.argpartition(
        row.data,
        -count
    )[-count:]

    positions = positions[
        np.argsort(row.data[positions])[::-1]
    ]

    return set(ids[row.indices[positions]])


print("Loading US samples...", flush=True)

start_time = time.time()

queries = load_country_sample(
    f"{TRAIN_DIR}/train_source1.tsv",
    COUNTRY,
    QUERY_LIMIT
)

targets = load_country_sample(
    f"{TRAIN_DIR}/train_source2.tsv",
    COUNTRY,
    TARGET_LIMIT
)

print(f"Source 1 queries: {len(queries):,}", flush=True)
print(f"Source 2 targets: {len(targets):,}", flush=True)


print("Normalizing names and addresses...", flush=True)

for df in [queries, targets]:

    df["name_clean"] = df["business_name"].map(
        normalize_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )


print("Building name TF-IDF...", flush=True)

name_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    dtype=np.float32,
    max_features=100000
)

target_name = name_vectorizer.fit_transform(
    targets["name_clean"]
)

query_name = name_vectorizer.transform(
    queries["name_clean"]
)


print("Building address TF-IDF...", flush=True)

address_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    dtype=np.float32,
    max_features=100000
)

target_address = address_vectorizer.fit_transform(
    targets["address_clean"]
)

query_address = address_vectorizer.transform(
    queries["address_clean"]
)


target_ids = targets["entity_id"].to_numpy()

results = []

print("Retrieving candidates...", flush=True)

for start in range(0, len(queries), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(queries))

    name_sim = (
        query_name[start:end] @ target_name.T
    ).tocsr()

    address_sim = (
        query_address[start:end] @ target_address.T
    ).tocsr()

    for local_i in range(end - start):

        query_id = queries.iloc[start + local_i]["entity_id"]

        name_ids = top_candidates(
            name_sim.getrow(local_i),
            target_ids,
            TOP_K_NAME
        )

        address_ids = top_candidates(
            address_sim.getrow(local_i),
            target_ids,
            TOP_K_ADDRESS
        )

        combined_ids = name_ids | address_ids

        for candidate_id in combined_ids:

            results.append(
                (query_id, candidate_id)
            )

    print(
        f"Processed {end:,}/{len(queries):,} queries",
        flush=True
    )

    del name_sim, address_sim
    gc.collect()


result_df = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

result_df = result_df.drop_duplicates()

output_path = "output/scalable_blocking_2000q_500k.tsv"

result_df.to_csv(
    output_path,
    sep="\t",
    index=False
)

elapsed = time.time() - start_time

print("\nSCALABLE BLOCKING PILOT COMPLETED")
print(f"Candidate pairs: {len(result_df):,}")
print(f"Average candidates/query: {len(result_df) / len(queries):.2f}")
print(f"Execution time: {elapsed / 60:.2f} minutes")
print(f"Saved to: {output_path}")

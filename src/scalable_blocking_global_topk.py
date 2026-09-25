
import gc
import heapq
import os
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer

from preprocessing import normalize_name, normalize_address


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DIR = "dataset/train"

OUTPUT_PATH = "output/global_topk_100_2000q_dual_source.tsv"

COUNTRY = "US"

QUERY_LIMIT = 2000
TARGET_LIMIT_PER_SOURCE = 500000

READ_CHUNK_SIZE = 50000
TARGET_CHUNK_SIZE = 100000

TOP_K_NAME = 50
TOP_K_ADDRESS = 50

BATCH_SIZE = 5

SOURCE_FILES = {
    "source2": f"{TRAIN_DIR}/train_source2.tsv",
    "source3": f"{TRAIN_DIR}/train_source3.tsv",
}

OUTPUT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]


# --------------------------------------------------
# 1. Load Source 1 sample
# --------------------------------------------------

def load_country_sample(path, country, limit):

    parts = []
    collected = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
    ):

        selected = chunk.loc[
            chunk["country"] == country
        ]

        remaining = limit - collected

        selected = selected.head(remaining)

        if not selected.empty:
            parts.append(selected.copy())
            collected += len(selected)

        if collected >= limit:
            break

    if not parts:
        raise ValueError(
            f"No records found for {country} in {path}"
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# --------------------------------------------------
# 2. Stream target chunks
# --------------------------------------------------

def iter_country_target_chunks(path, country, limit):

    collected = 0
    buffer = []
    buffer_count = 0
    chunk_number = 0

    for raw_chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
    ):

        remaining = limit - collected

        if remaining <= 0:
            break

        selected = raw_chunk.loc[
            raw_chunk["country"] == country
        ].head(remaining)

        if selected.empty:
            continue

        buffer.append(selected.copy())

        selected_count = len(selected)

        collected += selected_count
        buffer_count += selected_count

        if buffer_count >= TARGET_CHUNK_SIZE:

            chunk_number += 1

            targets = pd.concat(
                buffer,
                ignore_index=True,
            )

            yield chunk_number, targets

            buffer = []
            buffer_count = 0

        if collected >= limit:
            break

    if buffer:

        chunk_number += 1

        targets = pd.concat(
            buffer,
            ignore_index=True,
        )

        yield chunk_number, targets


# --------------------------------------------------
# 3. Normalize records
# --------------------------------------------------

def normalize_records(df):

    df = df.copy()

    df["name_clean"] = df["business_name"].map(
        normalize_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )

    return df


# --------------------------------------------------
# 4. Create consistent vectorizers
# --------------------------------------------------

# Unlike fitting a new TF-IDF model for every chunk,
# hashing uses the same feature mapping across chunks.
# This allows similarity scores to be compared globally.

name_vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32,
)

address_vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32,
)


# --------------------------------------------------
# 5. Update a global top-K heap
# --------------------------------------------------

def update_topk(heap, similarity_row, target_ids, k):

    if similarity_row.nnz == 0:
        return

    count = min(k, similarity_row.nnz)

    # Only inspect the strongest entries in this chunk.
    positions = np.argpartition(
        similarity_row.data,
        -count,
    )[-count:]

    for position in positions:

        score = float(
            similarity_row.data[position]
        )

        if score <= 0:
            continue

        candidate_id = str(
            target_ids[
                similarity_row.indices[position]
            ]
        )

        item = (score, candidate_id)

        if len(heap) < k:

            heapq.heappush(
                heap,
                item,
            )

        elif item > heap[0]:

            heapq.heapreplace(
                heap,
                item,
            )


# --------------------------------------------------
# 6. Main pipeline
# --------------------------------------------------

def main():

    start_time = time.time()

    os.makedirs(
        "output",
        exist_ok=True,
    )

    print(
        "Loading Source 1 queries...",
        flush=True,
    )

    queries = load_country_sample(
        f"{TRAIN_DIR}/train_source1.tsv",
        COUNTRY,
        QUERY_LIMIT,
    )

    queries = normalize_records(queries)

    query_ids = queries["entity_id"].to_numpy()

    print(
        f"Source 1 queries: {len(queries):,}",
        flush=True,
    )

    print(
        "Vectorizing Source 1 queries...",
        flush=True,
    )

    query_name = name_vectorizer.transform(
        queries["name_clean"]
    )

    query_address = address_vectorizer.transform(
        queries["address_clean"]
    )

    # One heap per query for each similarity channel.
    # Each heap retains only the strongest 20 matches.

    name_heaps = [
        [] for _ in range(len(queries))
    ]

    address_heaps = [
        [] for _ in range(len(queries))
    ]

    total_targets = 0

    # --------------------------------------------------
    # Process both target sources
    # --------------------------------------------------

    for source_name, source_path in SOURCE_FILES.items():

        print(
            f"\nPROCESSING {source_name.upper()}",
            flush=True,
        )

        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)

        for chunk_number, targets in iter_country_target_chunks(
            source_path,
            COUNTRY,
            TARGET_LIMIT_PER_SOURCE,
        ):

            chunk_start = time.time()

            print(
                f"\n{source_name} | "
                f"Chunk {chunk_number} | "
                f"Targets: {len(targets):,}",
                flush=True,
            )

            total_targets += len(targets)

            targets = normalize_records(targets)

            target_ids = targets[
                "entity_id"
            ].to_numpy()

            print(
                "Vectorizing target chunk...",
                flush=True,
            )

            target_name = name_vectorizer.transform(
                targets["name_clean"]
            )

            target_address = address_vectorizer.transform(
                targets["address_clean"]
            )

            target_name_t = target_name.T.tocsr()
            target_address_t = target_address.T.tocsr()

            # ------------------------------------------
            # Search queries in small batches
            # ------------------------------------------

            for start in range(
                0,
                len(queries),
                BATCH_SIZE,
            ):

                end = min(
                    start + BATCH_SIZE,
                    len(queries),
                )

                name_sim = (
                    query_name[start:end]
                    @ target_name_t
                ).tocsr()

                address_sim = (
                    query_address[start:end]
                    @ target_address_t
                ).tocsr()

                for local_i in range(end - start):

                    query_index = start + local_i

                    update_topk(
                        name_heaps[query_index],
                        name_sim.getrow(local_i),
                        target_ids,
                        TOP_K_NAME,
                    )

                    update_topk(
                        address_heaps[query_index],
                        address_sim.getrow(local_i),
                        target_ids,
                        TOP_K_ADDRESS,
                    )

                if end % 500 == 0 or end == len(queries):

                    print(
                        f"Processed "
                        f"{end:,}/{len(queries):,} queries",
                        flush=True,
                    )

                del name_sim, address_sim

            elapsed = time.time() - chunk_start

            print(
                f"Chunk completed in "
                f"{elapsed / 60:.2f} minutes",
                flush=True,
            )

            del (
                targets,
                target_name,
                target_address,
                target_name_t,
                target_address_t,
            )

            gc.collect()

    # --------------------------------------------------
    # 7. Merge global top-K results
    # --------------------------------------------------

    print(
        "\nMerging global top-K candidates...",
        flush=True,
    )

    results = []

    for query_index, query_id in enumerate(query_ids):

        name_ids = {
            candidate_id
            for score, candidate_id
            in name_heaps[query_index]
        }

        address_ids = {
            candidate_id
            for score, candidate_id
            in address_heaps[query_index]
        }

        combined_ids = name_ids | address_ids

        for candidate_id in sorted(combined_ids):

            results.append(
                (query_id, candidate_id)
            )

    result_df = pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS,
    )

    result_df = result_df.drop_duplicates()

    # --------------------------------------------------
    # 8. Save safely
    # --------------------------------------------------

    temporary_path = OUTPUT_PATH + ".tmp"

    result_df.to_csv(
        temporary_path,
        sep="\t",
        index=False,
    )

    os.replace(
        temporary_path,
        OUTPUT_PATH,
    )

    # --------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------

    elapsed = time.time() - start_time

    print("\nGLOBAL TOP-K BLOCKING COMPLETED")

    print(
        f"Queries: {len(queries):,}"
    )

    print(
        f"Targets processed: {total_targets:,}"
    )

    print(
        f"Candidate pairs: {len(result_df):,}"
    )

    print(
        f"Average candidates/query: "
        f"{len(result_df) / len(queries):.2f}"
    )

    print(
        f"Execution time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()


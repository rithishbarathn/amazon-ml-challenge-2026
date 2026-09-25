
import gc
import os
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import normalize_name, normalize_address


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DIR = "dataset/train"
OUTPUT_DIR = "output/dual_source_pilot"

COUNTRY = "US"

QUERY_LIMIT = 2000
TARGET_LIMIT_PER_SOURCE = 500000

TARGET_CHUNK_SIZE = 100000
READ_CHUNK_SIZE = 50000

TOP_K_NAME = 20
TOP_K_ADDRESS = 20

BATCH_SIZE = 5
MAX_FEATURES = 100000

SOURCE_FILES = {
    "source2": f"{TRAIN_DIR}/train_source2.tsv",
    "source3": f"{TRAIN_DIR}/train_source3.tsv",
}

OUTPUT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]


# --------------------------------------------------
# 1. Load first N records for a country
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
            f"No records found for country {country} in {path}"
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# --------------------------------------------------
# 2. Stream target chunks without loading all targets
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

        selected = raw_chunk.loc[
            raw_chunk["country"] == country
        ]

        remaining = limit - collected

        selected = selected.head(remaining)

        if selected.empty:
            continue

        buffer.append(selected.copy())

        count = len(selected)

        collected += count
        buffer_count += count

        if buffer_count >= TARGET_CHUNK_SIZE:

            targets = pd.concat(
                buffer,
                ignore_index=True,
            )

            chunk_number += 1

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
# 3. Retrieve highest-scoring nonzero candidates
# --------------------------------------------------

def top_candidates(row, ids, k):

    if row.nnz == 0:
        return set()

    count = min(k, row.nnz)

    positions = np.argpartition(
        row.data,
        -count,
    )[-count:]

    positions = positions[
        np.argsort(row.data[positions])[::-1]
    ]

    return set(
        ids[row.indices[positions]]
    )


# --------------------------------------------------
# 4. Normalize records
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
# 5. Build TF-IDF safely
# --------------------------------------------------

def build_tfidf(target_text, query_text):

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        dtype=np.float32,
        max_features=MAX_FEATURES,
    )

    # Handle a chunk where every text value is empty.
    if not target_text.str.strip().any():
        return None, None

    target_matrix = vectorizer.fit_transform(
        target_text
    )

    query_matrix = vectorizer.transform(
        query_text
    )

    return target_matrix, query_matrix


# --------------------------------------------------
# 6. Process one target chunk
# --------------------------------------------------

def process_target_chunk(
    queries,
    targets,
    output_path,
):

    print("Normalizing target records...", flush=True)

    targets = normalize_records(targets)

    print("Building name TF-IDF...", flush=True)

    target_name, query_name = build_tfidf(
        targets["name_clean"],
        queries["name_clean"],
    )

    print("Building address TF-IDF...", flush=True)

    target_address, query_address = build_tfidf(
        targets["address_clean"],
        queries["address_clean"],
    )

    target_ids = targets["entity_id"].to_numpy()
    query_ids = queries["entity_id"].to_numpy()

    # Write to a temporary file first.
    # Only rename it after the entire chunk finishes.
    temporary_path = output_path + ".tmp"

    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    total_pairs = 0

    for start in range(0, len(queries), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(queries),
        )

        if target_name is not None:

            name_sim = (
                query_name[start:end] @ target_name.T
            ).tocsr()

        else:
            name_sim = None

        if target_address is not None:

            address_sim = (
                query_address[start:end] @ target_address.T
            ).tocsr()

        else:
            address_sim = None

        batch_results = []

        for local_i in range(end - start):

            query_id = query_ids[start + local_i]

            name_ids = set()
            address_ids = set()

            if name_sim is not None:

                name_ids = top_candidates(
                    name_sim.getrow(local_i),
                    target_ids,
                    TOP_K_NAME,
                )

            if address_sim is not None:

                address_ids = top_candidates(
                    address_sim.getrow(local_i),
                    target_ids,
                    TOP_K_ADDRESS,
                )

            combined_ids = name_ids | address_ids

            for candidate_id in combined_ids:

                batch_results.append(
                    (query_id, candidate_id)
                )

        batch_df = pd.DataFrame(
            batch_results,
            columns=OUTPUT_COLUMNS,
        )

        batch_df.to_csv(
            temporary_path,
            sep="\t",
            index=False,
            mode="a",
            header=(start == 0),
        )

        total_pairs += len(batch_df)

        if end % 250 == 0 or end == len(queries):

            print(
                f"Processed {end:,}/{len(queries):,} queries",
                flush=True,
            )

        del name_sim, address_sim, batch_df

    # Mark this chunk complete only after successful writing.
    os.replace(
        temporary_path,
        output_path,
    )

    del (
        target_name,
        query_name,
        target_address,
        query_address,
    )

    gc.collect()

    return total_pairs


# --------------------------------------------------
# 7. Main pipeline
# --------------------------------------------------

def main():

    start_time = time.time()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("Loading Source 1 queries...", flush=True)

    queries = load_country_sample(
        f"{TRAIN_DIR}/train_source1.tsv",
        COUNTRY,
        QUERY_LIMIT,
    )

    queries = normalize_records(queries)

    print(
        f"Source 1 queries: {len(queries):,}",
        flush=True,
    )

    grand_total = 0

    for source_name, source_path in SOURCE_FILES.items():

        print(
            f"\nPROCESSING {source_name.upper()}",
            flush=True,
        )

        if not os.path.exists(source_path):

            raise FileNotFoundError(
                f"Missing source file: {source_path}"
            )

        for chunk_number, targets in iter_country_target_chunks(
            source_path,
            COUNTRY,
            TARGET_LIMIT_PER_SOURCE,
        ):

            output_path = (
                f"{OUTPUT_DIR}/"
                f"{source_name}_chunk_{chunk_number:03d}.tsv"
            )

            # Resume: skip chunks already completed.
            if os.path.exists(output_path):

                print(
                    f"Skipping completed chunk: {output_path}",
                    flush=True,
                )

                continue

            print(
                f"\n{source_name} | "
                f"Chunk {chunk_number} | "
                f"Targets: {len(targets):,}",
                flush=True,
            )

            chunk_start = time.time()

            pair_count = process_target_chunk(
                queries,
                targets,
                output_path,
            )

            grand_total += pair_count

            elapsed = time.time() - chunk_start

            print(
                f"Chunk candidate pairs: {pair_count:,}",
                flush=True,
            )

            print(
                f"Chunk time: {elapsed / 60:.2f} minutes",
                flush=True,
            )

            print(
                f"Saved to: {output_path}",
                flush=True,
            )

            del targets
            gc.collect()

    total_elapsed = time.time() - start_time

    print("\nDUAL-SOURCE PILOT COMPLETED")

    print(
        f"New candidate pairs written this run: "
        f"{grand_total:,}"
    )

    print(
        f"Execution time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

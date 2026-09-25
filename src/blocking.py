
import gc
from collections import defaultdict
from pathlib import Path

import pandas as pd

from preprocessing import normalize_name


TRAIN_DIR = "dataset/train"
OUTPUT_PATH = "output/exact_name_candidates.tsv"
CHUNK_SIZE = 100000


def build_name_index():
    """
    Build an exact normalized-name index using Source 2 and Source 3.

    Key: (country, normalized business name)
    Value: list of target entity IDs
    """

    print("Building exact-name index...", flush=True)

    index = defaultdict(list)

    for source in [2, 3]:

        path = f"{TRAIN_DIR}/train_source{source}.tsv"

        print(f"Reading Source {source}...", flush=True)

        processed = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "country"
            ],
            chunksize=CHUNK_SIZE
        ):

            for entity_id, name, country in zip(
                chunk["entity_id"],
                chunk["business_name"],
                chunk["country"]
            ):

                normalized_name = normalize_name(name)

                if not normalized_name:
                    continue

                key = (
                    country.casefold().strip(),
                    normalized_name
                )

                index[key].append(entity_id)

            processed += len(chunk)

            print(
                f"Source {source}: {processed:,} records indexed",
                flush=True
            )

            del chunk

        gc.collect()

    print(
        f"Unique blocking keys: {len(index):,}",
        flush=True
    )

    return index


def generate_candidates(index):
    """
    Generate exact-name candidate pairs for Source 1.

    Write candidates incrementally to avoid keeping all
    candidate pairs in memory.
    """

    print("Generating exact-name candidates...", flush=True)

    Path("output").mkdir(
        parents=True,
        exist_ok=True
    )

    total_queries = 0
    total_pairs = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

    path = f"{TRAIN_DIR}/train_source1.tsv"

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        rows = []

        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"]
        ):

            normalized_name = normalize_name(name)

            if not normalized_name:
                continue

            key = (
                country.casefold().strip(),
                normalized_name
            )

            matched_ids = index.get(key, [])

            for candidate_id in matched_ids:

                rows.append(
                    (entity_id, candidate_id)
                )

        if rows:

            candidate_df = pd.DataFrame(
                rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            )

            candidate_df.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                header=False,
                mode="a"
            )

            total_pairs += len(candidate_df)

            del candidate_df

        total_queries += len(chunk)

        print(
            f"Processed {total_queries:,} Source 1 records | "
            f"Candidate pairs: {total_pairs:,}",
            flush=True
        )

        del rows, chunk
        gc.collect()

    print("\nCANDIDATE GENERATION COMPLETED")
    print(f"Source 1 records: {total_queries:,}")
    print(f"Candidate pairs: {total_pairs:,}")
    print(f"Saved to: {OUTPUT_PATH}")


def evaluate_recall():
    """
    Evaluate candidate recall against training ground truth.
    """

    print("\nLoading ground truth...", flush=True)

    truth = pd.read_csv(
        f"{TRAIN_DIR}/train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Loading generated candidate pairs...", flush=True)

    candidates = pd.read_csv(
        OUTPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    retrieved = (
        candidates.groupby("source1_entity_id")[
            "candidate_entity_id"
        ]
        .agg(set)
        .to_dict()
    )

    found = 0
    total = 0

    for entity_id, matches in zip(
        truth["source1_entity_id"],
        truth["matched_entity_ids"]
    ):

        if not matches.strip():
            continue

        true_matches = {
            match.strip()
            for match in matches.split(",")
            if match.strip()
        }

        predicted = retrieved.get(
            entity_id,
            set()
        )

        found += len(true_matches & predicted)
        total += len(true_matches)

    recall = found / total if total else 0

    print("\nEXACT-NAME BLOCKING RESULTS")
    print(f"Total true pairs: {total:,}")
    print(f"Retrieved true pairs: {found:,}")
    print(f"Candidate recall: {recall:.4%}")

    del truth, candidates, retrieved
    gc.collect()


if __name__ == "__main__":

    index = build_name_index()

    generate_candidates(index)

    del index
    gc.collect()

    evaluate_recall()

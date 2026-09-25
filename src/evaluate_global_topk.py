
import os

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DIR = "dataset/train"

SOURCE1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
SOURCE2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
SOURCE3_PATH = f"{TRAIN_DIR}/train_source3.tsv"

GROUND_TRUTH_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

CANDIDATE_FILES = {
    "Global top-40": "output/global_topk_2000q_dual_source.tsv",
    "Global top-100": "output/global_topk_100_2000q_dual_source.tsv",
}

COUNTRY = "US"

QUERY_LIMIT = 2000
TARGET_LIMIT = 500000

READ_CHUNK_SIZE = 50000

PREVIOUS_CANDIDATE_COUNT = 798576
PREVIOUS_FOUND = 1104
PREVIOUS_ELIGIBLE = 1107


# --------------------------------------------------
# Load pilot IDs
# --------------------------------------------------

def load_country_ids(path, country, limit):

    ids = set()
    collected = 0

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

        remaining = limit - collected

        selected = selected.head(remaining)

        ids.update(selected.tolist())

        collected += len(selected)

        if collected >= limit:
            break

    return ids


# --------------------------------------------------
# Evaluate one candidate file
# --------------------------------------------------

def evaluate_candidate_file(
    label,
    path,
    eligible_true_pairs,
    query_count,
):

    if not os.path.exists(path):

        print(
            f"\n{label}: FILE NOT FOUND",
            flush=True,
        )

        print(f"Expected path: {path}")

        return None

    print(
        f"\nLoading {label} candidates...",
        flush=True,
    )

    candidates = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing_columns = required_columns - set(
        candidates.columns
    )

    if missing_columns:

        raise ValueError(
            f"{label} is missing columns: "
            f"{missing_columns}"
        )

    retrieved_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    found_pairs = (
        eligible_true_pairs & retrieved_pairs
    )

    missed_pairs = (
        eligible_true_pairs - retrieved_pairs
    )

    eligible_count = len(eligible_true_pairs)

    recall = (
        len(found_pairs) / eligible_count
        if eligible_count
        else 0.0
    )

    reduction = (
        1 - len(retrieved_pairs)
        / PREVIOUS_CANDIDATE_COUNT
    ) * 100

    print(f"\n{label.upper()} RECALL RESULTS")

    print(
        f"Retrieved genuine pairs: "
        f"{len(found_pairs):,}"
    )

    print(
        f"Missed genuine pairs: "
        f"{len(missed_pairs):,}"
    )

    print(
        f"Conditional candidate recall: "
        f"{recall:.4%}"
    )

    print(
        f"Candidate rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Unique candidate pairs: "
        f"{len(retrieved_pairs):,}"
    )

    print(
        f"Average candidates/query: "
        f"{len(retrieved_pairs) / query_count:.2f}"
    )

    print(
        f"Candidate reduction vs original: "
        f"{reduction:.2f}%"
    )

    return {
        "label": label,
        "found": len(found_pairs),
        "missed": len(missed_pairs),
        "recall": recall,
        "candidates": len(retrieved_pairs),
        "reduction": reduction,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "Loading Source 1 pilot query IDs...",
        flush=True,
    )

    query_ids = load_country_ids(
        SOURCE1_PATH,
        COUNTRY,
        QUERY_LIMIT,
    )

    print(
        f"Queries: {len(query_ids):,}",
        flush=True,
    )

    print(
        "Loading Source 2 target IDs...",
        flush=True,
    )

    source2_ids = load_country_ids(
        SOURCE2_PATH,
        COUNTRY,
        TARGET_LIMIT,
    )

    print(
        f"Source 2 targets: {len(source2_ids):,}",
        flush=True,
    )

    print(
        "Loading Source 3 target IDs...",
        flush=True,
    )

    source3_ids = load_country_ids(
        SOURCE3_PATH,
        COUNTRY,
        TARGET_LIMIT,
    )

    print(
        f"Source 3 targets: {len(source3_ids):,}",
        flush=True,
    )

    eligible_target_ids = (
        source2_ids | source3_ids
    )

    # --------------------------------------------------
    # Load ground truth
    # --------------------------------------------------

    print(
        "Loading ground truth...",
        flush=True,
    )

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

            pair = (
                source_id,
                target_id,
            )

            all_true_pairs.add(pair)

            if target_id in eligible_target_ids:

                eligible_true_pairs.add(pair)

    print("\nPILOT GROUND TRUTH")

    print(
        f"Queries evaluated: "
        f"{len(query_ids):,}"
    )

    print(
        f"All genuine pairs: "
        f"{len(all_true_pairs):,}"
    )

    print(
        f"Eligible genuine pairs: "
        f"{len(eligible_true_pairs):,}"
    )

    # --------------------------------------------------
    # Evaluate available candidate files
    # --------------------------------------------------

    results = []

    for label, path in CANDIDATE_FILES.items():

        result = evaluate_candidate_file(
            label,
            path,
            eligible_true_pairs,
            len(query_ids),
        )

        if result is not None:
            results.append(result)

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    print("\nCOMPARISON WITH PREVIOUS PILOTS")

    previous_recall = (
        PREVIOUS_FOUND / PREVIOUS_ELIGIBLE
    )

    print(
        f"Original dual-source recall: "
        f"{previous_recall:.4%}"
    )

    print(
        f"Original candidate pairs: "
        f"{PREVIOUS_CANDIDATE_COUNT:,}"
    )

    for result in results:

        print(
            f"\n{result['label']}:"
        )

        print(
            f"  Recall: "
            f"{result['recall']:.4%}"
        )

        print(
            f"  Retrieved genuine pairs: "
            f"{result['found']:,}"
        )

        print(
            f"  Missed genuine pairs: "
            f"{result['missed']:,}"
        )

        print(
            f"  Candidate pairs: "
            f"{result['candidates']:,}"
        )

        print(
            f"  Candidate reduction: "
            f"{result['reduction']:.2f}%"
        )

    print(
        "\nGLOBAL TOP-K EVALUATION COMPLETED"
    )


if __name__ == "__main__":
    main()

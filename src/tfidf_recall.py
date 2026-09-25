
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import normalize_name


TRAIN_DIR = "dataset/train"

QUERY_COUNT = 200
DISTRACTORS_PER_SOURCE = 10000
TOP_K_VALUES = [5, 10, 20, 50]
BATCH_SIZE = 20

print("Loading Source 1 and ground truth...", flush=True)

s1 = pd.read_csv(
    f"{TRAIN_DIR}/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

truth = pd.read_csv(
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Select queries with at least one genuine match.
truth = truth[truth["matched_entity_ids"].str.strip() != ""]

queries = truth.sample(
    n=min(QUERY_COUNT, len(truth)),
    random_state=42
).copy()

queries = queries.merge(
    s1,
    left_on="source1_entity_id",
    right_on="entity_id",
    how="inner",
    validate="one_to_one"
)

del s1, truth

print(f"Selected {len(queries)} queries", flush=True)

# Collect genuine matching IDs.
true_ids = set()

for matches in queries["matched_entity_ids"]:
    true_ids.update(
        x.strip() for x in matches.split(",") if x.strip()
    )

print(f"Genuine target records: {len(true_ids):,}", flush=True)

print("Collecting genuine matches and distractors...", flush=True)

candidate_parts = []

for source in [2, 3]:

    path = f"{TRAIN_DIR}/train_source{source}.tsv"

    genuine_parts = []
    distractor_parts = []
    distractor_count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):

        genuine = chunk[chunk["entity_id"].isin(true_ids)]

        if not genuine.empty:
            genuine_parts.append(genuine)

        if distractor_count < DISTRACTORS_PER_SOURCE:

            unrelated = chunk[
                ~chunk["entity_id"].isin(true_ids)
            ]

            remaining = DISTRACTORS_PER_SOURCE - distractor_count

            selected = unrelated.head(remaining)

            if not selected.empty:
                distractor_parts.append(selected)
                distractor_count += len(selected)

    parts = genuine_parts + distractor_parts

    if parts:
        candidate_parts.append(
            pd.concat(parts, ignore_index=True)
        )

    print(
        f"Source {source}: collected "
        f"{sum(len(part) for part in parts):,} records",
        flush=True
    )

candidates = pd.concat(
    candidate_parts,
    ignore_index=True
)

candidates = candidates.drop_duplicates(
    subset="entity_id"
).reset_index(drop=True)

candidate_ids = set(candidates["entity_id"])

missing = true_ids - candidate_ids

print(f"Total candidates: {len(candidates):,}", flush=True)
print(f"Missing genuine target IDs: {len(missing):,}", flush=True)

if missing:
    raise RuntimeError(
        "Some genuine matches were not found in Source 2/3. "
        "Check the input files and ground truth."
    )

print("Normalizing business names...", flush=True)

queries["name_clean"] = queries["business_name"].map(
    normalize_name
)

candidates["name_clean"] = candidates["business_name"].map(
    normalize_name
)

print("Building TF-IDF vectors...", flush=True)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    dtype=np.float32
)

candidate_vectors = vectorizer.fit_transform(
    candidates["name_clean"]
)

query_vectors = vectorizer.transform(
    queries["name_clean"]
)

candidate_id_array = candidates["entity_id"].to_numpy()

found = {k: 0 for k in TOP_K_VALUES}
total = 0

print("Evaluating retrieval...", flush=True)

for start in range(0, len(queries), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(queries))

    similarities = (
        query_vectors[start:end] @ candidate_vectors.T
    ).tocsr()

    for local_i in range(end - start):

        query = queries.iloc[start + local_i]

        true_matches = {
            x.strip()
            for x in query["matched_entity_ids"].split(",")
            if x.strip()
        }

        row = similarities.getrow(local_i)

        if row.nnz == 0:
            total += len(true_matches)
            continue

        # Rank candidates with nonzero cosine similarity.
        ranking = np.argsort(row.data)[::-1]

        for k in TOP_K_VALUES:

            positions = ranking[:k]

            retrieved_ids = set(
                candidate_id_array[row.indices[positions]]
            )

            found[k] += len(
                true_matches & retrieved_ids
            )

        total += len(true_matches)

    print(
        f"Processed {end}/{len(queries)} queries",
        flush=True
    )

print("\nGROUND-TRUTH-AWARE TF-IDF RESULTS")
print(f"Total genuine pairs: {total:,}")

for k in TOP_K_VALUES:

    recall = found[k] / total if total else 0

    print(
        f"Recall@{k}: {recall:.4%} "
        f"({found[k]:,}/{total:,})"
    )

print("\nTF-IDF RECALL EXPERIMENT COMPLETED")

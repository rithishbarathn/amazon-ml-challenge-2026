
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import normalize_name, normalize_address


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

true_ids = set()

for matches in queries["matched_entity_ids"]:
    true_ids.update(
        x.strip() for x in matches.split(",") if x.strip()
    )

print(f"Selected queries: {len(queries)}", flush=True)
print(f"Genuine target records: {len(true_ids)}", flush=True)

print("Collecting genuine matches and distractors...", flush=True)

candidate_parts = []

for source in [2, 3]:

    genuine_parts = []
    distractor_parts = []
    distractor_count = 0

    path = f"{TRAIN_DIR}/train_source{source}.tsv"

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

    print(f"Source {source} collected.", flush=True)

candidates = pd.concat(
    candidate_parts,
    ignore_index=True
)

candidates = candidates.drop_duplicates(
    subset="entity_id"
).reset_index(drop=True)

missing = true_ids - set(candidates["entity_id"])

print(f"Total candidates: {len(candidates):,}", flush=True)
print(f"Missing genuine IDs: {len(missing)}", flush=True)

if missing:
    raise RuntimeError("Some genuine matches are missing.")

print("Normalizing names and addresses...", flush=True)

for df in [queries, candidates]:

    df["name_clean"] = df["business_name"].map(
        normalize_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )

print("Building name TF-IDF vectors...", flush=True)

name_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    dtype=np.float32
)

candidate_name_vectors = name_vectorizer.fit_transform(
    candidates["name_clean"]
)

query_name_vectors = name_vectorizer.transform(
    queries["name_clean"]
)

print("Building address TF-IDF vectors...", flush=True)

address_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    dtype=np.float32
)

candidate_address_vectors = address_vectorizer.fit_transform(
    candidates["address_clean"]
)

query_address_vectors = address_vectorizer.transform(
    queries["address_clean"]
)

candidate_ids = candidates["entity_id"].to_numpy()

methods = ["name", "address", "combined"]

found = {
    method: {k: 0 for k in TOP_K_VALUES}
    for method in methods
}

total = 0


def get_top_ids(row, k):

    if row.nnz == 0:
        return set()

    positions = np.argsort(row.data)[-k:][::-1]

    return set(
        candidate_ids[row.indices[positions]]
    )


print("Evaluating retrieval...", flush=True)

for start in range(0, len(queries), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(queries))

    name_sim = (
        query_name_vectors[start:end]
        @ candidate_name_vectors.T
    ).tocsr()

    address_sim = (
        query_address_vectors[start:end]
        @ candidate_address_vectors.T
    ).tocsr()

    for local_i in range(end - start):

        query = queries.iloc[start + local_i]

        true_matches = {
            x.strip()
            for x in query["matched_entity_ids"].split(",")
            if x.strip()
        }

        name_row = name_sim.getrow(local_i)
        address_row = address_sim.getrow(local_i)

        total += len(true_matches)

        for k in TOP_K_VALUES:

            name_ids = get_top_ids(name_row, k)
            address_ids = get_top_ids(address_row, k)

            # Union of both retrieval methods.
            combined_ids = name_ids | address_ids

            found["name"][k] += len(
                true_matches & name_ids
            )

            found["address"][k] += len(
                true_matches & address_ids
            )

            found["combined"][k] += len(
                true_matches & combined_ids
            )

    print(
        f"Processed {end}/{len(queries)} queries",
        flush=True
    )


print("\nCOMBINED RETRIEVAL RESULTS")
print(f"Total genuine pairs: {total:,}")

for method in methods:

    print(f"\n{method.upper()} RETRIEVAL")

    for k in TOP_K_VALUES:

        recall = found[method][k] / total if total else 0

        print(
            f"Recall@{k}: {recall:.4%} "
            f"({found[method][k]:,}/{total:,})"
        )

print("\nCOMBINED EXPERIMENT COMPLETED")

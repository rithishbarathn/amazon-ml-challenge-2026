import pandas as pd

TRAIN_DIR = "dataset/train"

# Match these settings with scalable_blocking.py
COUNTRY = "US"
TARGET_LIMIT = 500000

CANDIDATE_PATH = "output/scalable_blocking_2000q_500k.tsv"


print("Loading 500K pilot candidates...", flush=True)

pairs = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Loading ground truth...", flush=True)

truth = pd.read_csv(
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Loading Source 2 target IDs...", flush=True)

# Reconstruct the same first 500,000 US records
# selected by scalable_blocking.py.
target_ids = set()
collected = 0

for chunk in pd.read_csv(
    f"{TRAIN_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=50000,
    usecols=["entity_id", "country"]
):

    us_ids = chunk.loc[
        chunk["country"] == COUNTRY,
        "entity_id"
    ]

    remaining = TARGET_LIMIT - collected

    selected = us_ids.head(remaining)

    target_ids.update(selected)
    collected += len(selected)

    if collected >= TARGET_LIMIT:
        break

print(f"Target IDs: {len(target_ids):,}", flush=True)

print("Grouping retrieved candidates...", flush=True)

retrieved = (
    pairs.groupby("source1_entity_id")["candidate_entity_id"]
    .agg(set)
    .to_dict()
)

query_ids = set(pairs["source1_entity_id"])

truth = truth[
    truth["source1_entity_id"].isin(query_ids)
]

found = 0
eligible = 0
all_true = 0

for row in truth.itertuples(index=False):

    true_ids = {
        x.strip()
        for x in row.matched_entity_ids.split(",")
        if x.strip()
    }

    # All genuine matches for these queries.
    all_true += len(true_ids)

    # Only genuine matches present in the 500K
    # Source 2 target collection.
    eligible_ids = true_ids & target_ids

    eligible += len(eligible_ids)

    predicted_ids = retrieved.get(
        row.source1_entity_id,
        set()
    )

    found += len(eligible_ids & predicted_ids)


print("\n500K PILOT RECALL RESULTS")

print(f"Queries evaluated: {len(truth):,}")
print(f"All genuine pairs: {all_true:,}")
print(f"Eligible genuine pairs in pilot targets: {eligible:,}")
print(f"Retrieved eligible genuine pairs: {found:,}")

if eligible:
    recall = found / eligible
    print(f"Conditional candidate recall: {recall:.4%}")
else:
    print("Conditional candidate recall: N/A (no eligible pairs)")

print("\n500K PILOT EVALUATION COMPLETED")
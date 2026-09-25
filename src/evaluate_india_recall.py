
import pandas as pd

CANDIDATE_PATH = "output/global_top100_india_2000q.tsv"
GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"

print("Loading candidate pairs...", flush=True)

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

required = {"source1_entity_id", "candidate_entity_id"}
if not required.issubset(candidates.columns):
    raise ValueError(f"Missing columns: {required - set(candidates.columns)}")

candidate_sets = (
    candidates.groupby("source1_entity_id")["candidate_entity_id"]
    .agg(set)
    .to_dict()
)

print("Loading ground truth...", flush=True)

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(candidate_sets)
]

total_true = 0
retrieved_true = 0
queries_with_matches = 0
queries_with_all_matches = 0

for row in ground_truth.itertuples(index=False):
    true_ids = {
        x.strip()
        for x in row.matched_entity_ids.split(",")
        if x.strip()
    }

    retrieved_ids = candidate_sets.get(row.source1_entity_id, set())

    if true_ids:
        queries_with_matches += 1
        if true_ids.issubset(retrieved_ids):
            queries_with_all_matches += 1

    total_true += len(true_ids)
    retrieved_true += len(true_ids & retrieved_ids)

recall = retrieved_true / total_true if total_true else 0

print("\nINDIA FULL-TARGET BLOCKING EVALUATION")
print("-" * 45)
print(f"Queries evaluated: {len(ground_truth):,}")
print(f"Candidate pairs: {len(candidates):,}")
print(f"Total genuine matches: {total_true:,}")
print(f"Genuine matches retrieved: {retrieved_true:,}")
print(f"Genuine matches missed: {total_true - retrieved_true:,}")
print(f"Candidate recall: {recall:.4%}")
print(f"Queries with genuine matches: {queries_with_matches:,}")
print(f"Queries with ALL matches retrieved: {queries_with_all_matches:,}")
print("-" * 45)


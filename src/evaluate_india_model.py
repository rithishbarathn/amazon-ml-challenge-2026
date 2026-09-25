
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, fbeta_score


FEATURE_PATH = "output/features_india_2000q.tsv"
TRUTH_PATH = "dataset/train/train_ground_truth.tsv"
MODEL_PATH = "output/xgboost_full_targets_2000q.json"
PREDICTION_PATH = "output/india_model_predictions.tsv"

THRESHOLD = 0.980

FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
    "exact_name",
    "exact_address",
    "same_country",
    "address_number_overlap",
    "name_length_difference",
    "address_length_difference",
    "name_missing_1",
    "name_missing_2",
    "address_missing_1",
    "address_missing_2",
]


def parse_matches(value):
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def macro_f05(query_ids, predicted_map, truth_map):
    scores = []

    for query_id in query_ids:
        actual = truth_map[query_id]
        predicted = predicted_map.get(query_id, set())

        if not actual:
            scores.append(1.0 if not predicted else 0.0)
            continue

        tp = len(actual & predicted)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        denominator = 1.25 * tp + 0.25 * fn + fp

        score = (
            1.25 * tp / denominator
            if denominator > 0
            else 0.0
        )

        scores.append(score)

    return float(np.mean(scores))


print("Loading India features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_entity_id": str,
    },
)

missing_features = set(FEATURE_COLUMNS) - set(df.columns)

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

if df[["source1_entity_id", "candidate_entity_id"]].duplicated().any():
    raise ValueError("Duplicate candidate pairs found.")

print("Loading ground truth...", flush=True)

truth = pd.read_csv(
    TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

truth_map = {
    row.source1_entity_id: parse_matches(row.matched_entity_ids)
    for row in truth.itertuples(index=False)
}

query_ids = df["source1_entity_id"].unique().tolist()

missing_queries = set(query_ids) - set(truth_map)

if missing_queries:
    raise ValueError(
        f"Ground truth missing for {len(missing_queries)} India queries."
    )

print("Loading existing US-trained XGBoost model...", flush=True)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X = df[FEATURE_COLUMNS].fillna(0).astype(np.float32)

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

actual_labels = np.array(
    [
        int(candidate_id in truth_map[query_id])
        for query_id, candidate_id in zip(
            df["source1_entity_id"],
            df["candidate_entity_id"],
        )
    ],
    dtype=int,
)

predicted_map = {}

for query_id, candidate_id, predicted in zip(
    df["source1_entity_id"],
    df["candidate_entity_id"],
    predictions,
):
    if predicted:
        predicted_map.setdefault(query_id, set()).add(candidate_id)

macro_score = macro_f05(
    query_ids,
    predicted_map,
    truth_map,
)

precision = precision_score(
    actual_labels,
    predictions,
    zero_division=0,
)

candidate_recall = recall_score(
    actual_labels,
    predictions,
    zero_division=0,
)

candidate_f05 = fbeta_score(
    actual_labels,
    predictions,
    beta=0.5,
    zero_division=0,
)

tp = int(((actual_labels == 1) & (predictions == 1)).sum())
fp = int(((actual_labels == 0) & (predictions == 1)).sum())
fn = int(((actual_labels == 1) & (predictions == 0)).sum())

all_true_matches = sum(
    len(truth_map[query_id])
    for query_id in query_ids
)

retrieved_true_matches = int(actual_labels.sum())

blocking_recall = (
    retrieved_true_matches / all_true_matches
    if all_true_matches
    else 0.0
)

end_to_end_recall = (
    tp / all_true_matches
    if all_true_matches
    else 0.0
)

singleton_queries = sum(
    not truth_map[query_id]
    for query_id in query_ids
)

print("\nINDIA CROSS-COUNTRY MODEL EVALUATION")
print("-" * 50)
print(f"Model: {MODEL_PATH}")
print(f"Threshold: {THRESHOLD:.3f}")
print(f"Queries evaluated: {len(query_ids):,}")
print(f"Singleton queries: {singleton_queries:,}")
print(f"Candidate pairs: {len(df):,}")
print(f"Total genuine matches: {all_true_matches:,}")
print(f"Genuine matches retrieved by blocking: {retrieved_true_matches:,}")
print(f"Blocking recall: {blocking_recall:.4%}")
print(f"Macro F0.5: {macro_score:.4%}")
print(f"Pair precision: {precision:.4%}")
print(f"Candidate-level recall: {candidate_recall:.4%}")
print(f"Candidate-level F0.5: {candidate_f05:.4%}")
print(f"End-to-end recall: {end_to_end_recall:.4%}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives among candidates: {fn:,}")
print("-" * 50)

results = df[
    ["source1_entity_id", "candidate_entity_id"]
].copy()

results["label"] = actual_labels
results["match_probability"] = probabilities
results["predicted_label"] = predictions

results.to_csv(
    PREDICTION_PATH,
    sep="\t",
    index=False,
)

print(f"Predictions saved: {PREDICTION_PATH}")
print("INDIA MODEL EVALUATION COMPLETED")

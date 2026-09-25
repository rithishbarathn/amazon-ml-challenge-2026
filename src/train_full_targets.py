
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score


FEATURE_PATH = "output/features_full_targets_2000q.tsv"
TRUTH_PATH = "dataset/train/train_ground_truth.tsv"

MODEL_PATH = "output/xgboost_full_targets_2000q.json"
PREDICTION_PATH = "output/full_targets_holdout_predictions.tsv"

RANDOM_STATE = 42

THRESHOLDS = [
    0.50, 0.70, 0.80, 0.90, 0.95,
    0.97, 0.98, 0.99, 0.995, 0.999,
]


def parse_matches(value):
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def macro_f05(query_ids, candidate_df, probabilities, threshold, truth_map):
    predicted_map = {}

    for query_id, candidate_id, probability in zip(
        candidate_df["source1_entity_id"],
        candidate_df["candidate_entity_id"],
        probabilities,
    ):
        if probability >= threshold:
            predicted_map.setdefault(query_id, set()).add(candidate_id)

    scores = []

    for query_id in query_ids:
        actual = truth_map.get(query_id, set())
        predicted = predicted_map.get(query_id, set())

        # Correctly predicting no matches for a singleton scores 1.
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


print("Loading full-target features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_entity_id": str,
    },
)

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

positive_pairs = {
    (query_id, candidate_id)
    for query_id, matches in truth_map.items()
    for candidate_id in matches
}

df["label"] = [
    int((query_id, candidate_id) in positive_pairs)
    for query_id, candidate_id in zip(
        df["source1_entity_id"],
        df["candidate_entity_id"],
    )
]

feature_columns = [
    column
    for column in df.columns
    if column not in {
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    }
]

if len(feature_columns) != 18:
    raise ValueError(
        f"Expected 18 features, found {len(feature_columns)}"
    )

X = df[feature_columns].fillna(0).astype(np.float32)
y = df["label"]
groups = df["source1_entity_id"]

print(f"Candidate pairs: {len(df):,}")
print(f"Positive candidate pairs: {int(y.sum()):,}")
print(f"Feature columns: {len(feature_columns)}")

# Split by Source 1 entity, never by individual candidate pair.
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

development_idx, holdout_idx = next(
    outer_split.split(X, y, groups)
)

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=123,
)

train_local, valid_local = next(
    inner_split.split(
        X.iloc[development_idx],
        y.iloc[development_idx],
        groups.iloc[development_idx],
    )
)

train_idx = development_idx[train_local]
valid_idx = development_idx[valid_local]

train_queries = set(groups.iloc[train_idx])
valid_queries = set(groups.iloc[valid_idx])
holdout_queries = set(groups.iloc[holdout_idx])

assert train_queries.isdisjoint(valid_queries)
assert train_queries.isdisjoint(holdout_queries)
assert valid_queries.isdisjoint(holdout_queries)

print("\nGROUP-BASED SPLIT")
print(f"Training queries: {len(train_queries):,}")
print(f"Validation queries: {len(valid_queries):,}")
print(f"Holdout queries: {len(holdout_queries):,}")

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]
X_holdout = X.iloc[holdout_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]
y_holdout = y.iloc[holdout_idx]

if min(y_train.sum(), y_valid.sum(), y_holdout.sum()) == 0:
    raise ValueError("One split has no positive examples.")

positive_count = int(y_train.sum())
negative_count = int((y_train == 0).sum())

print("\nTraining full-target XGBoost...", flush=True)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=4,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

# Select the threshold using validation data only.
valid_probabilities = model.predict_proba(X_valid)[:, 1]
valid_pairs = df.iloc[valid_idx]

print("\nVALIDATION THRESHOLD COMPARISON")
print("-" * 45)

threshold_results = []

for threshold in THRESHOLDS:
    score = macro_f05(
        valid_queries,
        valid_pairs,
        valid_probabilities,
        threshold,
        truth_map,
    )

    threshold_results.append((score, threshold))
    print(f"Threshold {threshold:.3f}: Macro F0.5 = {score:.4%}")

best_valid_score, best_threshold = max(
    threshold_results,
    key=lambda item: item[0],
)

print("-" * 45)
print(f"Selected threshold: {best_threshold:.3f}")
print(f"Validation macro F0.5: {best_valid_score:.4%}")

# Evaluate the selected model and threshold on holdout queries.
holdout_probabilities = model.predict_proba(X_holdout)[:, 1]
holdout_predictions = (
    holdout_probabilities >= best_threshold
).astype(int)

holdout_pairs = df.iloc[holdout_idx]

holdout_macro = macro_f05(
    holdout_queries,
    holdout_pairs,
    holdout_probabilities,
    best_threshold,
    truth_map,
)

precision = precision_score(
    y_holdout,
    holdout_predictions,
    zero_division=0,
)

candidate_recall = recall_score(
    y_holdout,
    holdout_predictions,
    zero_division=0,
)

candidate_f05 = fbeta_score(
    y_holdout,
    holdout_predictions,
    beta=0.5,
    zero_division=0,
)

tp = int(
    ((y_holdout.to_numpy() == 1) & (holdout_predictions == 1)).sum()
)
fp = int(
    ((y_holdout.to_numpy() == 0) & (holdout_predictions == 1)).sum()
)
fn = int(
    ((y_holdout.to_numpy() == 1) & (holdout_predictions == 0)).sum()
)

all_true_holdout = sum(
    len(truth_map.get(query_id, set()))
    for query_id in holdout_queries
)

end_to_end_recall = (
    tp / all_true_holdout
    if all_true_holdout
    else 0.0
)

singleton_queries = sum(
    not truth_map.get(query_id, set())
    for query_id in holdout_queries
)

print("\nFULL-TARGET HOLDOUT RESULTS")
print("-" * 45)
print(f"Threshold: {best_threshold:.3f}")
print(f"Holdout queries: {len(holdout_queries):,}")
print(f"Singleton queries: {singleton_queries:,}")
print(f"Macro F0.5: {holdout_macro:.4%}")
print(f"Pair precision: {precision:.4%}")
print(f"Candidate-level recall: {candidate_recall:.4%}")
print(f"Candidate-level F0.5: {candidate_f05:.4%}")
print(f"End-to-end recall: {end_to_end_recall:.4%}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives among candidates: {fn:,}")
print(f"All genuine holdout matches: {all_true_holdout:,}")
print("-" * 45)

os.makedirs("output", exist_ok=True)

model.save_model(MODEL_PATH)

results = holdout_pairs[
    ["source1_entity_id", "candidate_entity_id", "label"]
].copy()

results["match_probability"] = holdout_probabilities
results["predicted_label"] = holdout_predictions

results.to_csv(
    PREDICTION_PATH,
    sep="\t",
    index=False,
)

print(f"\nModel saved: {MODEL_PATH}")
print(f"Predictions saved: {PREDICTION_PATH}")
print("FULL-TARGET TRAINING COMPLETED")

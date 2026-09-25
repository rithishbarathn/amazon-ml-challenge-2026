
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score

FEATURE_PATH = "output/features_global_top100_2000q.tsv"
TRUTH_PATH = "dataset/train/train_ground_truth.tsv"

MODEL_PATH = "output/xgboost_independent_holdout.json"
PREDICTION_PATH = "output/independent_holdout_predictions.tsv"

RANDOM_STATE = 42
THRESHOLD = 0.99

print("Loading features...", flush=True)

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

positive_pairs = set()

for row in truth.itertuples(index=False):
    for candidate_id in row.matched_entity_ids.split(","):
        candidate_id = candidate_id.strip()

        if candidate_id:
            positive_pairs.add(
                (row.source1_entity_id, candidate_id)
            )

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

X = df[feature_columns].fillna(0).astype(np.float32)
y = df["label"]
groups = df["source1_entity_id"]

# Split 1: 80% development, 20% untouched holdout
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

development_idx, holdout_idx = next(
    outer_split.split(X, y, groups)
)

# Split 2: separate training and early-stopping validation
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

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]
X_holdout = X.iloc[holdout_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]
y_holdout = y.iloc[holdout_idx]

print("\nINDEPENDENT SPLIT")
print(f"Training queries: {groups.iloc[train_idx].nunique():,}")
print(f"Validation queries: {groups.iloc[valid_idx].nunique():,}")
print(f"Holdout queries: {groups.iloc[holdout_idx].nunique():,}")

assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[holdout_idx])
)
assert set(groups.iloc[valid_idx]).isdisjoint(
    set(groups.iloc[holdout_idx])
)
assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[valid_idx])
)

if y_train.sum() == 0 or y_valid.sum() == 0 or y_holdout.sum() == 0:
    raise ValueError("One split has no positive examples.")

positive_count = int(y_train.sum())
negative_count = int((y_train == 0).sum())

print("\nTraining independent XGBoost model...", flush=True)

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

# Holdout is used only after training and threshold selection.
probabilities = model.predict_proba(X_holdout)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

precision = precision_score(
    y_holdout, predictions, zero_division=0
)
recall = recall_score(
    y_holdout, predictions, zero_division=0
)
f05 = fbeta_score(
    y_holdout, predictions, beta=0.5, zero_division=0
)

tp = int(((y_holdout == 1) & (predictions == 1)).sum())
fp = int(((y_holdout == 0) & (predictions == 1)).sum())
fn = int(((y_holdout == 1) & (predictions == 0)).sum())

print("\nINDEPENDENT HOLDOUT RESULTS")
print(f"Threshold: {THRESHOLD:.3f}")
print(f"Precision: {precision:.4%}")
print(f"Candidate-level recall: {recall:.4%}")
print(f"Candidate-level F0.5: {f05:.4%}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")

# Include genuine matches missed by blocking.
holdout_query_ids = set(groups.iloc[holdout_idx])

all_holdout_positives = sum(
    query_id in holdout_query_ids
    for query_id, candidate_id in positive_pairs
)

end_to_end_recall = (
    tp / all_holdout_positives
    if all_holdout_positives
    else 0.0
)

end_to_end_f05 = (
    1.25 * precision * end_to_end_recall
    / (0.25 * precision + end_to_end_recall)
    if precision + end_to_end_recall > 0
    else 0.0
)

print("\nEND-TO-END HOLDOUT RESULTS")
print(f"All genuine holdout pairs: {all_holdout_positives:,}")
print(f"End-to-end precision: {precision:.4%}")
print(f"End-to-end recall: {end_to_end_recall:.4%}")
print(f"End-to-end micro F0.5: {end_to_end_f05:.4%}")

os.makedirs("output", exist_ok=True)

model.save_model(MODEL_PATH)

results = df.iloc[holdout_idx][
    ["source1_entity_id", "candidate_entity_id", "label"]
].copy()

results["match_probability"] = probabilities
results["predicted_label"] = predictions

results.to_csv(
    PREDICTION_PATH,
    sep="\t",
    index=False,
)

print(f"\nModel saved: {MODEL_PATH}")
print(f"Predictions saved: {PREDICTION_PATH}")
print("INDEPENDENT HOLDOUT EVALUATION COMPLETED")

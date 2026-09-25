
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FEATURE_PATH = "output/features_2000q_500k.tsv"
GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"

MODEL_PATH = "output/xgboost_2000q_500k.json"
OUTPUT_PATH = "output/holdout_predictions_2000q_500k.tsv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLD = 0.970


# --------------------------------------------------
# 1. Load features
# --------------------------------------------------

print("Loading features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_entity_id": str,
    },
)

print(f"Candidate pairs: {len(df):,}")


# --------------------------------------------------
# 2. Load ground truth and create labels
# --------------------------------------------------

print("Loading ground truth...", flush=True)

truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

query_ids = set(df["source1_entity_id"])

truth = truth[
    truth["source1_entity_id"].isin(query_ids)
]

positive_pairs = set()

for row in truth.itertuples(index=False):
    for matched_id in row.matched_entity_ids.split(","):
        matched_id = matched_id.strip()

        if matched_id:
            positive_pairs.add(
                (row.source1_entity_id, matched_id)
            )

df["label"] = [
    int((source_id, candidate_id) in positive_pairs)
    for source_id, candidate_id in zip(
        df["source1_entity_id"],
        df["candidate_entity_id"],
    )
]


# --------------------------------------------------
# 3. Reconstruct original validation split
# --------------------------------------------------

print("Reconstructing original validation split...", flush=True)

exclude_columns = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
}

feature_columns = [
    column
    for column in df.columns
    if column not in exclude_columns
]

X = df[feature_columns].fillna(0).astype(np.float32)
y = df["label"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
)

_, valid_idx = next(
    splitter.split(
        X,
        y,
        groups=df["source1_entity_id"],
    )
)

X_valid = X.iloc[valid_idx]
y_valid = y.iloc[valid_idx]


# --------------------------------------------------
# 4. Load model and predict
# --------------------------------------------------

print("Loading trained XGBoost model...", flush=True)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

probabilities = model.predict_proba(X_valid)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)


# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------

precision = precision_score(
    y_valid,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_valid,
    predictions,
    zero_division=0,
)

f05 = fbeta_score(
    y_valid,
    predictions,
    beta=0.5,
    zero_division=0,
)

tn, fp, fn, tp = confusion_matrix(
    y_valid,
    predictions,
    labels=[0, 1],
).ravel()

print("\nVALIDATION RESULTS AT THRESHOLD 0.970")
print(f"Validation pairs: {len(y_valid):,}")
print(f"Precision: {precision:.4%}")
print(f"Recall: {recall:.4%}")
print(f"F0.5: {f05:.4%}")

print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True negatives: {tn:,}")


# --------------------------------------------------
# 6. Save predictions
# --------------------------------------------------

os.makedirs("output", exist_ok=True)

results = df.iloc[valid_idx][
    [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ]
].copy()

results["match_probability"] = probabilities
results["predicted_label"] = predictions

results.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)

print("\nEVALUATION COMPLETED")
print(f"Saved to: {OUTPUT_PATH}")

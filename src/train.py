
import os
import time

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

TRAIN_DIR = "dataset/train"

FEATURE_PATH = "output/features_2000q_500k.tsv"
GROUND_TRUTH_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

MODEL_PATH = "output/xgboost_2000q_500k.json"
VALIDATION_PATH = "output/validation_predictions_2000q_500k.tsv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLD = 0.50


# --------------------------------------------------
# 1. Load features
# --------------------------------------------------

print("Loading similarity features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_entity_id": str,
    },
)

print(f"Candidate pairs: {len(df):,}", flush=True)


# --------------------------------------------------
# 2. Load ground truth
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


# --------------------------------------------------
# 3. Create labels
# --------------------------------------------------

print("Creating positive match labels...", flush=True)

positive_pairs = set()

for row in truth.itertuples(index=False):

    source1_id = row.source1_entity_id

    matched_ids = [
        value.strip()
        for value in row.matched_entity_ids.split(",")
        if value.strip()
    ]

    for matched_id in matched_ids:
        positive_pairs.add((source1_id, matched_id))


df["label"] = [
    int((source1_id, candidate_id) in positive_pairs)
    for source1_id, candidate_id in zip(
        df["source1_entity_id"],
        df["candidate_entity_id"],
    )
]

positive_count = int(df["label"].sum())
negative_count = int((df["label"] == 0).sum())

print(f"Positive candidate pairs: {positive_count:,}")
print(f"Negative candidate pairs: {negative_count:,}")


# --------------------------------------------------
# 4. Select model features
# --------------------------------------------------

EXCLUDE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
}

feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDE_COLUMNS
]

X = df[feature_columns].fillna(0).astype(np.float32)
y = df["label"]

groups = df["source1_entity_id"]

print(f"Features selected: {len(feature_columns)}")


# --------------------------------------------------
# 5. Split by Source 1 entity ID
# --------------------------------------------------

print("\nSplitting by Source 1 entity ID...", flush=True)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, valid_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

print(f"Training pairs: {len(X_train):,}")
print(f"Validation pairs: {len(X_valid):,}")
print(f"Training positives: {y_train.sum():,}")
print(f"Validation positives: {y_valid.sum():,}")


if y_train.sum() == 0 or y_valid.sum() == 0:
    raise ValueError(
        "The split has no positive pairs in training or validation. "
        "Try changing RANDOM_STATE or expanding the pilot."
    )


# --------------------------------------------------
# 6. Train XGBoost
# --------------------------------------------------

print("\nTraining XGBoost...", flush=True)

start_time = time.time()

train_positive = int((y_train == 1).sum())
train_negative = int((y_train == 0).sum())

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=train_negative / train_positive,
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


# --------------------------------------------------
# 7. Predict validation pairs
# --------------------------------------------------

print("Predicting validation pairs...", flush=True)

probabilities = model.predict_proba(X_valid)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# --------------------------------------------------
# 8. Evaluate
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


print("\nXGBOOST 2000-QUERY VALIDATION RESULTS")

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Precision: {precision:.4%}")
print(f"Recall: {recall:.4%}")
print(f"F0.5 score: {f05:.4%}")

print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True negatives: {tn:,}")


# --------------------------------------------------
# 9. Save model and predictions
# --------------------------------------------------

print("\nSaving model and validation predictions...", flush=True)

os.makedirs("output", exist_ok=True)

model.save_model(MODEL_PATH)

validation_results = df.iloc[valid_idx][
    [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ]
].copy()

validation_results["match_probability"] = probabilities
validation_results["predicted_label"] = predictions

validation_results.to_csv(
    VALIDATION_PATH,
    sep="\t",
    index=False,
)


# --------------------------------------------------
# 10. Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

print("\nTOP 10 FEATURES")
print(importance.head(10).to_string(index=False))


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

elapsed = time.time() - start_time

print("\nXGBOOST 2000-QUERY TRAINING COMPLETED")
print(f"Features used: {len(feature_columns)}")
print(f"Best iteration: {model.best_iteration}")
print(f"Training time: {elapsed:.2f} seconds")
print(f"Model saved to: {MODEL_PATH}")
print(f"Validation predictions saved to: {VALIDATION_PATH}")

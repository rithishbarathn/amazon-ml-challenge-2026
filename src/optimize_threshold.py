
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "output/validation_predictions_2000q_500k.tsv"
OUTPUT_PATH = "output/threshold_results_2000q_500k_fine.tsv"

# Test thresholds from 0.800 to 1.000 in steps of 0.005.
THRESHOLDS = np.round(
    np.arange(0.800, 1.001, 0.005),
    3,
)

BASELINE_THRESHOLD = 0.95


# --------------------------------------------------
# 1. Load validation predictions
# --------------------------------------------------

print("Loading validation predictions...", flush=True)

df = pd.read_csv(
    INPUT_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "candidate_entity_id": str,
    },
)

y_true = df["label"].astype(int).to_numpy()
probabilities = df["match_probability"].to_numpy()

print(f"Validation candidate pairs: {len(df):,}")
print(f"Genuine matches: {int(y_true.sum()):,}")


# --------------------------------------------------
# 2. Evaluate thresholds
# --------------------------------------------------

print("\nEvaluating thresholds...", flush=True)

results = []

for threshold in THRESHOLDS:

    y_pred = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    })


# --------------------------------------------------
# 3. Find best threshold
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Primary metric: F0.5
# Tie-breakers: higher precision, then higher threshold.
ranked = results_df.sort_values(
    by=["f0.5", "precision", "threshold"],
    ascending=[False, False, False],
)

best = ranked.iloc[0]

baseline = results_df.loc[
    np.isclose(
        results_df["threshold"],
        BASELINE_THRESHOLD,
    )
].iloc[0]


# --------------------------------------------------
# 4. Display baseline results
# --------------------------------------------------

print(
    f"\nBASELINE RESULTS "
    f"(THRESHOLD = {BASELINE_THRESHOLD:.3f})"
)

print(f"Precision: {baseline['precision']:.4%}")
print(f"Recall: {baseline['recall']:.4%}")
print(f"F0.5: {baseline['f0.5']:.4%}")

print(
    f"True positives: "
    f"{int(baseline['true_positives']):,}"
)

print(
    f"False positives: "
    f"{int(baseline['false_positives']):,}"
)


# --------------------------------------------------
# 5. Display best threshold
# --------------------------------------------------

print("\nBEST VALIDATION THRESHOLD")

print(f"Threshold: {best['threshold']:.3f}")
print(f"Precision: {best['precision']:.4%}")
print(f"Recall: {best['recall']:.4%}")
print(f"F0.5: {best['f0.5']:.4%}")

print(
    f"True positives: "
    f"{int(best['true_positives']):,}"
)

print(
    f"False positives: "
    f"{int(best['false_positives']):,}"
)

print(
    f"False negatives: "
    f"{int(best['false_negatives']):,}"
)

print(
    f"True negatives: "
    f"{int(best['true_negatives']):,}"
)


# --------------------------------------------------
# 6. Display top 10 thresholds
# --------------------------------------------------

print("\nTOP 10 THRESHOLDS")

print(
    ranked[
        ["threshold", "precision", "recall", "f0.5"]
    ].head(10).to_string(
        index=False,
        formatters={
            "threshold": "{:.3f}".format,
            "precision": "{:.4%}".format,
            "recall": "{:.4%}".format,
            "f0.5": "{:.4%}".format,
        },
    )
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

os.makedirs("output", exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)

print("\nFINE THRESHOLD OPTIMIZATION COMPLETED")
print(f"Thresholds tested: {len(THRESHOLDS)}")
print(f"Saved to: {OUTPUT_PATH}")

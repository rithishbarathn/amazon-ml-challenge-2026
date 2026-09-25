
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score

INPUT_PATH = "output/validation_predictions_global_top100_2000q.tsv"
OUTPUT_PATH = "output/threshold_comparison_global_top100.tsv"

df = pd.read_csv(INPUT_PATH, sep="\t")

y_true = df["label"]
probabilities = df["match_probability"]

thresholds = [
    0.50, 0.60, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.92, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99, 0.995
]

results = []

print("\nGLOBAL TOP-100 THRESHOLD COMPARISON\n")

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f05 = fbeta_score(
        y_true, predictions, beta=0.5, zero_division=0
    )

    tp = int(((y_true == 1) & (predictions == 1)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    })

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4%}".format,
            "recall": "{:.4%}".format,
            "f05": "{:.4%}".format,
        },
    )
)

results_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)

print(f"\nSaved to: {OUTPUT_PATH}")

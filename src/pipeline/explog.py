"""One machine-readable row per experiment in submissions/experiments.csv."""

import csv
import time

from pipeline.common import ROOT

LOG_PATH = ROOT / "submissions" / "experiments.csv"
COLUMNS = [
    "time", "experiment", "phase", "encoder_config", "feature_groups", "n_features",
    "depth", "learning_rate", "other_params", "threshold", "inner_f05", "holdout_f05",
    "us_f05", "india_f05", "india_transfer_f05", "france_proxy", "block_recall",
    "addr_channel_recall", "candidates_per_s1", "notes",
]


def log_row(**values):
    unknown = set(values) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown experiment columns: {sorted(unknown)}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        row = {k: "" for k in COLUMNS}
        row.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in values.items()})
        row["time"] = time.strftime("%Y-%m-%d %H:%M")
        w.writerow(row)

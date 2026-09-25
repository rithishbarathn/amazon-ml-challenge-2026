
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")

files = {
    "Train Source 1": DATA_DIR / "train/train_source1.tsv",
    "Train Source 2": DATA_DIR / "train/train_source2.tsv",
    "Train Source 3": DATA_DIR / "train/train_source3.tsv",
    "Ground Truth": DATA_DIR / "train/train_ground_truth.tsv",
    "Test Source 1": DATA_DIR / "test/test_source1.tsv",
    "Test Source 2": DATA_DIR / "test/test_source2.tsv",
    "Test Source 3": DATA_DIR / "test/test_source3.tsv",
}

print("\nAMAZON ML CHALLENGE 2026")
print("BUSINESS ENTITY RESOLUTION - DATA ANALYSIS")

for name, path in files.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Column names:", df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print("\nEmpty values:")
    print(df.eq("").sum())

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts())

    if "matched_entity_ids" in df.columns:

        match_count = df["matched_entity_ids"].apply(
            lambda x: len(x.split(",")) if x.strip() else 0
        )

        print("\nGround Truth Analysis:")
        print("Entities with no matches:", (match_count == 0).sum())
        print("Entities with one match:", (match_count == 1).sum())
        print("Entities with multiple matches:", (match_count > 1).sum())
        print("Total positive matching pairs:", match_count.sum())

print("\nDATA ANALYSIS COMPLETED")

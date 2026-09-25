
import os
import time

import pandas as pd
from rapidfuzz import fuzz

from preprocessing import normalize_name, normalize_address


TRAIN_DIR = "dataset/train"

CANDIDATE_PATH = "output/scalable_blocking_2000q_500k.tsv"
OUTPUT_PATH = "output/features_2000q_500k.tsv"

CHUNK_SIZE = 50000


def token_jaccard(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def number_overlap(a, b):
    a_numbers = {token for token in a.split() if token.isdigit()}
    b_numbers = {token for token in b.split() if token.isdigit()}

    if not a_numbers or not b_numbers:
        return 0

    return int(bool(a_numbers & b_numbers))


def load_selected_records(path, required_ids):
    parts = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        selected = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise ValueError(f"No matching records found in {path}")

    return pd.concat(parts, ignore_index=True)


def prepare_records(df):
    df = df.copy()

    df["name_clean"] = df["business_name"].map(normalize_name)
    df["address_clean"] = df["business_address"].map(normalize_address)

    return df.set_index("entity_id")


def make_features(row):
    name1 = row["name_clean_1"]
    name2 = row["name_clean_2"]

    address1 = row["address_clean_1"]
    address2 = row["address_clean_2"]

    return {
        "name_ratio": fuzz.ratio(name1, name2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(name1, name2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(name1, name2) / 100.0,
        "name_jaccard": token_jaccard(name1, name2),

        "address_ratio": fuzz.ratio(address1, address2) / 100.0,
        "address_token_sort": fuzz.token_sort_ratio(address1, address2) / 100.0,
        "address_token_set": fuzz.token_set_ratio(address1, address2) / 100.0,
        "address_jaccard": token_jaccard(address1, address2),

        "exact_name": int(bool(name1) and name1 == name2),
        "exact_address": int(bool(address1) and address1 == address2),
        "same_country": int(row["country_1"] == row["country_2"]),

        "address_number_overlap": number_overlap(address1, address2),

        "name_length_difference": abs(len(name1) - len(name2)),
        "address_length_difference": abs(len(address1) - len(address2)),

        "name_missing_1": int(not name1),
        "name_missing_2": int(not name2),
        "address_missing_1": int(not address1),
        "address_missing_2": int(not address2),
    }


start_time = time.time()

print("Loading 500K candidate pairs...", flush=True)

pairs = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
).drop_duplicates()

query_ids = set(pairs["source1_entity_id"])
target_ids = set(pairs["candidate_entity_id"])

print(f"Candidate pairs: {len(pairs):,}", flush=True)
print(f"Required Source 1 records: {len(query_ids):,}", flush=True)
print(f"Required Source 2 records: {len(target_ids):,}", flush=True)

print("Loading required Source 1 records...", flush=True)

source1 = load_selected_records(
    f"{TRAIN_DIR}/train_source1.tsv",
    query_ids,
)

print("Loading required Source 2 records...", flush=True)

source2 = load_selected_records(
    f"{TRAIN_DIR}/train_source2.tsv",
    target_ids,
)

print("Normalizing selected records...", flush=True)

source1 = prepare_records(source1)
source2 = prepare_records(source2)

print("Joining candidate pairs with business records...", flush=True)

source1_columns = source1[
    ["name_clean", "address_clean", "country"]
].add_suffix("_1")

source2_columns = source2[
    ["name_clean", "address_clean", "country"]
].add_suffix("_2")

joined = pairs.join(
    source1_columns,
    on="source1_entity_id",
)

joined = joined.join(
    source2_columns,
    on="candidate_entity_id",
)

if joined[
    ["name_clean_1", "name_clean_2"]
].isna().any().any():
    raise ValueError("Some candidate entity IDs were not found in the source files.")

print("Calculating similarity features...", flush=True)

feature_rows = []

for i, row in enumerate(joined.to_dict("records"), start=1):
    feature_rows.append(make_features(row))

    if i % 5000 == 0:
        print(f"Processed {i:,}/{len(joined):,} pairs", flush=True)

features = pd.DataFrame(feature_rows)

result = pd.concat(
    [
        pairs.reset_index(drop=True),
        features.reset_index(drop=True),
    ],
    axis=1,
)

os.makedirs("output", exist_ok=True)

result.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)

elapsed = time.time() - start_time

print("\nFEATURE ENGINEERING COMPLETED")
print(f"Candidate pairs: {len(result):,}")
print(f"Feature columns: {len(features.columns)}")
print(f"Execution time: {elapsed / 60:.2f} minutes")
print(f"Saved to: {OUTPUT_PATH}")

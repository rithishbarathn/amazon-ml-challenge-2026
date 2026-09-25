
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import normalize_name


TRAIN_DIR = "dataset/train"

S1_SAMPLE = 1000
S2_SAMPLE = 20000
TOP_K = 5
BATCH_SIZE = 50


print("Loading sample data...", flush=True)

s1 = pd.read_csv(
    f"{TRAIN_DIR}/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S1_SAMPLE
)

s2 = pd.read_csv(
    f"{TRAIN_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S2_SAMPLE
)


print("Normalizing names...", flush=True)

s1["name_clean"] = s1["business_name"].map(normalize_name)
s2["name_clean"] = s2["business_name"].map(normalize_name)


print("Building TF-IDF vectors...", flush=True)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1,
    dtype=np.float32
)

s2_vectors = vectorizer.fit_transform(s2["name_clean"])
s1_vectors = vectorizer.transform(s1["name_clean"])


print("Finding similar businesses...", flush=True)

for start in range(0, len(s1), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(s1))

    # TF-IDF vectors are L2-normalized by default.
    # Their dot product is cosine similarity.
    similarities = (
        s1_vectors[start:end] @ s2_vectors.T
    ).tocsr()

    # Display results only for the first five Source 1 records.
    if start == 0:

        for local_i in range(min(5, end - start)):

            row = similarities.getrow(local_i)

            if row.nnz == 0:
                print(
                    "No nonzero candidates for:",
                    s1.iloc[start + local_i]["business_name"]
                )
                continue

            top_positions = np.argsort(row.data)[-TOP_K:][::-1]

            print(
                "\nSource 1:",
                s1.iloc[start + local_i]["business_name"]
            )

            for pos in top_positions:

                candidate_idx = row.indices[pos]
                score = row.data[pos]

                match = s2.iloc[candidate_idx]

                print(
                    f"  Candidate: {match['business_name']} "
                    f"| Similarity: {score:.4f}"
                )

            print("-" * 60)

    print(
        f"Processed {end:,}/{len(s1):,} Source 1 records",
        flush=True
    )


print("\nTF-IDF EXPERIMENT COMPLETED")

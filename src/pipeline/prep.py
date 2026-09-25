"""Stage 1: read every raw TSV once, normalise it and cache it as parquet.

Output columns per record:
    entity_id, country, name_n, addr_n   (normalised name / address)

Usage:  python -m pipeline.prep            (from the src/ directory)
"""

import time
from multiprocessing import Pool

import pandas as pd

from pipeline.common import (
    SOURCES, SPLITS, normalize_address, normalize_name, prep_path, raw_path,
)

CHUNK = 200_000
WORKERS = 12


def _normalize_chunk(chunk):
    names = [normalize_name(n) for n in chunk["business_name"]]
    addrs = [
        normalize_address(a, c)
        for a, c in zip(chunk["business_address"], chunk["country"])
    ]
    return pd.DataFrame({
        "entity_id": chunk["entity_id"].to_numpy(),
        "country": chunk["country"].to_numpy(),
        "name_n": names,
        "addr_n": addrs,
    })


def prepare(split, source, pool):
    out = prep_path(split, source)
    if out.exists():
        print(f"skip {out.name} (exists)", flush=True)
        return
    start = time.time()
    reader = pd.read_csv(
        raw_path(split, source), sep="\t", dtype=str,
        keep_default_na=False, chunksize=CHUNK, quoting=3,
    )
    parts = pool.map(_normalize_chunk, reader)
    df = pd.concat(parts, ignore_index=True)
    if df["entity_id"].duplicated().any():
        raise ValueError(f"duplicate entity ids in {split} {source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"{split}_{source}: {len(df):,} rows in {time.time() - start:.0f}s "
        f"{df['country'].value_counts().to_dict()}",
        flush=True,
    )


def main():
    with Pool(WORKERS) as pool:
        for split in SPLITS:
            for source in SOURCES:
                prepare(split, source, pool)


if __name__ == "__main__":
    main()

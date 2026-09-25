"""Stage 3: candidate generation (blocking) with GPU nearest-neighbour search.

Records are partitioned by their country label (treated as an open set -
whatever labels appear in Source 1 are used, nothing is hard-coded). Within a
country, two k-NN searches over the learned encoder vectors are run:

  A) target -> Source 1 : every Source 2/3 record proposes its K_A closest
     Source 1 entities as possible owners. Because Source 1 is deduplicated,
     each Source 2/3 record belongs to at most one Source 1 entity, so this
     direction alone captures most matches with very few pairs.
  B) Source 1 -> target : every Source 1 entity keeps its K_B closest targets
     (catches targets whose best owner in A was a near-duplicate).

The union is saved with both ranks so the final (smaller) cut-off can be
chosen on the training data without re-running the search.

Usage (from src/):
    python -m pipeline.block train|test
    python -m pipeline.block eval          # recall vs size on train holdout
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

from pipeline.common import DATA_DIR, WORK_DIR, bucket, emb_path, prep_path
from pipeline.model import DEVICE, combine, load_encoder

K_A = 5          # owners proposed per target (search depth)
K_B = 20         # targets kept per Source 1 (search depth)
SCORE_BUDGET = 400_000_000
TARGET_CHUNK = 400_000
NO_RANK = 99


def cand_path(split):
    return WORK_DIR / "cand" / f"{split}_candidates.parquet"


def load_side(split, sources):
    """Concatenate prep ids/countries and embeddings for the given sources."""
    meta, embs = [], []
    for source in sources:
        meta.append(pd.read_parquet(prep_path(split, source), columns=["entity_id", "country"]))
        embs.append(np.load(emb_path(split, source), mmap_mode="r"))
    meta = pd.concat(meta, ignore_index=True)
    return meta, embs


def gather(embs, rows):
    """Fetch rows (global index over the concatenated sources) from memmaps."""
    out = np.empty((len(rows), embs[0].shape[1]), dtype=np.float16)
    start = 0
    for e in embs:
        mask = (rows >= start) & (rows < start + len(e))
        if mask.any():
            out[mask] = e[rows[mask] - start]
        start += len(e)
    return out


@torch.no_grad()
def to_combined(block, addr_w):
    x = torch.from_numpy(block).to(DEVICE).float()
    half = x.shape[1] // 2
    return combine(x[:, :half], x[:, half:], addr_w).half()


@torch.no_grad()
def search_country(zs, target_chunks):
    """k-NN in both directions, streaming target chunks through the GPU.

    zs            : combined Source 1 vectors of one country (on GPU)
    target_chunks : iterator of (offset, combined target vectors on GPU)
    Returns (A pairs, B pairs) as local-index arrays + scores + ranks.
    """
    ns = zs.shape[0]
    ka = min(K_A, ns)
    # Keep each score matrix at ~SCORE_BUDGET elements (fp16) on the GPU.
    batch_a = max(64, SCORE_BUDGET // ns)
    batch_b = max(64, SCORE_BUDGET // TARGET_CHUNK)
    a_t, a_s, a_v = [], [], []
    b_val = torch.full((ns, K_B), -2.0, device=DEVICE, dtype=torch.float16)
    b_idx = torch.zeros((ns, K_B), device=DEVICE, dtype=torch.int64)

    for offset, zt in target_chunks:
        # A: every target in this chunk -> top K_A Source 1 entities.
        for i in range(0, zt.shape[0], batch_a):
            v, idx = (zt[i:i + batch_a] @ zs.T).topk(ka, dim=1)
            a_s.append(idx.cpu().numpy())
            a_v.append(v.float().cpu().numpy())
            a_t.append(np.arange(offset + i, offset + i + len(idx)))
        # B: update every Source 1's running top K_B with this chunk.
        kb = min(K_B, zt.shape[0])
        for i in range(0, ns, batch_b):
            v, idx = (zs[i:i + batch_b] @ zt.T).topk(kb, dim=1)
            cat_v = torch.cat([b_val[i:i + batch_b], v], dim=1)
            cat_i = torch.cat([b_idx[i:i + batch_b], idx + offset], dim=1)
            best_v, pos = cat_v.topk(K_B, dim=1)
            b_val[i:i + batch_b] = best_v
            b_idx[i:i + batch_b] = cat_i.gather(1, pos)
        del zt

    a_s = np.concatenate(a_s)
    a_v = np.concatenate(a_v)
    a_t = np.repeat(np.concatenate(a_t), ka)
    a_rank = np.tile(np.arange(ka), len(a_s))

    b_v = b_val.float().cpu().numpy()
    b_t = b_idx.cpu().numpy()
    valid = b_v > -2          # fewer targets than K_B in this country
    b_s = np.repeat(np.arange(ns), K_B).reshape(ns, K_B)
    b_rank = np.tile(np.arange(K_B), (ns, 1))
    return (
        (a_s.ravel(), a_t, a_v.ravel(), a_rank),
        (b_s[valid], b_t[valid], b_v[valid], b_rank[valid]),
    )


def run(split):
    start = time.time()
    addr_w = load_encoder().log_addr_w.exp().item()
    s1_meta, s1_embs = load_side(split, ["source1"])
    tg_meta, tg_embs = load_side(split, ["source2", "source3"])
    print(f"{split}: {len(s1_meta):,} S1, {len(tg_meta):,} targets, addr_w={addr_w:.2f}", flush=True)

    frames = []
    for country in sorted(s1_meta["country"].unique()):
        s_rows = np.flatnonzero(s1_meta["country"].to_numpy() == country)
        t_rows = np.flatnonzero(tg_meta["country"].to_numpy() == country)
        if len(t_rows) == 0:
            continue
        t0 = time.time()
        zs = to_combined(gather(s1_embs, s_rows), addr_w)
        chunks = (
            (i, to_combined(gather(tg_embs, t_rows[i:i + TARGET_CHUNK]), addr_w))
            for i in range(0, len(t_rows), TARGET_CHUNK)
        )
        (as_, at, av, ar), (bs, bt, bv, br) = search_country(zs, chunks)
        del zs
        torch.cuda.empty_cache()

        a = pd.DataFrame({"s1": s_rows[as_], "tgt": t_rows[at], "cos": av, "rank_a": ar})
        b = pd.DataFrame({"s1": s_rows[bs], "tgt": t_rows[bt], "cos": bv, "rank_b": br})
        # Per-target score of its best / second-best owner (margin features).
        owner = a.loc[a.rank_a == 0, ["tgt", "cos"]].rename(columns={"cos": "t_top1"})
        second = a.loc[a.rank_a == 1, ["tgt", "cos"]].rename(columns={"cos": "t_top2"})
        owner = owner.merge(second, on="tgt", how="left")
        # Per-Source-1 score of its best target.
        s_top = b[b.rank_b == 0][["s1", "cos"]].rename(columns={"cos": "s_top1"})

        m = a.merge(b, on=["s1", "tgt"], how="outer", suffixes=("", "_b"))
        m["cos"] = m["cos"].fillna(m.pop("cos_b"))
        m["rank_a"] = m["rank_a"].fillna(NO_RANK).astype(np.int8)
        m["rank_b"] = m["rank_b"].fillna(NO_RANK).astype(np.int8)
        m = m.merge(owner, on="tgt", how="left").merge(s_top, on="s1", how="left")
        frames.append(m)
        print(
            f"  {country}: {len(s_rows):,} S1 x {len(t_rows):,} targets -> "
            f"{len(m):,} pairs ({len(m) / len(s_rows):.1f}/S1) in {time.time() - t0:.0f}s",
            flush=True,
        )

    cand = pd.concat(frames, ignore_index=True)
    for col in ["cos", "t_top1", "t_top2", "s_top1"]:
        cand[col] = cand[col].astype(np.float32)
    cand["s1"] = cand["s1"].astype(np.int32)
    cand["tgt"] = cand["tgt"].astype(np.int32)
    out = cand_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out, index=False)
    print(f"saved {len(cand):,} pairs to {out} in {time.time() - start:.0f}s", flush=True)


# --------------------------------------------------------------------------
# Recall / size trade-off on the training holdout bucket
# --------------------------------------------------------------------------

def truth_pairs(s1_ids, tg_ids, holdout_only=True):
    truth = pd.read_csv(
        DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
        keep_default_na=False,
    )
    if holdout_only:
        truth = truth[[bucket(i) == 9 for i in truth["source1_entity_id"]]]
    s1_index = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    tg_index = pd.Series(np.arange(len(tg_ids)), index=tg_ids)
    exploded = truth.assign(
        tgt=truth["matched_entity_ids"].str.split(",")
    ).explode("tgt")
    exploded = exploded[exploded["tgt"] != ""]
    return (
        s1_index.loc[truth["source1_entity_id"]].to_numpy(),
        pd.DataFrame({
            "s1": s1_index.loc[exploded["source1_entity_id"]].to_numpy(),
            "tgt": tg_index.loc[exploded["tgt"]].to_numpy(),
        }),
    )


def evaluate():
    s1_meta, _ = load_side("train", ["source1"])
    tg_meta, _ = load_side("train", ["source2", "source3"])
    queries, pos = truth_pairs(s1_meta["entity_id"].to_numpy(), tg_meta["entity_id"].to_numpy())
    cand = pd.read_parquet(cand_path("train"))
    cand = cand[cand["s1"].isin(set(queries))]
    pos["hit"] = 1
    cand = cand.merge(pos, on=["s1", "tgt"], how="left")
    cand["hit"] = cand["hit"].fillna(0)
    print(f"holdout S1: {len(queries):,}, true pairs: {len(pos):,}")
    print(f"{'k_a':>4} {'k_b':>4} {'recall':>8} {'cand/S1':>8}")
    for ka in [1, 2, 3, 5]:
        for kb in [0, 1, 2, 3, 5, 10, 20]:
            keep = (cand["rank_a"] < ka) | (cand["rank_b"] < kb)
            print(f"{ka:>4} {kb:>4} {cand.loc[keep, 'hit'].sum() / len(pos):8.4f} "
                  f"{keep.sum() / len(queries):8.2f}")


if __name__ == "__main__":
    arg = sys.argv[1]
    evaluate() if arg == "eval" else run(arg)

"""Stage 4b: cross-encoder match probability for candidate pairs.

A multilingual transformer (intfloat/multilingual-e5-small, MIT licence,
118M parameters) reads BOTH records of a pair at once,

    "<name 1> · <address 1>" [SEP] "<name 2> · <address 2>"

on the raw text (Indic scripts, French accents and abbreviations kept), and is
fine-tuned to output the probability that they are the same business. Its
score is added as one feature of the XGBoost matcher, which keeps the
number / decoy features.

Leakage control: the cross-encoder is trained on candidate pairs of Source 1
entities in hash buckets 0-5 only; the matcher trains on buckets 6-8 and is
evaluated on bucket 9, so every score the matcher sees is out-of-sample.

Usage (from src/):
    python -m pipeline.crossenc train [--country US] [--max-pairs N] [--tag T]
    python -m pipeline.crossenc score train|test [--tag T] [--country US] [--max-queries N]
"""

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pipeline.block import load_candidates
from pipeline.common import WORK_DIR, bucket, raw_path

BASE_MODEL = WORK_DIR / "e5-small"
CE_DIR = WORK_DIR / "crossenc"
MAX_LEN = 128
TRAIN_BATCH = 64
LR = 3e-5
WARMUP = 0.05
SCORE_BATCH = 512


def model_dir(tag):
    return CE_DIR / f"model_{tag}"


def scores_path(split, tag):
    return CE_DIR / f"{split}_scores_{tag}.parquet"


class RawText:
    """Raw "name · address" strings in the same row order as pipeline.prep."""

    def __init__(self, split, sources):
        frames = [
            pd.read_csv(raw_path(split, s), sep="\t", dtype=str, keep_default_na=False,
                        usecols=["business_name", "business_address"], quoting=3)
            for s in sources
        ]
        df = pd.concat(frames, ignore_index=True)
        name = df["business_name"].str.strip()
        addr = df["business_address"].str.strip()
        self.text = np.where(addr == "", name, name + " · " + addr).astype(object)


def load(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).cuda()
    return tok, model


def labelled_pairs(split_buckets, country=None, max_queries=None, seed=0):
    """Filtered candidate pairs of training S1 in the given buckets, with labels."""
    from pipeline.features import filter_candidates
    from pipeline.matcher import load_truth, pair_keys

    s1 = pd.read_parquet(WORK_DIR / "prep" / "train_source1.parquet", columns=["entity_id", "country"])
    tg_ids = pd.concat([pd.read_parquet(WORK_DIR / "prep" / f"train_{s}.parquet", columns=["entity_id"])
                        for s in ("source2", "source3")], ignore_index=True)["entity_id"].to_numpy()
    positives, _ = load_truth(s1["entity_id"].to_numpy(), tg_ids)
    b = np.array([bucket(i) for i in s1["entity_id"]])
    keep = np.isin(b, split_buckets)
    if country is not None:
        keep &= s1["country"].to_numpy() == country
    queries = np.flatnonzero(keep)
    if max_queries is not None and len(queries) > max_queries:
        queries = np.random.default_rng(seed).choice(queries, max_queries, replace=False)
    cand = filter_candidates(load_candidates("train"))
    cand = cand[cand["s1"].isin(queries)].reset_index(drop=True)
    y = np.isin(pair_keys(cand), positives).astype(np.float32)
    return cand[["s1", "tgt"]], y


def _encode(tok, a, b):
    return tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True,
               return_tensors="pt").to("cuda")


def train(country=None, max_pairs=1_000_000, tag="all"):
    start = time.time()
    pairs, y = labelled_pairs(split_buckets=list(range(6)), country=country)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))[:max_pairs]
    pairs, y = pairs.iloc[idx].reset_index(drop=True), y[idx]
    s1_txt = RawText("train", ["source1"]).text
    tg_txt = RawText("train", ["source2", "source3"]).text
    a, b = s1_txt[pairs["s1"].to_numpy()], tg_txt[pairs["tgt"].to_numpy()]
    print(f"cross-encoder training pairs {len(y):,} (pos {y.mean():.3f}) ({time.time() - start:.0f}s)", flush=True)

    tok, model = load(BASE_MODEL)
    model.train()
    steps = math.ceil(len(y) / TRAIN_BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, WARMUP * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    y_t = torch.from_numpy(y)
    running = 0.0
    for step in range(steps):
        sl = slice(step * TRAIN_BATCH, (step + 1) * TRAIN_BATCH)
        enc = _encode(tok, a[sl], b[sl])
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y_t[sl].cuda())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  step {step:,}/{steps:,} loss {running:.4f} ({time.time() - start:.0f}s)", flush=True)
    out = model_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} ({time.time() - start:.0f}s)")


@torch.no_grad()
def score_pairs(tok, model, a, b):
    """Match probability for aligned text arrays, length-sorted for speed."""
    model.eval()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), SCORE_BATCH):
        idx = order[i:i + SCORE_BATCH]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**_encode(tok, a[idx], b[idx])).logits.squeeze(-1)
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        if (i // SCORE_BATCH) % 2000 == 0:
            print(f"  scored {min(i + SCORE_BATCH, len(a)):,}/{len(a):,}", flush=True)
    return out


def score(split, tag="all", country=None, max_queries=None):
    """Score the pairs the matcher needs: matcher-train + holdout (train) or all (test)."""
    from pipeline.features import filter_candidates

    start = time.time()
    cand = filter_candidates(load_candidates(split))
    if split == "train":
        s1 = pd.read_parquet(WORK_DIR / "prep" / "train_source1.parquet", columns=["entity_id", "country"])
        b = np.array([bucket(i) for i in s1["entity_id"]])
        mq = np.flatnonzero((b >= 6) & (b <= 8) & ((s1["country"] == country).to_numpy() if country else True))
        if max_queries is not None and len(mq) > max_queries:
            mq = np.random.default_rng(0).choice(mq, max_queries, replace=False)
        hold_targets = cand.loc[np.isin(b[cand["s1"].to_numpy()], [9]), "tgt"].unique()
        keep = cand["s1"].isin(mq) | cand["tgt"].isin(hold_targets)
        cand = cand[keep].reset_index(drop=True)
    s1_txt = RawText(split, ["source1"]).text
    tg_txt = RawText(split, ["source2", "source3"]).text
    print(f"scoring {len(cand):,} {split} pairs ({time.time() - start:.0f}s)", flush=True)
    tok, model = load(model_dir(tag))
    ce = score_pairs(tok, model, s1_txt[cand["s1"].to_numpy()], tg_txt[cand["tgt"].to_numpy()])
    pd.DataFrame({"s1": cand["s1"].to_numpy(), "tgt": cand["tgt"].to_numpy(), "ce": ce}).to_parquet(
        scores_path(split, tag), index=False)
    print(f"saved {scores_path(split, tag)} ({time.time() - start:.0f}s)")


def attach_scores(cand, split, tag="all"):
    """Left-join cross-encoder scores onto a filtered candidate table (NaN if unscored)."""
    path = scores_path(split, tag)
    if not path.exists():
        return cand
    sc = pd.read_parquet(path)
    key = lambda d: (d["s1"].to_numpy().astype(np.int64) << 32) | d["tgt"].to_numpy()
    pos = pd.Series(np.arange(len(sc)), index=key(sc)).reindex(key(cand)).to_numpy()
    ce = np.full(len(cand), np.nan, dtype=np.float32)
    hit = ~np.isnan(pos)
    ce[hit] = sc["ce"].to_numpy()[pos[hit].astype(np.int64)]
    return cand.assign(ce=ce)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "score"])
    p.add_argument("split", nargs="?")
    p.add_argument("--country")
    p.add_argument("--max-pairs", type=int, default=1_000_000)
    p.add_argument("--max-queries", type=int)
    p.add_argument("--tag", default="all")
    args = p.parse_args()
    if args.cmd == "train":
        train(args.country, args.max_pairs, args.tag)
    else:
        score(args.split, args.tag, args.country, args.max_queries)

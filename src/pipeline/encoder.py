"""Stage 2: learned record encoder for blocking (GPU).

Each record is turned into hashed character n-grams + word tokens for its
name and its address (pipeline.textvec). Two EmbeddingBag layers
(pipeline.model) map these sparse bags to dense vectors. The model is trained
with a contrastive (InfoNCE) loss on Source 1 <-> matched Source 2/3 pairs
from the labelled training data, so that records of the same business land
close to each other - it learns abbreviations, transliterations and typos.

After training, every record of every split is embedded and stored as a
float16 matrix [name_vec | addr_vec] (both L2-normalised, addr zero if empty).

torch is imported lazily inside the functions: on Windows every
multiprocessing worker re-imports this module, and loading the CUDA
libraries in 12 workers exhausts the paging file.

Usage (from src/):
    python -m pipeline.encoder train
    python -m pipeline.encoder embed
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pipeline.common import DATA_DIR, SOURCES, SPLITS, bucket, emb_path, prep_path
from pipeline.textvec import vectorize, vectorize_parallel

# Source 1 entities (buckets 0-5) used to train; 0 = all (~1.25M).
ENCODER_QUERIES = int(os.environ.get("ER_ENC_QUERIES", 0))
BATCH = 4096
EPOCHS = int(os.environ.get("ER_ENC_EPOCHS", 4))
# Hard-negative batches: after a random warm-up epoch, batches are cut from
# pairs sorted by locality so in-batch negatives share a city / state.
HARD_NEGATIVES = os.environ.get("ER_ENC_HARD", "0") == "1"   # off: no gain in the Phase 1 sweep
LR = 0.05
WORKERS = 4
EMBED_CHUNK = 100_000

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _load_training_pairs():
    truth = pd.read_csv(
        DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
        keep_default_na=False,
    )
    truth = truth[truth["matched_entity_ids"] != ""]
    truth = truth[[bucket(i) < 6 for i in truth["source1_entity_id"]]]
    if ENCODER_QUERIES:
        truth = truth.sample(n=min(ENCODER_QUERIES, len(truth)), random_state=0)
    pairs = truth.assign(
        target=truth["matched_entity_ids"].str.split(",")
    ).explode("target")[["source1_entity_id", "target"]]
    return pairs.rename(columns={"source1_entity_id": "s1"})


def _load_records(split, source, ids):
    df = pd.read_parquet(prep_path(split, source))
    return df[df["entity_id"].isin(ids)]


def train():
    import torch
    from pipeline.model import DEVICE, MODEL_PATH, Encoder

    start = time.time()
    torch.manual_seed(0)
    pairs = _load_training_pairs()
    print(f"training pairs: {len(pairs):,} from {pairs.s1.nunique():,} S1", flush=True)

    s1 = _load_records("train", "source1", set(pairs["s1"]))
    tgt_ids = set(pairs["target"])
    tgt = pd.concat([
        _load_records("train", "source2", tgt_ids),
        _load_records("train", "source3", tgt_ids),
    ], ignore_index=True)

    s1_row = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    tgt_row = pd.Series(np.arange(len(tgt)), index=tgt["entity_id"])
    pair_s1 = s1_row.loc[pairs["s1"]].to_numpy()
    pair_tg = tgt_row.loc[pairs["target"]].to_numpy()
    # Group batches by country so in-batch negatives are harder.
    pair_country = s1["country"].to_numpy()[pair_s1]
    # Locality key of each pair's S1: the last two address tokens (city / state /
    # postcode), taken from the data for every country alike.
    locality = np.array([" ".join(a.split()[-2:]) for a in s1["addr_n"]], dtype=object)
    _, pair_locality = np.unique(locality[pair_s1].astype(str), return_inverse=True)

    with Pool(WORKERS) as pool:
        s1_name, s1_addr = vectorize_parallel(pool, list(s1.name_n), list(s1.addr_n))
        tg_name, tg_addr = vectorize_parallel(pool, list(tgt.name_n), list(tgt.addr_n))
    print(f"vectorised in {time.time() - start:.0f}s", flush=True)

    model = Encoder().to(DEVICE)
    sparse_params = [model.name_bag.weight, model.addr_bag.weight]
    dense_params = [model.log_addr_w, model.log_tau]
    opt_sparse = torch.optim.SparseAdam(sparse_params, lr=LR)
    opt_dense = torch.optim.Adam(dense_params, lr=0.01)

    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        batches = []
        if HARD_NEGATIVES and epoch > 0:
            # Contiguous batches of locality-sorted pairs (random order inside a locality).
            order = np.lexsort((rng.random(len(pair_s1)), pair_locality, pair_country))
            batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        else:
            for country in np.unique(pair_country):
                idx = np.flatnonzero(pair_country == country)
                rng.shuffle(idx)
                batches += [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]
        rng.shuffle(batches)
        total, seen = 0.0, 0
        for step, b in enumerate(batches):
            if len(b) < 64:
                continue
            q = pair_s1[b]
            t = pair_tg[b]
            qn, qa = model.fields(s1_name[q], s1_addr[q])
            tn, ta = model.fields(tg_name[t], tg_addr[t])
            zq = model.combine(qn, qa)
            zt = model.combine(tn, ta)
            logits = zq @ zt.T / model.log_tau.exp()
            # Several targets of the same S1 can share a batch: all positive.
            qt = torch.from_numpy(q).to(DEVICE)
            pos = (qt[:, None] == qt[None, :]).float()
            log_p_row = logits.log_softmax(dim=1)
            log_p_col = logits.log_softmax(dim=0)
            loss = -(
                (torch.logsumexp(log_p_row + (pos - 1) * 1e4, dim=1)).mean()
                + (torch.logsumexp(log_p_col + (pos - 1) * 1e4, dim=0)).mean()
            ) / 2
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()
            total += loss.item() * len(b)
            seen += len(b)
            if step % 200 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(batches)} "
                    f"loss {total / max(seen, 1):.4f} "
                    f"tau {model.log_tau.exp().item():.3f} "
                    f"addr_w {model.log_addr_w.exp().item():.2f}",
                    flush=True,
                )
        print(f"epoch {epoch} mean loss {total / seen:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}", flush=True)


# --------------------------------------------------------------------------
# Embedding every record
# --------------------------------------------------------------------------

def embed_all():
    import torch
    from pipeline.model import DEVICE, load_encoder

    torch.set_grad_enabled(False)
    model = load_encoder()
    dim = model.name_bag.embedding_dim
    with Pool(WORKERS) as pool:
        for split in SPLITS:
            for source in SOURCES:
                out = emb_path(split, source)
                if out.exists():
                    print(f"skip {out.name}", flush=True)
                    continue
                start = time.time()
                df = pd.read_parquet(prep_path(split, source), columns=["name_n", "addr_n"])
                names, addrs = list(df.name_n), list(df.addr_n)
                del df
                result = np.zeros((len(names), 2 * dim), dtype=np.float16)
                jobs = [
                    (names[i:i + EMBED_CHUNK], addrs[i:i + EMBED_CHUNK])
                    for i in range(0, len(names), EMBED_CHUNK)
                ]
                row = 0
                for name_csr, addr_csr in pool.imap(vectorize, jobs):
                    for j in range(0, name_csr.shape[0], 20_000):
                        n, a = model.fields(name_csr[j:j + 20_000], addr_csr[j:j + 20_000])
                        block = torch.cat([n, a], dim=1).half().cpu().numpy()
                        result[row:row + len(block)] = block
                        row += len(block)
                assert row == len(names)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, result)
                print(f"{split}_{source}: {row:,} in {time.time() - start:.0f}s", flush=True)


# --------------------------------------------------------------------------
# Retrieval evaluation on held-out entities (encoder selection)
# --------------------------------------------------------------------------

EVAL_S1 = 20_000          # fixed sample of bucket-9 S1 with matches
EVAL_RANDOM_TARGETS = 1_500_000   # per country, plus every true target


def evaluate(tag):
    """Owner recall@1 and target recall@5 on a fixed held-out sample.

    owner@1   : a true target's nearest S1 (among ALL S1 of its country) is
                its true owner - the main blocking channel (rank_a < 1).
    target@5  : share of a sample S1's true targets among its 5 nearest
                targets (true targets + a fixed random set of the country).
    """
    import torch
    from pipeline.explog import log_row
    from pipeline.model import DEVICE, load_encoder

    torch.set_grad_enabled(False)
    start = time.time()
    model = load_encoder()
    addr_w = model.log_addr_w.exp()
    truth = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                        keep_default_na=False)
    truth = truth[(truth["matched_entity_ids"] != "").to_numpy() & np.array([bucket(i) == 9 for i in truth["source1_entity_id"]])]
    truth = truth.sample(n=EVAL_S1, random_state=0)
    pairs = truth.assign(t=truth["matched_entity_ids"].str.split(",")).explode("t")
    s1 = pd.read_parquet(prep_path("train", "source1"))
    tg = pd.concat([pd.read_parquet(prep_path("train", s)) for s in ("source2", "source3")], ignore_index=True)
    s1_row = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    tg_row = pd.Series(np.arange(len(tg)), index=tg["entity_id"])
    ps = s1_row.loc[pairs["source1_entity_id"]].to_numpy()
    pt = tg_row.loc[pairs["t"]].to_numpy()
    rng = np.random.default_rng(0)

    def embed(df, pool):
        out = []
        for name_csr, addr_csr in pool.imap(vectorize, [
                (list(df.name_n[i:i + EMBED_CHUNK]), list(df.addr_n[i:i + EMBED_CHUNK]))
                for i in range(0, len(df), EMBED_CHUNK)]):
            for j in range(0, name_csr.shape[0], 20_000):
                n, a = model.fields(name_csr[j:j + 20_000], addr_csr[j:j + 20_000])
                out.append(model.combine(n, a).half())
        return torch.cat(out)

    results = {}
    with Pool(WORKERS) as pool:
        for country in sorted(s1["country"].iloc[np.unique(ps)].unique()):
            m = s1["country"].to_numpy()[ps] == country
            cps, cpt = ps[m], pt[m]
            s_all = np.flatnonzero(s1["country"].to_numpy() == country)
            zs = embed(s1.iloc[s_all], pool)                       # all S1 of the country
            s_pos = pd.Series(np.arange(len(s_all)), index=s_all)
            # owner@1: nearest S1 of each true target
            zt_true = embed(tg.iloc[cpt], pool)
            best = torch.cat([(zt_true[i:i + 512] @ zs.T).argmax(1) for i in range(0, len(cpt), 512)])
            owner1 = (best.cpu().numpy() == s_pos.loc[cps].to_numpy()).mean()
            # target@5: true targets + fixed random targets of the country
            t_country = np.flatnonzero(tg["country"].to_numpy() == country)
            t_rand = rng.choice(np.setdiff1d(t_country, cpt), EVAL_RANDOM_TARGETS, replace=False)
            t_pool = np.concatenate([np.unique(cpt), t_rand])
            zt = embed(tg.iloc[t_pool], pool)
            t_pos = pd.Series(np.arange(len(t_pool)), index=t_pool)
            q = np.unique(cps)
            zq = zs[torch.from_numpy(s_pos.loc[q].to_numpy().copy()).to(zs.device)]
            top5 = torch.cat([(zq[i:i + 512] @ zt.T).topk(5, dim=1).indices for i in range(0, len(q), 512)]).cpu().numpy()
            q_pos = pd.Series(np.arange(len(q)), index=q)
            hit = [t_pos[t] in set(top5[q_pos[s]]) for s, t in zip(cps, cpt)]
            results[country] = (owner1, float(np.mean(hit)), len(cpt))
            del zs, zt_true, zt, zq, best
            torch.cuda.empty_cache()
            print(f"  {country}: owner@1 {owner1:.4f}  target@5 {np.mean(hit):.4f}  ({len(cpt):,} pairs)", flush=True)
    n = sum(r[2] for r in results.values())
    owner = sum(r[0] * r[2] for r in results.values()) / n
    target = sum(r[1] * r[2] for r in results.values()) / n
    print(f"{tag}: owner@1 {owner:.4f} target@5 {target:.4f} ({time.time() - start:.0f}s)")
    log_row(experiment=f"encoder_{tag}", phase="1-encoder", encoder_config=tag,
            block_recall=owner,
            notes="owner@1 all / target@5 all / per country owner@1: " + "; ".join(
                f"{c} {r[0]:.4f}/{r[1]:.4f}" for c, r in results.items()) + f"; target@5 {target:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "current")
    else:
        {"train": train, "embed": embed_all}[cmd]()

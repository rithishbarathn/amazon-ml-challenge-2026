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

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pipeline.common import DATA_DIR, SOURCES, SPLITS, bucket, emb_path, prep_path
from pipeline.textvec import vectorize, vectorize_parallel

ENCODER_QUERIES = 200_000     # Source 1 entities (bucket 0-5) used to train
BATCH = 4096
EPOCHS = 4
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
    truth = truth.sample(
        n=min(ENCODER_QUERIES, len(truth)), random_state=0,
    )
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
    from pipeline.model import DEVICE, DIM, load_encoder

    torch.set_grad_enabled(False)
    model = load_encoder()
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
                result = np.zeros((len(names), 2 * DIM), dtype=np.float16)
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


if __name__ == "__main__":
    {"train": train, "embed": embed_all}[sys.argv[1]]()

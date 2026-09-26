"""Stage 2b: multilingual sentence embeddings of names and addresses.

The hashed n-gram encoder only sees transliterated text, so an Indic-script
name ("राम मार्केटिंग प्राइवेट लिमिटेड") or a French abbreviation ("R." for
"rue") reaches it as a rough phonetic spelling. This stage embeds the *raw*
business name and address with intfloat/multilingual-e5-small (MIT licence,
118M parameters, 100+ languages incl. Hindi, Tamil, Telugu, Bengali, French)
and stores them for the matcher's similarity features.

To keep disk use reasonable the 384-d outputs are centred and projected to
PCA_DIM dimensions (PCA fitted once on a sample of training records), then
re-normalised, so a dot product is still a cosine. Centring also removes the
large component shared by all texts, which spreads the cosines out.

The model is loaded from MODEL_DIR (download once with
`huggingface_hub.snapshot_download("intfloat/multilingual-e5-small",
local_dir="work/e5-small")`); nothing is fetched at run time.

Usage (from src/):
    python -m pipeline.mlemb train
    python -m pipeline.mlemb test
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pipeline.common import SOURCES, WORK_DIR, raw_path

MODEL_DIR = WORK_DIR / "e5-small"
OUT_DIR = WORK_DIR / "mlemb"
PCA_DIM = 128
FIELDS = {"name": ("business_name", 32), "addr": ("business_address", 64)}  # column, max tokens
BATCH = 1024
PCA_SAMPLE = 200_000


def mlemb_path(split, source, field):
    return OUT_DIR / f"{split}_{source}_{field}.npy"


def pca_path(field):
    return OUT_DIR / f"pca_{field}.pt"


class Embedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer  # only needed to embed

        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, texts, max_len):
        """Mean-pooled, L2-normalised 384-d embeddings (float32, on GPU).

        E5 expects a "query: " prefix on both sides of a symmetric
        comparison. Texts are processed in length order to minimise padding.
        """
        order = np.argsort([len(t) for t in texts])
        out = torch.empty((len(texts), self.model.config.hidden_size), device="cuda")
        for i in range(0, len(texts), BATCH):
            idx = order[i:i + BATCH]
            batch = self.tok(["query: " + texts[k] for k in idx], padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt").to("cuda")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out[torch.from_numpy(idx).cuda()] = F.normalize(pooled.float(), dim=1)
        return out


def read_texts(split, source, column):
    # Same reader options as pipeline.prep, so rows line up with the prep parquet.
    df = pd.read_csv(raw_path(split, source), sep="\t", dtype=str, keep_default_na=False,
                     usecols=[column], quoting=3)
    return df[column].str.strip().tolist()


def fit_pca(embed):
    """Mean + top PCA_DIM principal directions per field, from training records."""
    rng = np.random.default_rng(0)
    for field, (column, max_len) in FIELDS.items():
        if pca_path(field).exists():
            continue
        texts = []
        for source in SOURCES:
            t = read_texts("train", source, column)
            texts += [t[k] for k in rng.choice(len(t), PCA_SAMPLE // len(SOURCES), replace=False)]
        x = embed(texts, max_len)
        mean = x.mean(0)
        _, _, v = torch.linalg.svd(x - mean, full_matrices=False)
        torch.save({"mean": mean.cpu(), "components": v[:PCA_DIM].T.contiguous().cpu()}, pca_path(field))
        print(f"PCA {field}: fitted on {len(texts):,} texts", flush=True)


def run(split):
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    embed = Embedder()
    fit_pca(embed)
    for field, (column, max_len) in FIELDS.items():
        pca = torch.load(pca_path(field))
        mean, comps = pca["mean"].cuda(), pca["components"].cuda()
        for source in SOURCES:
            path = mlemb_path(split, source, field)
            if path.exists():
                continue
            t0 = time.time()
            texts = read_texts(split, source, column)
            out = np.empty((len(texts), PCA_DIM), dtype=np.float16)
            chunk = 500_000
            for i in range(0, len(texts), chunk):
                x = embed(texts[i:i + chunk], max_len)
                out[i:i + chunk] = F.normalize((x - mean) @ comps, dim=1).half().cpu().numpy()
            np.save(path, out)
            print(f"{split}_{source} {field}: {len(texts):,} in {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1])

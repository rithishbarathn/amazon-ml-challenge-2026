"""Hashed character n-gram / word bags for the encoder (no torch import, so
it is cheap to load in multiprocessing workers on Windows)."""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from pipeline.common import squash

N_FEATURES = 2 ** 19

# --------------------------------------------------------------------------
# Sparse featurisation (runs in worker processes)
# --------------------------------------------------------------------------

_name_char = HashingVectorizer(
    analyzer="char_wb", ngram_range=(2, 4), n_features=N_FEATURES,
    alternate_sign=False, norm="l2", dtype=np.float32,
)
_addr_char = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), n_features=N_FEATURES,
    alternate_sign=False, norm="l2", dtype=np.float32,
)
_word = HashingVectorizer(
    analyzer="word", token_pattern=r"\S+", n_features=N_FEATURES,
    alternate_sign=False, norm="l2", dtype=np.float32,
)


def vectorize(args):
    """(names, addresses) -> (name_csr, addr_csr) sparse bags."""
    names, addrs = args
    names_sq = [squash(n) for n in names]
    addrs_sq = [squash(a) for a in addrs]
    name_csr = (_name_char.transform(names_sq) + _word.transform(names)).tocsr()
    addr_csr = (_addr_char.transform(addrs_sq) + _word.transform(addrs)).tocsr()
    name_csr.sort_indices()
    addr_csr.sort_indices()
    return name_csr, addr_csr


def vectorize_parallel(pool, names, addrs, chunk=50_000):
    jobs = [
        (names[i:i + chunk], addrs[i:i + chunk])
        for i in range(0, len(names), chunk)
    ]
    parts = pool.map(vectorize, jobs)
    return (
        sp.vstack([p[0] for p in parts]).tocsr(),
        sp.vstack([p[1] for p in parts]).tocsr(),
    )



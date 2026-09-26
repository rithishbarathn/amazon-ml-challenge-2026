"""Stage 4: pairwise features for (Source 1, candidate) pairs.

Feature groups
  * blocking signals : combined cosine, ranks in both search directions,
                       margins to the best owner / best target
  * encoder fields   : name-vector and address-vector cosine
  * string similarity: RapidFuzz ratios on normalised, core (legal-form
                       stripped) and squashed (transliteration-robust) names,
                       and on addresses
  * numbers          : house numbers / ZIP / PIN agreement in addresses
  * context          : candidate counts, missing-field flags, target source

Nothing here depends on the country label, so the model transfers to
countries that are absent from training (France).
"""

import os
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from pipeline.common import core_name, emb_path, prep_path, squash
from pipeline.mlemb import mlemb_path

PAIR_CHUNK = 1_000_000   # pairs featurised at once (bounds peak RAM)

# Optional feature groups (experiments; ER_FEATURE_GROUPS="A,B,C"):
#   A number decoys, B name structure, C acronym / domain names.
# A+B are on by default (Phase 3: inner +0.10, unseen-country +1.09); C hurt transfer.
FEATURE_GROUPS = set(filter(None, os.environ.get("ER_FEATURE_GROUPS", "A,B").split(",")))
DOMAIN_TOKENS = {"com", "net", "org", "in", "co", "fr", "biz", "info", "io"}

# Final blocking cut-off applied to the saved search results
# (chosen on the training holdout with `python -m pipeline.block eval`).
KEEP_A = 1
KEEP_B = 7
KEEP_C = 1   # address-only channel (pipeline.block addr); ignored if not computed


def filter_candidates(cand):
    """Apply the final blocking cut-off and add candidate-count features.

    Counts are computed here, on the complete candidate table, so they stay
    correct when a subset of pairs is featurised later.
    """
    if "rank_c" not in cand:
        cand = cand.assign(rank_c=np.int8(99), addr_nn_cos=np.float32(np.nan))
    keep = (cand["rank_a"] < KEEP_A) | (cand["rank_b"] < KEEP_B) | (cand["rank_c"] < KEEP_C)
    cand = cand[keep].reset_index(drop=True)
    cand["n_cand_s1"] = cand.groupby("s1")["tgt"].transform("size").astype(np.float32)
    cand["n_cand_tgt"] = cand.groupby("tgt")["s1"].transform("size").astype(np.float32)
    return cand


class Records:
    """Normalised strings + embeddings of one side (S1 or targets)."""

    def __init__(self, split, sources):
        frames = [pd.read_parquet(prep_path(split, s)) for s in sources]
        self.sizes = [len(f) for f in frames]
        df = pd.concat(frames, ignore_index=True)
        del frames
        self.ids = df["entity_id"].to_numpy(dtype=object)
        self.name = df["name_n"].to_numpy(dtype=object)
        self.addr = df["addr_n"].to_numpy(dtype=object)
        self.country = df["country"].to_numpy(dtype=object)  # reporting only, never a feature
        del df
        self.embs = [np.load(emb_path(split, s), mmap_mode="r") for s in sources]
        self.source = np.concatenate([np.full(n, i, dtype=np.int8) for i, n in enumerate(self.sizes)])
        # Multilingual embeddings (pipeline.mlemb). Opt-in with ER_USE_MLEMB=1: they did not
        # help the holdout (0.9847 -> 0.9850) and hurt transfer to an unseen country
        # (US-only model on India 0.9392 -> 0.9346), so the submitted model does not use them.
        self.ml = {}
        for field in (("name", "addr") if os.environ.get("ER_USE_MLEMB") == "1" else ()):
            paths = [mlemb_path(split, s, field) for s in sources]
            if all(p.exists() for p in paths):
                arrays = [np.load(p, mmap_mode="r") for p in paths]
                if [len(a) for a in arrays] != self.sizes:
                    raise ValueError(f"mlemb {field} rows do not match prep for {split}")
                self.ml[field] = arrays
        self._idf = None
        self._addr_idf = None

    @property
    def idf(self):
        if self._idf is None:
            self._idf = token_idf(self.name, core_tokens)
        return self._idf

    @property
    def addr_idf(self):
        if self._addr_idf is None:
            self._addr_idf = token_idf(self.addr, addr_tokens)
        return self._addr_idf

    @property
    def idf_default(self):
        """IDF of a token seen once (unseen tokens are treated as rarest)."""
        return float(np.log(len(self.name)))

    def emb(self, rows, arrays=None):
        arrays = self.embs if arrays is None else arrays
        out = np.empty((len(rows), arrays[0].shape[1]), dtype=np.float32)
        start = 0
        for e in arrays:
            mask = (rows >= start) & (rows < start + len(e))
            if mask.any():
                sub = rows[mask] - start
                order = np.argsort(sub)
                block = np.asarray(e[sub[order]], dtype=np.float32)
                tmp = np.empty_like(block)
                tmp[order] = block
                out[mask] = tmp
            start += len(e)
        return out


_DIGITS = re.compile(r"\d+")


def _numbers(addr):
    """Digit runs with leading zeros stripped ("0069" == "69", "8c" -> "8")."""
    return {d.lstrip("0") or "0" for d in _DIGITS.findall(addr)}


def _postcode(addr):
    """Trailing 5/6-digit token (US ZIP / French code postal / Indian PIN).

    Only the last number in the address counts, and never the first token,
    so a zero-padded house number ("05131 copper meadow lane") is not
    mistaken for a postcode.
    """
    tokens = addr.split()
    for pos in range(len(tokens) - 1, 0, -1):
        if tokens[pos].isdigit():
            return tokens[pos] if len(tokens[pos]) in (5, 6) else ""
    return ""


def _num_diff(x, y):
    """Smallest |a - b| between numbers found on only one side (-1 if none).

    Near-duplicate decoys often share the street but differ by a few house
    numbers (4747 vs 4749), which a plain "any number in common" misses.
    """
    ox, oy = x - y, y - x
    if not ox or not oy:
        return -1.0
    return float(min(abs(int(a[:12]) - int(b[:12])) for a in ox for b in oy))


def core_tokens(name):
    return core_name(name).split()


def addr_tokens(addr):
    """Address words without numbers (street / locality words)."""
    return [t for t in addr.split() if not any(ch.isdigit() for ch in t)]


def token_idf(texts, tokenize):
    """log(N / df) of tokens, computed on the records being matched."""
    df = {}
    for text in texts:
        for t in set(tokenize(text)):
            df[t] = df.get(t, 0) + 1
    n = len(texts)
    return {t: float(np.log(n / c)) for t, c in df.items()}


def _idf_diff(k1, k2, idf, default):
    """IDF mass of core-name tokens shared / not shared by the two names."""
    common, diff, diff_max = [], [], []
    for a, b in zip(k1, k2):
        ta, tb = set(a.split()), set(b.split())
        c = sum(idf.get(t, default) for t in ta & tb)
        d = [idf.get(t, default) for t in ta ^ tb]
        common.append(c)
        diff.append(sum(d))
        diff_max.append(max(d) if d else 0.0)
    return (np.array(common, np.float32), np.array(diff, np.float32),
            np.array(diff_max, np.float32))


def _ordered_numbers(addr):
    return [d.lstrip("0") or "0" for d in _DIGITS.findall(addr)]


def _number_decoy_features(a1, a2):
    """Group A: truncated / split numbers (102 vs 10, 133 vs 33, "3 05" vs 305)."""
    n = len(a1)
    prefix = np.zeros(n, np.float32)
    split = np.zeros(n, np.float32)
    first_eq = np.full(n, -1, np.float32)
    cnt1 = np.zeros(n, np.float32)
    cnt2 = np.zeros(n, np.float32)
    for k, (x, y) in enumerate(zip(a1, a2)):
        lx, ly = _ordered_numbers(x), _ordered_numbers(y)
        cnt1[k], cnt2[k] = len(lx), len(ly)
        if lx and ly:
            first_eq[k] = float(lx[0] == ly[0])
        sx, sy = set(lx), set(ly)
        ox, oy = sx - sy, sy - sx
        prefix[k] = float(any(a != b and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a))
                              for a in ox for b in oy))
        rx, ry = _DIGITS.findall(x), _DIGITS.findall(y)   # raw digits: "3" + "05" -> "305"
        jx = {(rx[i] + rx[i + 1]).lstrip("0") for i in range(len(rx) - 1)}
        jy = {(ry[i] + ry[i + 1]).lstrip("0") for i in range(len(ry) - 1)}
        split[k] = float(bool(jx & oy) or bool(jy & ox))
    return {"num_prefix": prefix, "num_split_match": split, "num_first_eq": first_eq,
            "num_count1": cnt1, "num_count2": cnt2}


def _name_structure_features(k1, k2):
    """Group B: edit distance and token structure of the core names."""
    t1 = [x.split() for x in k1]
    t2 = [x.split() for x in k2]
    return {
        "core_lev": process.cpdist(list(k1), list(k2), scorer=Levenshtein.normalized_similarity,
                                   workers=-1).astype(np.float32),
        "first_tok_eq": np.fromiter((float(bool(a) and bool(b) and a[0] == b[0]) for a, b in zip(t1, t2)), np.float32, len(t1)),
        "last_tok_eq": np.fromiter((float(bool(a) and bool(b) and a[-1] == b[-1]) for a, b in zip(t1, t2)), np.float32, len(t1)),
        "tok_count_diff": np.fromiter((abs(len(a) - len(b)) for a, b in zip(t1, t2)), np.float32, len(t1)),
        "core_contained": np.fromiter((float(bool(a) and bool(b) and (a in b or b in a)) for a, b in zip(k1, k2)), np.float32, len(t1)),
    }


def _acronym_features(n1, n2, k1, k2):
    """Group C: acronyms ("ucprivate" ~ universal constructions private) and web domains."""
    def acr(tokens, k):
        return "".join(t[0] for t in tokens[:k]) if len(tokens) >= k else ""

    def match(core_a, name_b):
        ta, tb = core_a.split(), name_b.split()
        for k in (2, 3, 4):
            a = acr(ta, k)
            if len(a) >= 2 and any(t.startswith(a) and t != ta[0] for t in tb):
                return 1.0
        return 0.0

    n = len(n1)
    return {
        "acronym_match": np.fromiter((max(match(a, y), match(b, x)) for a, b, x, y in zip(k1, k2, n1, n2)), np.float32, n),
        "domain1": np.fromiter((float(bool(set(x.split()) & DOMAIN_TOKENS)) for x in n1), np.float32, n),
        "domain2": np.fromiter((float(bool(set(y.split()) & DOMAIN_TOKENS)) for y in n2), np.float32, n),
        "concat_ratio": process.cpdist([a.replace(" ", "") for a in k1], [b.replace(" ", "") for b in k2],
                                       scorer=fuzz.ratio, workers=-1).astype(np.float32) / 100.0,
    }


def _pairwise(scorer, a, b):
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1).astype(np.float32) / 100.0


def _chunk_features(s1, tg, c):
    i = c["s1"].to_numpy()
    j = c["tgt"].to_numpy()
    f = {}

    # Blocking signals
    f["cos"] = c["cos"].to_numpy()
    f["rank_a"] = c["rank_a"].to_numpy().astype(np.float32)
    f["rank_b"] = c["rank_b"].to_numpy().astype(np.float32)
    f["rank_c"] = c["rank_c"].to_numpy().astype(np.float32)
    f["addr_nn_cos"] = c["addr_nn_cos"].to_numpy().astype(np.float32)
    if "ce" in c.columns:  # cross-encoder probability (pipeline.crossenc, ER_USE_CE=1)
        f["ce"] = c["ce"].to_numpy().astype(np.float32)
    f["t_top1"] = c["t_top1"].to_numpy()
    f["t_top2"] = c["t_top2"].to_numpy()
    f["s_top1"] = c["s_top1"].to_numpy()
    f["margin_owner"] = f["cos"] - f["t_top1"]
    f["margin_s1"] = f["cos"] - f["s_top1"]
    f["owner_gap"] = f["t_top1"] - f["t_top2"]

    # Encoder field cosines
    e1 = s1.emb(i)
    e2 = tg.emb(j)
    half = e1.shape[1] // 2
    f["name_cos"] = (e1[:, :half] * e2[:, :half]).sum(1)
    f["addr_cos"] = (e1[:, half:] * e2[:, half:]).sum(1)
    del e1, e2

    # Multilingual embedding cosines on the raw text (Indic scripts, French)
    for field in ("name", "addr"):
        if field in s1.ml and field in tg.ml:
            f[f"ml_{field}_cos"] = (s1.emb(i, s1.ml[field]) * tg.emb(j, tg.ml[field])).sum(1)
    if "name" in s1.ml and "addr" in tg.ml:
        # Name written into the other record's address field (and vice versa)
        f["ml_cross_cos"] = np.maximum(
            (s1.emb(i, s1.ml["name"]) * tg.emb(j, tg.ml["addr"])).sum(1),
            (s1.emb(i, s1.ml["addr"]) * tg.emb(j, tg.ml["name"])).sum(1))

    # String similarity
    n1, n2 = s1.name[i], tg.name[j]
    k1 = np.array([core_name(x) for x in n1], dtype=object)
    k2 = np.array([core_name(x) for x in n2], dtype=object)
    q1 = [squash(x) for x in n1]
    q2 = [squash(x) for x in n2]
    a1, a2 = s1.addr[i], tg.addr[j]
    f["name_ratio"] = _pairwise(fuzz.ratio, n1, n2)
    f["name_token_set"] = _pairwise(fuzz.token_set_ratio, n1, n2)
    f["name_token_sort"] = _pairwise(fuzz.token_sort_ratio, n1, n2)
    f["name_partial"] = _pairwise(fuzz.partial_ratio, n1, n2)
    f["core_ratio"] = _pairwise(fuzz.ratio, k1, k2)
    f["core_token_set"] = _pairwise(fuzz.token_set_ratio, k1, k2)
    f["core_jw"] = process.cpdist(list(k1), list(k2), scorer=JaroWinkler.similarity, workers=-1).astype(np.float32)
    f["squash_ratio"] = _pairwise(fuzz.ratio, q1, q2)
    f["addr_ratio"] = _pairwise(fuzz.ratio, a1, a2)
    f["addr_token_set"] = _pairwise(fuzz.token_set_ratio, a1, a2)
    f["addr_partial"] = _pairwise(fuzz.partial_ratio, a1, a2)
    f["core_exact"] = (k1 == k2).astype(np.float32)
    f["idf_common"], f["idf_diff"], f["idf_diff_max"] = _idf_diff(k1, k2, tg.idf, tg.idf_default)
    f["idf_diff_ratio"] = f["idf_diff"] / (f["idf_common"] + f["idf_diff"] + 1e-6)
    w1 = np.array([" ".join(addr_tokens(x)) for x in a1], dtype=object)
    w2 = np.array([" ".join(addr_tokens(x)) for x in a2], dtype=object)
    f["addr_idf_common"], f["addr_idf_diff"], f["addr_idf_diff_max"] = _idf_diff(
        w1, w2, tg.addr_idf, tg.idf_default)
    f["addr_idf_diff_ratio"] = f["addr_idf_diff"] / (f["addr_idf_common"] + f["addr_idf_diff"] + 1e-6)

    # Numbers in the address
    num1 = [_numbers(x) for x in a1]
    num2 = [_numbers(x) for x in a2]
    inter = np.fromiter((len(x & y) for x, y in zip(num1, num2)), np.float32, len(i))
    union = np.fromiter((len(x | y) for x, y in zip(num1, num2)), np.float32, len(i))
    f["num_jaccard"] = np.where(union > 0, inter / np.maximum(union, 1), -1)
    f["num_conflict"] = ((inter == 0) & (union > 0) & np.fromiter(
        (bool(x) and bool(y) for x, y in zip(num1, num2)), bool, len(i))).astype(np.float32)
    f["num_only1"] = np.fromiter((len(x - y) for x, y in zip(num1, num2)), np.float32, len(i))
    f["num_only2"] = np.fromiter((len(y - x) for x, y in zip(num1, num2)), np.float32, len(i))
    f["num_min_diff"] = np.log1p(np.fromiter(
        (_num_diff(x, y) for x, y in zip(num1, num2)), np.float64, len(i)).clip(-1 + 1e-9)).astype(np.float32)
    p1 = np.array([_postcode(x) for x in a1], dtype=object)
    p2 = np.array([_postcode(x) for x in a2], dtype=object)
    f["postcode"] = np.where((p1 == "") | (p2 == ""), -1, (p1 == p2).astype(np.float32)).astype(np.float32)

    # Optional feature groups (see FEATURE_GROUPS)
    if "A" in FEATURE_GROUPS:
        f.update(_number_decoy_features(a1, a2))
    if "B" in FEATURE_GROUPS:
        f.update(_name_structure_features(k1, k2))
    if "C" in FEATURE_GROUPS:
        f.update(_acronym_features(n1, n2, k1, k2))

    # Context
    f["name_len1"] = np.fromiter((len(x) for x in n1), np.float32, len(i))
    f["name_len2"] = np.fromiter((len(x) for x in n2), np.float32, len(i))
    f["addr_len1"] = np.fromiter((len(x) for x in a1), np.float32, len(i))
    f["addr_len2"] = np.fromiter((len(x) for x in a2), np.float32, len(i))
    f["target_source"] = tg.source[j].astype(np.float32)
    return pd.DataFrame(f)


def iter_features(s1, tg, cand):
    """Yield (start, features) chunks for the rows of `cand` (from filter_candidates)."""
    for start in range(0, len(cand), PAIR_CHUNK):
        c = cand.iloc[start:start + PAIR_CHUNK]
        feats = _chunk_features(s1, tg, c)
        feats["n_cand_s1"] = c["n_cand_s1"].to_numpy()
        feats["n_cand_tgt"] = c["n_cand_tgt"].to_numpy()
        print(f"  features {min(start + PAIR_CHUNK, len(cand)):,}/{len(cand):,}", flush=True)
        yield start, feats


def build_features(s1, tg, cand):
    return pd.concat([f for _, f in iter_features(s1, tg, cand)], ignore_index=True)

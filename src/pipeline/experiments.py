"""Feature-group, stage-2 and hyperparameter experiments on cached matcher features.

Every experiment reports (and logs to submissions/experiments.csv):
  inner_f05          train S1 buckets 6-7, validate bucket 8 (one-owner rule
                     within bucket-8 pairs, best threshold)
  india_transfer_f05 train on US rows of buckets 6-8, score India holdout
                     (bucket 9) at the US-best threshold -> unseen-country check
  us_f05             the US holdout score of that same US-only model

Uses the train / holdout features cached by `matcher train` in work/features
(pair ids are recovered by re-applying the matcher's candidate filter).

Usage (from src/):
    python -m pipeline.experiments groups           # base, A, B, C, cumulative
    python -m pipeline.experiments stage2 A,B       # stage-2 relational model
    python -m pipeline.experiments tune A,B [n]     # random search, n configs
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.block import load_candidates
from pipeline.common import WORK_DIR, bucket, prep_path
from pipeline.explog import log_row
from pipeline.features import (Records, _acronym_features, _name_structure_features,
                               _number_decoy_features, filter_candidates)
from pipeline.common import core_name
from pipeline.matcher import (FEATURE_CACHE, HOLDOUT_PRED, HOLDOUT_QUERIES_PATH, THRESHOLDS,
                              assign, load_truth, macro_f05)

BASE_PARAMS = dict(n_estimators=6000, learning_rate=0.05, max_depth=8, min_child_weight=5,
                   subsample=0.8, colsample_bytree=0.8)
GROUP_FNS = {"A": "number decoys", "B": "name structure", "C": "acronym / domain"}


class Data:
    def __init__(self):
        t0 = time.time()
        s1 = pd.read_parquet(prep_path("train", "source1"), columns=["entity_id", "country"])
        tg_ids = pd.concat([pd.read_parquet(prep_path("train", s), columns=["entity_id"])
                            for s in ("source2", "source3")], ignore_index=True)["entity_id"].to_numpy()
        self.pos, self.n_true = load_truth(s1["entity_id"].to_numpy(), tg_ids)
        self.ctry = s1["country"].to_numpy()
        self.bkt = np.array([bucket(i) for i in s1["entity_id"]])

        self.tr = pd.read_parquet(FEATURE_CACHE / "train.parquet")
        self.tr = self.tr[[c for c in self.tr.columns if not c.startswith("ml_")]]
        cand = filter_candidates(load_candidates("train"))
        pairs = cand[cand["s1"].isin(np.unique(self.tr["s1"].to_numpy()))].reset_index(drop=True)
        if not (pairs["s1"].to_numpy() == self.tr["s1"].to_numpy()).all():
            raise ValueError("cached train features do not match the candidate order")
        self.tr["tgt"] = pairs["tgt"].to_numpy()
        self.ho = pd.read_parquet(FEATURE_CACHE / "holdout.parquet")
        self.ho = self.ho[[c for c in self.ho.columns if not c.startswith("ml_")]]
        hp = pd.read_parquet(HOLDOUT_PRED)
        if len(hp) != len(self.ho):
            raise ValueError("holdout features / predictions out of sync")
        self.ho["s1"], self.ho["tgt"] = hp["s1"].to_numpy(), hp["tgt"].to_numpy()
        self.hold_q = np.load(HOLDOUT_QUERIES_PATH)
        self.base = [c for c in self.tr.columns if c not in ("y", "s1", "tgt")]
        print(f"data ready: {len(self.tr):,} train / {len(self.ho):,} holdout pairs "
              f"({time.time() - t0:.0f}s)", flush=True)

    def add_groups(self, groups):
        """Compute (and cache) feature groups for the train / holdout pairs."""
        need = [g for g in groups if not any(c.startswith(f"g{g}_") for c in self.tr.columns)]
        if not need:
            return
        recs = None
        for name, df in (("train", self.tr), ("holdout", self.ho)):
            for g in need:
                path = FEATURE_CACHE / f"group{g}_{name}.parquet"
                if not path.exists():
                    if recs is None:
                        recs = (Records("train", ["source1"]), Records("train", ["source2", "source3"]))
                    s1, tg = recs
                    i, j = df["s1"].to_numpy(), df["tgt"].to_numpy()
                    n1, n2, a1, a2 = s1.name[i], tg.name[j], s1.addr[i], tg.addr[j]
                    k1 = np.array([core_name(x) for x in n1], dtype=object)
                    k2 = np.array([core_name(x) for x in n2], dtype=object)
                    feats = {"A": lambda: _number_decoy_features(a1, a2),
                             "B": lambda: _name_structure_features(k1, k2),
                             "C": lambda: _acronym_features(n1, n2, k1, k2)}[g]()
                    pd.DataFrame(feats).to_parquet(path)
                    print(f"  computed group {g} for {name}", flush=True)
                gdf = pd.read_parquet(path)
                for c in gdf.columns:
                    df[f"g{g}_{c}"] = gdf[c].to_numpy()

    def cols(self, groups):
        return self.base + [c for c in self.tr.columns if any(c.startswith(f"g{g}_") for g in groups)]


def fit(X, y, groups_s1, params, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(groups_s1)
    val_s1 = rng.choice(u, max(1, len(u) // 10), replace=False)
    v = np.isin(groups_s1, val_s1)
    m = xgb.XGBClassifier(**params, tree_method="hist", device="cuda", eval_metric="logloss",
                          early_stopping_rounds=50, random_state=seed)
    m.fit(X[~v], y[~v], eval_set=[(X[v], y[v])], verbose=False)
    return m


def best_f05(pairs, p, queries, d):
    scores = {t: macro_f05(queries, d.n_true, assign(pairs, p, t), d.pos) for t in THRESHOLDS}
    t = max(scores, key=scores.get)
    return scores[t], t


def inner_eval(d, cols, params, extra_tr=None, extra_va=None):
    """Train buckets 6-7, validate bucket 8 (pairs of bucket-8 S1 only)."""
    b = d.bkt[d.tr["s1"].to_numpy()]
    tr, va = d.tr[np.isin(b, [6, 7])], d.tr[b == 8]
    Xtr, Xva = tr[cols], va[cols]
    if extra_tr is not None:
        Xtr = pd.concat([Xtr.reset_index(drop=True), extra_tr.reset_index(drop=True)], axis=1)
        Xva = pd.concat([Xva.reset_index(drop=True), extra_va.reset_index(drop=True)], axis=1)
    m = fit(Xtr, tr["y"].to_numpy(), tr["s1"].to_numpy(), params)
    p = m.predict_proba(Xva)[:, 1]
    q = np.unique(va["s1"].to_numpy())
    return best_f05(va[["s1", "tgt"]].reset_index(drop=True), p, q, d)


def transfer_eval(d, cols, params):
    """US-only model: US holdout and India holdout (unseen country) at the US threshold."""
    use = d.ctry[d.tr["s1"].to_numpy()] == "US"
    tr = d.tr[use]
    m = fit(tr[cols], tr["y"].to_numpy(), tr["s1"].to_numpy(), params)
    p = m.predict_proba(d.ho[cols])[:, 1]
    q_us = d.hold_q[d.ctry[d.hold_q] == "US"]
    q_in = d.hold_q[d.ctry[d.hold_q] == "India"]
    pairs = d.ho[["s1", "tgt"]]
    us, t = best_f05(pairs, p, q_us, d)
    india = macro_f05(q_in, d.n_true, assign(pairs, p, t), d.pos)
    return us, india, t


def run_groups():
    d = Data()
    d.add_groups(list(GROUP_FNS))
    results = {}
    for groups in ([], ["A"], ["B"], ["C"]):
        results[tuple(groups)] = evaluate(d, groups, BASE_PARAMS, "3-features")
    base_inner, base_india = results[()][0], results[()][2]
    keep = [g for g in "ABC" if results[(g,)][0] > base_inner and results[(g,)][2] >= base_india - 0.0005]
    print(f"groups passing individually: {keep}", flush=True)
    for k in range(2, len(keep) + 1):
        evaluate(d, keep[:k], BASE_PARAMS, "3-features")


def evaluate(d, groups, params, phase, tag=None):
    cols = d.cols(groups)
    t0 = time.time()
    inner, thr = inner_eval(d, cols, params)
    us, india, _ = transfer_eval(d, cols, params)
    name = tag or ("base" if not groups else "+".join(groups))
    print(f"{name:12s} inner {inner:.4f} | US {us:.4f} | India transfer {india:.4f} | "
          f"{len(cols)} features ({time.time() - t0:.0f}s)", flush=True)
    log_row(experiment=f"features_{name}", phase=phase, encoder_config="v6",
            feature_groups="+".join(groups) or "base", n_features=len(cols),
            depth=params["max_depth"], learning_rate=params["learning_rate"],
            other_params=json.dumps({k: v for k, v in params.items() if k not in ("max_depth", "learning_rate")}),
            threshold=thr, inner_f05=inner, us_f05=us, india_transfer_f05=india)
    return inner, us, india


# --------------------------------------------------------------------------
# Stage 2: relational features from out-of-fold stage-1 probabilities
# --------------------------------------------------------------------------

def relational(pairs, p):
    """Features of a pair relative to its S1's other candidates and its target's other owners."""
    df = pd.DataFrame({"s1": pairs["s1"].to_numpy(), "tgt": pairs["tgt"].to_numpy(),
                       "p": np.asarray(p, dtype=np.float32)})
    df["hi"] = (df["p"] > 0.5).astype(np.float32)
    g1, gt = df.groupby("s1"), df.groupby("tgt")
    out = pd.DataFrame(index=df.index)
    out["r_p"] = df["p"]
    out["r_rank_s1"] = g1["p"].rank(ascending=False, method="first")
    out["r_max_s1"] = g1["p"].transform("max")
    out["r_n50_s1"] = g1["hi"].transform("sum")
    out["r_share_s1"] = df["p"] / (g1["p"].transform("sum") + 1e-6)
    # Best competing owner of the same target: the target's top p, or its
    # second-best p when this pair is the top one.
    rank_t = gt["p"].rank(ascending=False, method="first")
    top1 = gt["p"].transform("max")
    second = df["p"].where(rank_t == 2).groupby(df["tgt"]).transform("max").fillna(0.0)
    out["r_minus_best_other_owner"] = df["p"] - np.where(rank_t == 1, second, top1)
    out["r_n50_tgt"] = gt["hi"].transform("sum")
    return out.astype(np.float32).reset_index(drop=True)


def oof_probs(X, y, s1, params, folds=3, seed=0):
    """Out-of-fold probabilities with folds split by S1 (never by pair)."""
    rng = np.random.default_rng(seed)
    u = np.unique(s1)
    fold_of = pd.Series(rng.integers(0, folds, len(u)), index=u).loc[s1].to_numpy()
    p = np.empty(len(y), dtype=np.float32)
    for k in range(folds):
        te = fold_of == k
        m = fit(X[~te], y[~te], s1[~te], params, seed=seed + k)
        p[te] = m.predict_proba(X[te])[:, 1]
        print(f"    fold {k}: {te.sum():,} pairs", flush=True)
    return p


def run_stage2(groups, confirm_holdout=True):
    d = Data()
    d.add_groups(groups)
    cols = d.cols(groups)
    b = d.bkt[d.tr["s1"].to_numpy()]
    tr, va = d.tr[np.isin(b, [6, 7])].reset_index(drop=True), d.tr[b == 8].reset_index(drop=True)
    s1_tr = tr["s1"].to_numpy()
    # Stage-1 OOF on the inner-train rows; stage-1 on all inner-train rows for validation rows.
    p_oof = oof_probs(tr[cols], tr["y"].to_numpy(), s1_tr, BASE_PARAMS)
    m1 = fit(tr[cols], tr["y"].to_numpy(), s1_tr, BASE_PARAMS)
    p_va = m1.predict_proba(va[cols])[:, 1]
    R_tr, R_va = relational(tr, p_oof), relational(va, p_va)
    q = np.unique(va["s1"].to_numpy())
    s1_only, _ = best_f05(va[["s1", "tgt"]], p_va, q, d)
    m2 = fit(pd.concat([tr[cols], R_tr], axis=1), tr["y"].to_numpy(), s1_tr, BASE_PARAMS)
    p2 = m2.predict_proba(pd.concat([va[cols], R_va], axis=1))[:, 1]
    s2, thr = best_f05(va[["s1", "tgt"]], p2, q, d)
    print(f"stage-1 inner {s1_only:.4f} -> stage-2 inner {s2:.4f}", flush=True)
    hold1 = hold2 = None
    if confirm_holdout:
        # Leakage check on bucket 9: stage-1 on all 6-8 (OOF for stage-2 training rows).
        full = d.tr.reset_index(drop=True)
        s1_all = full["s1"].to_numpy()
        p_oof_all = oof_probs(full[cols], full["y"].to_numpy(), s1_all, BASE_PARAMS)
        m1a = fit(full[cols], full["y"].to_numpy(), s1_all, BASE_PARAMS)
        p_ho = m1a.predict_proba(d.ho[cols])[:, 1]
        m2a = fit(pd.concat([full[cols], relational(full, p_oof_all)], axis=1), full["y"].to_numpy(), s1_all, BASE_PARAMS)
        p_ho2 = m2a.predict_proba(pd.concat([d.ho[cols].reset_index(drop=True), relational(d.ho, p_ho)], axis=1))[:, 1]
        pairs = d.ho[["s1", "tgt"]]
        hold1, _ = best_f05(pairs, p_ho, d.hold_q, d)
        hold2, _ = best_f05(pairs, p_ho2, d.hold_q, d)
        print(f"holdout (bucket 9): stage-1 {hold1:.4f} -> stage-2 {hold2:.4f}", flush=True)
    log_row(experiment=f"stage2_{'+'.join(groups) or 'base'}", phase="3-stage2", encoder_config="v6",
            feature_groups=("+".join(groups) or "base") + "+D", n_features=len(cols) + R_tr.shape[1],
            depth=BASE_PARAMS["max_depth"], learning_rate=BASE_PARAMS["learning_rate"], threshold=thr,
            inner_f05=s2, holdout_f05=hold2 if hold2 is not None else "",
            notes=f"stage-1 inner {s1_only:.4f}; holdout stage-1 {hold1} -> stage-2 {hold2}")


def run_stage2_transfer(groups):
    """Unseen-country check for stage-2: US-only stage-1 (OOF) + stage-2, scored on India."""
    d = Data()
    d.add_groups(groups)
    cols = d.cols(groups)
    us = d.tr[d.ctry[d.tr["s1"].to_numpy()] == "US"].reset_index(drop=True)
    s1_us, y_us = us["s1"].to_numpy(), us["y"].to_numpy()
    p_oof = oof_probs(us[cols], y_us, s1_us, BASE_PARAMS)
    m1 = fit(us[cols], y_us, s1_us, BASE_PARAMS)
    m2 = fit(pd.concat([us[cols], relational(us, p_oof)], axis=1), y_us, s1_us, BASE_PARAMS)
    p1 = m1.predict_proba(d.ho[cols])[:, 1]
    p2 = m2.predict_proba(pd.concat([d.ho[cols].reset_index(drop=True), relational(d.ho, p1)], axis=1))[:, 1]
    pairs = d.ho[["s1", "tgt"]]
    q_us = d.hold_q[d.ctry[d.hold_q] == "US"]
    q_in = d.hold_q[d.ctry[d.hold_q] == "India"]
    out = {}
    for tag, p in (("stage-1", p1), ("stage-2", p2)):
        us_f, t = best_f05(pairs, p, q_us, d)
        out[tag] = (us_f, macro_f05(q_in, d.n_true, assign(pairs, p, t), d.pos))
        print(f"{tag}: US {out[tag][0]:.4f} | India transfer {out[tag][1]:.4f}", flush=True)
    log_row(experiment=f"stage2_transfer_{'+'.join(groups)}", phase="3-stage2", encoder_config="v6",
            feature_groups="+".join(groups) + "+D", us_f05=out["stage-2"][0], india_transfer_f05=out["stage-2"][1],
            notes=f"stage-1 US {out['stage-1'][0]:.4f} India {out['stage-1'][1]:.4f}")


def run_sibling():
    """Stage-2 with vs without sibling features: inner, bucket-9 holdout, US -> India."""
    from pipeline.matcher import oof_probs as m_oof, relational as m_rel, sibling_features

    d = Data()
    tg = Records("train", ["source2", "source3"])
    cols = d.cols([])
    b = d.bkt[d.tr["s1"].to_numpy()]

    def stage2_pair(tr, te, tag):
        """Train stage-1 (+OOF) on tr, evaluate stage-2 variants on te pairs."""
        s1, y = tr["s1"].to_numpy(), tr["y"].to_numpy()
        p_oof = m_oof(tr[cols], y, s1, BASE_PARAMS)
        p_te = fit(tr[cols], y, s1, BASE_PARAMS).predict_proba(te[cols])[:, 1]
        feats_tr = {"rel": m_rel(tr, p_oof)}
        feats_te = {"rel": m_rel(te, p_te)}
        feats_tr["rel+sib"] = pd.concat([feats_tr["rel"], sibling_features(tr, p_oof, tg)], axis=1)
        feats_te["rel+sib"] = pd.concat([feats_te["rel"], sibling_features(te, p_te, tg)], axis=1)
        probs = {"stage-1": p_te}
        for k in ("rel", "rel+sib"):
            m2 = fit(pd.concat([tr[cols].reset_index(drop=True), feats_tr[k]], axis=1), y, s1, BASE_PARAMS)
            probs[k] = m2.predict_proba(pd.concat([te[cols].reset_index(drop=True), feats_te[k]], axis=1))[:, 1]
        print(f"  [{tag}] done", flush=True)
        return probs

    rows = {}
    # 1. inner: train buckets 6-7, validate bucket 8
    tr, va = d.tr[np.isin(b, [6, 7])].reset_index(drop=True), d.tr[b == 8].reset_index(drop=True)
    probs = stage2_pair(tr, va, "inner")
    q = np.unique(va["s1"].to_numpy())
    for k, p in probs.items():
        rows.setdefault(k, {})["inner"] = best_f05(va[["s1", "tgt"]], p, q, d)[0]
    # 2. holdout (bucket 9): train on all of 6-8
    probs = stage2_pair(d.tr.reset_index(drop=True), d.ho.reset_index(drop=True), "holdout")
    for k, p in probs.items():
        rows[k]["holdout"] = best_f05(d.ho[["s1", "tgt"]], p, d.hold_q, d)[0]
    # 3. unseen country: US-only training, India holdout at the US-best threshold
    us = d.tr[d.ctry[d.tr["s1"].to_numpy()] == "US"].reset_index(drop=True)
    probs = stage2_pair(us, d.ho.reset_index(drop=True), "transfer")
    q_us = d.hold_q[d.ctry[d.hold_q] == "US"]
    q_in = d.hold_q[d.ctry[d.hold_q] == "India"]
    for k, p in probs.items():
        us_f, t = best_f05(d.ho[["s1", "tgt"]], p, q_us, d)
        rows[k]["us"], rows[k]["india"] = us_f, macro_f05(q_in, d.n_true, assign(d.ho[["s1", "tgt"]], p, t), d.pos)
    for k, r in rows.items():
        print(f"{k:8s} inner {r['inner']:.4f} | holdout {r['holdout']:.4f} | US {r['us']:.4f} | "
              f"India transfer {r['india']:.4f}", flush=True)
        log_row(experiment=f"sibling_{k}", phase="v10-step1", encoder_config="a_alldata",
                feature_groups="A+B" + ("+D" if k != "stage-1" else "") + ("+sib" if "sib" in k else ""),
                depth=BASE_PARAMS["max_depth"], learning_rate=BASE_PARAMS["learning_rate"],
                inner_f05=r["inner"], holdout_f05=r["holdout"], us_f05=r["us"], india_transfer_f05=r["india"])


# --------------------------------------------------------------------------
# Hyperparameter random search
# --------------------------------------------------------------------------

def run_tune(groups, n=20):
    d = Data()
    d.add_groups(groups)
    cols = d.cols(groups)
    rng = np.random.default_rng(42)
    trials = []
    for k in range(n):
        params = dict(n_estimators=6000,
                      max_depth=int(rng.integers(6, 11)),
                      learning_rate=float(np.round(rng.uniform(0.03, 0.1), 3)),
                      min_child_weight=float(np.round(rng.choice([1, 3, 5, 10, 20, 30]), 1)),
                      subsample=float(np.round(rng.uniform(0.6, 0.9), 2)),
                      colsample_bytree=float(np.round(rng.uniform(0.6, 0.9), 2)),
                      reg_lambda=float(np.round(rng.uniform(1, 10), 2)),
                      reg_alpha=float(np.round(rng.uniform(0, 1), 2)),
                      gamma=float(np.round(rng.uniform(0, 1), 2)),
                      max_bin=int(rng.choice([256, 512])))
        if k == 0:
            params = dict(BASE_PARAMS)
        t0 = time.time()
        inner, thr = inner_eval(d, cols, params)
        trials.append((inner, params))
        print(f"trial {k:2d} inner {inner:.4f} ({time.time() - t0:.0f}s) {params}", flush=True)
        log_row(experiment=f"tune_{k}", phase="4-tune", encoder_config="v6",
                feature_groups="+".join(groups) or "base", n_features=len(cols),
                depth=params["max_depth"], learning_rate=params["learning_rate"],
                other_params=json.dumps({a: b for a, b in params.items() if a not in ("max_depth", "learning_rate")}),
                threshold=thr, inner_f05=inner)
    trials.sort(key=lambda t: -t[0])
    print("top 3 -> unseen-country check", flush=True)
    for rank, (inner, params) in enumerate(trials[:3]):
        us, india, _ = transfer_eval(d, cols, params)
        print(f"  #{rank + 1} inner {inner:.4f} | US {us:.4f} | India transfer {india:.4f} | {params}", flush=True)
        log_row(experiment=f"tune_top{rank + 1}", phase="4-tune", encoder_config="v6",
                feature_groups="+".join(groups) or "base", n_features=len(cols),
                depth=params["max_depth"], learning_rate=params["learning_rate"],
                other_params=json.dumps({a: b for a, b in params.items() if a not in ("max_depth", "learning_rate")}),
                inner_f05=inner, us_f05=us, india_transfer_f05=india)


if __name__ == "__main__":
    cmd = sys.argv[1]
    groups = [g for g in (sys.argv[2].split(",") if len(sys.argv) > 2 else []) if g and g != "base"]
    if cmd == "prepare":      # CPU only: compute and cache feature groups
        Data().add_groups(list(GROUP_FNS))
    elif cmd == "groups":
        run_groups()
    elif cmd == "stage2":
        run_stage2(groups)
    elif cmd == "sibling":
        run_sibling()
    elif cmd == "stage2_transfer":
        run_stage2_transfer(groups)
    elif cmd == "tune":
        run_tune(groups, int(sys.argv[3]) if len(sys.argv) > 3 else 20)

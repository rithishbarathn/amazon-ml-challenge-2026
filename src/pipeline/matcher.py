"""Stage 5: pairwise matching model + decision rule, and submission writing.

Model     : XGBoost (GPU) on the features of pipeline.features.
Decision  : a Source 2/3 record can belong to at most one Source 1 entity
            (Source 1 is deduplicated), so each target is assigned only to its
            highest-probability Source 1 candidate, and only if that
            probability clears a threshold tuned for macro F0.5 on a held-out
            set of training Source 1 entities.

Usage (from src/):
    python -m pipeline.matcher train                  # fit + tune decision rule
    python -m pipeline.matcher tune                   # re-tune rule on saved holdout preds
    python -m pipeline.matcher predict [--reuse-prob] # --reuse-prob skips re-scoring
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.block import load_candidates
from pipeline.common import DATA_DIR, OUTPUT_DIR, WORK_DIR, bucket, prep_path
from pipeline.features import Records, build_features, filter_candidates, iter_features

MODEL_PATH = WORK_DIR / "matcher.json"
CONFIG_PATH = WORK_DIR / "matcher_config.json"
HOLDOUT_PRED = WORK_DIR / "holdout_pred.parquet"
HOLDOUT_QUERIES_PATH = WORK_DIR / "holdout_queries.npy"
TEST_PROB = WORK_DIR / "test_prob.npy"
FEATURE_CACHE = WORK_DIR / "features"
STAGE1_PATH = WORK_DIR / "matcher_stage1.json"
# XGBoost settings for both stages. The Phase 4 random search found no setting
# better than these on the inner split (submissions/experiments.csv, tune_*).
MATCHER_PARAMS = dict(n_estimators=6000, learning_rate=0.05, max_depth=8, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8)
STAGE2 = True          # relational second stage on out-of-fold stage-1 probabilities
OOF_FOLDS = 3
TRAIN_QUERIES = 10_000_000  # capped at all matcher-bucket S1
HOLDOUT_QUERIES = 100_000
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.025), 3)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_truth(s1_ids, tg_ids):
    """Ground truth as (sorted array of positive (s1 << 32 | tgt) keys, #matches per S1)."""
    truth = pd.read_csv(
        DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
        keep_default_na=False,
    )
    s1_index = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    tg_index = pd.Series(np.arange(len(tg_ids)), index=tg_ids)
    ex = truth.assign(tgt=truth["matched_entity_ids"].str.split(",")).explode("tgt")
    ex = ex[ex["tgt"] != ""]
    s = s1_index.loc[ex["source1_entity_id"]].to_numpy().astype(np.int64)
    t = tg_index.loc[ex["tgt"]].to_numpy().astype(np.int64)
    n_true = np.zeros(len(s1_ids), dtype=np.int32)
    np.add.at(n_true, s, 1)
    return np.unique((s << 32) | t), n_true


def with_ce(cand, split):
    """Attach cross-encoder scores when enabled (ER_USE_CE=1, tag ER_CE_TAG)."""
    if os.environ.get("ER_USE_CE") != "1":
        return cand
    from pipeline.crossenc import attach_scores
    return attach_scores(cand, split, os.environ.get("ER_CE_TAG", "all"))


# --------------------------------------------------------------------------
# Stage 2: relational features from stage-1 probabilities
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


def fit_xgb(X, y, s1, params=None, seed=0, verbose=False):
    """XGBoost with early stopping on a 10% validation split grouped by S1."""
    rng = np.random.default_rng(seed)
    u = np.unique(s1)
    v = np.isin(s1, rng.choice(u, max(1, len(u) // 10), replace=False))
    model = xgb.XGBClassifier(**(params or MATCHER_PARAMS), tree_method="hist", device="cuda",
                              eval_metric="logloss", early_stopping_rounds=50, random_state=seed)
    model.fit(X[~v], y[~v], eval_set=[(X[v], y[v])], verbose=200 if verbose else False)
    return model


def oof_probs(X, y, s1, params=None, folds=OOF_FOLDS, seed=0):
    """Out-of-fold stage-1 probabilities; folds are split by S1, never by pair,
    so no pair's relational features are built from a model that saw its label."""
    rng = np.random.default_rng(seed)
    u = np.unique(s1)
    fold_of = pd.Series(rng.integers(0, folds, len(u)), index=u).loc[s1].to_numpy()
    p = np.empty(len(y), dtype=np.float32)
    for k in range(folds):
        te = fold_of == k
        p[te] = fit_xgb(X[~te], y[~te], s1[~te], params, seed=seed + k).predict_proba(X[te])[:, 1]
        print(f"  OOF fold {k}: {te.sum():,} pairs", flush=True)
    return p


def pair_keys(cand):
    return (cand["s1"].to_numpy().astype(np.int64) << 32) | cand["tgt"].to_numpy().astype(np.int64)


def assign(cand, prob, threshold):
    """One-owner rule: keep each target only for its best S1, if p >= threshold."""
    df = pd.DataFrame({"s1": cand["s1"].to_numpy(), "tgt": cand["tgt"].to_numpy(), "p": prob})
    best = df.loc[df.groupby("tgt")["p"].idxmax()]
    return best[best["p"] >= threshold]


def assign_expected(cand, prob, floor, extra):
    """Per-S1 choice that maximises the expected F0.5 of that S1.

    After the one-owner rule, each S1's surviving candidates (p >= floor) are
    sorted by p. Keeping the top k gives expected F0.5 of roughly
        1.25 * sum(p_1..p_k) / (k + 0.25 * N),
    where N (expected number of true matches) is the sum of p over all of the
    S1's candidates plus `extra` for matches blocking missed. Predicting
    nothing scores 1 only if the S1 is a singleton, i.e. prod(1 - p). The
    best of these options is kept.
    """
    df = pd.DataFrame({"s1": cand["s1"].to_numpy(), "tgt": cand["tgt"].to_numpy(), "p": prob})
    grp = df.groupby("s1")["p"]
    n_exp = grp.sum() + extra
    p_none = np.exp(np.log1p(-df["p"].clip(upper=1 - 1e-6)).groupby(df["s1"]).sum())

    best = df.loc[df.groupby("tgt")["p"].idxmax()]
    best = best[best["p"] >= floor].sort_values(["s1", "p"], ascending=[True, False])
    k = best.groupby("s1").cumcount().to_numpy() + 1
    csum = best.groupby("s1")["p"].cumsum().to_numpy()
    s1 = best["s1"].to_numpy()
    gain = 1.25 * csum / (k + 0.25 * n_exp.loc[s1].to_numpy())
    best = best.assign(gain=gain, k=k)
    top = best.groupby("s1")["gain"].transform("max").to_numpy()
    k_best = best.loc[best["gain"].to_numpy() == top].groupby("s1")["k"].min()
    keep = (best["k"].to_numpy() <= k_best.loc[s1].to_numpy()) & \
        (top > p_none.loc[s1].to_numpy())
    return best.loc[keep, ["s1", "tgt", "p"]]


def decide(cand, prob, rule):
    if rule["mode"] == "expected":
        return assign_expected(cand, prob, rule["floor"], rule["extra"])
    return assign(cand, prob, rule["threshold"])


def macro_f05(queries, n_true, predicted, positives):
    """Official metric: per-S1 F0.5 averaged over `queries` (singletons included)."""
    pred = predicted[predicted["s1"].isin(queries)]
    hit = np.isin(pair_keys(pred), positives)
    per = pd.DataFrame({"s1": pred["s1"].to_numpy(), "tp": hit}).groupby("s1").agg(
        n_pred=("tp", "size"), tp=("tp", "sum"))
    per = per.reindex(queries, fill_value=0)
    tp = per["tp"].to_numpy()
    fp = per["n_pred"].to_numpy() - tp
    fn = n_true[queries] - tp
    denom = 1.25 * tp + 0.25 * fn + fp
    score = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-9), 1.0)  # empty/empty -> 1
    return float(score.mean())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train():
    start = time.time()
    s1 = Records("train", ["source1"])
    tg = Records("train", ["source2", "source3"])
    print(f"records loaded ({time.time() - start:.0f}s)", flush=True)
    positives, n_true = load_truth(s1.ids, tg.ids)
    cand = with_ce(filter_candidates(load_candidates("train")), "train")
    print(f"train candidates: {len(cand):,}", flush=True)

    rng = np.random.default_rng(0)
    buckets = np.array([bucket(i) for i in s1.ids])
    train_q = rng.permutation(np.flatnonzero((buckets >= 6) & (buckets <= 8)))[:TRAIN_QUERIES]
    hold_q = rng.choice(np.flatnonzero(buckets == 9), HOLDOUT_QUERIES, replace=False)

    tr = cand[cand["s1"].isin(train_q)].reset_index(drop=True)
    # Holdout pairs + every competing owner of the same targets, so that the
    # one-owner rule sees the same competition as it will on the test set.
    hold_targets = cand.loc[cand["s1"].isin(hold_q), "tgt"].unique()
    ho = cand[cand["tgt"].isin(hold_targets)].reset_index(drop=True)

    X_tr = build_features(s1, tg, tr)
    y_tr = np.isin(pair_keys(tr), positives).astype(np.int8)
    X_ho = build_features(s1, tg, ho)
    y_ho = np.isin(pair_keys(ho), positives).astype(np.int8)
    # Cache features so model variants / ensembles can be tried without re-featurising.
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    X_tr.assign(y=y_tr, s1=tr["s1"].to_numpy()).to_parquet(FEATURE_CACHE / "train.parquet")
    X_ho.assign(y=y_ho).to_parquet(FEATURE_CACHE / "holdout.parquet")
    print(f"train pairs {len(tr):,} (pos {y_tr.mean():.3f}), holdout pairs {len(ho):,} "
          f"({time.time() - start:.0f}s)", flush=True)

    s1_tr = tr["s1"].to_numpy()
    stage1 = fit_xgb(X_tr, y_tr, s1_tr, verbose=True)
    prob = stage1.predict_proba(X_ho)[:, 1]
    print(f"stage-1 best iteration {stage1.best_iteration}", flush=True)
    model = stage1
    if STAGE2:
        p_oof = oof_probs(X_tr, y_tr, s1_tr)
        X_tr = pd.concat([X_tr.reset_index(drop=True), relational(tr, p_oof)], axis=1)
        X_ho = pd.concat([X_ho.reset_index(drop=True), relational(ho, prob)], axis=1)
        stage1.save_model(STAGE1_PATH)
        model = fit_xgb(X_tr, y_tr, s1_tr, verbose=True)
        prob = model.predict_proba(X_ho)[:, 1]
    print(f"best iteration {model.best_iteration}; holdout mean p pos "
          f"{prob[y_ho == 1].mean():.3f} neg {prob[y_ho == 0].mean():.3f}")

    imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print(imp.head(15).to_string())
    model.save_model(MODEL_PATH)

    # Keep holdout predictions so decision rules can be re-tuned without retraining.
    ho.assign(p=prob, y=y_ho)[["s1", "tgt", "rank_a", "rank_b", "rank_c", "p", "y"]].to_parquet(HOLDOUT_PRED)
    np.save(HOLDOUT_QUERIES_PATH, hold_q)
    CONFIG_PATH.write_text(json.dumps({
        "features": [c for c in X_tr.columns if not c.startswith("r_")],
        "stage2": STAGE2, "stage2_features": list(X_tr.columns), "params": MATCHER_PARAMS,
    }, indent=2))
    print(f"saved model ({time.time() - start:.0f}s)")
    tune(s1, tg, positives, n_true)


def tune(s1=None, tg=None, positives=None, n_true=None):
    """Pick the decision rule on the saved holdout predictions and store it in the config."""
    if s1 is None:
        s1 = Records("train", ["source1"])
        tg = Records("train", ["source2", "source3"])
        positives, n_true = load_truth(s1.ids, tg.ids)
    ho = pd.read_parquet(HOLDOUT_PRED)
    hold_q = np.load(HOLDOUT_QUERIES_PATH)
    prob = ho["p"].to_numpy()
    blocking_recall = ho.loc[ho["s1"].isin(hold_q), "y"].sum() / n_true[hold_q].sum()
    print(f"holdout blocking recall: {blocking_recall:.4f}")

    rules = [{"mode": "threshold", "threshold": float(t)} for t in THRESHOLDS]
    rules += [{"mode": "expected", "floor": f, "extra": e}
              for f in (0.2, 0.3, 0.4, 0.5) for e in (0.0, 0.05, 0.15, 0.3)]
    results = []
    for rule in rules:
        score = macro_f05(hold_q, n_true, decide(ho, prob, rule), positives)
        results.append((score, rule))
        print(f"{json.dumps(rule)}: macro F0.5 {score:.4f}", flush=True)
    best_score, best_rule = max(results, key=lambda r: r[0])
    best_thr = max((r for r in results if r[1]["mode"] == "threshold"), key=lambda r: r[0])
    print(f"BEST {json.dumps(best_rule)} -> {best_score:.4f} "
          f"(best plain threshold {best_thr[1]['threshold']} -> {best_thr[0]:.4f})")

    # Per-country breakdown (reporting only - the rule is shared by all countries).
    pred = decide(ho, prob, best_rule)
    for country in sorted(set(s1.country[hold_q])):
        q = hold_q[s1.country[hold_q] == country]
        singles = (n_true[q] == 0).mean()
        print(f"  {country}: {len(q):,} S1 ({singles:.1%} singletons) "
              f"macro F0.5 {macro_f05(q, n_true, pred, positives):.4f}")

    # Smaller candidate sets (organisers rank smaller sets higher).
    rank_c = ho["rank_c"] if "rank_c" in ho else pd.Series(99, index=ho.index)
    for kb in (0, 3, 5):
        for use_c in (False, True):
            keep = ((ho["rank_a"] < 1) | (ho["rank_b"] < kb) | (use_c & (rank_c < 1))).to_numpy()
            sub = ho[keep]
            n_c = sub["s1"].isin(hold_q).sum() / len(hold_q)
            score = macro_f05(hold_q, n_true, decide(sub, prob[keep], best_rule), positives)
            print(f"  candidates rank_a<1 | rank_b<{kb}{' | rank_c<1' if use_c else ''}: "
                  f"{n_c:.2f}/S1 -> macro F0.5 {score:.4f}")

    config = json.loads(CONFIG_PATH.read_text())
    config.update({"rule": best_rule, "holdout_f05": best_score})
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


# --------------------------------------------------------------------------
# Test prediction + submission files
# --------------------------------------------------------------------------

def write_id_lists(path, s1_ids, pairs, id_col):
    grouped = pairs.groupby("s1_id")["t_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[id_col] = out["source1_entity_id"].map(grouped).fillna("")
    out.to_csv(path, sep="\t", index=False)


def write_variants(s1, tg, pred):
    """Leaderboard probes for countries absent from training (no labels offline).

    Same matches as the main file except for S1 whose country never appears in
    the training data: 'strict' keeps only their pairs with p >= 0.9, 'empty'
    predicts no matches for them.
    """
    train_countries = set(pd.read_parquet(
        prep_path("train", "source1"), columns=["country"])["country"].unique())
    unseen = ~np.isin(s1.country[pred["s1"].to_numpy()], list(train_countries))
    out = OUTPUT_DIR / "variants"
    out.mkdir(parents=True, exist_ok=True)
    for name, keep in [("unseen_strict", ~unseen | (pred["p"].to_numpy() >= 0.9)),
                       ("unseen_empty", ~unseen)]:
        sub = pred[keep]
        write_id_lists(
            out / f"matching_results_{name}.tsv", s1.ids,
            pd.DataFrame({"s1_id": s1.ids[sub["s1"]], "t_id": tg.ids[sub["tgt"]]}),
            "matched_entity_ids",
        )
    print(f"variants written to {out} (unseen-country pairs: {unseen.sum():,})")


def predict():
    start = time.time()
    config = json.loads(CONFIG_PATH.read_text())
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cuda")

    s1 = Records("test", ["source1"])
    tg = Records("test", ["source2", "source3"])
    cand = with_ce(filter_candidates(load_candidates("test")), "test")
    print(f"test candidates: {len(cand):,} ({len(cand) / len(s1.ids):.2f} per S1)", flush=True)

    if len(sys.argv) > 2 and sys.argv[2] == "--reuse-prob":
        prob = np.load(TEST_PROB)
    else:
        if config.get("stage2"):
            stage1 = xgb.XGBClassifier()
            stage1.load_model(STAGE1_PATH)
            stage1.set_params(device="cuda")
            chunks = [feats[config["features"]] for _, feats in iter_features(s1, tg, cand)]
            X = pd.concat(chunks, ignore_index=True)
            del chunks
            p1 = np.concatenate([stage1.predict_proba(X.iloc[i:i + 2_000_000])[:, 1]
                                 for i in range(0, len(X), 2_000_000)])
            X = pd.concat([X, relational(cand, p1)], axis=1)[config["stage2_features"]]
            prob = np.concatenate([model.predict_proba(X.iloc[i:i + 2_000_000])[:, 1]
                                   for i in range(0, len(X), 2_000_000)]).astype(np.float32)
            del X
        else:
            prob = np.empty(len(cand), dtype=np.float32)
            for begin, feats in iter_features(s1, tg, cand):
                prob[begin:begin + len(feats)] = model.predict_proba(feats[config["features"]])[:, 1]
        np.save(TEST_PROB, prob)

    rule = config.get("rule", {"mode": "threshold", "threshold": config.get("threshold")})
    print(f"decision rule: {json.dumps(rule)}")
    pred = decide(cand, prob, rule)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_variants(s1, tg, pred)
    write_id_lists(
        OUTPUT_DIR / "candidate_pairs.tsv", s1.ids,
        pd.DataFrame({"s1_id": s1.ids[cand["s1"]], "t_id": tg.ids[cand["tgt"]]}),
        "candidate_entity_ids",
    )
    write_id_lists(
        OUTPUT_DIR / "matching_results.tsv", s1.ids,
        pd.DataFrame({"s1_id": s1.ids[pred["s1"]], "t_id": tg.ids[pred["tgt"]]}),
        "matched_entity_ids",
    )
    print(f"matched pairs: {len(pred):,}; S1 with >=1 match: {pred['s1'].nunique():,}/{len(s1.ids):,}")
    print(f"written to {OUTPUT_DIR} ({time.time() - start:.0f}s)")

    validator = DATA_DIR.parent / "utils" / "validate_submission.py"
    subprocess.run([
        sys.executable, str(validator),
        "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(DATA_DIR / "test"),
    ], check=False)


if __name__ == "__main__":
    {"train": train, "tune": tune, "predict": predict}[sys.argv[1]]()

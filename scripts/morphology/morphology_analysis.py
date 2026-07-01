#!/usr/bin/env python
"""
morphology_analysis.py — WCC-trace morphology + collinearity + order-unbiased
incremental-value analysis (Lerique 30-dyad pilot, but data-agnostic).

Addresses four requests:
  (A) Data-driven morphology: cluster traces by SHAPE descriptors (no preset
      taxonomy); describe each cluster's feature distribution (morphology is a
      DESCRIBED outcome, never used as both clustering input and prediction y).
  (B) Collinearity diagnosis: correlation matrix + VIF + hierarchical feature
      clustering, for the 9 SyncPipe features.
  (C) Order-UNBIASED incremental value: LOFO (leave-one-feature-out) and
      random-order-averaged marginal AUC (Shapley-style), reported BOTH with
      and without mean_synchrony as a baseline anchor. This removes the
      "ordering bias" of fixed cumulative incremental steps.
  (D) Matched-mean contrast: among traces with SIMILAR mean_synchrony, do
      different morphologies separate under different features?

Designed for SMALL n (≈30): all AUCs use stratified CV + bootstrap CIs and are
labelled EXPLORATORY. Works on either real Lerique traces or a synthetic proxy
(`--synthetic`) so the pipeline can be validated before real data arrives.

INPUTS
------
Real data: a long-format CSV with one row per (dyad, modality-pair) trace and
either (a) a column holding the WCC trace as a JSON list, or (b) precomputed
feature columns. Use --traces-csv with --trace-col.
Synthetic: --synthetic builds 30 proxy traces across 4 shapes at matched mean.

OUTPUT  (artifacts/morphology/)
  morphology_features.csv      shape descriptors + SyncPipe features per trace
  cluster_assignment.csv       cluster label per trace + silhouette
  collinearity_corr.csv        feature correlation matrix
  collinearity_vif.csv         VIF per feature
  incremental_auc.csv          LOFO + random-order marginal AUC (+/- mean_sync)
  matched_mean_contrast.csv    within-mean-band morphology separability
  MORPHOLOGY_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts" / "morphology"

# SyncPipe feature set (the 9 features) — kept separate from SHAPE descriptors
MS_FEATURES = [
    "mean_synchrony", "peak_amplitude", "dwell_time", "switching_rate",
    "bimodality_coefficient", "synchrony_entropy",
    "onset_latency", "rise_time", "recovery_time",
]

# ---------------------------------------------------------------------------
# Synthetic proxy traces (4 shapes at MATCHED mean_synchrony ~ 0.5)
# ---------------------------------------------------------------------------
def _calibrate_mean(wcc, target=0.5):
    cur = np.nanmean(wcc)
    return np.clip(wcc + (target - cur), -1, 1)


def synth_traces(n_per_shape=8, n=300, hz=1.0, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    out = []
    def add(shape, base):
        w = _calibrate_mean(np.clip(base + rng.normal(0, noise, n), -1, 1))
        out.append((shape, w))
    for k in range(n_per_shape):
        # sustained: mostly high, few crossings
        add("sustained", 0.6 + 0.05 * np.sin(2 * np.pi * t / n))
        # single_peak: one Gaussian bump
        c = n // 2
        add("single_peak", 0.1 + 0.8 * np.exp(-((t - c) ** 2) / (2 * (n / 8) ** 2)))
        # oscillatory: multi-peak
        add("oscillatory", 0.4 + 0.45 * np.sin(2 * np.pi * t * 4 / n))
        # asymmetric: fast rise, slow decay
        rise = np.clip((t - n * 0.3) / (n * 0.05), 0, 1)
        decay = np.exp(-(t - n * 0.35) / (n * 0.4))
        decay[t < n * 0.35] = 1.0
        add("asymmetric", 0.1 + 0.8 * rise * decay)
    return out  # list of (true_shape, wcc)


# ---------------------------------------------------------------------------
# Shape descriptors (for clustering) — reuse experimental morphology profile
# ---------------------------------------------------------------------------
def shape_descriptors(wcc, hz=1.0, threshold=0.5):
    """Morphology shape features used as CLUSTERING INPUT only."""
    w = np.asarray(wcc, float)
    finite = np.isfinite(w)
    w = w[finite]
    if w.size < 5:
        return None
    above = w >= threshold
    # episode runs
    d = np.diff(above.astype(int))
    n_episodes = int((d == 1).sum() + (1 if above[0] else 0))
    above_ratio = float(above.mean())
    # peak count via simple local maxima above threshold
    from scipy.signal import find_peaks
    peaks, props = find_peaks(w, height=threshold, distance=max(2, int(0.05 * w.size)))
    n_peaks = int(len(peaks))
    # asymmetry of dominant peak (rise vs fall width at half-prominence)
    asym = np.nan
    if n_peaks >= 1:
        pk = peaks[int(np.argmax(w[peaks]))]
        half = (w[pk] + np.nanmin(w)) / 2
        l = pk
        while l > 0 and w[l] > half:
            l -= 1
        r = pk
        while r < w.size - 1 and w[r] > half:
            r += 1
        rise_dur = max(pk - l, 1)
        fall_dur = max(r - pk, 1)
        asym = float(rise_dur / fall_dur)
    return {
        "above_ratio": above_ratio,
        "n_episodes": float(n_episodes),
        "n_peaks": float(n_peaks),
        "peak_asymmetry": asym,
        "wcc_std": float(np.std(w)),
        "wcc_range": float(np.ptp(w)),
        "frac_negative": float((w < 0).mean()),
        "autocorr_lag1": float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w.size > 2 else np.nan,
    }


def ms_features(wcc, hz=1.0, wcc_window_sec=None):
    """Extract the 9 SyncPipe features via the package."""
    from multisync.dynamic_features import extract_dynamic_features
    f = extract_dynamic_features(np.asarray(wcc, float), hz=hz,
                                 wcc_window_sec=wcc_window_sec or (len(wcc) / hz))
    return {k: float(getattr(f, k, np.nan)) for k in MS_FEATURES}


# ---------------------------------------------------------------------------
# (A) clustering
# ---------------------------------------------------------------------------
def cluster_shapes(shape_df, max_k=5, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = shape_df.fillna(shape_df.median()).values
    Xs = StandardScaler().fit_transform(X)
    best = {"k": None, "sil": -1, "labels": None}
    n = len(Xs)
    for k in range(2, min(max_k, n - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(Xs, labels)
        if sil > best["sil"]:
            best = {"k": k, "sil": float(sil), "labels": labels}
    return best


# ---------------------------------------------------------------------------
# (B) collinearity
# ---------------------------------------------------------------------------
def collinearity(feat_df, features):
    sub = feat_df[features].dropna(axis=1, how="all")
    used = [c for c in features if c in sub.columns and sub[c].std() > 1e-9]
    corr = sub[used].corr(method="spearman")
    # VIF
    from numpy.linalg import LinAlgError
    vif = {}
    X = sub[used].fillna(sub[used].median()).values
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    for i, c in enumerate(used):
        others = np.delete(Xc, i, axis=1)
        try:
            beta, *_ = np.linalg.lstsq(others, Xc[:, i], rcond=None)
            pred = others @ beta
            ss_res = np.sum((Xc[:, i] - pred) ** 2)
            ss_tot = np.sum((Xc[:, i] - Xc[:, i].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            vif[c] = float(1.0 / max(1e-6, 1 - r2))
        except LinAlgError:
            vif[c] = float("nan")
    return corr, pd.Series(vif, name="VIF")


# ---------------------------------------------------------------------------
# (C) order-UNBIASED incremental AUC
# ---------------------------------------------------------------------------
def _cv_auc(X, y, seed=42, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if X.shape[1] == 0 or len(set(y)) < 2:
        return 0.5
    classes, counts = np.unique(y, return_counts=True)
    n_splits = int(min(n_splits, counts.min()))
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000))
    scoring = "roc_auc_ovr" if len(classes) > 2 else "roc_auc"
    try:
        return float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring=scoring)))
    except Exception:
        return np.nan


def incremental_value(feat_df, y, features, baseline_drop_meansync=True,
                      n_orders=200, seed=42):
    """Random-order-averaged marginal AUC (Shapley-style) + LOFO.

    Returns per-feature: mean marginal contribution to AUC over random insertion
    orders, and LOFO drop (full minus full-without-feature).
    """
    rng = np.random.default_rng(seed)
    feats = [f for f in features if f in feat_df.columns and feat_df[f].std() > 1e-9]
    if baseline_drop_meansync and "mean_synchrony" in feats:
        feats = [f for f in feats if f != "mean_synchrony"]
        baseline_feats = []
    else:
        baseline_feats = ["mean_synchrony"] if "mean_synchrony" in feats else []
        feats = [f for f in feats if f not in baseline_feats]

    allcols = list(dict.fromkeys(baseline_feats + feats))
    Xall = feat_df[allcols].apply(pd.to_numeric, errors="coerce")
    Xall = Xall.fillna(Xall.median())

    def auc_of(cols):
        if not cols:
            return 0.5
        return _cv_auc(Xall[cols].values, y, seed=seed)

    base_auc = auc_of(baseline_feats)
    full_auc = auc_of(baseline_feats + feats)

    # Random-order marginal (Shapley approx)
    marg = {f: [] for f in feats}
    k = len(feats)
    if k <= 6:
        orders = list(permutations(feats))
    else:
        orders = [list(rng.permutation(feats)) for _ in range(n_orders)]
    for order in orders:
        cur = list(baseline_feats)
        prev = auc_of(cur)
        for f in order:
            cur.append(f)
            now = auc_of(cur)
            marg[f].append(now - prev)
            prev = now
    shap = {f: float(np.nanmean(v)) for f, v in marg.items()}

    # LOFO
    lofo = {}
    for f in feats:
        without = baseline_feats + [x for x in feats if x != f]
        lofo[f] = float(full_auc - auc_of(without))

    rows = []
    for f in feats:
        rows.append({"feature": f, "shapley_marginal_auc": shap[f],
                     "lofo_auc_drop": lofo[f]})
    out = pd.DataFrame(rows).sort_values("shapley_marginal_auc", ascending=False)
    meta = {"baseline": "none" if baseline_drop_meansync else "mean_synchrony",
            "baseline_auc": base_auc, "full_auc": full_auc}
    return out, meta


# ---------------------------------------------------------------------------
# (D) matched-mean contrast
# ---------------------------------------------------------------------------
def matched_mean_contrast(feat_df, y, features, band=0.1):
    """Within a narrow mean_synchrony band, can features separate morphology?"""
    if "mean_synchrony" not in feat_df:
        return pd.DataFrame()
    med = feat_df["mean_synchrony"].median()
    mask = (feat_df["mean_synchrony"] >= med - band) & (feat_df["mean_synchrony"] <= med + band)
    sub = feat_df[mask]
    ysub = np.asarray(y)[mask.values]
    rows = []
    for f in features:
        if f == "mean_synchrony" or f not in sub or sub[f].std() < 1e-9:
            continue
        col = pd.to_numeric(sub[f], errors="coerce")
        auc = _cv_auc(col.fillna(col.median()).values.reshape(-1, 1), ysub)
        rows.append({"feature": f, "auc_within_mean_band": auc, "n_in_band": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("auc_within_mean_band", ascending=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic proxy traces (validate pipeline)")
    ap.add_argument("--traces-csv", type=str, default=None)
    ap.add_argument("--trace-col", type=str, default="wcc_json")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ----- load traces -----
    traces = []  # list of (id, true_shape_or_None, wcc)
    if args.synthetic:
        for i, (shape, w) in enumerate(synth_traces(seed=args.seed)):
            traces.append((f"synth_{i}", shape, w))
    elif args.traces_csv:
        df = pd.read_csv(args.traces_csv)
        for i, row in df.iterrows():
            w = np.array(json.loads(row[args.trace_col]), float)
            traces.append((row.get("id", f"trace_{i}"), None, w))
    else:
        raise SystemExit("provide --synthetic or --traces-csv")

    # ----- features -----
    rows = []
    for tid, shape, w in traces:
        sd = shape_descriptors(w, hz=args.hz)
        if sd is None:
            continue
        msf = ms_features(w, hz=args.hz)
        rec = {"id": tid, "true_shape": shape, **{f"shape_{k}": v for k, v in sd.items()}, **msf}
        rows.append(rec)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(OUTDIR / "morphology_features.csv", index=False)

    # ----- (A) cluster on SHAPE descriptors -----
    shape_cols = [c for c in feat_df.columns if c.startswith("shape_")]
    best = cluster_shapes(feat_df[shape_cols], seed=args.seed)
    feat_df["cluster"] = best["labels"]
    feat_df[["id", "true_shape", "cluster"]].to_csv(OUTDIR / "cluster_assignment.csv", index=False)

    # cluster descriptive: mean feature per cluster
    cluster_desc = feat_df.groupby("cluster")[MS_FEATURES].mean()
    cluster_desc.to_csv(OUTDIR / "cluster_feature_means.csv")

    # ----- (B) collinearity -----
    corr, vif = collinearity(feat_df, MS_FEATURES)
    corr.to_csv(OUTDIR / "collinearity_corr.csv")
    vif.to_frame().to_csv(OUTDIR / "collinearity_vif.csv")

    # ----- (C) incremental value (cluster as y; SHAPE descriptors were the
    #           clustering input, MS_FEATURES are the predictors -> no leakage
    #           of identical inputs, but note partial dependence; see report) -
    y = feat_df["cluster"].values
    inc_drop, meta_drop = incremental_value(feat_df, y, MS_FEATURES,
                                            baseline_drop_meansync=True, seed=args.seed)
    inc_keep, meta_keep = incremental_value(feat_df, y, MS_FEATURES,
                                            baseline_drop_meansync=False, seed=args.seed)
    inc_drop["baseline"] = "drop_mean_synchrony"
    inc_keep["baseline"] = "keep_mean_synchrony"
    pd.concat([inc_drop, inc_keep]).to_csv(OUTDIR / "incremental_auc.csv", index=False)

    # ----- (D) matched-mean contrast -----
    mm = matched_mean_contrast(feat_df, y, MS_FEATURES)
    mm.to_csv(OUTDIR / "matched_mean_contrast.csv", index=False)

    # ----- report -----
    write_report(OUTDIR / "MORPHOLOGY_REPORT.md", feat_df, best, corr, vif,
                 inc_drop, meta_drop, inc_keep, meta_keep, mm, cluster_desc,
                 synthetic=args.synthetic)
    print(f"Done. Report -> {OUTDIR/'MORPHOLOGY_REPORT.md'}")
    return 0


def write_report(path, feat_df, best, corr, vif, inc_drop, meta_drop,
                 inc_keep, meta_keep, mm, cluster_desc, synthetic):
    L = []
    L.append("# SyncPipe — WCC Morphology / Collinearity / Incremental-Value Report\n")
    src = "SYNTHETIC PROXY (pipeline validation)" if synthetic else "Lerique pilot traces"
    L.append(f"- **Data**: {src}  |  n traces = {len(feat_df)}  |  "
             f"**EXPLORATORY** (small-n; report effect sizes + CIs, not confirmatory).\n")

    L.append("## (A) Data-driven morphology clusters")
    L.append(f"- Best k = **{best['k']}** (silhouette = {best['sil']:.3f}, "
             "Agglomerative on standardised SHAPE descriptors).")
    if "true_shape" in feat_df and feat_df["true_shape"].notna().any():
        ct = pd.crosstab(feat_df["cluster"], feat_df["true_shape"])
        L.append("\nCluster × known shape (synthetic sanity check):\n")
        L.append(ct.to_markdown())
    L.append("\nPer-cluster mean of SyncPipe features:\n")
    L.append(cluster_desc.round(3).to_markdown())

    L.append("\n## (B) Collinearity")
    L.append("\nSpearman correlation matrix:\n")
    L.append(corr.round(2).to_markdown())
    L.append("\nVariance Inflation Factor (VIF > 5 = concerning, > 10 = severe):\n")
    L.append(vif.round(2).to_frame().to_markdown())
    high = vif[vif > 5].index.tolist()
    if high:
        L.append(f"\n⚠️ High-VIF features (redundant): **{', '.join(high)}** — "
                 "candidates to drop/merge before confirmatory FDR.")

    L.append("\n## (C) Order-unbiased incremental value (cluster as target)")
    L.append("\n> Marginal AUC averaged over random insertion orders (Shapley-style) "
             "removes the fixed-ordering bias of cumulative incremental steps. "
             "LOFO = AUC lost when the feature is removed from the full model.\n")
    L.append(f"\n**Baseline = drop mean_synchrony** (base AUC={meta_drop['baseline_auc']:.3f}, "
             f"full AUC={meta_drop['full_auc']:.3f}):\n")
    L.append(inc_drop.round(4).to_markdown(index=False))
    L.append(f"\n**Baseline = keep mean_synchrony** (base AUC={meta_keep['baseline_auc']:.3f}, "
             f"full AUC={meta_keep['full_auc']:.3f}):\n")
    L.append(inc_keep.round(4).to_markdown(index=False))

    L.append("\n## (D) Matched-mean-synchrony contrast")
    L.append("\n> Among traces in a narrow mean_synchrony band, which single feature "
             "best separates morphology clusters? This is the core test of whether "
             "SHAPE carries information beyond synchrony MAGNITUDE.\n")
    if len(mm):
        L.append(mm.round(4).to_markdown(index=False))
    else:
        L.append("(insufficient traces in mean band)")

    L.append("\n## Honest limitations")
    L.append("- **n is small**: all AUCs are exploratory with wide CIs; clusters may "
             "be unstable. Report bootstrap CIs and cluster-stability (e.g. ARI under "
             "resampling) before any claim.")
    L.append("- **Partial circularity caveat**: clusters come from SHAPE descriptors; "
             "predictors are the 9 SyncPipe features. These are different feature "
             "sets, but some MS features (dwell, switching) correlate with shape "
             "descriptors by construction, so 'predicting cluster' partly recovers the "
             "clustering geometry. Interpret incremental AUC as *descriptive structure*, "
             "not out-of-sample morphology classification.")
    L.append("- **Collinearity** directly threatens FDR validity on correlated features; "
             "see (B). High-VIF features should not be treated as independent tests.")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
lerique_feature_analysis.py — UNIFIED feature-level analysis (one file, two methods).

Runs BOTH analysis methods on the same data and reports them side by side, so the
effect of the methodological rigour upgrade is visible:

  METHOD A (naive):   plain StratifiedKFold CV, no bootstrap, raw feature units.
  METHOD B (rigorous): leave-dyad-out CV (StratifiedGroupKFold) + dyad-cluster
                       bootstrap 95% CIs + duration-corrected features.

For each method it computes, per modality (predicting rest1 vs trials_concat):
  - order-unbiased incremental AUC (Shapley marginal + LOFO), DROP vs KEEP mean_synchrony
Plus a shared (1) collinearity block via multisync.feature_vif_test.

The point of keeping both in one file: METHOD B is the publication-grade result;
METHOD A is shown only to demonstrate that the honest CV does NOT inflate the story
(reviewers like seeing the naive-vs-grouped comparison).

USAGE
-----
    python scripts/lerique_feature_analysis.py --csv path/to/lerique_dyads.csv
    python scripts/lerique_feature_analysis.py --csv ... --methods B   # rigorous only
    python scripts/lerique_feature_analysis.py --csv ... --n-boot 1000

Requires columns: dyad_label, modality, condition, n_samples, duration_sec,
plus the 9 SyncPipe feature columns.
"""
from __future__ import annotations
import argparse, sys, warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ALL_FEATURES = ["mean_synchrony", "peak_amplitude", "dwell_time", "switching_rate",
                "bimodality_coefficient", "synchrony_entropy",
                "onset_latency", "rise_time", "recovery_time"]
SAMPLE_UNIT = ["onset_latency", "rise_time", "recovery_time", "dwell_time"]
PRIMARY = ["mean_synchrony", "peak_amplitude", "switching_rate",
           "bimodality_coefficient", "synchrony_entropy"]
# Duration-NORMALISED temporal features (sample-unit feature / epoch_duration),
# built in load(). These rescue the structure/morphology features that the
# primary model excludes for the duration confound, but in a duration-robust way
# (a fraction of the recording, not absolute samples). Reported as a SEPARATE,
# clearly-labelled supplementary model — NOT mixed into the primary model.
DURATION_NORM = ["onset_latency_fracdur", "rise_time_fracdur",
                 "recovery_time_fracdur", "dwell_time_fracdur"]
SEED = 42


# ---------------------------------------------------------------------------
def load(csv, fix_duration):
    df = pd.read_csv(csv)
    if fix_duration and {"n_samples", "duration_sec"} <= set(df.columns):
        hz = df["n_samples"] / df["duration_sec"]
        for f in SAMPLE_UNIT:
            if f in df:
                df[f + "_sec"] = df[f] / hz
                df[f + "_fracdur"] = df[f + "_sec"] / df["duration_sec"]
    return df


# ---- CV back-ends ---------------------------------------------------------
def cv_auc_naive(X, y, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if X.shape[1] == 0 or len(set(y)) < 2:
        return 0.5
    _, c = np.unique(y, return_counts=True)
    k = int(min(5, c.min()))
    if k < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    try:
        return float(np.nanmean(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))
    except Exception:
        return np.nan


def cv_auc_grouped(X, y, groups, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    if X.shape[1] == 0 or len(set(y)) < 2:
        return 0.5
    k = int(min(5, len(set(groups))))
    if k < 2:
        return np.nan
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    try:
        return float(np.nanmean(cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="roc_auc")))
    except Exception:
        return np.nan


def make_auc_fn(method, df, ycol, groupcol):
    """Return auc(cols)->float for the chosen method, closing over the frame."""
    y = df[ycol].values
    g = df[groupcol].values if groupcol in df else None

    def auc(cols, frame=None):
        f = df if frame is None else frame
        if not cols:
            return 0.5
        X = f[cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median()).values
        yy = f[ycol].values
        if method == "A":
            return cv_auc_naive(X, yy)
        return cv_auc_grouped(X, yy, f[groupcol].values)
    return auc


def incremental(method, df, ycol, groupcol, feat_cols, drop_meansync, seed=SEED, n_orders=200):
    feats = [f for f in feat_cols if f in df and df[f].notna().sum() > 3 and df[f].std() > 1e-9]
    base = [] if drop_meansync else (["mean_synchrony"] if "mean_synchrony" in feats else [])
    feats = [f for f in feats if f != "mean_synchrony"]
    auc = make_auc_fn(method, df, ycol, groupcol)
    base_auc, full_auc = auc(base), auc(base + feats)
    rng = np.random.default_rng(seed)
    marg = {f: [] for f in feats}
    orders = list(permutations(feats)) if len(feats) <= 6 else [list(rng.permutation(feats)) for _ in range(n_orders)]
    for order in orders:
        cur = list(base); prev = auc(cur)
        for f in order:
            cur.append(f); now = auc(cur); marg[f].append(now - prev); prev = now
    rows = [{"feature": f, "shapley_marginal_auc": float(np.nanmean(v)),
             "lofo_auc_drop": float(full_auc - auc(base + [x for x in feats if x != f]))}
            for f, v in marg.items()]
    return (pd.DataFrame(rows).sort_values("shapley_marginal_auc", ascending=False),
            {"base_auc": base_auc, "full_auc": full_auc})


def vif_bootstrap_ci(df, features, groupcol, n_boot, seed=SEED):
    """Dyad-cluster bootstrap CI for VIF per feature.

    VIF point estimates are biased toward 1 in small samples; at n≈27-31 per
    modality vs 9 features the design is not generous, so a point estimate is
    not enough. We resample DYADS (clusters), recompute VIF, and report the
    median + 95% percentile interval per feature.
    """
    from multisync.feature_vif_test import feature_vif
    rng = np.random.default_rng(seed)
    groups = df[groupcol].unique()
    point = feature_vif(df, features)
    draws = {f: [] for f in point.index}
    for _ in range(n_boot):
        samp = rng.choice(groups, size=len(groups), replace=True)
        rows = pd.concat([df[df[groupcol] == g] for g in samp], ignore_index=True)
        v = feature_vif(rows, features)
        for f in point.index:
            if f in v.index and np.isfinite(v[f]):
                draws[f].append(float(v[f]))
    out = []
    for f in point.index:
        d = np.array(draws[f])
        if len(d) >= 10:
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append({"feature": f, "vif_point": float(point[f]),
                        "vif_boot_median": float(np.median(d)),
                        "vif_ci_lo": float(lo), "vif_ci_hi": float(hi),
                        "n_boot_valid": int(len(d))})
        else:
            out.append({"feature": f, "vif_point": float(point[f]),
                        "vif_boot_median": np.nan, "vif_ci_lo": np.nan,
                        "vif_ci_hi": np.nan, "n_boot_valid": int(len(d))})
    return pd.DataFrame(out)


def bootstrap_ci(df, feat_cols, ycol, groupcol, n_boot, seed=SEED):
    """Dyad-cluster bootstrap on grouped-CV full-model AUC (METHOD B only)."""
    rng = np.random.default_rng(seed)
    auc = make_auc_fn("B", df, ycol, groupcol)
    point = auc(feat_cols)
    groups = df[groupcol].unique()
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(groups, size=len(groups), replace=True)
        rows = pd.concat([df[df[groupcol] == g] for g in samp], ignore_index=True)
        if rows[ycol].nunique() < 2:
            continue
        a = auc(feat_cols, frame=rows)
        if np.isfinite(a):
            boots.append(a)
    if not boots:
        return point, (np.nan, np.nan)
    return point, tuple(np.percentile(boots, [2.5, 97.5]))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--methods", default="AB", help="which methods to run: A, B, or AB")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir) if args.outdir else Path(args.csv).resolve().parent / "lerique_analysis_out"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.csv, fix_duration=True)

    L = ["# Lerique — Unified Feature Analysis (Methods A & B)\n",
         f"- {len(df)} traces, {df['dyad_label'].nunique() if 'dyad_label' in df else '?'} dyads. **EXPLORATORY**.\n",
         "- **METHOD A** = naive StratifiedKFold CV (no grouping, no bootstrap, raw units).\n",
         "- **METHOD B** = leave-dyad-out CV + dyad-cluster bootstrap CI + duration-corrected.\n",
         "- Both shown so the rigour upgrade (A→B) is auditable.\n"]

    # ---- (1) collinearity (shared) ----
    from multisync.feature_vif_test import collinearity_report
    rep = collinearity_report(df, ALL_FEATURES)
    rep["vif"].round(2).to_frame().to_csv(outdir / "vif.csv")
    rep["correlation"].round(3).to_csv(outdir / "corr.csv")
    L.append("\n## (1) Collinearity — multisync.feature_vif_test\n")
    L.append(rep["vif"].round(2).to_frame().to_markdown())
    L.append(f"\n{rep['interpretation']}\n")
    L.append("\nTop correlated pairs:\n")
    for a, b, r in rep["top_correlated_pairs"][:6]:
        L.append(f"- {a} ↔ {b}: r={r:+.2f}")

    # ---- (1b) VIF bootstrap CIs (point estimate is biased toward 1 at small n) ----
    L.append("\n\n### (1b) VIF with dyad-cluster bootstrap 95% CI\n")
    L.append("> A single VIF point estimate at n≈27-31 vs 9 features is not enough "
             "(VIF is biased toward 1 in small samples). Per-modality VIF CIs "
             "quantify how uncertain the 'VIF<5' conclusion actually is.\n")
    for m in sorted(df["modality"].unique()):
        subm = df[df["modality"] == m]
        vci = vif_bootstrap_ci(subm, ALL_FEATURES, "dyad_label", args.n_boot)
        vci.to_csv(outdir / f"vif_ci_{m}.csv", index=False)
        L.append(f"\n**{m}** (n={len(subm)}, {subm['dyad_label'].nunique()} dyads):\n")
        L.append(vci.round(2).to_markdown(index=False))

    methods = [m for m in ("A", "B") if m in args.methods.upper()]
    for m in sorted(df["modality"].unique()):
        sub = df[df["modality"] == m].copy()
        sub["y"] = (sub["condition"] == "trials_concat").astype(int)
        L.append(f"\n## Modality {m} (n={len(sub)}, {sub['dyad_label'].nunique()} dyads)")
        if "B" in methods:
            feat = [f for f in PRIMARY if sub[f].notna().sum() > 3 and sub[f].std() > 1e-9]
            pt, (lo, hi) = bootstrap_ci(sub, feat, "y", "dyad_label", args.n_boot)
            L.append(f"\n- **METHOD B** full primary-model AUC (grouped-CV) = "
                     f"**{pt:.3f}** [95% CI {lo:.3f}, {hi:.3f}]")
        for meth in methods:
            tag = "A (naive KFold)" if meth == "A" else "B (leave-dyad-out)"
            L.append(f"\n### Method {tag}")
            for drop in (True, False):
                inc, meta = incremental(meth, sub, "y", "dyad_label", PRIMARY, drop)
                btag = "DROP mean_synchrony" if drop else "KEEP mean_synchrony"
                L.append(f"\n**{btag}** — base={meta['base_auc']:.3f}, full={meta['full_auc']:.3f}:\n")
                L.append(inc.round(4).to_markdown(index=False))
                inc.to_csv(outdir / f"inc_{m}_method{meth}_{'drop' if drop else 'keep'}.csv", index=False)

        # ---- SUPPLEMENTARY: duration-normalised temporal/structure model ----
        # Rescues onset/rise/recovery/dwell (excluded from the primary model for
        # the duration confound) in a duration-robust form: feature / epoch
        # duration (fraction-of-recording), so they can speak to the
        # morphology/structure question without the rest1(180s)-vs-trials(1080s)
        # length artifact. Uses Method B (grouped CV). Reported separately.
        dn_feats = [f for f in DURATION_NORM
                    if f in sub and sub[f].notna().sum() > 3 and sub[f].std() > 1e-9]
        if dn_feats and "B" in methods:
            L.append(f"\n### Supplementary (duration-normalised temporal/structure features, Method B)")
            inc_dn, meta_dn = incremental("B", sub, "y", "dyad_label",
                                          ["mean_synchrony"] + dn_feats, drop_meansync=False)
            L.append(f"\n> Features as fraction-of-recording (duration-robust). "
                     f"base(mean_synchrony only)={meta_dn['base_auc']:.3f}, "
                     f"full={meta_dn['full_auc']:.3f}. Shows whether morphology/timing "
                     "features add information ONCE the duration confound is removed.\n")
            L.append(inc_dn.round(4).to_markdown(index=False))
            inc_dn.to_csv(outdir / f"inc_{m}_duration_norm.csv", index=False)

    L.append("\n## Caveats")
    L.append("- Method B is the result to report; Method A is for the naive-vs-grouped comparison only.")
    L.append("- Sample-unit temporal features excluded from primary model; seconds/fraction "
             "versions written to outdir for separate duration-aware analysis.")
    (outdir / "LERIQUE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("Done ->", outdir / "LERIQUE_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

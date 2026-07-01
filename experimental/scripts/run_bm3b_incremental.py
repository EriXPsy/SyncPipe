"""
BM3b: Incremental validity via hierarchical regression.
=======================================================
Motivation (2026-06-08)
-----------------------
BM3 argued "features carry information beyond mean_synchrony" using only
raw correlations |ρ(feature, mean)|.  A reviewer's real question is
sharper: AFTER controlling for mean_synchrony, does a feature still
predict the experimental condition?  This is the incremental-validity
test and it is the direct rebuttal to "SyncPipe adds nothing beyond the
average."

Design — TWO targets, so the test is not hostage to one weak label.

Target 1 (Andersen, n=300): is_close ∈ {0,1}.  NOTE: mean_synchrony
  itself barely predicts is_close (CV-AUC≈0.57), so this is a WEAK-
  SIGNAL target — useful only to show features add nothing when even
  the baseline is near chance (an honest null).

Target 2 (Lerique, per-record): rest1 vs trials_concat.  This is a
  STRONG synchrony contrast (the rest/task effect), so it is the
  proper venue to ask whether temporal features add predictive power
  ON TOP OF mean_synchrony.

For each (dataset, feature):
  - Baseline model:    y ~ mean_synchrony
  - Augmented model:   y ~ mean_synchrony + feature
  - Incremental: ΔAUC (5-fold CV) + likelihood-ratio test (χ²_1).

A feature with ΔAUC>0 and LR p<0.05 carries predictive information
mean_synchrony does NOT already contain.

Output: artifacts/bm3b_incremental_validity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ANDERSEN = r'<OSF_ROOT>\Andersen-hj4k6\multisync_results\multisync_andersen_full.csv'
LERIQUE = (r'<REPO>'
           r'\artifacts\realtest\lerique_2024\per_record_features.csv')

FEATURES = ["peak_amplitude", "dwell_time", "recovery_time", "onset_latency",
            "switching_rate", "rise_time", "synchrony_entropy"]
BASE = "mean_synchrony"


def _deviance(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _cv_auc(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def _lr_test(X_base, X_aug, y):
    """In-sample likelihood-ratio test (Δdeviance ~ χ²_1)."""
    sc_b = StandardScaler().fit(X_base)
    sc_a = StandardScaler().fit(X_aug)
    clf_b = LogisticRegression(max_iter=1000).fit(sc_b.transform(X_base), y)
    clf_a = LogisticRegression(max_iter=1000).fit(sc_a.transform(X_aug), y)
    dev_b = _deviance(y, clf_b.predict_proba(sc_b.transform(X_base))[:, 1])
    dev_a = _deviance(y, clf_a.predict_proba(sc_a.transform(X_aug))[:, 1])
    chi2 = dev_b - dev_a
    p = stats.chi2.sf(max(chi2, 0.0), df=1)
    return chi2, p


def analyse(name, df, y):
    """Run baseline-vs-augmented incremental validity on one dataset."""
    base_auc = _cv_auc(df[[BASE]].to_numpy(), y)
    print(f"\n=== {name} ===")
    print(f"n={len(df)}  prevalence={y.mean():.3f}  "
          f"baseline AUC (mean_synchrony only) = {base_auc:.3f}")
    print(f"{'feature':18s} {'aug_AUC':>8s} {'dAUC':>7s} "
          f"{'LR_chi2':>8s} {'LR_p':>9s} {'rho(feat,mean)':>15s}")
    rows = []
    for f in FEATURES:
        if f not in df.columns:
            continue
        d = df[[BASE, f]].dropna()
        if len(d) < 30 or d[f].nunique() < 5:
            continue
        yy = y[d.index.to_numpy()]
        if len(np.unique(yy)) < 2:
            continue
        aug_auc = _cv_auc(d.to_numpy(), yy)
        chi2, p = _lr_test(d[[BASE]].to_numpy(), d.to_numpy(), yy)
        rho = stats.spearmanr(d[f], d[BASE])[0]
        d_auc = aug_auc - base_auc
        rows.append({"dataset": name, "feature": f, "n": len(d),
                     "baseline_auc": base_auc, "augmented_auc": aug_auc,
                     "delta_auc": d_auc, "lr_chi2": chi2, "lr_p": p,
                     "rho_feat_mean": rho})
        print(f"{f:18s} {aug_auc:>8.3f} {d_auc:>+7.3f} "
              f"{chi2:>8.2f} {p:>9.2e} {rho:>+15.3f}")
    sig = [r["feature"] for r in rows if r["lr_p"] < 0.05 and r["delta_auc"] > 0]
    print(f"incremental-valid features (LR p<0.05 AND ΔAUC>0): {sig}")
    return rows


def main():
    all_rows = []

    # Target 1: Andersen is_close (weak-signal control)
    a = pd.read_csv(ANDERSEN)
    if "status" in a.columns:
        a = a[a["status"] == "ok"].copy()
    a = a[["is_close", BASE] + FEATURES].apply(pd.to_numeric, errors="coerce")
    a = a.dropna(subset=["is_close", BASE]).reset_index(drop=True)
    all_rows += analyse("Andersen:is_close", a, a["is_close"].astype(int).to_numpy())

    # Target 2: Lerique rest1 vs trials_concat (strong synchrony contrast)
    l = pd.read_csv(LERIQUE)
    l = l[l["condition_unit"].isin(["rest1", "trials_concat"])].copy()
    l["y"] = (l["condition_unit"] == "trials_concat").astype(int)
    num = l[[BASE] + FEATURES].apply(pd.to_numeric, errors="coerce")
    num["y"] = l["y"].to_numpy()
    num = num.dropna(subset=[BASE]).reset_index(drop=True)
    all_rows += analyse("Lerique:rest_vs_task", num, num["y"].to_numpy())

    out = Path(REPO) / "artifacts" / "bm3b_incremental_validity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

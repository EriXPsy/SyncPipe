"""
GT/dim: Supervised compression (PLS-DA) for the dimensional model.
==================================================================
Motivation (2026-06-08)
-----------------------
DIMENSIONAL_MODEL.md derives 3 dimensions from UNSUPERVISED PCA, then
invokes the information-bottleneck / minimal-sufficient-representation
framing.  These are mathematically inconsistent: IB sufficiency is
defined w.r.t. a target Y, but PCA ignores Y.  This script replaces the
core evidence with a SUPERVISED compression (PLS-DA), which is what
"sufficient for predicting the condition" actually requires.

We:
  1. De-confound each feature against mean_synchrony (residualize).
  2. Fit PLS-DA: X = 7 de-confounded residual features, Y = condition.
  3. Report variance explained and the loadings of the first 2-3
     latent components, and check whether they line up with the
     a-priori dimensions (INTENSITY / STRUCTURE / TIMING).
  4. Contrast with unsupervised PCA on the same residuals.

Dataset: Lerique rest1 vs trials_concat (strong synchrony contrast,
n=176), the venue where a supervised target exists.

Output: artifacts/dim_pls_lerique.csv  (loadings)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

LERIQUE = (r'<REPO>'
           r'\artifacts\realtest\lerique_2024\per_record_features.csv')

FEATURES = ["peak_amplitude", "recovery_time", "onset_latency", "rise_time",
            "switching_rate", "dwell_time", "synchrony_entropy"]
BASE = "mean_synchrony"


def deconfound(df):
    """Residualize each feature against mean_synchrony (linear)."""
    res = pd.DataFrame(index=df.index)
    m = df[BASE].to_numpy()
    M = np.column_stack([np.ones_like(m), m])
    for f in FEATURES:
        y = df[f].to_numpy()
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        res[f] = y - M @ beta
    return res


def main():
    df = pd.read_csv(LERIQUE)
    df = df[df["condition_unit"].isin(["rest1", "trials_concat"])].copy()
    df["y"] = (df["condition_unit"] == "trials_concat").astype(int)
    cols = [BASE] + FEATURES
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols).reset_index(drop=True)

    res = deconfound(df)
    Xs = StandardScaler().fit_transform(res[FEATURES].to_numpy())
    y = df["y"].to_numpy().astype(float)

    # ── Supervised: PLS-DA ──
    pls = PLSRegression(n_components=3)
    pls.fit(Xs, y)
    # x_scores explain X variance; compute fraction per component
    x_var = np.var(pls.x_scores_, axis=0)
    x_var_frac = x_var / np.sum(np.var(Xs, axis=0))
    # correlation of each PLS component score with y (supervised relevance)
    comp_y_corr = [np.corrcoef(pls.x_scores_[:, i], y)[0, 1] for i in range(3)]

    print("=== PLS-DA (supervised, Lerique rest vs task, de-confounded) ===")
    print(f"n={len(df)}")
    load = pd.DataFrame(pls.x_loadings_, index=FEATURES,
                        columns=[f"LV{i+1}" for i in range(3)])
    print("\nX-loadings (which features define each latent variable):")
    print(load.round(2).to_string())
    print("\nX-variance fraction per LV:", np.round(x_var_frac, 3).tolist())
    print("corr(LV_score, condition) per LV:", np.round(comp_y_corr, 3).tolist())

    # ── Unsupervised: PCA on same residuals (for contrast) ──
    pca = PCA(n_components=3).fit(Xs)
    pca_load = pd.DataFrame(pca.components_.T, index=FEATURES,
                            columns=[f"PC{i+1}" for i in range(3)])
    print("\n=== PCA (unsupervised, same residuals) — for contrast ===")
    print(pca_load.round(2).to_string())
    print("PCA explained variance ratio:",
          np.round(pca.explained_variance_ratio_, 3).tolist())

    out = Path(REPO) / "artifacts" / "dim_pls_lerique.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    load.join(pca_load).to_csv(out)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

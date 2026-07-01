"""
multisync.feature_vif_test — feature collinearity / VIF diagnostics.

Scope: this module ONLY tests feature collinearity (correlation + Variance
Inflation Factor). It is deliberately narrow — named ``feature_vif_test`` to
avoid being mistaken for a general "diagnostics" grab-bag.

Distinct from :mod:`multisync.qc` (raw-signal quality) and
:mod:`multisync.association` (coupling). It operates on the EXTRACTED feature
matrix and answers: "are these features redundant, and is it safe to treat them
as independent tests in an FDR family?"

Public API
----------
feature_correlation(df, features, method="spearman") -> pd.DataFrame
feature_vif(df, features) -> pd.Series
collinearity_report(df, features, ...) -> dict
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

__all__ = ["feature_correlation", "feature_vif", "collinearity_report",
           "VIF_CONCERN", "VIF_SEVERE"]

# Conventional VIF cutoffs (rule-of-thumb; cf. O'Brien, 2007).
VIF_CONCERN = 5.0
VIF_SEVERE = 10.0


def _usable(df: pd.DataFrame, features: Sequence[str], min_n: int = 4) -> List[str]:
    out = []
    for c in features:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            if col.notna().sum() >= min_n and col.std(skipna=True) > 1e-9:
                out.append(c)
    return out


def feature_correlation(df: pd.DataFrame, features: Sequence[str],
                        method: str = "spearman") -> pd.DataFrame:
    """Pairwise correlation matrix among ``features`` (Spearman by default).

    Spearman is preferred for synchrony features: bounded, skewed, often
    non-linearly related.
    """
    used = _usable(df, features)
    sub = df[used].apply(pd.to_numeric, errors="coerce")
    return sub.corr(method=method)


def feature_vif(df: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """Variance Inflation Factor per feature.

    VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing feature j on all
    other features. VIF > 5 concerning, > 10 severe. Computed on standardised,
    complete-case rows. Returns NaN for a feature if the design is rank-deficient.
    """
    used = _usable(df, features)
    sub = df[used].apply(pd.to_numeric, errors="coerce").dropna()
    vif: Dict[str, float] = {c: float("nan") for c in used}
    if len(sub) <= len(used) + 1 or len(used) < 2:
        return pd.Series(vif, name="VIF")
    Z = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-12)
    for c in used:
        y = Z[c].values
        X = Z.drop(columns=[c]).values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            vif[c] = float(1.0 / max(1e-9, 1.0 - r2))
        except np.linalg.LinAlgError:
            vif[c] = float("nan")
    return pd.Series(vif, name="VIF")


def collinearity_report(df: pd.DataFrame, features: Sequence[str],
                        method: str = "spearman",
                        vif_concern: float = VIF_CONCERN,
                        vif_severe: float = VIF_SEVERE) -> Dict[str, object]:
    """Combined collinearity diagnostic: correlation + VIF + flags.

    Use BEFORE treating a feature set as an FDR family: high-VIF features are
    not independent tests, which can invalidate naive multiplicity correction.
    """
    corr = feature_correlation(df, features, method=method)
    vif = feature_vif(df, features)

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    concern = sorted(vif[(vif >= vif_concern) & (vif < vif_severe)].index.tolist())
    severe = sorted(vif[vif >= vif_severe].index.tolist())

    return {
        "correlation": corr,
        "vif": vif,
        "top_correlated_pairs": pairs[:10],
        "vif_concern": concern,
        "vif_severe": severe,
        "interpretation": (
            f"{len(severe)} feature(s) with severe VIF (>={vif_severe}), "
            f"{len(concern)} concerning (>={vif_concern}). High-VIF features "
            "should not be treated as independent tests in an FDR family."
        ),
    }

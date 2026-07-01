"""
L2 Between-Condition Null — Dyad-Paired Permutation + BH-FDR
==============================================================

L2 testing asks: "Does condition A reliably differ from condition B
across dyads, for each SyncPipe feature?"

Unlike L0 (signal-level) and L1 (WCC-level), L2 operates on the
*feature table* — one scalar value per feature per (dyad, condition).
The null model resamples condition labels within each dyad, preserving
the dyad-pairing structure while breaking any systematic condition
effect.

Levels summary:
  L0 — Does synchrony EXIST above noise?         (signal-level IAAFT)
  L1 — Is the temporal STRUCTURE real?            (WCC-level IAAFT)
  L2 — Do conditions RELIABLY DIFFER across dyads? (dyad-paired permutation)

Method
------
1. For each feature k, compute observed Δ_k = median(C1) - median(C2)
2. Permute condition labels within each dyad (swap C1↔C2), recompute Δ
3. Phipson-Smyth p-value: p = (|Δ_perm| >= |Δ_obs| + 1) / (n_perm + 1)
4. BH-FDR across features (family = 8 confirmatory)
5. Cohen's d = Δ_obs / SD(Δ_perm)

This is the correct between-condition null for dyadic designs where
each dyad contributes data to both conditions (paired design).
For unpaired designs (where different dyads are in different conditions),
use a two-sample permutation instead.

References
----------
Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never
  be zero. *Statistical Applications in Genetics and Molecular Biology*,
  9(1), Article 39.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
  rate. *Journal of the Royal Statistical Society: Series B*, 57(1),
  289–300.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ── Re-use BH-FDR from pgt1_intensity ──────────────────────────────────────
def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    m_fin = np.isfinite(p)
    if not m_fin.any():
        return out
    pm = p[m_fin]
    n = len(pm)
    order = np.argsort(pm)
    adj = pm[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    fdr = np.empty_like(pm)
    fdr[order] = adj
    out[m_fin] = np.minimum(fdr, 1.0)
    return out


def _phipson_smyth_p(obs_diff: float, null_diffs: np.ndarray) -> float:
    """Phipson-Smyth (2010) unbiased two-tailed permutation p-value.

    p = (|null >= |obs|| + 1) / (n_null + 1)
    """
    null_diffs = np.asarray(null_diffs, dtype=float)
    finite = np.isfinite(null_diffs)
    null_fin = null_diffs[finite]
    n = len(null_fin)
    if n < 10:
        return 1.0
    n_ge = np.sum(np.abs(null_fin) >= np.abs(obs_diff))
    return float((n_ge + 1) / (n + 1))


# ═══════════════════════════════════════════════════════════════════════════
# Public dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class L2Result:
    """L2 between-condition test result.

    Attributes
    ----------
    feature : str
        Feature name.
    condition_a, condition_b : str
        The two conditions being compared.
    observed_diff : float
        Median(A) - Median(B) across dyads.
    null_mean : float
        Mean of the permutation null distribution for the difference.
    null_sd : float
        Standard deviation of the permutation null distribution.
    p_raw : float
        Phipson-Smyth uncorrected p-value.
    p_fdr : float
        BH-FDR corrected p-value (across features).
    significant_05 : bool
        True if p_fdr < 0.05.
    cohens_d : float
        Effect size: observed_diff / null_sd.
    n_dyads : int
        Number of dyads with data in both conditions.
    """
    feature: str
    condition_a: str
    condition_b: str
    observed_diff: float
    null_mean: float
    null_sd: float
    p_raw: float
    p_fdr: float
    significant_05: bool
    cohens_d: float
    n_dyads: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Core L2 function
# ═══════════════════════════════════════════════════════════════════════════

def between_condition_fdr(
    df: pd.DataFrame,
    condition_col: str = "condition",
    dyad_col: str = "dyad_label",
    feature_cols: Optional[Sequence[str]] = None,
    n_permutations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    condition_values: Optional[Tuple[str, str]] = None,
) -> Dict[str, Union[List[L2Result], L2Result]]:
    """L2 between-condition permutation test with BH-FDR correction.

    Compares two conditions using dyad-paired permutation. Each dyad
    must have exactly one observation in each condition. The null
    model randomly flips the condition label within each dyad.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format feature table. Must contain ``dyad_col``,
        ``condition_col``, and all ``feature_cols``.
    condition_col : str
        Column name for condition labels (default "condition").
    dyad_col : str
        Column name for dyad/pair identifier (default "dyad_label").
    feature_cols : sequence of str, optional
        Which feature columns to test. Defaults to the SSoT
        confirmatory 6 + bimodality_coefficient + mean_synchrony.
    n_permutations : int
        Number of permutation iterations (default 10000).
    seed : int
        RNG seed for reproducibility.
    alpha : float
        Significance threshold (default 0.05).
    condition_values : tuple (str, str), optional
        Which two conditions to compare, e.g. ("rest1", "trials_concat").
        If None, uses the first two unique values in ``condition_col``.

    Returns
    -------
    dict
        ``"per_feature"``: list of L2Result objects (one per feature).
        ``"n_dyads"``: int, number of dyads.
        ``"n_significant"``: int, number of features with p_fdr < alpha.
        ``"condition_a"``, ``"condition_b"``: str.
        ``"n_permutations"``: int.
        ``"summary_df"``: pd.DataFrame with all per-feature results.

    Raises
    ------
    ValueError
        If fewer than 2 conditions are present, or if the specified
        condition_values are not found.
    """
    # ── Validate ───────────────────────────────────────────────────────
    if feature_cols is None:
        feature_cols = [
            "onset_latency", "rise_time", "peak_amplitude",
            "recovery_time", "dwell_time", "switching_rate",
            "bimodality_coefficient", "mean_synchrony",
        ]

    # Only use columns actually present
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError(f"No feature columns found in df. Looking for: {feature_cols}")

    unique_conditions = sorted(df[condition_col].dropna().unique())

    if condition_values is None:
        if len(unique_conditions) < 2:
            raise ValueError(
                f"Need at least 2 conditions, found {len(unique_conditions)}: "
                f"{unique_conditions}"
            )
        condition_a, condition_b = unique_conditions[0], unique_conditions[1]
    else:
        condition_a, condition_b = condition_values
        for c in (condition_a, condition_b):
            if c not in unique_conditions:
                raise ValueError(
                    f"Condition '{c}' not found in data. Available: {unique_conditions}"
                )

    # ── Build paired dyad table ─────────────────────────────────────────
    subset = df[[dyad_col, condition_col] + feature_cols].dropna(
        subset=[dyad_col, condition_col]
    )

    df_a = subset[subset[condition_col] == condition_a].set_index(dyad_col)
    df_b = subset[subset[condition_col] == condition_b].set_index(dyad_col)

    # Keep only dyads present in BOTH conditions
    common_dyads = df_a.index.intersection(df_b.index)
    if len(common_dyads) < 4:
        raise ValueError(
            f"Only {len(common_dyads)} dyads have data in both "
            f"'{condition_a}' and '{condition_b}'. Need >= 4."
        )

    df_a = df_a.loc[common_dyads, feature_cols]
    df_b = df_b.loc[common_dyads, feature_cols]

    # ── Aggregate duplicate dyad rows (multi-trial/multi-stimulus designs) ──
    # When a dyad × condition has multiple rows (e.g. Han's 4 stimuli per
    # dyad), aggregate to one scalar per dyad. This ensures permutation
    # operates at the correct unit of analysis (dyad = observation).
    has_duplicates = df_a.index.has_duplicates or df_b.index.has_duplicates
    if has_duplicates:
        df_a = df_a.groupby(df_a.index).mean()
        df_b = df_b.groupby(df_b.index).mean()
        # Re-intersect after aggregation
        common_dyads = df_a.index.intersection(df_b.index)
        df_a = df_a.loc[common_dyads]
        df_b = df_b.loc[common_dyads]

    n_dyads = len(common_dyads)

    # ── Permutation test per feature ────────────────────────────────────
    rng = np.random.default_rng(seed)
    results: List[L2Result] = []

    for feat in feature_cols:
        vals_a = df_a[feat].to_numpy(dtype=float)
        vals_b = df_b[feat].to_numpy(dtype=float)

        # Remove dyads where either condition has NaN
        valid = np.isfinite(vals_a) & np.isfinite(vals_b)
        if valid.sum() < 4:
            results.append(L2Result(
                feature=feat,
                condition_a=condition_a,
                condition_b=condition_b,
                observed_diff=np.nan,
                null_mean=np.nan,
                null_sd=np.nan,
                p_raw=1.0,
                p_fdr=1.0,
                significant_05=False,
                cohens_d=np.nan,
                n_dyads=int(valid.sum()),
            ))
            continue

        a_fin = vals_a[valid]
        b_fin = vals_b[valid]
        n = len(a_fin)

        # Observed difference (median paired)
        observed_diff = float(np.median(a_fin - b_fin))

        # Permutation: within each dyad, randomly swap condition labels
        null_diffs = np.empty(n_permutations)
        for i in range(n_permutations):
            flip = rng.choice([-1, 1], size=n)
            perm_diff = np.median(flip * (a_fin - b_fin))
            null_diffs[i] = float(perm_diff)

        null_mean = float(np.mean(null_diffs))
        null_sd = float(np.std(null_diffs, ddof=1))
        p_raw = _phipson_smyth_p(observed_diff, null_diffs)
        cohens_d = observed_diff / null_sd if null_sd > 1e-10 else np.nan

        results.append(L2Result(
            feature=feat,
            condition_a=condition_a,
            condition_b=condition_b,
            observed_diff=observed_diff,
            null_mean=null_mean,
            null_sd=null_sd,
            p_raw=p_raw,
            p_fdr=np.nan,  # filled after BH-FDR
            significant_05=False,  # filled after BH-FDR
            cohens_d=cohens_d,
            n_dyads=int(valid.sum()),
        ))

    # ── BH-FDR across features ─────────────────────────────────────────
    p_raw_arr = np.array([r.p_raw for r in results], dtype=float)
    p_fdr_arr = _bh_fdr(p_raw_arr)

    for i, r in enumerate(results):
        r.p_fdr = float(p_fdr_arr[i]) if np.isfinite(p_fdr_arr[i]) else 1.0
        r.significant_05 = bool(r.p_fdr < alpha)

    n_significant = sum(1 for r in results if r.significant_05)

    # ── Build summary dataframe ────────────────────────────────────────
    summary_df = pd.DataFrame([
        {
            "feature": r.feature,
            "observed_diff": r.observed_diff,
            "null_mean": r.null_mean,
            "null_sd": r.null_sd,
            "p_raw": r.p_raw,
            "p_fdr": r.p_fdr,
            "significant_05": r.significant_05,
            "cohens_d": r.cohens_d,
            "n_dyads": r.n_dyads,
        }
        for r in results
    ])

    return {
        "per_feature": results,
        "n_dyads": n_dyads,
        "n_significant": n_significant,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_permutations": n_permutations,
        "summary_df": summary_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: L2 within-modality
# ═══════════════════════════════════════════════════════════════════════════

def between_condition_by_modality(
    df: pd.DataFrame,
    modality_col: str = "modality",
    condition_col: str = "condition",
    dyad_col: str = "dyad_label",
    feature_cols: Optional[Sequence[str]] = None,
    n_permutations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    condition_values: Optional[Tuple[str, str]] = None,
) -> Dict[str, Dict]:
    """Run L2 between-condition test split by modality.

    Calls ``between_condition_fdr`` for each unique modality in the
    data, returning modality-keyed results.

    Parameters
    ----------
    df, condition_col, dyad_col, feature_cols, n_permutations, seed,
    alpha, condition_values
        Same as ``between_condition_fdr``.
    modality_col : str
        Column name for modality labels.

    Returns
    -------
    dict
        ``{modality: l2_result_dict}`` where each value matches the
        return format of ``between_condition_fdr``.
    """
    results = {}
    modalities = sorted(df[modality_col].dropna().unique())
    for mod in modalities:
        mod_df = df[df[modality_col] == mod]
        try:
            results[mod] = between_condition_fdr(
                mod_df,
                condition_col=condition_col,
                dyad_col=dyad_col,
                feature_cols=feature_cols,
                n_permutations=n_permutations,
                seed=seed + hash(mod) % 10000,
                alpha=alpha,
                condition_values=condition_values,
            )
        except ValueError as e:
            results[mod] = {"error": str(e)}
    return results

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
4. BH-FDR across the frozen PRIMARY confirmatory family (SyncPipe v1
   default = 1 primary descriptor, peak_amplitude; SECONDARY descriptors are
   reported in parallel with their own small-family correction)
5. Standardized permutation effect = Δ_obs / SD(Δ_perm)

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

import hashlib
import itertools
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..batch import _bh_fdr_correction, dedupe_fdr_input
from ..feature_definitions import (
    FDR_FAMILIES,
    PRIMARY_FDR_FAMILY,
    REFERENCE_FEATURE,
)

logger = logging.getLogger(__name__)


# ── Re-use the single canonical BH-FDR (batch._bh_fdr_correction) ──────────
def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Deprecated thin wrapper — delegates to ``batch._bh_fdr_correction``.

    Kept only so any external caller of the old local name still works.
    The canonical implementation now lives in ``syncpipe.batch``.
    """
    return np.asarray(_bh_fdr_correction(p_values)[0], dtype=float)


# Threshold (in #dyads) below which we enumerate *all* 2^n sign-flip
# combinations exactly, matching design_controls._paired_signflip_p_upper.
# For n=4 this gives a true null of 16 points (p resolution 1/17 ≈ 0.059)
# instead of the spurious 1/10001 implied by Monte-Carlo with n_permutations.
_ENUM_THRESHOLD = 12


def _signflip_null(diffs: np.ndarray, n_permutations: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Null distribution of median(flip * diffs) over all dyad sign flips.

    Enumerates every one of the 2^n sign assignments exactly when
    n <= _ENUM_THRESHOLD (honest discrete resolution); otherwise samples
    ``n_permutations`` Monte-Carlo draws.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n <= _ENUM_THRESHOLD:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)),
                          dtype=float)
        return np.median(signs * diffs, axis=1)
    flips = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    return np.median(flips * diffs, axis=1)


def _definedness_null(is_def_a: np.ndarray, is_def_b: np.ndarray,
                      n_permutations: int, rng: np.random.Generator) -> np.ndarray:
    """Null distribution of (sum p_a - sum p_b) under all label swaps.

    Enumerates every one of the 2^n swap assignments exactly when
    n <= _ENUM_THRESHOLD; otherwise Monte-Carlo samples ``n_permutations``.
    """
    a = np.asarray(is_def_a, dtype=float)
    b = np.asarray(is_def_b, dtype=float)
    n = a.size
    if n <= _ENUM_THRESHOLD:
        flips = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    else:
        flips = rng.choice([0, 1], size=(n_permutations, n)).astype(float)
    # flip==0 -> p_a=a, p_b=b ; flip==1 -> p_a=b, p_b=a
    p_a = (1.0 - flips) * a[None, :] + flips * b[None, :]
    p_b = flips * a[None, :] + (1.0 - flips) * b[None, :]
    return p_a.sum(axis=1) - p_b.sum(axis=1)


def _exact_p(obs: float, null: np.ndarray) -> float:
    """Phipson-Smyth (2010) two-tailed p from a (possibly exhaustive) null.

    p = (|null| >= |obs| + 1) / (N + 1), where N = len(null).  Works for
    both the exhaustive (N = 2^n) and Monte-Carlo (N = n_permutations) nulls.
    """
    null = np.asarray(null, dtype=float)
    finite = np.isfinite(null)
    null_fin = null[finite]
    if null_fin.size == 0:
        return 1.0
    n_ge = np.sum(np.abs(null_fin) >= np.abs(obs))
    return float(min((n_ge + 1) / (null_fin.size + 1), 1.0))


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
    perm_effect_size : float
        Permutation-standardized effect size = observed_diff / SD(null
        distribution).  NOTE: this is NOT classical Cohen's d.  Cohen's d
        uses (mean difference) / (pooled SD of the raw groups); this uses
        the SD of the *permutation null*, a standardization that depends on
        n_permutations / sample size / the permutation mechanism.  It is a
        self-contained effect-size index for the paired-permutation test,
        but must not be reported as "Cohen's d" in a manuscript.
    n_dyads : int
        Number of dyads with data in both conditions (paired).
    defined_a : int
        Number of dyads where this feature is defined in condition_a.
    defined_b : int
        Number of dyads where this feature is defined in condition_b.
    p_definedness : float
        P-value for the difference in definedness rates between conditions.
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
    perm_effect_size: float
    n_dyads: int = 0
    defined_a: int = 0
    defined_b: int = 0
    p_definedness: float = 1.0
    definedness_status: str = "complete"
    claimable: bool = True

    @property
    def cohens_d(self) -> float:
        """Deprecated alias for :attr:`perm_effect_size`.

        The legacy ``cohens_d`` name was misleading: the value is
        ``observed_diff / SD(null)``, not classical Cohen's d.  Use
        ``perm_effect_size`` and do not report it as Cohen's d.
        """
        warnings.warn(
            "L2Result.cohens_d is deprecated and mislabeled; it is a "
            "permutation-standardized effect (observed_diff / SD(null)), not "
            "Cohen's d. Use L2Result.perm_effect_size.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.perm_effect_size


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
    threshold_scope: str = "unknown",
    modality_col: Optional[str] = "modality",
    allow_multimodal_pool: bool = False,
    observation_col: Optional[str] = "n_wcc_points",
    observation_policy: str = "warn",
    undefined_policy: str = "flag",
    min_defined_fraction: float = 0.50,
    eligibility_policy: str = "warn",
    n_min_dyads: int = 10,
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
        Which feature columns to test. Defaults to the SSoT frozen
        PRIMARY confirmatory family (peak_amplitude only; dwell_time and
        switching_rate are reported in parallel as SECONDARY).
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
    if observation_policy not in {"ignore", "warn", "raise"}:
        raise ValueError("observation_policy must be 'ignore', 'warn', or 'raise'")
    if undefined_policy not in {"flag", "gate"}:
        raise ValueError("undefined_policy must be 'flag' or 'gate'")
    if eligibility_policy not in {"ignore", "warn", "raise"}:
        raise ValueError("eligibility_policy must be 'ignore', 'warn', or 'raise'")
    if n_min_dyads < 1:
        raise ValueError("n_min_dyads must be >= 1")
    if not 0.0 <= min_defined_fraction <= 1.0:
        raise ValueError("min_defined_fraction must lie in [0, 1]")
    if feature_cols is None:
        from ..feature_definitions import PRIMARY_FDR_FAMILY
        feature_cols = list(PRIMARY_FDR_FAMILY)

    # Only use columns actually present
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError(f"No feature columns found in df. Looking for: {feature_cols}")

    # ── FDR input guard: refuse duplicate feature keys ────────────────
    # Entering the same feature twice would inflate the BH test count m
    # and silently weaken the correction.  Fail loud (Karpathy Rule 12).
    feature_cols, _ = dedupe_fdr_input(
        list(feature_cols), [0.0] * len(feature_cols), on_duplicate="raise"
    )

    # ── P0-2: refuse silent multimodal pooling ───────────────────────────
    # If a modality column exists with >1 modality, groupby.mean() over
    # duplicate (dyad, condition) rows would average EDA+ECG+RESP into one
    # scalar and can cancel real per-modality effects (diff→0, p→1).
    # Call test_l2_by_modality / filter one modality, or pass
    # allow_multimodal_pool=True only with explicit scientific justification.
    if (
        modality_col is not None
        and modality_col in df.columns
        and not allow_multimodal_pool
    ):
        n_mod = df[modality_col].dropna().nunique()
        if n_mod > 1:
            mods = sorted(df[modality_col].dropna().astype(str).unique().tolist())
            raise ValueError(
                f"between_condition_fdr: found {n_mod} modalities in "
                f"column {modality_col!r} ({mods}). Pooling modalities by "
                f"averaging duplicate (dyad, condition) rows is refused by "
                f"default (P0-2). Filter to one modality, call "
                f"between_condition_by_modality / test_l2_by_modality, or "
                f"pass allow_multimodal_pool=True if you intentionally want "
                f"a pooled contrast (document the justification)."
            )

    # ── A11: per-dyad (non-pooled) threshold WARN for structure features ──
    # dwell_time / switching_rate are derived from an onset threshold. If that
    # threshold is PER-DYAD (each dyad/condition gets its own surrogate
    # threshold, as the legacy DynamicAnalyzer path computes), the resulting
    # descriptor means are NOT comparable across dyads/conditions — a
    # cross-condition difference may reflect threshold noise, not coupling.
    # The canonical pipeline_bridge path uses a single fixed (pooled)
    # onset threshold, so threshold_scope should be "fixed"/"pooled".
    # We only WARN (never hard-fail): this is advisory governance.
    _PER_DYAD_THRESHOLD_SCOPES = frozenset(
        {"within_dyad", "per_dyad", "per_pair", "non_pooled"}
    )
    _STRUCTURE_FEATURES = ("dwell_time", "switching_rate")
    _structure_in_fdr = [f for f in feature_cols if f in _STRUCTURE_FEATURES]
    if threshold_scope in _PER_DYAD_THRESHOLD_SCOPES and _structure_in_fdr:
        logger.warning(
            "L2 structure-feature warning: %s enter the group BH-FDR "
            "with a per-dyad (non-pooled) onset threshold. Per-dyad "
            "thresholds are not comparable across dyads/conditions, so "
            "cross-condition differences in dwell_time / switching_rate may "
            "reflect threshold noise rather than coupling. Prefer a "
            "session- or condition-pooled onset threshold (e.g. "
            "BatchComputationPipeline / ComputationPipeline with a single "
            "fixed onset_threshold) and always report the per-condition "
            "definedness rates (L2Result.defined_a / defined_b) "
            "alongside any significant result.",
            ", ".join(_structure_in_fdr),
        )

    # ── VIF gate (collinearity diagnostic for the FDR family) ──────────
    # High-VIF features are statistically redundant and inflate the
    # effective test count. We do NOT silently drop confirmatory features
    # (the FDR family is frozen); instead we flag severe/concern
    # collinearity so the analyst can interpret the FDR result honestly.
    vif_gate: Dict[str, object] = {"passed": True, "skipped": False}
    try:
        from ..feature_vif_test import VIF_SEVERE, collinearity_report
        if len(df.dropna(subset=feature_cols)) >= 4:
            rep = collinearity_report(df, feature_cols)
            severe = list(rep.get("vif_severe", []))
            concern = list(rep.get("vif_concern", []))
            vif_series = rep.get("vif", {})
            vif_dict = dict(vif_series) if hasattr(vif_series, "items") else {}
            vif_gate = {
                "passed": len(severe) == 0,
                "skipped": False,
                "vif_severe": severe,
                "vif_concern": concern,
                "vif": {k: float(v) for k, v in vif_dict.items()
                        if np.isfinite(float(v))},
            }
            if severe:
                logger.warning(
                    "VIF gate: %d FDR-family feature(s) show SEVERE "
                    "collinearity (VIF >= %.1f): %s. Their independent "
                    "significance is questionable and the effective test "
                    "count m may be inflated.",
                    len(severe), float(VIF_SEVERE), severe,
                )
        else:
            vif_gate["skipped"] = True
            vif_gate["reason"] = "insufficient finite rows for VIF"
    except Exception as exc:  # noqa: BLE001
        # Diagnostic only — never let the VIF gate break the L2 test.
        vif_gate = {"passed": True, "skipped": True,
                    "reason": f"vif computation skipped: {exc}"}

    unique_conditions = sorted(df[condition_col].dropna().unique())

    if condition_values is None:
        if len(unique_conditions) < 2:
            raise ValueError(
                f"Need at least 2 conditions, found {len(unique_conditions)}: "
                f"{unique_conditions}"
            )
        condition_a, condition_b = unique_conditions[0], unique_conditions[1]
        warnings.warn(
            f"between_condition_fdr: condition_values not specified; using "
            f"sorted labels ({condition_a!r}, {condition_b!r}). Difference "
            f"sign is condition_a - condition_b. Pass condition_values= "
            f"explicitly for confirmatory analyses.",
            UserWarning,
            stacklevel=2,
        )
    else:
        condition_a, condition_b = condition_values
        for c in (condition_a, condition_b):
            if c not in unique_conditions:
                raise ValueError(
                    f"Condition '{c}' not found in data. Available: {unique_conditions}"
                )

    # ── Observation-opportunity guard ─────────────────────────────────
    observation_guard = {"available": False, "policy": observation_policy}
    if observation_col is not None and observation_col in df.columns and observation_policy != "ignore":
        obs = df[[dyad_col, condition_col, observation_col]].dropna()
        if not obs.empty:
            cell_stats = (
                obs.groupby([dyad_col, condition_col])[observation_col]
                .agg(cell_min="min", cell_max="max", cell_nunique="nunique")
                .reset_index()
            )
            internal_variation = cell_stats[cell_stats["cell_nunique"] > 1]
            # Do not use first(): it would hide unequal trial observation
            # opportunity inside one dyad-condition cell.
            obs = obs.groupby([dyad_col, condition_col], as_index=False)[observation_col].mean()
            pivot = obs.pivot(index=dyad_col, columns=condition_col, values=observation_col)
            if condition_values is not None:
                pivot = pivot.reindex(columns=list(condition_values))
            unequal = []
            if pivot.shape[1] >= 2:
                for dyad_id, row in pivot.iterrows():
                    vals = row.dropna().to_numpy(dtype=float)
                    if vals.size >= 2 and not np.allclose(vals, vals[0]):
                        unequal.append(str(dyad_id))
            internal_dyads = sorted(internal_variation[dyad_col].astype(str).unique().tolist())
            observation_guard = {
                "available": True, "column": observation_col,
                "n_dyads_unequal": len(unequal),
                "unequal_dyads": unequal[:20],
                "n_cells_internal_variation": int(len(internal_variation)),
                "internal_variation_dyads": internal_dyads[:20],
                "policy": observation_policy,
            }
            if unequal or len(internal_variation):
                affected = sorted(set(unequal) | set(internal_dyads))
                msg = (
                    f"Observation opportunity differs across conditions or trials for "
                    f"{len(affected)} dyad(s) in {observation_col!r}; maximum/peak "
                    "features are not directly comparable."
                )
                if observation_policy == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
    elif observation_col is not None and observation_policy == "warn":
        observation_guard["warning"] = f"{observation_col!r} not present; opportunity was not checked."

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
    eligibility_status = "pass" if n_dyads >= n_min_dyads else "underpowered"
    if eligibility_status == "underpowered" and eligibility_policy != "ignore":
        msg = (
            f"Only {n_dyads} paired dyads; the configured v1 eligibility floor "
            f"is {n_min_dyads}. Results are exploratory/underpowered."
        )
        if eligibility_policy == "raise":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # ── Permutation test per feature ────────────────────────────────────
    rng = np.random.default_rng(seed)
    results: List[L2Result] = []

    for feat in feature_cols:
        vals_a = df_a[feat].to_numpy(dtype=float)
        vals_b = df_b[feat].to_numpy(dtype=float)

        # --- Scheme 1: Definedness Audit ---
        is_def_a = np.isfinite(vals_a)
        is_def_b = np.isfinite(vals_b)
        def_a_count = int(np.sum(is_def_a))
        def_b_count = int(np.sum(is_def_b))

        # Definedness p-value via permutation (H0: definedness is independent of condition).
        # Uses exact enumeration when n_dyads <= _ENUM_THRESHOLD for an honest
        # discrete p-value resolution (see _definedness_null).
        def_diff_obs = def_a_count - def_b_count
        def_null_diffs = _definedness_null(is_def_a, is_def_b, n_permutations, rng)
        p_def = _exact_p(def_diff_obs, def_null_diffs)
        min_rate = min(def_a_count, def_b_count) / max(n_dyads, 1)
        informative_undefinedness = (p_def < alpha) or (min_rate < min_defined_fraction)
        definedness_status = "informative_undefinedness" if informative_undefinedness else "complete"
        claimable = (
            eligibility_status == "pass"
            and (not informative_undefinedness or undefined_policy == "flag")
        )

        # --- Scheme 3: Continue with valid pairs only ---
        valid = is_def_a & is_def_b
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
                perm_effect_size=np.nan,
                n_dyads=int(valid.sum()),
                defined_a=def_a_count,
                defined_b=def_b_count,
                p_definedness=p_def,
                definedness_status=definedness_status,
                claimable=False if undefined_policy == "gate" else claimable,
            ))
            continue

        a_fin = vals_a[valid]
        b_fin = vals_b[valid]
        n = len(a_fin)

        # Observed difference (median paired)
        observed_diff = float(np.median(a_fin - b_fin))

        # Permutation null: median over all dyad sign-flips. Exact enumeration
        # when n <= _ENUM_THRESHOLD (honest discrete resolution); otherwise
        # Monte-Carlo with n_permutations.
        null_diffs = _signflip_null(a_fin - b_fin, n_permutations, rng)

        null_mean = float(np.mean(null_diffs))
        null_sd = float(np.std(null_diffs, ddof=1))
        p_raw = _exact_p(observed_diff, null_diffs)
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
            perm_effect_size=cohens_d,
            n_dyads=int(valid.sum()),
            defined_a=def_a_count,
            defined_b=def_b_count,
            p_definedness=p_def,
            definedness_status=definedness_status,
            claimable=claimable,
        ))

    # ── BH-FDR, stratified by SSoT FDR family ──────────────────────────
    # L0 (signal-level IAAFT null) and L1 (WCC-level IAAFT null) are
    # DIFFERENT null models, so they must NOT share one BH denominator —
    # mixing them dilutes the primary endpoint and lets the reference
    # feature occupy a correction slot. Correct within each SSoT family
    # instead (Axis D of feature_definitions). Reference features are
    # reported (p_raw) but never enter any BH denominator.
    family_of: Dict[str, str] = {
        feat: fam for fam, feats in FDR_FAMILIES.items() for feat in feats
    }
    reference_set = set(REFERENCE_FEATURE)

    # Group result indices by their SSoT family; unknown features get their
    # own singleton group (BH over a single p is identity — fail-safe).
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(results):
        if r.feature in reference_set:
            continue  # reference: reported, not corrected
        groups.setdefault(family_of.get(r.feature, r.feature), []).append(i)

    for fam, idxs in groups.items():
        p_raw_grp = np.array([results[i].p_raw for i in idxs], dtype=float)
        p_fdr_grp = np.asarray(_bh_fdr_correction(p_raw_grp)[0], dtype=float)
        for j, i in enumerate(idxs):
            r = results[i]
            r.p_fdr = float(p_fdr_grp[j]) if np.isfinite(p_fdr_grp[j]) else 1.0
            r.significant_05 = bool(r.p_fdr < alpha and r.claimable)

    # Reference features: report p_raw, but p_fdr is undefined (not corrected)
    # and they can never be declared significant in the confirmatory claim.
    for r in results:
        if r.feature in reference_set:
            r.p_fdr = float("nan")
            r.significant_05 = False

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
            "perm_effect_size": r.perm_effect_size,
            "n_dyads": r.n_dyads,
            "defined_a": r.defined_a,
            "defined_b": r.defined_b,
            "p_definedness": r.p_definedness,
            "definedness_status": r.definedness_status,
            "claimable": r.claimable,
        }
        for r in results
    ])

    return {
        "per_feature": results,
        "n_tested": len(results),
        "n_dyads": n_dyads,
        "n_significant": n_significant,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_permutations": n_permutations,
        "summary_df": summary_df,
        "vif_gate": vif_gate,
        "observation_guard": observation_guard,
        "eligibility_status": eligibility_status,
        "n_min_dyads": n_min_dyads,
        "undefined_policy": undefined_policy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: L2 within-modality
# ═══════════════════════════════════════════════════════════════════════════

def _modality_seed_offset(modality: str, modulus: int = 10000) -> int:
    """Deterministic, process-stable seed offset for a modality label.

    The built-in ``hash()`` is randomized per Python process by default
    (``PYTHONHASHSEED``) and is therefore NOT reproducible across runs or
    machines. Use a cryptographic digest of the modality name instead so
    that a fixed ``seed`` yields identical per-modality RNG offsets every
    time the analysis is re-run.
    """
    digest = hashlib.md5(str(modality).encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus


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
    threshold_scope: str = "unknown",
    observation_col: Optional[str] = "n_wcc_points",
    observation_policy: str = "warn",
    undefined_policy: str = "flag",
    min_defined_fraction: float = 0.50,
    eligibility_policy: str = "warn",
    n_min_dyads: int = 10,
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
                seed=seed + _modality_seed_offset(mod),
                alpha=alpha,
                condition_values=condition_values,
                threshold_scope=threshold_scope,
                # Subset is single-modality; disable the multimodal guard.
                modality_col=None,
                allow_multimodal_pool=False,
                observation_col=observation_col,
                observation_policy=observation_policy,
                undefined_policy=undefined_policy,
                min_defined_fraction=min_defined_fraction,
                eligibility_policy=eligibility_policy,
                n_min_dyads=n_min_dyads,
            )
        except ValueError as e:
            results[mod] = {"error": str(e)}
    return results

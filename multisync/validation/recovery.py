"""
Level 1 recovery validation — coupling × seed grid.

Does NOT vary SNR (Level 2) or autocorrelation (Level 3). Reproducible: same config → same numerics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd

from ..synthetic import generate_ground_truth_dyad
from ..dynamic_features import sliding_window_wcc
from ..feature_definitions import (
    FDR_FEATURES,
    REFERENCE_FEATURE,
    FEATURE_TIER,
    ONSET_THRESHOLD,
    extract_features as _ssot_extract_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONSET_THRESHOLD_DEFAULT: float = ONSET_THRESHOLD
"""Alias for ONSET_THRESHOLD (0.5) re-exported from feature_definitions (SSoT).
Override via Level1Config for sensitivity analysis only."""


@dataclass(frozen=True)
class Level1Config:
    """Frozen configuration for a Level 1 experiment.

    Frozen so that an instance is hashable and self-documenting; any
    deviation requires a new config object, which is easier to track
    than mutating fields in place.
    """
    duration_sec: float = 300.0
    hz: float = 1.0
    n_bursts: int = 5
    burst_sigma: float = 3.0
    noise_ratio: float = 0.3
    true_lag_sec: float = 0.0
    morphology: str = "identical"
    gap_prob: float = 0.0
    wcc_window_sec: float = 30.0
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT
    couplings: Sequence[float] = (0.0, 0.3, 0.7, 1.0)
    seeds: Sequence[int] = tuple(range(1000, 1030))   # 30 seeds

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz)))

    @property
    def n_cells(self) -> int:
        return len(self.couplings) * len(self.seeds)


# ---------------------------------------------------------------------------
# Feature extraction — thin wrapper delegating to the SSoT
# ---------------------------------------------------------------------------

def _extract_six_features(
    wcc: np.ndarray,
    hz: float,
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT,
    wcc_window_sec: float = 1.0,
) -> dict:
    """Return 8 FDR-family + 1 Reference feature + definedness flags for a WCC series.

    Thin wrapper delegating to :func:`feature_definitions.extract_features` (SSoT).
    Returns a dict (not DynamicFeatures) for backward compatibility with
    :func:`_run_single_cell`, :mod:`validation.snr`, and :mod:`validation.pgt1_intensity`.
    Name "six" is historical; actual output is 9 features + 3 flags.

    Parameters
    ----------
    wcc : np.ndarray
        WCC time series (may contain NaN).
    hz : float
        Sampling rate of the WCC series.
    onset_threshold : float, optional
        Onset/dwell/switching threshold (default 0.5).
    wcc_window_sec : float
        WCC window length in seconds for sustained-crossing scaling.
        :data:`ONSET_THRESHOLD_DEFAULT` (locked at 0.5, DECISION-01).
    wcc_window_sec : float, optional
        WCC window length in seconds — forwarded to the SSoT for the
        sustained-crossing scaling rule (DECISION-02).  Defaults to
        ``1.0`` to remain compatible with callers that have not yet been
        updated to forward their own ``wcc_window_sec``.

    Returns
    -------
    dict
        Keys: the 8 FDR-family features
        (``onset_latency, rise_time, peak_amplitude, recovery_time,
        dwell_time, switching_rate, synchrony_entropy, bimodality_coefficient``),
        the 1 Reference (``mean_synchrony``), and the 3 definedness
        flags (``onset_defined, rise_defined, recovery_defined``).
    """
    # Empty / all-non-finite guard (preserves legacy semantics)
    if wcc.size == 0 or not np.isfinite(wcc).any():
        return {
            "onset_latency": np.nan,
            "rise_time": np.nan,
            "peak_amplitude": np.nan,
            "recovery_time": np.nan,
            "dwell_time": np.nan,
            "switching_rate": np.nan,
            "mean_synchrony": np.nan,
            "synchrony_entropy": np.nan,
            "bimodality_coefficient": np.nan,
            "onset_defined": 0.0,
            "rise_defined": 0.0,
            "recovery_defined": 0.0,
        }

    feats = _ssot_extract_features(
        wcc,
        hz=hz,
        wcc_window_sec=wcc_window_sec,
        threshold=onset_threshold,
    )

    # ``to_dict`` includes the 8 FDR-family keys, 1 Reference, and 3
    # definedness flags.  Cast definedness flags to float to match the
    # legacy contract (pandas aggregations expected float means, not int).
    d = feats.to_dict()
    for k in ("onset_defined", "rise_defined", "recovery_defined"):
        d[k] = float(d.get(k, 0))
    # Drop internal meta keys so the returned dict is purely numeric and
    # safe to ``row.update(feats)`` into a tidy DataFrame row.
    d.pop("_notes", None)
    d.pop("_params", None)
    return d


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def _run_single_cell(
    coupling: float,
    seed: int,
    cfg: Level1Config,
) -> dict:
    """Generate one synthetic dyad, compute WCC + features, return a dict row."""
    ds = generate_ground_truth_dyad(
        lead_modality="lead",
        lag_modality="lag",
        true_lag_sec=cfg.true_lag_sec,
        noise_ratio=cfg.noise_ratio,
        duration_sec=cfg.duration_sec,
        hz=cfg.hz,
        seed=seed,
        n_bursts=cfg.n_bursts,
        burst_sigma=cfg.burst_sigma,
        gap_prob=cfg.gap_prob,
        morphology=cfg.morphology,
        coupling=coupling,
    )

    df_lead = ds.modalities["lead"]
    a = df_lead["person_a"].to_numpy()
    b = df_lead["person_b"].to_numpy()

    wcc = sliding_window_wcc(
        a, b,
        window_size=cfg.wcc_window_samples,
        hz=cfg.hz,
    )

    feats = _extract_six_features(
        wcc,
        hz=cfg.hz,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )

    row = {
        "coupling": coupling,
        "seed": seed,
        "onset_threshold": cfg.onset_threshold,
        "n_wcc_samples": int(np.sum(~np.isnan(wcc))),
        "wcc_min": float(np.nanmin(wcc)) if np.any(~np.isnan(wcc)) else float("nan"),
        "wcc_max": float(np.nanmax(wcc)) if np.any(~np.isnan(wcc)) else float("nan"),
    }
    row.update(feats)
    return row


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_level1_grid(cfg: Level1Config | None = None) -> pd.DataFrame:
    """Run the full Level 1 coupling × seed grid.

    Returns
    -------
    pd.DataFrame
        One row per (coupling, seed). Columns include the experimental
        knobs and all 6 dynamic features plus ``onset_defined``.
    """
    cfg = cfg or Level1Config()
    rows: List[dict] = []
    for coupling in cfg.couplings:
        for seed in cfg.seeds:
            rows.append(_run_single_cell(float(coupling), int(seed), cfg))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

# DECISION-09 / R-C (revised 2026-06-17):
#
# Feature columns for Level 1 summary (long format).
# Features summarised by summarise_level1: the confirmatory FDR family
# (Axis C) PLUS the reference comparator.  The reference (mean_synchrony)
# is reported with family="reference" and does NOT enter BH-FDR, but it is
# still summarised so its response curve can be inspected alongside the
# FDR features.
# mean_synchrony is a reported reference (not an FDR member), so it is added
# explicitly via REFERENCE_FEATURE rather than through FDR_FEATURES.
FEATURE_COLUMNS: tuple = tuple(FDR_FEATURES) + tuple(REFERENCE_FEATURE)


def summarise_level1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(coupling, feature) summary statistics for the Level 1 grid.

    DECISION-09 / R-C (revised 2026-06-17): returns a **long-format** table that
    makes the functional tier partition explicit in the
    schema.  Each row represents one (coupling, feature) cell.

    Columns
    -------
    coupling : float
        Coupling level (experimental knob).
    feature : str
        Feature name; one of the 8 SyncPipe features.
    family : category
        Functional tier from FEATURE_TIER: ``"core"``, ``"conditional"``,
        or ``"reference"``.  Core + Conditional enter BH-FDR;
        Reference is report-only.
    mean : float
        Across-seed mean of the feature value at this coupling.
    sd : float
        Across-seed standard deviation (ddof=1, pandas default).
    n_seeds : int
        Number of seeds in this (coupling) cell.
    onset_threshold : float
        Experimental knob, propagated from the input DataFrame.

    Definedness fractions (``onset_defined``, ``rise_defined``,
    ``recovery_defined``, ``dwell_defined``) are reported separately by
    :func:`summarise_definedness` to keep this table purely numeric.

    Raises
    ------
    ValueError
        If the DataFrame contains rows from multiple onset_threshold
        values. Split by threshold before summarising to prevent
        silent mis-aggregation across sensitivity sweeps.
    """
    if "onset_threshold" not in df.columns:
        raise ValueError(
            "summarise_level1 expects an 'onset_threshold' column in the "
            "results DataFrame. Re-run with the updated grid runner."
        )
    if df["onset_threshold"].nunique() > 1:
        raise ValueError(
            "summarise_level1 expects a single onset_threshold per call; "
            f"got {sorted(df['onset_threshold'].unique())}. "
            "Split by threshold before summarising."
        )

    threshold_val = float(df["onset_threshold"].iloc[0])

    rows: list[dict] = []
    grouped = df.groupby("coupling", sort=True)
    for coupling, sub in grouped:
        n_seeds = int(sub["seed"].count())
        for feat in FEATURE_COLUMNS:
            if feat not in sub.columns:
                # Defensive: a column may be absent if upstream changed.
                mean_val = float("nan")
                sd_val = float("nan")
            else:
                col = sub[feat]
                mean_val = float(col.mean())
                sd_val = float(col.std(ddof=1)) if n_seeds > 1 else float("nan")
            rows.append(
                {
                    "coupling": float(coupling),
                    "feature": feat,
                    "family": FEATURE_TIER[feat],
                    "mean": mean_val,
                    "sd": sd_val,
                    "n_seeds": n_seeds,
                    "onset_threshold": threshold_val,
                }
            )

    summary = pd.DataFrame(rows)
    # Pin the family column to category dtype so downstream groupbys
    # over family preserve the canonical "confirmatory" / "diagnostic"
    # ordering.
    summary["family"] = pd.Categorical(
        summary["family"],
        categories=["core", "conditional", "reference"],
        ordered=True,
    )
    return summary


def summarise_definedness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-coupling definedness fractions for the Level 1 grid.

    DECISION-09 / R-C (2026-05-24): split off from :func:`summarise_level1`
    so the main summary table can stay strictly numeric in long-format.

    Reports the fraction of seeds at which each conditionally-defined
    feature was computable.  This complements :func:`summarise_level1`,
    which silently averages over NaN-valued cells; this function answers
    the orthogonal question "for what fraction of seeds did the feature
    have a value at all?".

    Returned columns
    ----------------
    coupling : float
    onset_n_valid_fraction : float
        Fraction of seeds where ``onset_defined == 1`` (a WCC peak
        exceeded the onset threshold and an onset latency could be
        computed; DECISION-02).
    rise_n_valid_fraction : float
        Fraction of seeds where ``rise_defined == 1`` (the 25-75%
        quartile rise time was computable; DECISION-03).
    recovery_n_valid_fraction : float
        Fraction of seeds where ``recovery_defined == 1`` (the
        half-recovery time was computable; DECISION-05).
    dwell_n_valid_fraction : float
        Fraction of seeds where ``dwell_time`` was non-NaN.  Per the
        SSoT (:func:`feature_definitions._compute_dwell_time`),
        ``dwell_time`` is NaN whenever onset or recovery is undefined;
        non-NaN otherwise (including the legitimate zero-dwell case).
    n_seeds : int
    onset_threshold : float

    Note
    ----
    ``switching_rate`` does not need a dedicated definedness column:
    per the SSoT, it returns NaN only when the WCC series itself is
    empty / all-NaN, which is already reflected in ``n_wcc_samples``
    in the raw grid output.
    """
    if "onset_threshold" not in df.columns:
        raise ValueError(
            "summarise_definedness expects an 'onset_threshold' column "
            "in the results DataFrame. Re-run with the updated grid "
            "runner."
        )
    if df["onset_threshold"].nunique() > 1:
        raise ValueError(
            "summarise_definedness expects a single onset_threshold per "
            f"call; got {sorted(df['onset_threshold'].unique())}. "
            "Split by threshold before summarising."
        )

    threshold_val = float(df["onset_threshold"].iloc[0])
    grouped = df.groupby("coupling", as_index=False)

    summary = grouped.agg(
        onset_n_valid_fraction=("onset_defined", "mean"),
        rise_n_valid_fraction=("rise_defined", "mean"),
        recovery_n_valid_fraction=("recovery_defined", "mean"),
        n_seeds=("seed", "count"),
    )

    # dwell_time has no boolean definedness flag in the dict — its
    # definedness is encoded as NaN-vs-finite per the SSoT contract.
    dwell_frac = (
        df.assign(_dwell_defined=df["dwell_time"].notna().astype(float))
        .groupby("coupling", as_index=False)["_dwell_defined"]
        .mean()
        .rename(columns={"_dwell_defined": "dwell_n_valid_fraction"})
    )
    summary = summary.merge(dwell_frac, on="coupling", how="left")
    summary["onset_threshold"] = threshold_val
    return summary[
        [
            "coupling",
            "onset_n_valid_fraction",
            "rise_n_valid_fraction",
            "recovery_n_valid_fraction",
            "dwell_n_valid_fraction",
            "n_seeds",
            "onset_threshold",
        ]
    ]


def split_half_icc(
    values: np.ndarray,
    rng_seed: int = 0,
    ceiling_sd_threshold: float = 0.05,
) -> tuple:
    """
    Two-way random, single-rater ICC(2,1) on a single-coupling vector.

    Returns
    -------
    (value, status)
        status ∈ {"ok", "ceiling_undefined", "insufficient_seeds",
                   "all_undefined"}
        - "ok":                  value is a valid Pearson r.
        - "ceiling_undefined":   value is SD (precision estimate);
                                 ICC is mathematically undefined due to
                                 ceiling/floor effect (SD < ceiling_sd_threshold).
        - "insufficient_seeds":  value is NaN; not enough valid seeds (n < 4).
        - "all_undefined":       value is NaN; the feature column is
                                 entirely NaN at this coupling (e.g.,
                                 onset_latency when no baseline phase exists).

    Design note
    ------------
    When a feature's across-seed SD is very small (ceiling regime),
    the Pearson r denominator → 0 and the result is a 0/0 indeterminate
    form. The float-point realization (e.g. -0.327) carries no
    information. In that case we return SD itself as a precision estimate
    and label the status explicitly, so that the paper does not
    accidentally interpret a meaningless r as "poor reliability".

    This is the correct statistical practice; see also Bland-Altman
    (1986) for reporting precision instead of ICC under ceiling effects.
    """
    v = values[~np.isnan(values)]
    n = v.size
    if n == 0:
        return (float("nan"), "all_undefined")
    if n < 4:
        return (float("nan"), "insufficient_seeds")

    sd = float(np.std(v, ddof=1))
    if sd < ceiling_sd_threshold:
        # Ceiling/floor effect: return SD as precision; do NOT report ICC.
        return (sd, "ceiling_undefined")

    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    half = n // 2
    a = v[idx[:half]]
    b = v[idx[half : 2 * half]]
    if np.std(a, ddof=1) < 1e-12 or np.std(b, ddof=1) < 1e-12:
        return (sd, "ceiling_undefined")
    r = float(np.corrcoef(a, b)[0, 1])
    return (r, "ok")

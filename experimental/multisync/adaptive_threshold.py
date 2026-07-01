"""
adaptive_threshold.py
=====================

Surrogate-calibrated Epoch thresholding.

**SENSITIVITY ANALYSIS ONLY.  NOT FOR CONFIRMATORY TESTING.**
**DECISION-01 (confirmed 2026-06-03): The confirmatory pipeline uses**
**FIXED theta=0.5.  Functions here are for Appendix B / simulation only.**

Motivation
----------
The fixed threshold θ = 0.5 (Cohen 1988 large-effect benchmark) assumes
that all dyads and modalities have comparable WCC amplitude distributions.
In practice, HR, EDA, and behavioural signals produce systematically
different WCC ranges.  A dyad with naturally weak coupling (e.g., strangers
watching a neutral video) may never exceed 0.5, while a dyad with strong
coupling (e.g., romantic partners during conflict) may spend most of the
time above 0.5 — both cases lose discriminatory power.

Solution (Informed by Treur 2023 adaptive network logic)
--------------------------------------------------------
For each dyad, calibrate the threshold against its own null distribution,
obtained via IAAFT surrogate.  The 95th percentile of the null WCC
distribution becomes the dyad-specific θ_d.  This ensures that "elevated
synchrony" is defined relative to what that specific dyad's signals would
look like under the null of no coupling.

Constraints (to preserve cross-dyad comparability):
    θ_d := max(0.3, min(0.7, percentile(null_WCC, 95)))

The clamp [0.3, 0.7] prevents degenerate thresholds when:
    - null_WCC is extremely narrow  (→ θ_d would be near 0.0, every point is "Epoch")
    - null_WCC is extremely wide    (→ θ_d would be near 1.0, nothing is "Epoch")

Module Contract
---------------
This module is responsible for:
    - Computing dyad-specific thresholds from IAAFT surrogate WCC curves.
    - Providing a map from (dyad_id, modality) → θ_d.

This module MUST NOT:
    - Modify the locked ONSET_THRESHOLD constant.
    - Be imported by feature_definitions.py (circular dependency).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calibrate_threshold_from_signals(
    a: np.ndarray,
    b: np.ndarray,
    wcc_window_sec: float = 30.0,
    wcc_step_sec: float = 5.0,
    hz: float = 1.0,
    percentile: float = 95.0,
    clamp_min: float = 0.3,
    clamp_max: float = 0.7,
    seed: int = 42,
    n_surrogates: int = 499,
) -> float:
    """
    Dyad-specific threshold from a *signal-level* IAAFT null (correct null).

    This is the methodologically preferred calibrator: the IAAFT surrogate is
    applied to the raw signals ``a`` and ``b`` (destroying inter-signal phase
    coupling while preserving each signal's spectrum and amplitude
    distribution), and the WCC is *recomputed* on each surrogate pair.  The
    null therefore corresponds to "these two signals with no coupling", which
    is the hypothesis the threshold is meant to guard against.  Calibrating on
    the observed WCC curve itself (see :func:`calibrate_threshold_from_null`)
    instead randomises an already-derived quantity and does not represent the
    no-coupling null.

    Returns
    -------
    float
        theta_d in [clamp_min, clamp_max].
    """
    from .dynamic_features import _iaaft_surrogate, sliding_window_wcc

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 10 or len(b) < 10:
        return 0.5

    window_samples = max(3, int(round(wcc_window_sec * hz)))
    step_samples = max(1, int(round(wcc_step_sec * hz)))
    rng = np.random.default_rng(seed)

    null_values = np.full(n_surrogates, np.nan)
    for i in range(n_surrogates):
        sa = _iaaft_surrogate(a, rng)
        sb = _iaaft_surrogate(b, rng)
        wcc_surr = sliding_window_wcc(
            sa, sb, window_size=window_samples, hz=hz, step_samples=step_samples
        )
        finite = wcc_surr[np.isfinite(wcc_surr)]
        if finite.size:
            null_values[i] = np.percentile(np.abs(finite), percentile)

    valid = null_values[np.isfinite(null_values)]
    if valid.size == 0:
        return 0.5
    theta_d = float(np.median(valid))
    return max(clamp_min, min(clamp_max, theta_d))


def calibrate_threshold_from_null(
    wcc_curve: np.ndarray,
    percentile: float = 95.0,
    clamp_min: float = 0.3,
    clamp_max: float = 0.7,
    seed: int = 42,
    n_surrogates: int = 499,
) -> float:
    """
    DEPRECATED null-calibration on the WCC curve itself.

    .. deprecated::
        Prefer :func:`calibrate_threshold_from_signals`, which builds the null
        by surrogating the *raw signals* and recomputing WCC.  Surrogating the
        already-derived WCC curve does not instantiate the "no inter-signal
        coupling" null and will over- or under-state the threshold depending on
        the WCC curve's own autocorrelation.  Retained only for backward
        compatibility with earlier pilots.

    Parameters
    ----------
    wcc_curve : ndarray
        The observed WCC time series for a single dyad.
        Must be 1-D with finite values.
    percentile : float
        Percentile of the null distribution to use (default 95).
    clamp_min, clamp_max : float
        Allowable threshold range.
    seed : int
        RNG seed for surrogate generation.
    n_surrogates : int
        Number of surrogate realisations.

    Returns
    -------
    float
        Dyad-specific threshold θ_d in [clamp_min, clamp_max].
    """
    from .dynamic_features import _iaaft_surrogate

    wcc_clean = wcc_curve[~np.isnan(wcc_curve)]
    if len(wcc_clean) < 10:
        return 0.5  # fallback to default

    rng = np.random.default_rng(seed)
    null_values = np.zeros(n_surrogates)

    for i in range(n_surrogates):
        surr = _iaaft_surrogate(wcc_clean, rng)
        # The surrogate WCC represents the null of "no temporal structure beyond autocorrelation"
        null_values[i] = np.percentile(np.abs(surr), percentile)

    theta_d = float(np.median(null_values))
    theta_d = max(clamp_min, min(clamp_max, theta_d))

    return theta_d


def calibrate_threshold_batch(
    dyad_wcc_map: Dict[str, np.ndarray],
    percentile: float = 95.0,
    **kwargs,
) -> Dict[str, float]:
    """
    Batch calibrate thresholds for multiple dyads.

    Parameters
    ----------
    dyad_wcc_map : dict of {dyad_id: wcc_curve}
    percentile : float
    **kwargs
        Passed to calibrate_threshold_from_null.

    Returns
    -------
    dict of {dyad_id: theta_d}
    """
    thresholds: Dict[str, float] = {}
    for dyad_id, wcc in dyad_wcc_map.items():
        thresholds[dyad_id] = calibrate_threshold_from_null(
            wcc, percentile=percentile, **kwargs
        )
    return thresholds


# ---------------------------------------------------------------------------
# Modality-level calibration (pooled null)
# ---------------------------------------------------------------------------

def calibrate_modality_threshold(
    all_wcc_curves: List[np.ndarray],
    percentile: float = 95.0,
    **kwargs,
) -> float:
    """
    Pool all dyads of the same modality to get a shared threshold.

    This is computationally cheaper than per-dyad calibration and
    reasonable when the modality's WCC range is consistent across
    dyads (e.g., EDA is always in [-0.2, 0.9]).

    Parameters
    ----------
    all_wcc_curves : list of ndarray
        All WCC curves for a single modality across dyads.
    percentile : float

    Returns
    -------
    float
        Modality-level threshold.
    """
    if not all_wcc_curves:
        return 0.5

    thresholds = []
    for wcc in all_wcc_curves:
        theta = calibrate_threshold_from_null(wcc, percentile=percentile, **kwargs)
        thresholds.append(theta)

    return float(np.median(thresholds))


# ---------------------------------------------------------------------------
# Quick diagnostic: how different is θ_d from 0.5?
# ---------------------------------------------------------------------------

def threshold_deviation_report(
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    """Summarise deviation from the default θ=0.5."""
    values = np.array(list(thresholds.values()))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_below_0.4": int(np.sum(values < 0.4)),
        "n_above_0.6": int(np.sum(values > 0.6)),
        "n_at_default": int(np.sum(np.abs(values - 0.5) < 0.01)),
    }

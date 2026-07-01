"""
epoch_detection.py
==================

Dual-stream Epoch detection: WCC (0-lag) + WCLC (lagged) in parallel.

Motivation (Treur et al. 2023/2025):
    Synchrony transitions — not just mean levels — drive affiliation effects.
    Some dyadic coordination is lagged (leader-follower) and invisible to
    zero-lag WCC.  This module runs WCC and WCLC side-by-side, classifies
    Epochs by their lag profile, and extracts additional lag-aware features.

Design (DECISION-14 candidate):
    - Does NOT modify locked SSoT (feature_definitions.py).
    - Adds a thin dual-stream wrapper that delegates to existing SSoT
      functions for the 0-lag stream.
    - New lag-aware features are computed from the WCLC time series.
    - Eventually, lag_consistency and lag_direction can be promoted to
      diagnostic features once validated across datasets.

Module Contract
---------------
This module is responsible for:
    - Running WCC and WCLC in parallel on the same (a, b) pair.
    - Detecting Epochs on both streams.
    - Classifying Epochs as synchronous-only, lagged-only, or dual.
    - Computing three new lag-aware diagnostics.

This module MUST NOT:
    - Compute WCC / WCLC from raw signals (delegates to metrics.py /
      dynamic_features.py).
    - Modify the 6+2 confirmatory + diagnostic family (SSoT).
    - Produce figures or write files.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .feature_definitions import (
    DynamicFeatures,
    ONSET_THRESHOLD,
    compute_dwell_time,
    compute_switching_rate,
    extract_features,
    smoothed_wcc,
)
from .metrics import wclc_synchrony


# ---------------------------------------------------------------------------
# Helper: epoch classification
# ---------------------------------------------------------------------------

def _binary_epoch_mask(
    wcc_curve: np.ndarray, threshold: float
) -> np.ndarray:
    """Return boolean mask where |wcc_curve| >= threshold."""
    return np.abs(wcc_curve) >= threshold


def _overlap_ratio(
    mask_a: np.ndarray, mask_b: np.ndarray
) -> float:
    """Jaccard-like overlap: |intersection| / |union|."""
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return float(intersection / union) if union > 0 else 0.0


def _classify_epochs(
    wcc_0lag: np.ndarray,
    wclc_curve: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Classify each time point into one of four categories.

    Returns
    -------
    dict
        epoch_mask_sync   : bool[n_t]  — 0-lag Epoch (WCC only or dual)
        epoch_mask_lagged : bool[n_t]  — lagged Epoch (WCLC only or dual)
        overlap_ratio     : float      — Jaccard overlap
        n_transitions_sync   : int
        n_transitions_lagged : int
    """
    mask_sync = _binary_epoch_mask(wcc_0lag, threshold)
    mask_lag = _binary_epoch_mask(wclc_curve, threshold)
    overlap = _overlap_ratio(mask_sync, mask_lag)

    def _count_transitions(m: np.ndarray) -> int:
        diff = np.diff(m.astype(int))
        return int(np.sum(np.abs(diff)))  # each 0→1 or 1→0 = 1 transition edge

    return {
        "epoch_mask_sync": mask_sync,
        "epoch_mask_lagged": mask_lag,
        "overlap_ratio": overlap,
        "n_transitions_sync": _count_transitions(mask_sync),
        "n_transitions_lagged": _count_transitions(mask_lag),
    }


# ---------------------------------------------------------------------------
# New lag-aware diagnostics
# ---------------------------------------------------------------------------

def compute_lag_consistency(
    best_lags: np.ndarray, mask_lagged: np.ndarray
) -> float:
    """
    Standard deviation of best-lag indices within lagged Epochs.

    Low values  -> stable leader-follower relationship.
    High values -> erratic lag structure (lead alternates over time).

    Parameters
    ----------
    best_lags : ndarray
        Per-window best lag (samples) where max |r| occurs.  May contain NaN.
    mask_lagged : ndarray (bool)
        Window-level mask flagging lagged Epochs.  Must be index-aligned with
        ``best_lags``.

    Returns
    -------
    float
        Std of best lags inside lagged Epochs; NaN if fewer than two valid
        lagged windows exist.
    """
    if best_lags.size == 0 or mask_lagged.size == 0:
        return float("nan")
    n = min(len(best_lags), len(mask_lagged))
    lags = best_lags[:n]
    msk = mask_lagged[:n]
    sel = lags[msk & np.isfinite(lags)]
    if sel.size < 2:
        return float("nan")
    return float(np.std(sel))


def compute_lag_direction(
    best_lags: np.ndarray, mask_lagged: np.ndarray
) -> float:
    """
    Mean signed lag within lagged Epochs.

    > 0  → signal A lags behind signal B (B leads A).
    < 0  → signal A leads B.
    ~0  → balanced / unclear leader.
    """
    if not np.any(mask_lagged) or np.all(np.isnan(best_lags[mask_lagged])):
        return float("nan")
    return float(np.nanmean(best_lags[mask_lagged]))


def compute_lagged_dwell_time(
    wclc_curve: np.ndarray, hz: float, threshold: float = ONSET_THRESHOLD
) -> float:
    """Mean dwell time of elevated WCLC runs (lagged Epochs)."""
    return compute_dwell_time(wclc_curve, hz=hz, threshold=threshold)


def compute_lagged_switching_rate(
    wclc_curve: np.ndarray, hz: float, threshold: float = ONSET_THRESHOLD
) -> float:
    """Switching rate within the WCLC time series."""
    return compute_switching_rate(wclc_curve, hz=hz, threshold=threshold)


def compute_wclc_peak_amplitude(wclc_curve: np.ndarray) -> Tuple[float, Optional[int]]:
    """Maximum of 3-point smoothed WCLC curve."""
    sm = smoothed_wcc(wclc_curve)
    idx = int(np.argmax(sm)) if not np.all(np.isnan(sm)) else None
    val = float(sm[idx]) if idx is not None else float("nan")
    return val, idx


# ---------------------------------------------------------------------------
# Dual-stream entry point
# ---------------------------------------------------------------------------

def _wclc_with_lags(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 60,
    step: int = 10,
    max_lag_samples: int = 30,
    hz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run WCLC and return both the max-|r| curve and the best-lag indices.

    Returns
    -------
    wclc_curve : np.ndarray  (n_windows,)
        max |r| at each window
    best_lags  : np.ndarray  (n_windows,)
        lag index (in samples) at which max |r| occurs
    """
    # Use existing WCLC implementation
    wclc_curve = wclc_synchrony(
        a, b, window_size=window_size, step=step, max_lag_samples=max_lag_samples
    )

    # For best-lag trace, we need the raw lag-sweep within each window.
    # Re-implement the inner loop from wclc_synchrony to extract best lags.
    n_windows = (len(a) - window_size) // step + 1
    n_windows = max(n_windows, 0)
    best_lags = np.full(n_windows, np.nan)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > len(a):
            break
        slice_a = a[start:end]
        slice_b = b[start:end]
        if np.std(slice_a) < 1e-12 or np.std(slice_b) < 1e-12:
            continue
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        cors = np.zeros_like(lags, dtype=float)
        for j, lag in enumerate(lags):
            cors[j] = _lagged_pearson_r(slice_a, slice_b, lag)
        best_lags[i] = lags[int(np.argmax(np.abs(cors)))]

    return wclc_curve, best_lags


def _lagged_pearson_r(
    a: np.ndarray, b: np.ndarray, lag: int
) -> float:
    """Pearson r between a[t] and b[t+lag] over overlapping region."""
    if lag > 0:
        a_use, b_use = a[: -lag if lag != 0 else None], b[lag:]
    elif lag < 0:
        a_use, b_use = a[-lag:], b[:lag]
    else:
        a_use, b_use = a, b
    if len(a_use) < 3:
        return 0.0
    std_a, std_b = np.std(a_use), np.std(b_use)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    corr = np.corrcoef(a_use, b_use)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def _resample_to_grid(
    curve: np.ndarray, src_hz: float, dst_hz: float
) -> np.ndarray:
    """
    Linearly resample a curve from ``src_hz`` to ``dst_hz`` time resolution.

    Used to align the WCC stream (fine step) onto the WCLC stream (coarse step)
    so that mask comparison is done at matched wall-clock times rather than at
    matched array indices.  NaNs are interpolated over for resampling and the
    output keeps NaN where the source was entirely undefined.
    """
    n_src = len(curve)
    if n_src == 0 or src_hz <= 0 or dst_hz <= 0:
        return curve.copy()
    duration = (n_src - 1) / src_hz
    n_dst = int(np.floor(duration * dst_hz)) + 1
    if n_dst <= 1:
        return curve[:1].copy()
    t_src = np.arange(n_src) / src_hz
    t_dst = np.arange(n_dst) / dst_hz
    finite = np.isfinite(curve)
    if finite.sum() < 2:
        return np.full(n_dst, np.nan)
    return np.interp(t_dst, t_src[finite], curve[finite])


# ---------------------------------------------------------------------------
# Primary dual-stream API
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DualStreamResult:
    """Container for dual-stream WCC+WCLC Epoch analysis."""

    # 0-lag stream (delegated to SSoT)
    features_0lag: DynamicFeatures

    # WCLC stream (lag-aware)
    wclc_curve: np.ndarray
    best_lags: np.ndarray
    features_lagged: DynamicFeatures

    # Overlap & classification
    epoch_mask_sync: np.ndarray
    epoch_mask_lagged: np.ndarray
    overlap_ratio: float
    n_transitions_sync: int
    n_transitions_lagged: int

    # NEW lag-aware diagnostics (excluded from confirmatory family)
    lag_consistency: float = float("nan")
    lag_direction: float = float("nan")
    lagged_dwell_time: float = float("nan")
    lagged_switching_rate: float = float("nan")
    wclc_peak: float = float("nan")

    def to_dict(self) -> Dict[str, float]:
        base = self.features_0lag.to_dict()
        base.update(self.features_lagged.to_dict())
        base["overlap_ratio"] = self.overlap_ratio
        base["n_transitions_sync"] = float(self.n_transitions_sync)
        base["n_transitions_lagged"] = float(self.n_transitions_lagged)
        base["lag_consistency"] = self.lag_consistency
        base["lag_direction"] = self.lag_direction
        base["wclc_peak"] = self.wclc_peak
        return base


def dual_stream_epoch_analysis(
    a: np.ndarray,
    b: np.ndarray,
    wcc_window_sec: float = 30.0,
    wcc_step_sec: float = 5.0,
    hz: float = 1.0,
    wclc_window_sec: float = 60.0,
    wclc_step_sec: float = 10.0,
    max_lag_sec: float = 5.0,
    onset_threshold: float = ONSET_THRESHOLD,
) -> DualStreamResult:
    """
    Run WCC and WCLC Epoch detection in parallel on a dyad.

    Parameters
    ----------
    a, b : ndarray
        Pre-processed signal pair (same length, matched sampling rate).
    wcc_window_sec : float
        WCC sliding window in seconds (default 30).
    wcc_step_sec : float
        WCC step in seconds (default 5).
    hz : float
        Sampling rate in Hz.
    wclc_window_sec : float
        WCLC sliding window in seconds (default 60 — needs more data
        to detect lagged patterns).
    wclc_step_sec : float
        WCLC step in seconds.
    max_lag_sec : float
        Maximum lag to search in seconds (±).
    onset_threshold : float
        Threshold for binarizing WCC/WCLC into Epoch/non-Epoch
        (delegates to ONSET_THRESHOLD = 0.5 by default).

    Returns
    -------
    DualStreamResult
    """
    # ── 0-lag stream (WCC) ──────────────────────────────────────────
    from .dynamic_features import sliding_window_wcc

    window_samples = int(round(wcc_window_sec * hz))
    step_samples = int(round(wcc_step_sec * hz))
    window_samples = max(window_samples, 3)
    step_samples = max(step_samples, 1)

    wcc_curve = sliding_window_wcc(
        a, b, window_size=window_samples, hz=hz, step_samples=step_samples
    )
    wcc_hz = hz / step_samples  # effective sampling rate of WCC curve

    features_0lag = extract_features(
        wcc_curve, hz=wcc_hz, wcc_window_sec=wcc_window_sec,
        threshold=onset_threshold,
    )

    # ── Lagged stream (WCLC) ────────────────────────────────────────
    wclc_window_samples = int(round(wclc_window_sec * hz))
    wclc_step_samples = int(round(wclc_step_sec * hz))
    max_lag_samples = int(round(max_lag_sec * hz))

    wclc_curve, best_lags = _wclc_with_lags(
        a, b,
        window_size=wclc_window_samples,
        step=wclc_step_samples,
        max_lag_samples=max_lag_samples,
        hz=hz,
    )
    wclc_hz = hz / wclc_step_samples  # effective sampling rate

    features_lagged = extract_features(
        wclc_curve, hz=wclc_hz, wcc_window_sec=wclc_window_sec,
        threshold=onset_threshold,
    )

    # ── Classification ───────────────────────────────────────────────
    # WCC and WCLC are sampled on different grids (wcc_hz vs wclc_hz).
    # Resample both onto a common time axis before comparing masks so that
    # index i refers to the same wall-clock time in both streams.
    wcc_on_wclc = _resample_to_grid(wcc_curve, wcc_hz, wclc_hz)
    n_steps = min(len(wcc_on_wclc), len(wclc_curve))
    classification = _classify_epochs(
        wcc_on_wclc[:n_steps], wclc_curve[:n_steps], onset_threshold
    )

    # ── Lag-aware diagnostics ────────────────────────────────────────
    # best_lags is on the WCLC grid, already aligned with wclc_curve[:n_steps].
    mask_lagged = classification["epoch_mask_lagged"]
    if mask_lagged.any():
        lag_consistency = compute_lag_consistency(best_lags[:n_steps], mask_lagged)
        lag_direction = compute_lag_direction(best_lags[:n_steps], mask_lagged)
    else:
        lag_consistency = 0.0  # no lagged epochs -> never lags -> trivially consistent
        lag_direction = 0.0

    wclc_peak_val, _ = compute_wclc_peak_amplitude(wclc_curve)

    return DualStreamResult(
        features_0lag=features_0lag,
        wclc_curve=wclc_curve,
        best_lags=best_lags,
        features_lagged=features_lagged,
        epoch_mask_sync=classification["epoch_mask_sync"],
        epoch_mask_lagged=classification["epoch_mask_lagged"],
        overlap_ratio=classification["overlap_ratio"],
        n_transitions_sync=classification["n_transitions_sync"],
        n_transitions_lagged=classification["n_transitions_lagged"],
        lag_consistency=lag_consistency,
        lag_direction=lag_direction,
        lagged_dwell_time=features_lagged.dwell_time,
        lagged_switching_rate=features_lagged.switching_rate,
        wclc_peak=wclc_peak_val,
    )

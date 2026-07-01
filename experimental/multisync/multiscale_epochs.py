"""
multiscale_epochs.py
=====================

Multi-scale Epoch confirmation (P3).
Inspired by Treur NOM three-level architecture:
    Fine (5s)  → candidate micro-Epochs
    Medium (30s) → confirm meso-Epochs (default SyncPipe level)
    Coarse (condition) → macro-Epoch characterisation

Each scale produces its own Epoch mask.  Cross-scale consistency
provides a confidence score for every Epoch boundary, reducing the
blurring caused by WCC window smoothing.

Module Contract
---------------
This module is responsible for:
    - Computing WCC at multiple window sizes.
    - Detecting Epochs at each scale.
    - Computing cross-scale consistency scores.

This module MUST NOT:
    - Modify locked feature extraction (uses existing SSoT functions).
    - Import metrics.py directly (uses dynamic_features.sliding_window_wcc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_definitions import ONSET_THRESHOLD


# ---------------------------------------------------------------------------
# Single-scale helper
# ---------------------------------------------------------------------------

def _epoch_mask_at_scale(
    a: np.ndarray,
    b: np.ndarray,
    window_sec: float,
    step_sec: float,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
) -> Tuple[np.ndarray, float]:
    """
    Compute binary Epoch mask at a given WCC window size.

    Returns
    -------
    (mask, step_hz)
        mask    : bool[n_windows] where |WCC| >= threshold
        step_hz : effective sampling rate of the mask (1 / step_sec), used to
                  resample masks of different scales onto a common time grid.
    """
    from .dynamic_features import sliding_window_wcc

    window_samples = max(3, int(round(window_sec * hz)))
    step_samples = max(1, int(round(step_sec * hz)))

    wcc = sliding_window_wcc(
        a, b, window_size=window_samples, hz=hz, step_samples=step_samples
    )
    step_hz = hz / step_samples
    return (np.abs(wcc) >= threshold), step_hz


def _resample_mask_to_grid(
    mask: np.ndarray, src_hz: float, dst_hz: float, n_dst: int
) -> np.ndarray:
    """
    Nearest-neighbour resample a boolean mask from ``src_hz`` to ``dst_hz``.

    Aligns coarse-scale masks (few windows, large step) onto the fine-scale
    time grid so that cross-scale agreement is computed at matched wall-clock
    times instead of matched array indices.
    """
    n_src = len(mask)
    if n_src == 0:
        return np.zeros(n_dst, dtype=bool)
    if n_src == 1:
        return np.repeat(mask, n_dst)[:n_dst]
    t_src = np.arange(n_src) / src_hz
    t_dst = np.arange(n_dst) / dst_hz
    idx = np.clip(np.round(t_dst * src_hz).astype(int), 0, n_src - 1)
    return mask[idx]


# ---------------------------------------------------------------------------
# Cross-scale consistency
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleEpochResult:
    """Container for multi-scale Epoch analysis."""

    # Masks at each scale
    mask_fine: np.ndarray      # ~5s
    mask_meso: np.ndarray      # ~30s
    mask_coarse: np.ndarray    # condition-level

    # Confidence score: what fraction of scales agree at each time step
    confidence_2scale: np.ndarray  # fine ∩ meso (0, 0.5, 1.0)
    confidence_3scale: np.ndarray  # fine ∩ meso ∩ coarse (0, 0.33, 0.67, 1.0)

    # Global metrics
    consistency_2scale_mean: float
    consistency_3scale_mean: float

    # WCC curves
    wcc_fine: np.ndarray
    wcc_meso: np.ndarray
    wcc_coarse: np.ndarray

    # Parameters
    scales: Dict[str, float]     # {"fine": 5.0, "meso": 30.0, "coarse": "condition"}
    threshold: float


def multiscale_epoch_analysis(
    a: np.ndarray,
    b: np.ndarray,
    hz: float = 1.0,
    scales: Optional[Dict[str, float]] = None,
    threshold: float = ONSET_THRESHOLD,
) -> MultiScaleEpochResult:
    """
    Run Epoch detection at three temporal scales and compute
    cross-scale consistency.

    Parameters
    ----------
    a, b : ndarray
        Pre-processed signal pair.
    hz : float
        Sampling rate in Hz.
    scales : dict, optional
        Window sizes in seconds for each scale.
        Default: {"fine": 5.0, "meso": 30.0, "coarse": 120.0}
    threshold : float
        Epoch threshold.

    Returns
    -------
    MultiScaleEpochResult
    """
    if scales is None:
        scales = {"fine": 5.0, "meso": 30.0, "coarse": 120.0}

    step_ratio = 0.2  # step = 20% of window

    mask_fine, hz_fine = _epoch_mask_at_scale(
        a, b, scales["fine"], scales["fine"] * step_ratio, hz, threshold
    )
    mask_meso, hz_meso = _epoch_mask_at_scale(
        a, b, scales["meso"], scales["meso"] * step_ratio, hz, threshold
    )
    mask_coarse, hz_coarse = _epoch_mask_at_scale(
        a, b, scales["coarse"], scales["coarse"] * step_ratio, hz, threshold
    )

    # Resample meso and coarse masks onto the FINE time grid so that index i
    # refers to the same wall-clock time across all three scales.  (Previously
    # the masks were index-truncated to the shortest array, which silently
    # aligned t=1s of the fine stream with t=24s of the coarse stream.)
    n_grid = len(mask_fine)
    mask_f = mask_fine.astype(float)
    mask_m = _resample_mask_to_grid(mask_meso, hz_meso, hz_fine, n_grid).astype(float)
    mask_c = _resample_mask_to_grid(mask_coarse, hz_coarse, hz_fine, n_grid).astype(float)

    mask_meso = mask_m.astype(bool)
    mask_coarse = mask_c.astype(bool)

    confidence_2 = (mask_f + mask_m) / 2.0
    confidence_3 = (mask_f + mask_m + mask_c) / 3.0

    # WCC curves
    from .dynamic_features import sliding_window_wcc as _wcc

    wcc_fine = _wcc(
        a, b, window_size=max(3, int(scales["fine"] * hz)),
        hz=hz, step_samples=max(1, int(scales["fine"] * step_ratio * hz)),
    )
    wcc_meso = _wcc(
        a, b, window_size=max(3, int(scales["meso"] * hz)),
        hz=hz, step_samples=max(1, int(scales["meso"] * step_ratio * hz)),
    )
    wcc_coarse = _wcc(
        a, b, window_size=max(3, int(scales["coarse"] * hz)),
        hz=hz, step_samples=max(1, int(scales["coarse"] * step_ratio * hz)),
    )

    return MultiScaleEpochResult(
        mask_fine=mask_fine,
        mask_meso=mask_meso,
        mask_coarse=mask_coarse,
        confidence_2scale=confidence_2,
        confidence_3scale=confidence_3,
        consistency_2scale_mean=float(np.mean(confidence_2)),
        consistency_3scale_mean=float(np.mean(confidence_3)),
        wcc_fine=wcc_fine,
        wcc_meso=wcc_meso,
        wcc_coarse=wcc_coarse,
        scales=scales,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Boundary confidence: which Epoch boundaries are reliable?
# ---------------------------------------------------------------------------

def epoch_boundary_confidence(
    result: MultiScaleEpochResult,
    min_confidence_2scale: float = 0.5,
    min_confidence_3scale: float = 0.33,
) -> np.ndarray:
    """
    Flag time points where Epoch detection is consistent across scales.

    Returns
    -------
    bool array (n_t,) — True where both 2-scale AND 3-scale confidence
    exceed their minimum thresholds.
    """
    return (
        (result.confidence_2scale >= min_confidence_2scale)
        & (result.confidence_3scale >= min_confidence_3scale)
    )

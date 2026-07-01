"""
transition_detection.py
========================

Treur-style transition detection for synchrony Epoch boundaries.

Background
----------
Hendrikse, Treur, Wilderjans, Dikker & Koole (2023), "Switching In and Out
of Sync" (Complex Networks XI, Springer SCI 1078) showed that *transitions*
between synchrony states - not just mean synchrony levels - drive affiliation
effects.  Their self-modelling network uses three transition-detection
functions over a sliding window W with half-window size sigma (Table 1):

    Average transition : |mean(W[1..sigma]) - mean(W[sigma+1..2*sigma])|
    Maxmin  transition : max(W[1..sigma]) - min(W[1..sigma])
    Stdev   transition : sqrt(mean((W[1..sigma] - mean(W[1..sigma]))**2))

Why this helps SyncPipe
-------------------------
The fixed threshold theta=0.5 on a *smoothed* WCC curve produces blurred Epoch
boundaries: near a slow ramp the crossing point is ambiguous and shifts with
the smoothing window.  Treur's transition detectors measure the *rate of
change* of synchrony directly, so an Epoch boundary becomes a local maximum of
the transition signal rather than a single threshold crossing.  This gives
sharper, smoothing-robust boundaries and a continuous "transition strength"
that can refine the binary theta=0.5 decision.

Module Contract
---------------
This module is responsible for:
    - Computing the three Treur transition signals over a WCC (or WCLC) curve.
    - Locating Epoch boundaries as peaks of the transition signal.
    - Providing a continuous transition-strength trace for soft boundaries.

This module MUST NOT:
    - Modify the locked SSoT (feature_definitions.py).
    - Compute WCC / WCLC from raw signals (operates on an existing curve).
    - Produce figures or write files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Treur Table 1 transition functions (operate on a 1-D synchrony curve)
# ---------------------------------------------------------------------------

def average_transition(curve: np.ndarray, half_window: int) -> np.ndarray:
    """
    Treur "Average transition": |mean(first half) - mean(second half)|.

    For each centre index t the full window spans ``[t-half_window, t+half_window)``
    and is split into a left half and a right half of ``half_window`` samples
    each.  The result is the absolute difference of the two half-means, i.e. a
    step detector that peaks where the synchrony level jumps.

    Parameters
    ----------
    curve : ndarray
        1-D synchrony curve (e.g. signed WCC or |WCC|).
    half_window : int
        sigma in Treur's notation (samples).  Must be >= 1.

    Returns
    -------
    ndarray
        Same length as ``curve``; NaN where the full 2*sigma window does not
        fit (the first and last ``half_window`` samples).
    """
    sigma = int(half_window)
    n = len(curve)
    out = np.full(n, np.nan)
    if sigma < 1 or n < 2 * sigma:
        return out
    for t in range(sigma, n - sigma):
        left = curve[t - sigma:t]
        right = curve[t:t + sigma]
        lf = left[np.isfinite(left)]
        rf = right[np.isfinite(right)]
        if lf.size == 0 or rf.size == 0:
            continue
        out[t] = abs(float(np.mean(lf)) - float(np.mean(rf)))
    return out


def maxmin_transition(curve: np.ndarray, half_window: int) -> np.ndarray:
    """
    Treur "Maxmin transition": max(W) - min(W) over a sigma-sample window.

    A volatility detector: large where the synchrony curve swings widely
    within the window (an unstable Epoch), small in flat regions.

    Returns
    -------
    ndarray
        Centred trace, NaN at the edges where the window does not fit.
    """
    sigma = int(half_window)
    n = len(curve)
    out = np.full(n, np.nan)
    if sigma < 1 or n < sigma:
        return out
    half = sigma // 2
    for t in range(n):
        start = max(0, t - half)
        end = min(n, start + sigma)
        start = max(0, end - sigma)
        w = curve[start:end]
        wf = w[np.isfinite(w)]
        if wf.size == 0:
            continue
        out[t] = float(np.max(wf) - np.min(wf))
    return out


def stdev_transition(curve: np.ndarray, half_window: int) -> np.ndarray:
    """
    Treur "Standard deviation transition": std of W over a sigma-sample window.

    A continuous change-magnitude detector; unlike maxmin it is robust to a
    single outlier sample and provides a smooth transition-strength trace
    suitable for soft Epoch boundaries.

    Returns
    -------
    ndarray
        Centred trace, NaN at the edges where the window does not fit.
    """
    sigma = int(half_window)
    n = len(curve)
    out = np.full(n, np.nan)
    if sigma < 1 or n < sigma:
        return out
    half = sigma // 2
    for t in range(n):
        start = max(0, t - half)
        end = min(n, start + sigma)
        start = max(0, end - sigma)
        w = curve[start:end]
        wf = w[np.isfinite(w)]
        if wf.size == 0:
            continue
        out[t] = float(np.std(wf))
    return out


# ---------------------------------------------------------------------------
# Boundary localisation via transition peaks
# ---------------------------------------------------------------------------

@dataclass
class TransitionResult:
    """Container for Treur-style transition detection on one curve."""

    avg_transition: np.ndarray
    maxmin_transition: np.ndarray
    stdev_transition: np.ndarray

    boundary_indices: np.ndarray   # peak indices of the chosen transition signal
    boundary_strength: np.ndarray  # transition value at each boundary

    half_window: int
    method: str

    def to_dict(self) -> Dict[str, float]:
        avg = self.avg_transition[np.isfinite(self.avg_transition)]
        return {
            "n_boundaries": float(self.boundary_indices.size),
            "mean_boundary_strength": (
                float(np.mean(self.boundary_strength))
                if self.boundary_strength.size else float("nan")
            ),
            "max_transition": float(np.max(avg)) if avg.size else float("nan"),
            "mean_transition": float(np.mean(avg)) if avg.size else float("nan"),
        }


def _local_maxima(signal: np.ndarray, min_prominence: float, min_distance: int) -> np.ndarray:
    """
    Simple prominence/distance peak picker (no SciPy dependency).

    A point is a peak if it is strictly greater than both neighbours, its value
    exceeds ``min_prominence``, and it is at least ``min_distance`` samples from
    any previously accepted (stronger) peak.
    """
    finite = np.where(np.isfinite(signal), signal, -np.inf)
    n = finite.size
    cand = []
    for t in range(1, n - 1):
        v = finite[t]
        if v > min_prominence and v >= finite[t - 1] and v > finite[t + 1]:
            cand.append((v, t))
    cand.sort(reverse=True)  # strongest first
    accepted: List[int] = []
    for _, t in cand:
        if all(abs(t - a) >= min_distance for a in accepted):
            accepted.append(t)
    return np.array(sorted(accepted), dtype=int)


def detect_transitions(
    curve: np.ndarray,
    half_window: int,
    method: str = "average",
    min_prominence: Optional[float] = None,
    min_distance: Optional[int] = None,
) -> TransitionResult:
    """
    Compute all three Treur transition signals and locate Epoch boundaries
    as peaks of the selected transition signal.

    Parameters
    ----------
    curve : ndarray
        1-D synchrony curve (signed WCC, |WCC|, or WCLC).
    half_window : int
        sigma (samples).  A natural choice is half the WCC window length.
    method : {"average", "maxmin", "stdev"}
        Which transition signal drives boundary localisation.  "average" is
        the recommended default (step detector, sharpest boundaries).
    min_prominence : float, optional
        Minimum transition value to count as a boundary.  Defaults to the
        75th percentile of the finite transition signal (data-adaptive).
    min_distance : int, optional
        Minimum separation between boundaries in samples.  Defaults to
        ``half_window`` to avoid double-counting a single ramp.

    Returns
    -------
    TransitionResult
    """
    avg = average_transition(curve, half_window)
    mm = maxmin_transition(curve, half_window)
    sd = stdev_transition(curve, half_window)

    sig_map = {"average": avg, "maxmin": mm, "stdev": sd}
    if method not in sig_map:
        raise ValueError(f"method must be one of {list(sig_map)}, got {method!r}")
    signal = sig_map[method]

    finite_vals = signal[np.isfinite(signal)]
    if min_prominence is None:
        min_prominence = (
            float(np.percentile(finite_vals, 75)) if finite_vals.size else 0.0
        )
    if min_distance is None:
        min_distance = max(1, int(half_window))

    peaks = _local_maxima(signal, min_prominence, min_distance)
    strengths = signal[peaks] if peaks.size else np.array([], dtype=float)

    return TransitionResult(
        avg_transition=avg,
        maxmin_transition=mm,
        stdev_transition=sd,
        boundary_indices=peaks,
        boundary_strength=strengths,
        half_window=int(half_window),
        method=method,
    )


def refine_epoch_boundaries(
    curve: np.ndarray,
    binary_mask: np.ndarray,
    half_window: int,
    search_radius: Optional[int] = None,
) -> np.ndarray:
    """
    Snap fixed-threshold Epoch edges to the nearest transition peak.

    The fixed theta=0.5 mask gives approximate edges on a smoothed curve; this
    function shifts each edge to the closest local maximum of the average
    transition signal within +/- ``search_radius`` samples, sharpening the
    boundary while keeping the SSoT threshold decision intact.

    Parameters
    ----------
    curve : ndarray
        The synchrony curve the mask was computed from.
    binary_mask : ndarray (bool)
        Fixed-threshold Epoch mask (same length as ``curve``).
    half_window : int
        sigma for the transition signal.
    search_radius : int, optional
        Max samples an edge may move.  Defaults to ``half_window``.

    Returns
    -------
    ndarray (int)
        Refined edge indices (where the mask changes state), snapped to
        transition peaks.
    """
    if search_radius is None:
        search_radius = int(half_window)

    avg = average_transition(curve, half_window)
    edges = np.where(np.diff(binary_mask.astype(int)) != 0)[0] + 1

    refined: List[int] = []
    n = len(curve)
    for e in edges:
        lo = max(0, e - search_radius)
        hi = min(n, e + search_radius + 1)
        window = avg[lo:hi]
        if not np.any(np.isfinite(window)):
            refined.append(int(e))
            continue
        local = int(np.nanargmax(window))
        refined.append(lo + local)
    return np.array(sorted(set(refined)), dtype=int)

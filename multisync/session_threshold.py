"""
Session-level pooled surrogate thresholding.

Provides a single threshold shared across all dyads in a session (or condition),
addressing the cross-dyad comparability problem of per-dyad surrogate thresholds.

Rationale
---------
Per-dyad thresholds adapt to each dyad's own null distribution. This is useful for
within-dyad inference, but makes between-dyad feature values (e.g. dwell_time,
switching_rate) hard to compare because the "episode" definition differs per dyad.

A session-level pooled threshold:
1. generates surrogates for every dyad in the session,
2. pools all finite surrogate coupling values across dyads and replicates,
3. returns a single percentile-based threshold.

This preserves the null-hypothesis grounding of surrogate thresholds while giving
all dyads the same threshold, making group-level comparisons meaningful.

Two modes are supported:
- ``session`` : one threshold for the whole session (default for cross-condition
  comparability; Task A in the surrogate threshold design docs).
- ``condition`` : one threshold per condition, computed by pooling only the
  dyads/segments belonging to that condition (sensitivity analysis).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .dynamic_features import sliding_window_wcc
from .feature_definitions import compute_surrogate_threshold, ONSET_THRESHOLD
from .surrogate import iaaft_surrogate, ft_surrogate
from .wclr import wclr_coupling_trace


__all__ = [
    "compute_session_pooled_threshold",
    "compute_condition_pooled_thresholds",
]


def _generate_surrogate_coupling_matrix(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    hz: float,
    window_size: int,
    surrogate_n: int,
    seed: int,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
) -> np.ndarray:
    """Generate (surrogate_n, n_coupling_points) matrix of surrogate coupling values.

    Parameters
    ----------
    backend : {"wcc", "wclr"}
        If "wcc", compute sliding-window cross-correlation on surrogate pairs.
        If "wclr", compute windowed cross-lagged regression on surrogate pairs.
    """
    rng = np.random.default_rng(seed)
    _gen = iaaft_surrogate if surrogate_method == "iaaft" else ft_surrogate

    surrogate_couplings: List[np.ndarray] = []
    for _ in range(surrogate_n):
        a_surr = _gen(sig_a, rng)
        b_surr = _gen(sig_b, rng)
        if backend == "wclr":
            coup_s = wclr_coupling_trace(
                a_surr, b_surr,
                window_size=window_size,
                hz=hz,
                max_lag_samples=wclr_max_lag_samples,
            )
        else:
            coup_s = sliding_window_wcc(
                a_surr, b_surr,
                window_size=window_size,
                hz=hz,
            )
        surrogate_couplings.append(coup_s)

    return np.vstack(surrogate_couplings)


def compute_session_pooled_threshold(
    dyad_signals: List[Tuple[np.ndarray, np.ndarray]],
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = 95.0,
    seed: int = 42,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    fallback_threshold: float = ONSET_THRESHOLD,
) -> Tuple[float, Dict]:
    """Compute a single surrogate threshold pooled across all dyads.

    Parameters
    ----------
    dyad_signals : list of (sig_a, sig_b) tuples
        All dyad signal pairs in the session. Each pair must have the same
        length and a common sampling rate.
    hz : float
        Sampling rate (Hz).
    wcc_window_size : int
        WCC/WCLR window size in samples.
    surrogate_n : int
        Number of surrogates per dyad. Total surrogate replicates =
        ``len(dyad_signals) * surrogate_n``.
    percentile : float
        Quantile of the pooled surrogate coupling distribution (default 95).
    seed : int
        Base RNG seed. Per-dyad seeds are derived as ``seed + i`` so results
        are reproducible even when dyads are reordered.
    surrogate_method : {"iaaft", "ft"}
        Surrogate method. IAAFT (default) preserves spectrum and amplitude
        distribution; FT preserves only spectrum.
    backend : {"wcc", "wclr"}
        Computational backend used to compute the coupling trace on surrogate
        pairs. Must match the backend used for the observed analysis.
    wclr_max_lag_samples : int
        Max lag for WCLR backend (ignored for WCC).
    fallback_threshold : float
        Threshold to return if the pooled surrogate distribution is degenerate
        (fewer than 10 finite values).

    Returns
    -------
    Tuple[float, Dict]
        ``(threshold, meta)`` where ``meta`` contains:
        - ``mode``: "session_pooled"
        - ``n_dyads``: number of dyads
        - ``surrogate_n_per_dyad``: surrogates per dyad
        - ``total_replicates``: total number of surrogate coupling series
        - ``n_finite_coupling_values``: number of finite coupling values pooled
        - ``percentile``: percentile used
        - ``surrogate_method``: "iaaft" or "ft"
        - ``backend``: "wcc" or "wclr"
        - ``fallback_used``: whether the fallback threshold was used
    """
    if not dyad_signals:
        return fallback_threshold, {
            "mode": "session_pooled",
            "fallback_used": True,
            "reason": "empty dyad_signals",
        }

    pooled_values: List[np.ndarray] = []
    for i, (sig_a, sig_b) in enumerate(dyad_signals):
        sig_a = np.asarray(sig_a, dtype=float)
        sig_b = np.asarray(sig_b, dtype=float)
        if not (np.all(np.isfinite(sig_a)) and np.all(np.isfinite(sig_b))):
            continue
        if len(sig_a) != len(sig_b):
            continue
        coup_matrix = _generate_surrogate_coupling_matrix(
            sig_a, sig_b, hz, wcc_window_size,
            surrogate_n=surrogate_n,
            seed=seed + i,
            surrogate_method=surrogate_method,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
        )
        pooled_values.append(coup_matrix)

    if not pooled_values:
        return fallback_threshold, {
            "mode": "session_pooled",
            "fallback_used": True,
            "reason": "no dyads produced finite surrogate coupling values",
        }

    pooled = np.vstack(pooled_values)
    threshold, is_surrogate = compute_surrogate_threshold(pooled, percentile=percentile)

    meta = {
        "mode": "session_pooled",
        "n_dyads": len(dyad_signals),
        "surrogate_n_per_dyad": surrogate_n,
        "total_replicates": pooled.shape[0],
        "n_finite_coupling_values": int(np.isfinite(pooled).sum()),
        "percentile": percentile,
        "surrogate_method": surrogate_method,
        "backend": backend,
        "fallback_used": not is_surrogate,
    }
    return threshold, meta


def compute_condition_pooled_thresholds(
    condition_signals: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = 95.0,
    seed: int = 42,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    fallback_threshold: float = ONSET_THRESHOLD,
) -> Dict[str, Tuple[float, Dict]]:
    """Compute one pooled surrogate threshold per experimental condition.

    Parameters
    ----------
    condition_signals : dict
        Mapping ``condition_label -> list of (sig_a, sig_b) tuples``.
    hz, wcc_window_size, surrogate_n, percentile, seed, surrogate_method,
    backend, wclr_max_lag_samples, fallback_threshold
        Passed to :func:`compute_session_pooled_threshold` for each condition.

    Returns
    -------
    Dict[str, Tuple[float, Dict]]
        Mapping condition -> (threshold, meta).
    """
    results: Dict[str, Tuple[float, Dict]] = {}
    for cond, signals in condition_signals.items():
        threshold, meta = compute_session_pooled_threshold(
            signals,
            hz=hz,
            wcc_window_size=wcc_window_size,
            surrogate_n=surrogate_n,
            percentile=percentile,
            seed=seed,
            surrogate_method=surrogate_method,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
            fallback_threshold=fallback_threshold,
        )
        meta["mode"] = "condition_pooled"
        meta["condition"] = cond
        results[cond] = (threshold, meta)
    return results

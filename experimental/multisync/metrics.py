"""
Alternative synchrony metrics for SyncPipe.

Metrics:
    - WCLC (Windowed Cross-Lagged Correlation)
    - CRQA (Cross-Recurrence Quantification Analysis)
    - MI   (Mutual Information, binned estimator)
    - PLV  (Phase Locking Value, Hilbert-transform based)

Each metric follows the same interface: metric_func(a, b, **params) -> array
producing a time-varying coupling trace (like WCC).

Design note (DECISION-11):
    WCC remains the default metric across all paradigms.  WCLC is
    recommended for switching_rate in leader-follower paradigms.
    PLV is recommended for phase-dominated or oscillatory signals.
    CRQA and MI showed inferior detection in synthetic benchmarks
    (Appendix C) but remain available for nonlinear coupling research.

Benchmark framework matches the 5-layer surrogate validation approach.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# WCLC — Windowed Cross-Lagged Correlation
# ═══════════════════════════════════════════════════════════════════════

def wclc_synchrony(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 60,
    step: int = 10,
    max_lag_samples: int = 30,
    return_details: bool = False,
) -> np.ndarray:
    """
    Windowed Cross-Lagged Correlation.

    For each window, finds the lag ∈ [-L, +L] that maximizes |Pearson r|,
    capturing leader-follower dynamics that 0-lag WCC misses.

    **M1 FIX (2026-06)**: Now returns the SIGNED best_r (preserving polarity)
    by default.  This distinguishes in-phase (r > 0) from anti-phase (r < 0)
    coupling.  The lag selection criterion remains argmax|r| to handle both
    directions symmetrically.

    Recommended for switching_rate extraction in structured
    leader-follower paradigms (DECISION-11).

    Parameters
    ----------
    a, b : 1-D arrays, same length.
    window_size : window size in samples.
    step : step size between windows.
    max_lag_samples : maximum lag (in samples) to scan.
    return_details : bool, default False
        If True, return a dict with keys:
        - "wclc": 1-D array of signed best_r values
        - "best_lag": 1-D array of best lag (samples, positive = a leads b)
        - "best_r_abs": 1-D array of |best_r| (for backward compat)

    Returns
    -------
    np.ndarray or dict
        If return_details=False: 1-D array of signed best_r values.
        If return_details=True: dict with "wclc", "best_lag", "best_r_abs".
    """
    import warnings

    n = min(len(a), len(b))
    if n < window_size:
        if return_details:
            return {"wclc": np.array([]), "best_lag": np.array([]), "best_r_abs": np.array([])}
        return np.array([])

    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)

    n_windows = max(1, (n - window_size) // step + 1)
    rr = np.full(n_windows, np.nan)
    lags = np.full(n_windows, 0, dtype=int)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n:
            break
        aw = a[start:end]
        bw = b[start:end]
        best_r = 0.0
        best_lag = 0
        for lag in range(-max_lag_samples, max_lag_samples + 1):
            if lag < 0:
                x, y = aw[-lag:], bw[:end - start + lag] if end - start + lag > 0 else bw[:1]
            elif lag > 0:
                x, y = aw[:len(aw)-lag] if lag < len(aw) else aw, bw[lag:]
            else:
                x, y = aw, bw
            ml = min(len(x), len(y))
            if ml < 5:
                continue
            r = np.corrcoef(x[:ml], y[:ml])[0, 1]
            if np.isfinite(r) and abs(r) > abs(best_r):
                best_r = r
                best_lag = lag
        rr[i] = best_r  # SIGNED (M1 fix)
        lags[i] = best_lag

    if return_details:
        return {"wclc": rr, "best_lag": lags, "best_r_abs": np.abs(rr)}
    return rr


# ═══════════════════════════════════════════════════════════════════════
# CRQA — Cross-Recurrence Quantification Analysis
# ═══════════════════════════════════════════════════════════════════════

def crqa_synchrony(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 30,
    step: int = 5,
    threshold_percentile: float = 10.0,
    norm: bool = True,
) -> np.ndarray:
    """Sliding-window recurrence rate as a synchrony metric."""
    n = min(len(a), len(b))
    if n < window_size:
        return np.array([])

    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)

    if norm:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)

    dists = np.abs(np.subtract.outer(a, b))
    threshold = np.percentile(dists.ravel(), threshold_percentile)

    n_windows = max(1, (n - window_size) // step + 1)
    rr = np.full(n_windows, np.nan)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n:
            break
        aw = a[start:end]
        bw = b[start:end]
        dmat = np.abs(np.subtract.outer(aw, bw))
        rr[i] = np.mean(dmat < threshold)
    return rr


# ═══════════════════════════════════════════════════════════════════════
# MI — Mutual Information (binned estimator)
# ═══════════════════════════════════════════════════════════════════════

def mi_synchrony(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 30,
    step: int = 5,
    n_bins: int = 10,
    norm: bool = True,
) -> np.ndarray:
    """Sliding-window mutual information (binned, in nats)."""
    n = min(len(a), len(b))
    if n < window_size:
        return np.array([])

    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)

    if norm:
        a = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b = (b - np.mean(b)) / (np.std(b) + 1e-10)

    n_windows = max(1, (n - window_size) // step + 1)
    mi = np.full(n_windows, np.nan)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n:
            break
        aw = a[start:end]
        bw = b[start:end]
        H, _, _ = np.histogram2d(aw, bw, bins=n_bins, density=True)
        px = H.sum(axis=1) + 1e-10
        py = H.sum(axis=0) + 1e-10
        pxy = H + 1e-10
        Hx = -np.sum(px * np.log(px))
        Hy = -np.sum(py * np.log(py))
        Hxy = -np.sum(pxy.ravel() * np.log(pxy.ravel()))
        mi[i] = max(Hx + Hy - Hxy, 0.0)
    return mi


# ═══════════════════════════════════════════════════════════════════════
# PLV — Phase Locking Value
# ═══════════════════════════════════════════════════════════════════════

def plv_synchrony(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 30,
    step: int = 5,
    freq_band: Optional[Tuple[float, float]] = None,
    fs: float = 1.0,
) -> np.ndarray:
    """Sliding-window Phase Locking Value (0-1).

    **M2 FIX (2026-06)**: PLV assumes a narrow-band oscillatory signal so that
    the Hilbert-transform instantaneous phase is well-defined (Bedrosian's
    theorem).  When ``freq_band is None`` the broadband Hilbert phase is not
    physically meaningful and PLV can report spuriously high coupling that is a
    bandwidth artifact rather than true phase locking.  A ``RuntimeWarning`` is
    now emitted so callers do not silently report broadband PLV.
    """
    import warnings
    from scipy.signal import hilbert, butter, filtfilt

    n = min(len(a), len(b))
    if n < window_size:
        return np.array([])

    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)

    if freq_band is None:
        warnings.warn(
            "plv_synchrony called with freq_band=None (broadband). The Hilbert "
            "instantaneous phase is only well-defined for narrow-band signals; "
            "broadband PLV can report bandwidth-artifact coupling rather than "
            "true phase locking. Pass an explicit freq_band=(low, high) in Hz "
            "matching the oscillation of interest.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        nyq = fs / 2
        low, high = freq_band
        low_n = max(low / nyq, 0.01)
        high_n = min(high / nyq, 0.99)
        if low_n < high_n:
            b_bp, a_bp = butter(4, [low_n, high_n], btype="band")
            a = filtfilt(b_bp, a_bp, a)
            b = filtfilt(b_bp, a_bp, b)

    a_h = hilbert(a)
    b_h = hilbert(b)
    delta_phi = np.angle(a_h) - np.angle(b_h)

    n_windows = max(1, (n - window_size) // step + 1)
    plv = np.full(n_windows, np.nan)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n:
            break
        dp = delta_phi[start:end]
        plv[i] = np.abs(np.mean(np.exp(1j * dp)))
    return plv


# ═══════════════════════════════════════════════════════════════════════
# Benchmark framework
# ═══════════════════════════════════════════════════════════════════════

METRICS = {
    "wcc": None,
    "wclc": wclc_synchrony,
    "crqa": crqa_synchrony,
    "mi": mi_synchrony,
    "plv": plv_synchrony,
}


def benchmark_metrics(
    a: np.ndarray,
    b: np.ndarray,
    metrics: Optional[list] = None,
    window_size: int = 30,
    step: int = 5,
    hz: float = 1.0,
    max_lag: int = 30,
    crqa_threshold: float = 10.0,
    mi_bins: int = 10,
    plv_band: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute all selected synchrony metrics on the same input pair."""
    from multisync.dynamic_features import sliding_window_wcc

    if metrics is None:
        metrics = list(METRICS.keys())

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    results = {}
    kwargs_map = {
        "wcc": {"window_size": int(window_size * hz), "hz": hz, "step_samples": step},
        "wclc": {"window_size": int(window_size * hz), "step": step, "max_lag_samples": max_lag},
        "crqa": {"window_size": int(window_size * hz), "step": step, "threshold_percentile": crqa_threshold},
        "mi": {"window_size": int(window_size * hz), "step": step, "n_bins": mi_bins},
        "plv": {"window_size": int(window_size * hz), "step": step, "freq_band": plv_band, "fs": hz},
    }
    for m in metrics:
        if m == "wcc":
            results[m] = sliding_window_wcc(a, b, **kwargs_map[m])
        elif m in METRICS and METRICS[m] is not None:
            kwargs = kwargs_map.get(m, {})
            results[m] = METRICS[m](a, b, **kwargs)
    return results


def benchmark_feature_comparison(
    a: np.ndarray,
    b: np.ndarray,
    metrics: Optional[list] = None,
    window_size: int = 30,
    step: int = 5,
    hz: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Compute 6+2 features across all metrics and compare."""
    from multisync.dynamic_features import extract_dynamic_features

    traces = benchmark_metrics(a, b, metrics, window_size, step, hz)
    wcc_hz = hz / max(step, 1)
    results = {}
    for m, trace in traces.items():
        if len(trace) < 5 or np.all(np.isnan(trace)):
            results[m] = {}
            continue
        feats = extract_dynamic_features(trace, hz=wcc_hz, wcc_window_sec=window_size)
        d = {}
        if hasattr(feats, "__dict__"):
            d = dict(feats.__dict__)
        elif isinstance(feats, dict):
            d = feats
        else:
            for k in ["onset_latency", "rise_time", "peak_amplitude",
                       "recovery_time", "dwell_time", "switching_rate",
                       "mean_synchrony", "synchrony_entropy"]:
                v = getattr(feats, k, None)
                d[k] = float(v) if v is not None and np.isfinite(v) else np.nan
        results[m] = d
    return results

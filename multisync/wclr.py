"""
Windowed Cross-Lagged Regression (WCLR) backend.

WCLR estimates dyadic coupling while controlling for each partner's own
autocorrelation. It addresses a well-known limitation of windowed cross-
correlation (WCC): WCC can be inflated by slow autoregressive structure in
each signal, independent of true interpersonal coupling.

This module implements two WCLR variants:

1. ``metric="beta"`` (default): For each window and candidate lag, fit::

       y_t = beta_0 + beta_1 * y_{t-1} + beta_2 * x_{t-k} + epsilon

   and return the *standardized* beta_2. This is a partial-correlation-style
   measure that captures how much x at lag k predicts y after removing the
   autoregressive carry-over of y. It behaves well as a 1-D coupling trace
   for downstream feature extraction (mean, peak, dwell, etc.).

2. ``metric="r2"``: For each window and candidate lag, return the *increment*
   in R² from adding x_{t-k} to the model already containing y_{t-1}. This is
   the literature-standard WCLR summary (Schoenherr et al., 2019) and is
   bounded in [0, 1]. It is most informative when the research question is
   lead-lag prediction rather than concurrent correlation.

The final trace is the maximum absolute value across the candidate lag range,
capturing both concurrent and lead-lag coupling without requiring a single
pre-specified lag.

References
----------
Schoenherr, D., Paulick, J., Strauss, B. M., Deisenhofer, A. K., Schwartz, B.,
Stangier, U., & Rubel, J. A. (2019). Identification of movement synchrony:
Validation of windowed cross-lagged correlation and -regression with peak-
picking algorithm. PLoS ONE, 14(2), e0211494.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


__all__ = [
    "windowed_cross_lagged_regression",
    "wclr_coupling_trace",
]


def _standardized_beta(
    y: np.ndarray,
    X: np.ndarray,
    target_col: int,
) -> Optional[float]:
    """Return the standardized OLS coefficient for column ``target_col``.

    Returns None if regression is degenerate or non-finite.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yv = y[valid]
    Xv = X[valid]
    n = yv.size
    if n < Xv.shape[1] + 2:
        return None

    y_std = np.std(yv, ddof=1)
    if y_std <= 0:
        return None
    X_std = np.std(Xv, axis=0, ddof=1)
    X_std[X_std <= 0] = 1.0

    ys = (yv - np.mean(yv)) / y_std
    Xs = (Xv - np.mean(Xv, axis=0)) / X_std
    Xs = np.column_stack([np.ones(n), Xs])
    target_col += 1

    try:
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        if not np.isfinite(beta[target_col]):
            return None
        return float(beta[target_col])
    except Exception:
        return None


def _r2_increment(
    y: np.ndarray,
    X_ar: np.ndarray,
    x_lag: np.ndarray,
) -> Optional[float]:
    """Return the R² increment from adding x_lag to an AR-only model.

    Returns None if regression is degenerate.
    """
    y = np.asarray(y, dtype=float)
    X_ar = np.asarray(X_ar, dtype=float)
    x_lag = np.asarray(x_lag, dtype=float)

    valid = np.isfinite(y) & np.isfinite(X_ar) & np.isfinite(x_lag)
    yv = y[valid]
    arv = X_ar[valid]
    xv = x_lag[valid]
    n = yv.size

    if n < 4 or yv.std(ddof=1) <= 0:
        return None

    def _r2(resid, ycenter):
        return 1.0 - np.sum(resid ** 2) / np.sum(ycenter ** 2)

    yc = yv - yv.mean()

    X_ar_only = np.column_stack([np.ones(n), arv])
    try:
        beta_ar, *_ = np.linalg.lstsq(X_ar_only, yv, rcond=None)
        pred_ar = X_ar_only @ beta_ar
        r2_ar = _r2(yv - pred_ar, yc)
    except Exception:
        return None

    X_full = np.column_stack([np.ones(n), arv, xv])
    try:
        beta_full, *_ = np.linalg.lstsq(X_full, yv, rcond=None)
        pred_full = X_full @ beta_full
        r2_full = _r2(yv - pred_full, yc)
    except Exception:
        return None

    delta = float(r2_full - r2_ar)
    if delta < -1e-6:
        return None
    return max(delta, 0.0)


def windowed_cross_lagged_regression(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    max_lag_samples: int = 2,
    step_samples: int = 1,
    min_valid_ratio: float = 0.5,
    metric: str = "beta",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute WCLR time series between ``x`` and ``y``.

    Parameters
    ----------
    x, y : np.ndarray
        Two aligned time series (same length).
    window_size : int
        Window length in samples.
    hz : float
        Sampling rate (Hz); stored in output but not used for computation.
    max_lag_samples : int
        Maximum lag to explore in samples. Default 2. Lags are
        ``-max_lag_samples, ..., max_lag_samples``.
    step_samples : int
        Step between consecutive windows. Default 1 (every sample).
    min_valid_ratio : float
        Minimum fraction of valid finite pairs within a window for the
        coefficient to be computed. Default 0.5.
    metric : {"beta", "r2"}
        "beta" returns the standardized partial regression coefficient.
        "r2" returns the R² increment (literature standard, bounded [0,1]).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(wclr_trace, lag_trace)``. ``wclr_trace[i]`` is the maximum coupling
        estimate in the i-th window; ``lag_trace[i]`` is the lag (in samples)
        at which that maximum was achieved. Positive lag means ``x`` leads
        ``y``.
    """
    if metric not in ("beta", "r2"):
        raise ValueError(f"metric must be 'beta' or 'r2', got {metric!r}")

    n = len(x)
    if len(y) != n:
        raise ValueError(f"x and y must have same length: {n} vs {len(y)}")
    if window_size > n:
        return np.array([], dtype=float), np.array([], dtype=int)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lags = list(range(-max_lag_samples, max_lag_samples + 1))
    x_lagged: List[np.ndarray] = []
    for k in lags:
        xl = np.full(n, np.nan, dtype=float)
        if k > 0:
            xl[k:] = x[:-k]
        elif k < 0:
            xl[:k] = x[-k:]
        else:
            xl = x.copy()
        x_lagged.append(xl)

    y_lag = np.full(n, np.nan, dtype=float)
    y_lag[1:] = y[:-1]

    n_windows = (n - window_size) // step_samples + 1
    wclr_trace = np.full(n_windows, np.nan, dtype=float)
    lag_trace = np.full(n_windows, 0, dtype=int)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_size
        if end > n:
            break

        y_win = y[start:end]
        y_lag_win = y_lag[start:end]
        min_valid = int(min_valid_ratio * window_size)
        if (np.isfinite(y_win).sum() < min_valid or
                np.isfinite(y_lag_win).sum() < min_valid):
            continue

        best_value = 0.0
        best_lag_idx = 0
        for li, xl in enumerate(x_lagged):
            x_win = xl[start:end]
            if np.isfinite(x_win).sum() < min_valid:
                continue

            if metric == "beta":
                X = np.column_stack([y_lag_win, x_win])
                value = _standardized_beta(y_win, X, target_col=1)
                if value is None or not np.isfinite(value):
                    continue
                value = abs(value)
            else:  # metric == "r2"
                value = _r2_increment(y_win, y_lag_win, x_win)
                if value is None or not np.isfinite(value):
                    continue

            if value > best_value:
                best_value = value
                best_lag_idx = li

        wclr_trace[i] = best_value
        lag_trace[i] = lags[best_lag_idx]

    return wclr_trace, lag_trace


def wclr_coupling_trace(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    max_lag_samples: int = 2,
    step_samples: int = 1,
    min_valid_ratio: float = 0.5,
    metric: str = "beta",
) -> np.ndarray:
    """Convenience wrapper returning only the WCLR coupling trace.

    Parameters
    ----------
    metric : {"beta", "r2"}
        See :func:`windowed_cross_lagged_regression`.
    """
    trace, _ = windowed_cross_lagged_regression(
        sig_a, sig_b, window_size, hz, max_lag_samples, step_samples, min_valid_ratio, metric
    )
    return trace

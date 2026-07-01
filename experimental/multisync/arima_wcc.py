"""
ARIMA Pre-Whitening for WCC — reducing ISC confounding.
=========================================================

v0.1 — 2026-06-07

Purpose
-------
In interpersonal synchrony research, two subjects' physiological signals
often share low-frequency trends for reasons OTHER than genuine dyadic
coupling.  The most common confound is ISC (Inter-Subject Correlation):
both subjects respond similarly to a shared stimulus or task demand,
creating spurious synchrony in WCC/CCF estimates.

ARIMA pre-whitening addresses this by:
  1. Fitting an optimal ARIMA model to EACH subject individually
     (capturing their own autocorrelation structure and trends).
  2. Extracting the model RESIDUALS — the "unpredictable" component
     after removing the subject's own temporal dynamics.
  3. Computing WCC on the residuals — any remaining cross-correlation
     is more likely to reflect genuine interpersonal coupling rather
     than shared task responses.

This approach follows Boukarras et al. (2025, Psychophysiology), who
used auto.arima(d=1) pre-whitening with CCF to demonstrate that
physiological synchrony increases during joint action tasks.

When to use (and NOT to use)
----------------------------
USE pre-whitening when:
  - Your paradigm has a strong shared stimulus/task (ISC risk is high)
  - Both subjects are exposed to the same external events
  - You have LONG recordings (> 200 samples per subject)
  - You want to distinguish "genuine interpersonal coupling" from
    "shared stimulus response"

DO NOT use pre-whitening when:
  - Your recordings are SHORT (< 100 samples) — ARIMA needs enough data
    to reliably estimate model parameters
  - The subjects are in DIFFERENT environments (no shared stimulus →
    ISC is not a concern)
  - You PRIORITIZE sensitivity over specificity — pre-whitening makes
    WCC values smaller (removes shared low-frequency variance), which
    may cause more epoch features to be undefined (onset never crosses
    threshold)

Implementation notes
--------------------
- Uses statsmodels.tsa.arima.model.ARIMA (not pmdarima's auto_arima
  to avoid the heavy pmdarima dependency)
- Order selection: tries (1,1,1) as default (matching Boukarras's d=1),
  can optionally do a grid search over p,q ∈ {0,1,2}
- WARNING is raised if n_samples < 100
- NaN handling: ARIMA residuals are NaN where original data has NaN;
  filtered out before WCC computation

References
----------
Boukarras, S., et al. (2025). Interpersonal Physiological Synchrony
During Dyadic Joint Action Is Increased by Task. Psychophysiology.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum recommended sample size for reliable ARIMA fitting
MIN_SAMPLES_ARIMA: int = 100


def arima_residuals(
    x: np.ndarray,
    order: Tuple[int, int, int] = (1, 1, 1),
    auto_order: bool = False,
) -> np.ndarray:
    """
    Fit ARIMA and return whitened residuals.

    Parameters
    ----------
    x : 1-D array
        Time series to pre-whiten.
    order : tuple (p, d, q)
        ARIMA order.  Default (1,1,1) is a reasonable starting point
        for most physiological signals (matching Boukarras's d=1).
    auto_order : bool
        If True, try grid search over p,q ∈ {0,1,2} and select by AIC.
        Slower but more adaptive.

    Returns
    -------
    residuals : 1-D array, same length as x
        NaN where original data is NaN; finite elsewhere.

    Raises
    ------
    ValueError if n_valid < MIN_SAMPLES_ARIMA.
    """
    from statsmodels.tsa.arima.model import ARIMA

    x = np.asarray(x, dtype=float)
    valid_mask = np.isfinite(x)
    n_valid = valid_mask.sum()

    if n_valid < MIN_SAMPLES_ARIMA:
        raise ValueError(
            f"ARIMA pre-whitening requires at least {MIN_SAMPLES_ARIMA} "
            f"valid samples, got {n_valid}.  Use raw WCC or collect "
            f"longer recordings."
        )

    if auto_order:
        order = _select_arima_order(x[valid_mask])

    try:
        model = ARIMA(x[valid_mask], order=order)
        fitted = model.fit()
        resid_valid = fitted.resid
    except Exception as e:
        logger.warning(
            f"ARIMA({order}) fitting failed: {e}.  "
            f"Falling back to order=(1,0,0)."
        )
        model = ARIMA(x[valid_mask], order=(1, 0, 0))
        fitted = model.fit()
        resid_valid = fitted.resid

    # Reconstruct full-length array with NaN preserving
    residuals = np.full(len(x), np.nan)
    residuals[valid_mask] = resid_valid
    return residuals


def _select_arima_order(
    x_valid: np.ndarray,
    p_range: Tuple[int, int] = (0, 2),
    q_range: Tuple[int, int] = (0, 2),
) -> Tuple[int, int, int]:
    """Grid search over p,q for best AIC (d=1 fixed)."""
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(p_range[0], p_range[1] + 1):
        for q in range(q_range[0], q_range[1] + 1):
            try:
                model = ARIMA(x_valid, order=(p, 1, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 1, q)
            except Exception:
                continue
    return best_order


def sliding_window_wcc_arima(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 30,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    auto_order: bool = False,
    **wcc_kwargs,
) -> np.ndarray:
    """
    Compute sliding-window WCC on ARIMA-pre-whitened signals.

    This is the convenience entry point: fits ARIMA to each signal,
    extracts residuals, then calls the standard sliding_window_wcc.

    Parameters
    ----------
    a, b : 1-D arrays, same length.
    window_size : int
        WCC window size in samples.
    arima_order : tuple
        Passed to arima_residuals().
    auto_order : bool
        If True, auto-select ARIMA order via AIC grid search.
    **wcc_kwargs :
        Forwarded to sliding_window_wcc (hz, step_samples, etc.)

    Returns
    -------
    wcc : 1-D array
        Windowed cross-correlation of ARIMA residuals.

    Warns
    -----
    UserWarning if n_valid < 200 (recommended minimum for reliable
    ARIMA fitting in physiological applications).

    See Also
    --------
    multisync.dynamic_features.sliding_window_wcc
    """
    from .dynamic_features import sliding_window_wcc

    n = min(len(a), len(b))

    if n < 200:
        warnings.warn(
            f"ARIMA pre-whitening is recommended for n >= 200 samples. "
            f"Got n={n}.  Results may be unreliable due to insufficient "
            f"data for ARIMA model estimation.  Consider using the default "
            f"raw WCC path instead.",
            UserWarning,
            stacklevel=2,
        )

    a_resid = arima_residuals(a, order=arima_order, auto_order=auto_order)
    b_resid = arima_residuals(b, order=arima_order, auto_order=auto_order)

    # Remove NaN positions (ARIMA may leave edge NaN)
    valid = np.isfinite(a_resid) & np.isfinite(b_resid)
    if valid.sum() < window_size:
        warnings.warn(
            f"After ARIMA pre-whitening and NaN removal, only "
            f"{valid.sum()} valid samples remain (need ≥{window_size} "
            f"for WCC). Returning empty array.",
            UserWarning,
            stacklevel=2,
        )
        return np.array([])

    return sliding_window_wcc(
        a_resid[valid],
        b_resid[valid],
        window_size=window_size,
        **wcc_kwargs,
    )

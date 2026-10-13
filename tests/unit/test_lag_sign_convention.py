"""Lag sign convention: positive lag means x leads y (review R3, 2026-10-13).

Pinned by a known-answer fixture because the docstring and the
implementation once disagreed (audit m13, 2026-09-14, corrected with the
implementation in be8b76d): a regression here would silently mirror every
lagged analysis, and nothing else in the suite constrains the direction.

The convention is bsync-compatible ("positive lag = x leads y") so
surfaces from the two engines can be compared cell by cell.
"""
from __future__ import annotations

import numpy as np

from syncpipe.wcc import sliding_window_wcc


def _x_leads_y_by(n: int, lead: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y[t] = x[t - lead]: x's past appears in y's present (x leads y)."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    x = driver.copy()
    y = np.full(n, np.nan)
    y[lead:] = driver[:-lead]
    return x, y


def test_positive_lag_means_x_leads_y():
    """Known-answer: lag=+5 gives r≈1 (aligned), lag=-5 does not."""
    x, y = _x_leads_y_by(200, 5)
    r_pos = sliding_window_wcc(x, y, 20, lag_samples=5)
    r_neg = sliding_window_wcc(x, y, 20, lag_samples=-5)
    assert float(np.nanmean(r_pos)) > 0.999, (
        "lag=+5 must align an x-leads-y-by-5 pair (positive lag = x leads y); "
        f"got mean r = {float(np.nanmean(r_pos)):.4f}"
    )
    assert float(np.nanmean(r_neg)) < 0.5, (
        "lag=-5 must NOT align the pair; if it does, the lag sign convention "
        "has been mirrored"
    )


def test_negative_lag_recovers_a_negative_lead():
    """Symmetric check: y leads x by 5 -> lag=-5 aligns."""
    y, x = _x_leads_y_by(200, 5)  # swap roles: now x follows y
    r = sliding_window_wcc(x, y, 20, lag_samples=-5)
    assert float(np.nanmean(r)) > 0.999


def test_zero_lag_identical_series():
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    r = sliding_window_wcc(x, x, 20, lag_samples=0)
    assert float(np.nanmean(r)) > 0.999

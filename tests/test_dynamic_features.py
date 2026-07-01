"""Tests for dynamic feature extraction (WCC, episode detection, etc.)."""

import numpy as np
import pytest
from multisync.dynamic_features import sliding_window_wcc


# ============================================================
# WCC cumsum regression guard
# ============================================================

@pytest.mark.parametrize("offset", [0.0, 5.0, 72.0, 1000.0])
def test_wcc_invariant_under_global_mean_shift(offset):
    """
    Pearson correlation is invariant to additive shifts. The cumsum
    fast path MUST yield identical results to a naive per-window
    np.corrcoef regardless of the global mean of the signal.

    Regression guard for the v0.x covariance-formula bug, where
    omitting the ``- mean_x*mean_y`` correction caused a silent,
    direction-consistent ~0.02 bias on real physiological data
    (HR ~72 BPM, SCL ~5 microS) while remaining invisible on zero-mean
    synthetic data.
    """
    rng = np.random.default_rng(0)
    n, w = 600, 50
    x = rng.normal(0, 1, n) + offset
    y = 0.6 * (x - offset) + rng.normal(0, 0.4, n) + offset

    fast = sliding_window_wcc(x, y, window_size=w)
    naive = np.array([
        np.corrcoef(x[i:i + w], y[i:i + w])[0, 1]
        for i in range(n - w + 1)
    ])
    max_abs_err = float(np.max(np.abs(fast - naive)))
    assert max_abs_err < 1e-9, (
        f"WCC cumsum path drifts under global-mean shift "
        f"(offset={offset}): max abs error = {max_abs_err:.4e}"
    )

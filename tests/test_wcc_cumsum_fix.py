"""
Regression test for _sliding_window_wcc_cumsum covariance fix.

Ensures the cumsum (O(n) memory) path produces numerically
identical results to naive np.corrcoef for windows where
the signal mean is large relative to its variance — the exact
condition that exposed the missing "mean_x * mean_y" correction term.

See: dynamic_features.py :: _sliding_window_wcc_cumsum
"""
import numpy as np
import pytest

from multisync.dynamic_features import sliding_window_wcc


class TestCumsumWccCorrectness:
    """WCC cumsum must match naive Pearson correlation."""

    def _naive_wcc(self, x, y, window_size):
        """Ground-truth via numpy.corrcoef for every window."""
        n = len(x)
        return np.array([
            np.corrcoef(x[i:i + window_size], y[i:i + window_size])[0, 1]
            for i in range(n - window_size + 1)
        ])

    def test_large_mean_signal(self):
        """
        Stress catastrophic cancellation.

        Previously, var_x=sum_x2/w  (missing -mean_x**2) and
        cov=sum_xy/w  (missing -mean_x*mean_y) produced systematically
        wrong WCC when the global mean was large.
        """
        rng = np.random.default_rng(0)
        n, w = 500, 50
        x = rng.normal(0, 1.0, n) + 5000.0
        y = 0.7 * x + rng.normal(0, 0.5, n) + 10000.0

        wcc_fast = sliding_window_wcc(x, y, window_size=w)
        wcc_ref = self._naive_wcc(x, y, w)

        # All windows must agree to machine precision
        assert np.allclose(wcc_fast, wcc_ref, atol=1e-9), \
            f"max diff = {np.max(np.abs(wcc_fast - wcc_ref))}"

    def test_zero_mean_signal(self):
        """Zero-mean signals — the correction terms vanish, should still pass."""
        rng = np.random.default_rng(1)
        n, w = 300, 30
        x = rng.normal(0, 1.0, n)
        y = 0.5 * x + rng.normal(0, 0.3, n)

        wcc_fast = sliding_window_wcc(x, y, window_size=w)
        wcc_ref = self._naive_wcc(x, y, w)

        assert np.allclose(wcc_fast, wcc_ref, atol=1e-9)

    def test_against_stride_path(self):
        """
        Cumsum and stride paths must agree on clean (no-NaN) data.

        When NaNs are present, the stride path uses a pre-filtering
        strategy that computes valid_ratio before any arithmetic
        (H6 fix), so the set of valid windows may differ from the
        cumsum path's post-hoc filtering.  This is expected — the
        two paths are designed to handle NaN differently, and the
        stride path is more conservative.  We therefore only compare
        on clean data where both paths should produce identical results.
        """
        rng = np.random.default_rng(2)
        n, w = 400, 40
        x = rng.normal(5.0, 1.0, n)   # non-zero mean
        y = rng.normal(-3.0, 2.0, n)

        # Both inputs are clean — cumsum path should be triggered
        wcc_cumsum = sliding_window_wcc(x, y, window_size=w)

        # Force stride path by inserting a single NaN far from edges,
        # but compare only windows that don't overlap the NaN position.
        x_nan = x.copy()
        x_nan[10] = np.nan
        wcc_stride = sliding_window_wcc(x_nan, y, window_size=w)

        # Windows that don't overlap position 10 (index range [10-39, 10])
        # are unaffected by the NaN and should match cumsum exactly.
        unaffected = np.arange(len(wcc_cumsum))
        unaffected = unaffected[unaffected >= w]  # skip windows overlapping NaN

        assert np.allclose(wcc_cumsum[unaffected], wcc_stride[unaffected], atol=1e-9), \
            f"cumsum vs stride max diff on unaffected windows = " \
            f"{np.max(np.abs(wcc_cumsum[unaffected] - wcc_stride[unaffected]))}"

    def test_output_bounds(self):
        """WCC must always be in [-1, 1]."""
        rng = np.random.default_rng(3)
        n, w = 200, 25
        x = rng.normal(0, 1.0, n)
        y = rng.normal(0, 1.0, n)

        wcc = sliding_window_wcc(x, y, window_size=w)
        assert np.all(wcc >= -1.0) and np.all(wcc <= 1.0), \
            "WCC outside [-1, 1]"

"""Regression tests for BUG-1: tapered-window WCC backend consistency.

The cumsum WCC backend applies the taper kernel as a *globally tiled* weight
(weight[i] = kern[i % window_size]).  For a sliding window of stride 1 this is
only correct for the uniform ('rect') kernel; a non-rect taper is phase-shifted
for off-boundary windows, producing a WRONG WCC.  Tapered windows must use the
stride backend (per-window kernel alignment).  These tests encode that intent.

Intent (why this matters):
  * A researcher following the docstring's "use 'hann' for psychophysiology"
    recommendation must get the *correct* WCC, not a silently phase-shifted one.
  * Injecting a single NaN must NOT change the WCC of unaffected windows
    (previously NaN toggled the backend, changing every value).
"""

import numpy as np
import pytest

from multisync.dynamic_features import (
    sliding_window_wcc,
    _make_window_kernel,
    _sliding_window_wcc_stride,
    _sliding_window_wcc_cumsum,
)


def _bruteforce_tapered_wcc(x, y, w, window_type):
    """Independent ground truth: explicit per-window weighted Pearson r."""
    kern = _make_window_kernel(window_type, w)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    xg = x - mx
    yg = y - my
    n = len(x)
    out = np.full(n - w + 1, np.nan)
    for i in range(n - w + 1):
        xw = xg[i : i + w]
        yw = yg[i : i + w]
        we = kern
        Wt = float(we.sum())
        sx = float((we * xw).sum())
        sy = float((we * yw).sum())
        sxy = float((we * xw * yw).sum())
        sx2 = float((we * xw ** 2).sum())
        sy2 = float((we * yw ** 2).sum())
        mean_x = sx / Wt
        mean_y = sy / Wt
        cov = sxy / Wt - mean_x * mean_y
        vx = max(sx2 / Wt - mean_x ** 2, 0.0)
        vy = max(sy2 / Wt - mean_y ** 2, 0.0)
        d = (vx * vy) ** 0.5
        if d > 1e-10:
            out[i] = min(1.0, max(-1.0, cov / d))
    return out


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    n = 500
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50.0) + 0.3 * rng.standard_normal(n)
    y = np.sin(2 * np.pi * t / 50.0 + 0.5) + 0.3 * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize("window_type", ["hann", "hamming", "triang"])
def test_tapered_wcc_matches_bruteforce_per_window(pair, window_type):
    """Tapered WCC must equal the correct per-window weighted Pearson r."""
    x, y = pair
    w = 50
    got = sliding_window_wcc(x, y, w, window_type=window_type)
    exp = _bruteforce_tapered_wcc(x, y, w, window_type)
    assert got.shape == exp.shape
    mask = np.isfinite(exp)
    diff = np.max(np.abs(got[mask] - exp[mask]))
    assert np.allclose(got[mask], exp[mask], atol=1e-6), (
        f"{window_type}: tapered WCC deviates from correct per-window value "
        f"(max diff {diff:.2e})"
    )


def test_tapered_wcc_does_not_use_buggy_cumsum_backend(pair):
    """Tapered WCC must NOT equal the (phase-shifted) cumsum output."""
    x, y = pair
    w = 50
    got = sliding_window_wcc(x, y, w, window_type="hann")
    cumsum_out = _sliding_window_wcc_cumsum(x, y, w, "hann")
    # The cumsum backend applies the taper globally-tiled, which is wrong for
    # non-rect windows.  If the dispatcher still routed hann -> cumsum, these
    # would match.  They must NOT.
    assert not np.allclose(got, cumsum_out, atol=1e-6), (
        "tapered WCC unexpectedly matches the cumsum (phase-shifted) output"
    )


def test_tapered_wcc_routes_to_stride_backend(pair):
    """Both NaN-free and NaN-infested tapered inputs use the stride backend."""
    x, y = pair
    w = 50
    clean = sliding_window_wcc(x, y, w, window_type="hann")
    clean_stride = _sliding_window_wcc_stride(x, y, w, 0.5, "hann")
    assert np.allclose(clean, clean_stride, atol=1e-9)

    xn, yn = x.copy(), y.copy()
    xn[10] = np.nan
    yn[10] = np.nan
    dirty = sliding_window_wcc(xn, yn, w, window_type="hann")
    dirty_stride = _sliding_window_wcc_stride(xn, yn, w, 0.5, "hann")
    # NaN window positions must match; finite parts must match to 1e-9.
    assert np.array_equal(
        np.isfinite(dirty), np.isfinite(dirty_stride)
    ), "NaN window positions differ between dispatcher and stride backend"
    finite = np.isfinite(dirty) & np.isfinite(dirty_stride)
    assert np.allclose(dirty[finite], dirty_stride[finite], atol=1e-9)


def test_rect_wcc_keeps_fast_cumsum_path(pair):
    """'rect' (no NaN) must stay on the fast cumsum backend and stay correct."""
    x, y = pair
    w = 50
    got = sliding_window_wcc(x, y, w, window_type="rect")
    exp_cumsum = _sliding_window_wcc_cumsum(x, y, w, "rect")
    exp_brute = _bruteforce_tapered_wcc(x, y, w, "rect")
    assert np.allclose(got, exp_cumsum, atol=1e-9)
    assert np.allclose(got, exp_brute, atol=1e-9)

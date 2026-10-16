"""Properties a correlation surface must have whatever the data (the theory-facing contracts)."""
from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from syncpipe.surface import wcc_surface  # noqa: E402

# float64 keeps ~16 significant digits. A signal riding on an offset N times its own scale keeps only
# 16 - log10(N) digits of its variation, so exact invariance cannot hold for arbitrary offsets (hypothesis
# found scale=1e-9 on an offset of 2.0 losing the 8th digit). The property is therefore stated for offsets
# up to 1e3 signal scales -- far beyond any physiological recording -- and that limit is part of the claim.
_offset_in_scales = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


def _pair(seed: int, n: int = 90):
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, mode="same")[4:4 + n]
    return x, 0.5 * x + np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, mode="same")[4:4 + n]


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), scale_x=st.floats(1e-9, 1e6), scale_y=st.floats(1e-9, 1e6),
       offset_x=_offset_in_scales, offset_y=_offset_in_scales)
def test_surface_is_invariant_to_positive_affine_changes_of_either_series(seed, scale_x, scale_y, offset_x, offset_y):
    """Pearson r does not depend on units: rescaling or shifting a series must not move any cell."""
    x, y = _pair(seed)
    ref = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=5)
    other = wcc_surface(x * scale_x + offset_x * scale_x, y * scale_y + offset_y * scale_y, hz=1.0, window_samples=14, lag_max_samples=5)
    assert np.allclose(other.values, ref.values, atol=1e-7, equal_nan=True)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_negating_one_series_negates_every_cell(seed):
    x, y = _pair(seed)
    ref = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=5)
    assert np.allclose(wcc_surface(x, -y, hz=1.0, window_samples=14, lag_max_samples=5).values, -ref.values, atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), lag=st.integers(1, 5))
def test_swapping_the_series_mirrors_the_lag_axis_with_a_shifted_grid(seed, lag):
    """S_yx(-tau)[k] = S_xy(tau)[k - tau]: the same pair of windows, seen from the other series."""
    x, y = _pair(seed, 100)
    xy = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=5)
    yx = wcc_surface(y, x, hz=1.0, window_samples=14, lag_max_samples=5)
    row_xy, row_yx = int(np.where(xy.lags_samples == lag)[0][0]), int(np.where(yx.lags_samples == -lag)[0][0])
    assert np.allclose(yx.values[row_yx, lag:], xy.values[row_xy, :-lag], atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), hz=st.sampled_from([0.5, 1.0, 4.0, 32.0]))
def test_the_sampling_rate_only_relabels_the_time_axes(seed, hz):
    """The values depend on samples, not on Hz; only the second-based axes move."""
    x, y = _pair(seed)
    a = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=5)
    b = wcc_surface(x, y, hz=hz, window_samples=14, lag_max_samples=5)
    assert np.array_equal(a.values, b.values)
    assert np.allclose(b.lags_sec * hz, a.lags_sec) and np.allclose(b.window_center_sec * hz, a.window_center_sec)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), trial_len=st.integers(24, 40), n_trials=st.integers(2, 4))
def test_stacking_trials_with_seam_markers_equals_per_trial_correlation(seed, trial_len, n_trials):
    """Concatenation with the seam convention is the same measurement as computing each trial alone.

    Every window that lies inside one trial (and does not start on the marked first sample of a later
    trial) must equal that window computed on the trial by itself; no window may straddle a seam.
    """
    from syncpipe.wcc import _apply_discontinuity_mask, sliding_window_wcc

    window = 8
    rng = np.random.default_rng(seed)
    trials = [(rng.normal(size=trial_len), rng.normal(size=trial_len)) for _ in range(n_trials)]
    a = np.concatenate([t[0] for t in trials])
    b = np.concatenate([t[1] for t in trials])
    mask = np.ones(a.size, dtype=bool)
    mask[trial_len::trial_len] = False
    stacked = _apply_discontinuity_mask(sliding_window_wcc(a, b, window), mask, window)
    for k, (ta, tb) in enumerate(trials):
        alone = sliding_window_wcc(ta, tb, window)
        first = 0 if k == 0 else 1                         # a later trial's first sample is the marked one
        for start in range(first, trial_len - window + 1):
            assert stacked[k * trial_len + start] == pytest.approx(alone[start], abs=1e-9)
    straddling = [s for s in range(a.size - window + 1) if (s // trial_len) != ((s + window - 1) // trial_len)]
    assert np.isnan(stacked[straddling]).all()


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), small=st.integers(1, 3), extra=st.integers(1, 3))
def test_the_maximum_over_a_fixed_window_set_is_nondecreasing_in_the_lag_range(seed, small, extra):
    """On the SAME windows, opening more lags can only raise the surface maximum (trivially a superset).

    The bsync grid trims windows when the lag range grows, so the raw maximum is not monotone in
    lag_max across surfaces; this property is the exact statement that holds, and the reason a peak is
    always referenced to a null built on the same grid.
    """
    x, y = _pair(seed, 110)
    narrow = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=small)
    wide = wcc_surface(x, y, hz=1.0, window_samples=14, lag_max_samples=small + extra)
    common = np.intersect1d(narrow.window_start_samples, wide.window_start_samples)
    cols_n = np.searchsorted(narrow.window_start_samples, common)
    cols_w = np.searchsorted(wide.window_start_samples, common)
    rows_n = np.isin(wide.lags_samples, narrow.lags_samples)
    assert np.nanmax(wide.values[:, cols_w]) >= np.nanmax(narrow.values[:, cols_n]) - 1e-12
    assert np.allclose(wide.values[rows_n][:, cols_w], narrow.values[:, cols_n], atol=1e-12, equal_nan=True)

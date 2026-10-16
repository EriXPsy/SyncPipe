"""SyncSurface: grid conventions, derived quantities, validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syncpipe.surface import Z_CLIP, SurfaceSettings, SyncSurface, r_to_z, wcc_surface
from syncpipe.wcc import sliding_window_wcc


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    base = np.convolve(rng.normal(size=n + 20), np.ones(5) / 5, mode="same")
    return base[10:10 + n], base[10:10 + n] * 0.6 + 0.8 * np.convolve(rng.normal(size=n + 20), np.ones(5) / 5, mode="same")[10:10 + n]


# ---------------------------------------------------------------- oracle ----
def test_every_cell_matches_a_brute_force_pearson():
    """Independent oracle: np.corrcoef on explicit slices, with bsync's grid convention."""
    x, y = _series(70)
    window, lag_max, inc = 12, 4, 1
    surface = wcc_surface(x, y, hz=1.0, window_samples=window, lag_max_samples=lag_max, window_increment=inc)
    n_windows = (x.size - window + 1 - 2 * lag_max) // inc
    assert surface.values.shape == (2 * lag_max + 1, n_windows)
    for row, tau in enumerate(range(-lag_max, lag_max + 1)):
        for col in range(n_windows):
            start = lag_max + col * inc
            expected = np.corrcoef(x[start:start + window], y[start + tau:start + tau + window])[0, 1]
            assert surface.values[row, col] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("window_increment, lag_increment", [(1, 1), (3, 2), (5, 1)])
def test_grid_follows_bsync_formula_for_increments(window_increment, lag_increment):
    x, y = _series(100)
    s = wcc_surface(x, y, hz=1.0, window_samples=20, lag_max_samples=6,
                    window_increment=window_increment, lag_increment=lag_increment)
    assert s.values.shape[1] == (100 - 20 + 1 - 12) // window_increment
    assert list(s.lags_samples) == list(range(-6, 7, lag_increment))
    assert s.window_start_samples[0] == 6 and np.all(np.diff(s.window_start_samples) == window_increment)


def test_single_lag_surface_is_bitwise_the_existing_zero_lag_wcc():
    x, y = _series(150)
    s = wcc_surface(x, y, hz=1.0, window_samples=25)
    assert s.values.shape == (1, 126)
    assert np.array_equal(s.values[0], sliding_window_wcc(x, y, 25))


# --------------------------------------------------------- lag convention ----
def test_positive_lag_means_x_leads_y():
    rng = np.random.default_rng(3)
    base = np.convolve(rng.normal(size=300), np.ones(6) / 6, mode="same")
    x, y = base[20:260], base[17:257]                 # y(t) = x(t-3): x happens first
    s = wcc_surface(x, y, hz=2.0, window_samples=40, lag_max_samples=8)
    assert s.summary().lag_of_peak_sec == pytest.approx(3 / 2.0)
    swapped = wcc_surface(y, x, hz=2.0, window_samples=40, lag_max_samples=8)
    assert swapped.summary().lag_of_peak_sec == pytest.approx(-3 / 2.0)


# ------------------------------------------------------------------- axes ----
def test_time_axes_use_the_window_centre_and_the_origin():
    x, y = _series(80)
    s = wcc_surface(x, y, hz=4.0, window_samples=9, lag_max_samples=2, time_origin_sec=10.0)
    assert s.window_start_sec[0] == pytest.approx(10.0 + 2 / 4.0)
    assert s.window_center_sec[0] == pytest.approx(10.0 + (2 + 4) / 4.0)       # (9-1)/2 = 4 samples
    assert np.allclose(s.lags_sec, np.arange(-2, 3) / 4.0)


def test_window_and_lag_can_be_given_in_seconds():
    x, y = _series(200)
    by_sec = wcc_surface(x, y, hz=4.0, window_sec=5.0, lag_max_sec=1.0)
    by_samples = wcc_surface(x, y, hz=4.0, window_samples=20, lag_max_samples=4)
    assert by_sec.content_hash() == by_samples.content_hash()


# -------------------------------------------------------------- summaries ----
def test_summary_matches_a_hand_computation():
    x, y = _series(90)
    s = wcc_surface(x, y, hz=1.0, window_samples=16, lag_max_samples=5)
    z = np.arctanh(np.clip(s.values, -Z_CLIP, Z_CLIP))
    out = s.summary()
    assert out.mean_z == pytest.approx(np.mean(z))
    assert out.mean_abs_z == pytest.approx(np.mean(np.abs(z)))
    assert out.peak_abs_z == pytest.approx(np.mean(np.max(np.abs(z), axis=0)))      # bsync 'peak'
    assert out.peak_pos_z == pytest.approx(np.mean(np.max(z, axis=0)))
    assert out.lag_of_peak_sec == s.lags_sec[int(np.argmax(np.mean(z, axis=1)))]
    assert out.n_cells_valid == z.size and out.fraction_valid == 1.0


def test_summary_of_a_single_lag_surface_reports_lag_zero():
    x, y = _series(60)
    assert wcc_surface(x, y, hz=1.0, window_samples=10).summary().lag_of_peak_sec == 0.0


def test_perfect_fit_is_finite_in_z():
    assert np.isfinite(r_to_z(1.0)) and np.isfinite(r_to_z(-1.0))
    assert r_to_z(1.0) == pytest.approx(np.arctanh(Z_CLIP))


def test_missing_data_yields_nan_cells_not_errors_and_all_nan_summary_is_nan_without_warnings(recwarn):
    x, y = _series(60)
    y = y.copy()
    y[:] = np.nan
    s = wcc_surface(x, y, hz=1.0, window_samples=10, lag_max_samples=2)
    assert np.isnan(s.values).all()
    out = s.summary()
    assert out.n_cells_valid == 0 and np.isnan(out.mean_z) and np.isnan(out.lag_of_peak_sec)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ------------------------------------------------------------ tabular ----
def test_tidy_glance_and_bsync_frames_have_the_documented_shape():
    x, y = _series(80)
    s = wcc_surface(x, y, hz=2.0, window_samples=10, lag_max_samples=3)
    tidy, bs, one = s.tidy(), s.to_bsync_frame(), s.glance()
    assert len(tidy) == s.values.size == len(bs)
    assert list(bs.columns) == ["i", "tau", "wcc"]
    assert bs["i"].iloc[0] == s.window_start_samples[0] + 1                       # 1-based, as bsync
    assert bs["tau"].iloc[0] == -3 and bs["tau"].iloc[s.values.shape[1]] == -2       # lag-major order
    assert {"window_center_sec", "lag_sec", "value"} <= set(tidy.columns)
    assert len(one) == 1 and one["surface_hash"].iloc[0] == s.content_hash()
    pd.testing.assert_series_equal(tidy["value"], pd.Series(s.values.ravel(), name="value"), check_names=False)


# ---------------------------------------------------------------- hashing ----
def test_hash_is_deterministic_and_sensitive_to_values_and_settings():
    x, y = _series(80)
    a = wcc_surface(x, y, hz=1.0, window_samples=10, lag_max_samples=2)
    assert a.content_hash() == wcc_surface(x, y, hz=1.0, window_samples=10, lag_max_samples=2).content_hash()
    assert a.content_hash() != wcc_surface(x + 0.0, y[::-1].copy(), hz=1.0, window_samples=10, lag_max_samples=2).content_hash()
    assert a.content_hash() != wcc_surface(x, y, hz=2.0, window_samples=10, lag_max_samples=2).content_hash()


def test_arrays_are_read_only():
    x, y = _series(60)
    s = wcc_surface(x, y, hz=1.0, window_samples=10)
    with pytest.raises(ValueError):
        s.values[0, 0] = 0.0


# ------------------------------------------------------------- validation ----
@pytest.mark.parametrize("kwargs, message", [
    (dict(window_samples=10, window_sec=2.0), "either in samples or in seconds"),
    (dict(), "window length is required"),
    (dict(window_samples=2), "at least 3 samples"),
    (dict(window_sec=2.3), "whole number of samples"),
    (dict(window_samples=10, lag_max_samples=6), "exceeds half the window"),
    (dict(window_samples=60, lag_max_samples=20), "series too short"),
])
def test_invalid_requests_fail_loudly(kwargs, message):
    x, y = _series(50)
    with pytest.raises(ValueError, match=message):
        wcc_surface(x, y, hz=1.0, **kwargs)


def test_the_half_window_limit_can_be_waived_deliberately():
    x, y = _series(60)
    assert wcc_surface(x, y, hz=1.0, window_samples=10, lag_max_samples=6, enforce_susy_limit=False).values.shape[0] == 13


def test_unequal_lengths_and_infinities_are_rejected():
    x, y = _series(50)
    with pytest.raises(ValueError, match="same length"):
        wcc_surface(x, y[:-1], hz=1.0, window_samples=10)
    bad = y.copy()
    bad[3] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        wcc_surface(x, bad, hz=1.0, window_samples=10)


def test_settings_validate_themselves():
    with pytest.raises(ValueError, match="hz must be positive"):
        SurfaceSettings(hz=0, window_samples=10, window_increment=1, lag_max_samples=0, lag_increment=1, n_samples=50)
    with pytest.raises(ValueError, match="only scale='r'"):
        SurfaceSettings(hz=1, window_samples=10, window_increment=1, lag_max_samples=0, lag_increment=1, n_samples=50, scale="z")


def test_surface_rejects_inconsistent_arrays_and_out_of_range_correlations():
    settings = SurfaceSettings(hz=1, window_samples=10, window_increment=1, lag_max_samples=0, lag_increment=1, n_samples=50)
    with pytest.raises(ValueError, match="shape"):
        SyncSurface(np.zeros((2, 3)), np.array([0]), np.array([0, 1, 2]), settings)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        SyncSurface(np.array([[1.5]]), np.array([0]), np.array([0]), settings)


def test_plot_returns_axes_for_both_surface_kinds():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    x, y = _series(80)
    heat = wcc_surface(x, y, hz=1.0, window_samples=12, lag_max_samples=4).plot()
    line = wcc_surface(x, y, hz=1.0, window_samples=12).plot()
    assert heat.get_ylabel().startswith("lag") and line.get_ylabel().startswith("correlation")


def test_z_clamp_is_pinned_to_bsyncs_value_by_a_literal():
    """A mutation that changed 0.9999 to 0.999 survived tests that compared the constant with itself."""
    assert Z_CLIP == 0.9999
    assert r_to_z(1.0) == pytest.approx(4.951718775643098, rel=1e-12)      # atanh(0.9999), computed independently
    assert r_to_z(0.5) == pytest.approx(0.5493061443340549, rel=1e-12)     # below the clamp: plain Fisher z

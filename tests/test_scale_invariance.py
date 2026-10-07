"""Unit (scale) invariance (audit P0-1, 2026-10-07; ported from Claude's
scale-invariance patch, DECISION_LOG 2026-10-04).

Pearson correlation does not depend on the units of the signals, so neither
may anything derived from it. The flat-window guard used to be an *absolute*
threshold (``denom > 1e-10``), so the same data in raw units (e.g. Siemens,
std ~1e-5) lost most windows to NaN, and surrogate-derived quantities
(existence direction counts, the pooled onset threshold, dwell_time)
drifted with the unit.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from syncpipe.design_controls import synchrony_existence_audit
from syncpipe.qc import DEFAULT_CONFIG, _check_signal_integrity
from syncpipe.wcc import sliding_window_wcc

SCALES = [1e6, 1.0, 1e-3, 1e-5, 1e-7, 1e-9]


def _ar1(n, seed, phi=0.9):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("scale", SCALES)
def test_wcc_cumsum_backend_is_unit_free(scale):
    a, b = _ar1(300, 1), _ar1(300, 2)
    ref = sliding_window_wcc(a, b, 20)
    out = sliding_window_wcc(a * scale, b * scale, 20)
    assert np.isnan(out).sum() == np.isnan(ref).sum() == 0
    assert np.allclose(out, ref, atol=1e-9, rtol=0)


@pytest.mark.parametrize("scale", SCALES)
def test_wcc_stride_backend_is_unit_free(scale):
    """NaN present -> stride/pairwise-deletion backend."""
    a, b = _ar1(300, 3), _ar1(300, 4)
    a[100:104] = np.nan
    ref = sliding_window_wcc(a, b, 20)
    out = sliding_window_wcc(a * scale, b * scale, 20)
    assert np.array_equal(np.isnan(out), np.isnan(ref))
    assert np.allclose(out, ref, atol=1e-9, rtol=0, equal_nan=True)


@pytest.mark.parametrize("scale", [1.0, 1e-7])
def test_truly_constant_window_is_still_masked_at_any_scale(scale):
    a, b = _ar1(200, 5) * scale, _ar1(200, 6) * scale
    a[80:120] = a[80]                       # 40 constant samples
    out = sliding_window_wcc(a, b, 20)
    assert np.isnan(out[85:100]).all()      # windows fully inside the run
    assert np.isfinite(out[:60]).all()      # untouched region unaffected


@pytest.mark.parametrize("scale", [1.0, 1e-2])
def test_window_flat_in_one_signal_yields_nan_not_zero(scale):
    """A window flat in ONE signal must be NaN even when the other varies
    (legacy product-only guard returned a meaningless ~0 at ANY scale)."""
    a = _ar1(200, 7) * scale
    b = _ar1(200, 8) * scale
    b[80:120] = b[80]                       # flat in b only
    out = sliding_window_wcc(a, b, 20)
    assert np.isnan(out[85:100]).all()
    assert np.isfinite(out[:60]).all()


def _dataset(a, b):
    return types.SimpleNamespace(
        modalities={"EDA": pd.DataFrame({"x": a, "y": b})},
        feature_columns={"EDA": ["x", "y"]}, target_hz=1.0,
    )


def _verdict(report):
    return str(getattr(report.verdict, "value", report.verdict)).upper()


@pytest.mark.parametrize("scale", [1.0, 1e-5, 1e-7])
def test_qc_does_not_flag_a_valid_small_unit_signal(scale):
    a, b = _ar1(600, 9) * scale, _ar1(600, 10) * scale
    assert _verdict(_check_signal_integrity(_dataset(a, b), dict(DEFAULT_CONFIG))) == "PASS"


def test_qc_still_fails_a_real_flatline_and_a_constant_signal():
    a, b = _ar1(600, 11), _ar1(600, 12)
    flat = a.copy()
    flat[:400] = flat[0]                    # 67% constant run
    assert _verdict(_check_signal_integrity(_dataset(flat, b), dict(DEFAULT_CONFIG))) == "FAIL"
    assert _verdict(_check_signal_integrity(
        _dataset(np.full(600, 3.0), b), dict(DEFAULT_CONFIG))) == "FAIL"


@pytest.mark.parametrize("scale", [1.0, 1e-7])
def test_iaaft_surrogates_are_unit_free(scale):
    """IAAFT convergence is variance-relative, so the surrogate draw sequence
    (and hence the null) must not drift with the unit."""
    from syncpipe.surrogate import iaaft_surrogate
    a = _ar1(300, 13)
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    ref = iaaft_surrogate(a, rng1)
    small = iaaft_surrogate(a * scale, rng2)
    # amplitude distribution preserved per-unit -> compare after rescaling
    assert np.allclose(small / scale, ref, atol=1e-9)


def test_existence_audit_is_identical_across_units():
    """Full surrogate machinery (IAAFT + WCC + peak): same seed, different
    units, same answer."""
    a, b = _ar1(300, 14), _ar1(300, 15)
    ref = synchrony_existence_audit(
        a, b, hz=1.0, window_size=20, surrogate_n=49, seed=3)
    small = synchrony_existence_audit(
        a * 1e-7, b * 1e-7, hz=1.0, window_size=20, surrogate_n=49, seed=3)
    assert small["obs_peak_amplitude"] == pytest.approx(
        ref["obs_peak_amplitude"], abs=1e-9)
    assert np.allclose(
        small["null_peak_amplitude"], ref["null_peak_amplitude"], atol=1e-9)
    assert small["p_values"]["peak_amplitude"] == pytest.approx(
        ref["p_values"]["peak_amplitude"])

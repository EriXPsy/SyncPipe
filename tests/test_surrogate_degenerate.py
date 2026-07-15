"""
Edge / degenerate-input regression tests for the surrogate & permutation
p-value machinery (gstack #8 / #10).

Goal: guarantee the null-model p-value path never *silently* emits a wrong
number or *crashes* on pathological inputs — the two failure modes we care
about most.  Covers:
  - phipson_smyth_p: NaN observed / empty null / constant null / bounds / tails
  - _wcc_level_surrogate_test: all-NaN WCC, constant WCC, sub-minimum length,
    surrogate_n=1
  - _signal_level_surrogate_test: all-NaN signals, constant signals,
    surrogate_n=1
"""
import numpy as np
import pytest

from multisync.dynamic_features import (
    _wcc_level_surrogate_test,
    _signal_level_surrogate_test,
)
from multisync.validation.pgt1_intensity import phipson_smyth_p


# ---------------------------------------------------------------------------
# phipson_smyth_p — the unbiased estimator's boundary behaviour
# ---------------------------------------------------------------------------

def test_phipson_nan_observed_returns_nan():
    assert np.isnan(phipson_smyth_p(np.nan, np.array([0.1, 0.2, 0.3])))


def test_phipson_empty_null_returns_nan():
    assert np.isnan(phipson_smyth_p(0.5, np.array([])))
    assert np.isnan(phipson_smyth_p(0.5, np.array([np.nan, np.nan])))


def test_phipson_constant_null_p_equals_one():
    # observed exactly at the (only) null value -> all null >= obs -> k = n
    null = np.full(50, 0.4)
    assert phipson_smyth_p(0.4, null) == 1.0


def test_phipson_minimum_p_is_one_over_n_plus_one():
    null = np.full(999, 0.0)
    # observed far above every null -> k = 0 -> (1+0)/(1+999)
    p = phipson_smyth_p(1e6, null)
    assert p == pytest.approx(1.0 / (1.0 + 999))


def test_phipson_p_in_unit_interval_all_tails():
    rng = np.random.default_rng(0)
    null = rng.normal(0.0, 1.0, 200)
    for tail in ("upper", "lower", "two"):
        p = phipson_smyth_p(0.3, null, tail=tail)
        assert 0.0 < p <= 1.0


def test_phipson_two_tailed_symmetric_about_null_mean():
    rng = np.random.default_rng(1)
    # construct a null EXACTLY symmetric about 0 (values mirrored) so the
    # reflection symmetry of the two-tailed p is exact, free of sampling noise
    base = rng.normal(0.0, 1.0, 2500)
    null = np.concatenate([base, -base])
    d = 1.0
    p_a = phipson_smyth_p(d, null, tail="two")
    p_b = phipson_smyth_p(-d, null, tail="two")
    assert p_a == pytest.approx(p_b, rel=1e-9)


# ---------------------------------------------------------------------------
# _wcc_level_surrogate_test — L1 null on pathological WCC
# ---------------------------------------------------------------------------

def test_wcc_all_nan_does_not_crash():
    wcc = np.full(200, np.nan)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)
    # if p-values are finite they must be in (0, 1]
    for key in ("p_dwell_time", "p_switching_rate"):
        pv = res[key]
        if np.isfinite(pv):
            assert 0.0 < pv <= 1.0


def test_wcc_constant_does_not_crash():
    wcc = np.full(200, 0.5)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)
    assert "p_dwell_time" in res and "p_switching_rate" in res


def test_wcc_sub_minimum_length_returns_l1_shape():
    # below min_wcc_points -> early return; must still carry L1 keys
    wcc = np.zeros(10)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert "p_dwell_time" in res
    assert "p_switching_rate" in res
    assert res["p_dwell_time"] == 1.0  # fail-loud-safe neutral value


def test_wcc_surrogate_n_one_runs():
    wcc = np.random.default_rng(2).normal(0.3, 0.1, 200)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=1, seed=1)
    assert isinstance(res, dict)
    assert res["n_surrogates"] == 1


# ---------------------------------------------------------------------------
# _signal_level_surrogate_test — L0 null on pathological signals
# ---------------------------------------------------------------------------

def test_signal_all_nan_does_not_crash():
    a = np.full(200, np.nan)
    b = np.full(200, np.nan)
    wcc = np.full(150, np.nan)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)


def test_signal_constant_does_not_crash():
    # constant signals -> WCC is undefined (std=0); the test must degrade
    # gracefully rather than raise
    a = np.full(200, 1.0)
    b = np.full(200, 1.0)
    wcc = np.full(150, np.nan)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)


def test_signal_surrogate_n_one_runs():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    wcc = rng.normal(0.3, 0.1, 150)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=1, seed=1)
    assert isinstance(res, dict)
    assert res["n_surrogates"] == 1

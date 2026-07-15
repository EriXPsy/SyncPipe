"""Regression tests: per-feature significance (no OR) for BOTH L0 and L1,
two-tailed p, no silent surrogate cap, BC moved to L0 family.
"""
import numpy as np
import pytest

from multisync.dynamic_features import (
    wcc_surrogate_test, _wcc_level_surrogate_test, sliding_window_wcc,
)
from multisync.feature_definitions import FDR_FAMILIES, MATHEMATICAL_TIER, FDR_FEATURES


def _structured_wcc(seed=0):
    rng = np.random.default_rng(seed)
    w = np.tile(np.r_[np.full(50, 0.9), np.full(50, 0.1)], 5)
    return np.clip(w + rng.normal(0, 0.02, w.size), -1, 1)


# ---- L1 path -------------------------------------------------------------
def test_l1_emits_per_feature_significant_not_or():
    res = wcc_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=200, seed=1)
    assert "per_feature_significant" in res
    assert "surrogate_is_significant" not in res  # no OR aggregate flag
    assert set(res["per_feature_significant"]) == {"dwell_time", "switching_rate"}


def test_l1_two_tailed_p_bounded():
    res = wcc_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=300, seed=1)
    for k in ("p_dwell_time", "p_switching_rate"):
        assert 0.0 <= res[k] <= 1.0


def test_no_silent_surrogate_cap():
    res = _wcc_level_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=1100, seed=1)
    assert res["n_surrogates"] == 1100


# ---- gstack Finding 6: short-WCC early-return shape parity -------------
def test_l1_short_wcc_early_return_matches_normal_shape():
    """A WCC shorter than min_wcc_points must still return an L1-shaped dict
    (p_dwell_time / p_switching_rate present and == 1.0), so downstream
    consumers that index those keys don't KeyError a whole batch. Regression
    for gstack Finding 6."""
    short = np.zeros(10)  # far below default min_wcc_points=30
    res = _wcc_level_surrogate_test(short, hz=1.0, surrogate_n=50, seed=1)
    for k in ("p_dwell_time", "p_switching_rate"):
        assert k in res, f"short-WCC early return missing {k}"
        assert res[k] == 1.0  # non-significant: no evidence to reject H0
    for k in ("null_dwell_time", "null_switching_rate"):
        assert k in res
    assert res["per_feature_significant"] == {"dwell_time": False, "switching_rate": False}
    assert np.isnan(res["obs_dwell_time"])
    assert res["notes"].startswith("WCC too short")


def test_l1_short_wcc_via_dispatcher_no_keyerror():
    """Same guarantee through the public wcc_surrogate_test dispatcher (L1 path,
    no raw_signals)."""
    short = np.zeros(10)
    res = wcc_surrogate_test(short, hz=1.0, surrogate_n=50, seed=1)
    assert "p_dwell_time" in res and "p_switching_rate" in res
    assert res["p_dwell_time"] == 1.0


# ---- gstack: dispatcher-level shape-contract (prevents Finding 6 class) ----
def _p_feature_keys(res):
    return {k[2:] for k in res if k.startswith("p_")}


def test_dispatcher_shape_contract_l0_l1_and_short():
    """wcc_surrogate_test must honor a STABLE key contract per level so
    downstream consumers can rely on it across refactors. Contract:
      (1) every p_<f> key has matching null_<f> and obs_<f> companions;
      (2) per_feature_significant is keyed by exactly the same feature set;
      (3) the L1 short-WCC early-return shares the SAME p_* key set as the
          L1 normal path (exactly what Finding 6 violated).
    This is a regression guard: if any future refactor diverges one path's
    shape, the assertion fails instead of silently corrupting a whole batch.
    """
    rng = np.random.default_rng(0)
    n = 600
    shared = np.cumsum(rng.normal(0, 1, n))
    a = shared + rng.normal(0, 2, n)
    b = shared + rng.normal(0, 2, n)
    wcc = sliding_window_wcc(a, b, window_size=30)

    res_l0 = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=60, seed=1,
                                raw_signals=(a, b), wcc_window_size=30)
    res_l1 = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=60, seed=1)
    res_l1_short = wcc_surrogate_test(np.zeros(10), hz=1.0, surrogate_n=50, seed=1)

    for res, level in ((res_l0, "L0"), (res_l1, "L1"), (res_l1_short, "L1-short")):
        p_keys = _p_feature_keys(res)
        assert p_keys, f"{level}: no p_* keys found"
        assert set(res["per_feature_significant"]) == p_keys, (
            f"{level}: per_feature_significant keys "
            f"{set(res['per_feature_significant'])} != p_* keys {p_keys}"
        )
        for f in p_keys:
            assert f"null_{f}" in res, f"{level}: missing null_{f}"
            assert f"obs_{f}" in res, f"{level}: missing obs_{f}"

    # The invariant Finding 6 broke: short and normal L1 paths agree.
    assert _p_feature_keys(res_l1_short) == _p_feature_keys(res_l1), (
        "L1 short-WCC early-return shape diverges from L1 normal path"
    )


# ---- L0 path -------------------------------------------------------------
def test_l0_emits_per_feature_significant_for_three_features():
    rng = np.random.default_rng(0)
    n = 600
    shared = np.cumsum(rng.normal(0, 1, n))
    a = shared + rng.normal(0, 2, n)
    b = shared + rng.normal(0, 2, n)
    wcc = sliding_window_wcc(a, b, window_size=30)
    res = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=100, seed=1,
                             raw_signals=(a, b), wcc_window_size=30)
    assert res["null_model"] == "signal_level_iaaft"
    pfs = res["per_feature_significant"]
    assert set(pfs) == {"mean_synchrony", "peak_amplitude", "bimodality_coefficient"}
    assert "surrogate_is_significant" not in res  # no OR aggregate flag


# ---- BC tier consistency -------------------------------------------------
def test_bimodality_is_l0_math_tier_but_not_in_confirmatory_fdr():
    """2026-06-29 (SSoT Option B): bimodality_coefficient remains a
    permutation-invariant L0 feature for the synchrony-existence audit
    (MATHEMATICAL_TIER + _NULL_MODEL_L0), but was removed from the
    confirmatory group-condition FDR family (FDR_FAMILIES / FDR_FEATURES)."""
    assert MATHEMATICAL_TIER["bimodality_coefficient"] == "L0"
    assert "bimodality_coefficient" not in FDR_FAMILIES["L0"]
    assert "bimodality_coefficient" not in FDR_FAMILIES["L1"]
    assert "bimodality_coefficient" not in FDR_FEATURES
    assert FDR_FAMILIES["L0"] == ("peak_amplitude",)
    assert FDR_FAMILIES["L1"] == ("dwell_time", "switching_rate")

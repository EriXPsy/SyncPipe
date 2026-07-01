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

"""Regression tests for P0/P1 technical fixes (2026-07-22).

Copy into SyncPipe/tests/ and run:
    pytest tests/test_p0_technical_fixes.py -q
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_p0_1_l0_peak_matches_ssot_smoothed_peak():
    """Existence-null obs peak must equal SSoT peak_amplitude (smoothed)."""
    from multisync.feature_definitions import extract_features, smoothed_wcc, compute_peak_amplitude
    from multisync.dynamic_features import _signal_level_surrogate_test

    rng = np.random.default_rng(0)
    # Spiky WCC: raw max >> smoothed peak
    wcc = np.zeros(120)
    wcc[50] = 0.95
    wcc[51:56] = 0.45
    wcc += rng.normal(0, 0.02, size=120)

    feat = extract_features(wcc, hz=1.0, wcc_window_sec=20.0, threshold=0.5)
    ssot_peak = feat.peak_amplitude
    raw_max = float(np.max(wcc[np.isfinite(wcc)]))
    assert abs(raw_max - ssot_peak) > 0.1, "fixture must create raw vs smoothed gap"

    # Build trivial raw signals long enough for signal-level path
    n = 200
    sig_a = rng.normal(size=n)
    sig_b = 0.6 * sig_a + rng.normal(scale=0.3, size=n)
    # Use the spiky series as "observed WCC" with matching window size heuristic off
    result = _signal_level_surrogate_test(
        sig_A=sig_a,
        sig_B=sig_b,
        wcc=wcc,
        hz=1.0,
        surrogate_n=20,
        seed=1,
        wcc_window_size=30,
    )
    obs_peak = result["obs_peak_amplitude"]
    assert np.isfinite(obs_peak)
    assert abs(obs_peak - ssot_peak) < 1e-12, (
        f"L0 obs peak {obs_peak} != SSoT peak {ssot_peak} (raw would be {raw_max})"
    )


def test_p0_2_l2_refuses_multimodal_pool():
    from multisync.validation.l2_between_condition import between_condition_fdr

    rows = []
    for d in range(8):
        for cond in ("rest", "task"):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=0.2 if cond == "rest" else 0.8,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="ECG",
                    peak_amplitude=0.8 if cond == "rest" else 0.2,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="modalit"):
        between_condition_fdr(
            df,
            condition_col="condition",
            dyad_col="dyad_id",
            feature_cols=["peak_amplitude"],
            n_permutations=50,
            seed=0,
            condition_values=("rest", "task"),
        )


def test_p0_2_single_modality_still_works():
    from multisync.validation.l2_between_condition import between_condition_fdr

    rows = []
    for d in range(8):
        for cond, val in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=val,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    df = pd.DataFrame(rows)
    out = between_condition_fdr(
        df,
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=100,
        seed=0,
        condition_values=("rest", "task"),
    )
    assert out["n_dyads"] == 8
    assert abs(out["per_feature"][0].observed_diff - (0.2 - 0.8)) < 1e-9


def test_p0_3_finite_pair_warns_on_length_mismatch():
    from multisync.design_controls import _finite_pair

    a = np.arange(100.0)
    b = np.arange(70.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aa, bb = _finite_pair(a, b)
        assert any("unequal lengths" in str(x.message) for x in w)
    assert len(aa) == len(bb) == 70


def test_p0_3_finite_pair_raise_mode():
    from multisync.design_controls import _finite_pair

    with pytest.raises(ValueError, match="unequal lengths"):
        _finite_pair(np.arange(10.0), np.arange(7.0), on_length_mismatch="raise")


def test_p1_1_dwell_splits_across_nan_seam():
    from multisync.feature_definitions import compute_dwell_time

    # two elevated runs of length 3 separated by NaN seam
    w = np.array([0.1, 0.1, 0.8, 0.8, 0.8, np.nan, 0.8, 0.8, 0.8, 0.1, 0.1])
    d = compute_dwell_time(w, hz=1.0, threshold=0.5, hysteresis_delta=0.0, gap_policy="segment")
    # explicit gap_policy="segment": mean of two length-3 runs = 3.0
    assert d == pytest.approx(3.0)
    # opt-in merge_valid restores 2026-07-13 glue-across-gap behaviour
    d_merge = compute_dwell_time(
        w, hz=1.0, threshold=0.5, hysteresis_delta=0.0, gap_policy="merge_valid"
    )
    assert d_merge == pytest.approx(6.0)


def test_p1_4_nan_fraction_set_in_ssot():
    from multisync.feature_definitions import extract_features

    w = np.array([0.1, 0.2, np.nan, 0.8, 0.9, 0.7, np.nan, 0.1])
    f = extract_features(w, hz=1.0, wcc_window_sec=5.0)
    assert f.nan_fraction == pytest.approx(2.0 / 8.0)

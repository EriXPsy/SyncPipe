"""v1.0 hardening regression tests.

Covers the four review-driven hardening items:
  1. prediction.py  — feature-to-sample ratio (FSR) + overparameterization warning
  2. importer.py    — merge_person_files offset_b_sec (relative-time correction)
  3. qc.py          — marker-channel zero-variance exemption + dead-code cleanup
  4. wclr.py        — opt-in sign_stable flag (no effect on default trace)
"""
import csv
import os
import tempfile

import numpy as np
import pandas as pd
from types import SimpleNamespace

from multisync.prediction import (
    MIN_SAMPLES_PER_FEATURE,
    PredictionResult,
    _compose_warning,
    _fsr,
    _overparam_warning,
    rolling_origin_cv,
)
from multisync.importer import DataImporter
from multisync.qc import _check_signal_integrity
from multisync.wclr import windowed_cross_lagged_regression as wclr


# ---------------------------------------------------------------------------
# 1. prediction.py — FSR + overparameterization guard
# ---------------------------------------------------------------------------
def test_fsr_helper_basic():
    assert _fsr(20, 10) == 2.0
    assert _fsr(0, 10) == 0.0
    assert _fsr(20, 0) == 0.0


def test_overparam_warning_threshold():
    # 20 samples, 10 features -> 20 < 3*10 -> warn
    msg = _overparam_warning(20, 10)
    assert msg is not None
    assert "overparameterized" in msg
    # 40 samples, 10 features -> 40 >= 30 -> no warn
    assert _overparam_warning(40, 10) is None
    # undefined -> no warn
    assert _overparam_warning(0, 10) is None
    assert _overparam_warning(20, 0) is None


def test_compose_warning():
    assert _compose_warning("a", None) == "a"
    assert _compose_warning(None, "b") == "b"
    assert _compose_warning("a", "b") == "a; b"
    assert _compose_warning(None, None) is None


def test_prediction_result_fsr_roundtrip():
    r = PredictionResult(n_samples=25, feature_to_sample_ratio=2.5)
    d = r.to_dict()
    assert d["n_samples"] == 25
    assert d["feature_to_sample_ratio"] == 2.5
    r2 = PredictionResult.from_dict(d)
    assert r2.n_samples == 25
    assert r2.feature_to_sample_ratio == 2.5


def test_rolling_origin_cv_reports_fsr_and_overparam():
    rng = np.random.default_rng(1)
    wcc = np.concatenate([
        rng.normal(0, 0.3, 30),
        rng.normal(0.5, 0.2, 10),
        rng.normal(0, 0.3, 30),
    ])
    res = rolling_origin_cv(
        wcc[:40], window_size=30, hz=1.0, horizon_windows=1,
        n_splits=3, gap=0, threshold=0.0, mode="intra",
    )
    assert res.n_samples > 0
    assert res.feature_to_sample_ratio > 0.0
    assert "overparameterized" in (res.warning or "")
    # ratio equals n_samples / n_features_used
    assert abs(res.feature_to_sample_ratio - res.n_samples / res.n_features_used) < 1e-9


# ---------------------------------------------------------------------------
# 2. importer.py — offset_b_sec
# ---------------------------------------------------------------------------
def _write_csv(path, vals):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "signal"])
        for i, v in enumerate(vals):
            w.writerow([i, v])


def test_merge_person_files_offset_b_sec():
    imp = DataImporter(default_hz=1.0, force_zero_start=False)
    d = tempfile.mkdtemp()
    a = os.path.join(d, "a.csv")
    b = os.path.join(d, "b.csv")
    # distinct values so a shift is detectable
    _write_csv(a, list(range(10)))        # signal 0..9
    _write_csv(b, list(range(10, 20)))     # signal 10..19

    m0 = imp.merge_person_files(a, b, offset_b_sec=None)
    m3 = imp.merge_person_files(a, b, offset_b_sec=3.0)

    # offset_b_sec=None and offset_b_sec=0 must be identical
    m0b = imp.merge_person_files(a, b, offset_b_sec=0.0)
    assert np.allclose(m0.person_b.values, m0b.person_b.values, equal_nan=True)

    # with offset 3, person_b at merged time t (t>=3) equals person_b at t-3
    # without offset (B's timeline was shifted by +3s before alignment)
    for t in range(3, 10):
        assert abs(m3.person_b.iloc[t] - m0.person_b.iloc[t - 3]) < 1e-9


# ---------------------------------------------------------------------------
# 3. qc.py — marker-channel exemption
# ---------------------------------------------------------------------------
def test_marker_channel_exemption():
    rng = np.random.default_rng(2)
    modalities = {
        "phys": pd.DataFrame({
            "marker_ch": np.zeros(50),          # legitimately zero-variance
            "real_ch": rng.normal(0, 1, 50),    # has variance
        }),
        "bio": pd.DataFrame({
            "flat": np.zeros(50),               # zero-variance, NOT exempt
        }),
    }
    feature_columns = {"phys": ["marker_ch", "real_ch"], "bio": ["flat"]}
    dataset = SimpleNamespace(
        modalities=modalities, feature_columns=feature_columns, target_hz=10.0,
    )
    cfg = {"marker_channels": ["phys/marker_ch"]}
    res = _check_signal_integrity(dataset, cfg)

    types = [d["type"] for d in res.details]
    assert "marker_exempt" in types          # marker channel exempted
    assert "zero_variance" in types          # non-exempt flat channel still fails
    # the exempted one must not appear as a zero_variance failure
    zero_var_keys = [
        (d.get("modality"), d.get("feature"))
        for d in res.details if d["type"] == "zero_variance"
    ]
    assert ("phys", "marker_ch") not in zero_var_keys


# ---------------------------------------------------------------------------
# 4. wclr.py — sign_stable invariant
# ---------------------------------------------------------------------------
def _sign_flips(trace):
    s = np.sign(trace)
    s = s[np.isfinite(s)]
    return int(np.sum(np.diff(s) != 0))


def test_wclr_sign_stable_invariant_and_inert_default():
    rng = np.random.default_rng(3)
    n_mismatch = 0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        N = 160
        y = np.zeros(N)
        for t in range(1, N):
            y[t] = 0.4 * y[t - 1] + rng.normal(0, 0.3)
        x = 0.6 * np.roll(y, 1) - 0.5 * np.roll(y, -1) + rng.normal(0, 0.2, N)

        # Default (absolute_beta=True): non-negative, and sign_stable is inert
        tr_abs_off, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                             step_samples=5, metric="beta", absolute_beta=True,
                             sign_stable=False)
        tr_abs_on, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                            step_samples=5, metric="beta", absolute_beta=True,
                            sign_stable=True)
        assert np.all(tr_abs_off[np.isfinite(tr_abs_off)] >= 0)
        assert np.allclose(tr_abs_off, tr_abs_on, equal_nan=True)

        # Signed mode: sign_stable can only reduce (never increase) sign flips
        tr_off, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                         step_samples=5, metric="beta", absolute_beta=False,
                         sign_stable=False)
        tr_on, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                        step_samples=5, metric="beta", absolute_beta=False,
                        sign_stable=True)
        assert tr_off.shape == tr_on.shape
        assert np.all(np.isfinite(tr_off)) and np.all(np.isfinite(tr_on))
        if _sign_flips(tr_on) > _sign_flips(tr_off):
            n_mismatch += 1
    assert n_mismatch == 0, "sign_stable increased sign flips in %d/25 cases" % n_mismatch

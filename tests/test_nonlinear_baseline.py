"""
Tests for the nonlinear prediction baseline (critique E, 2026-07-07).

A linear-only prediction pipeline cannot distinguish "no signal" from
"signal present but nonlinear". ``rolling_origin_cv`` and
``cross_modal_prediction`` therefore run RandomForest / SVM-RBF on the SAME
train/test splits in parallel and aggregate them into the ``nonlinear``
field of :class:`PredictionResult`. When the linear ``delta_AUC`` is ~0 but a
nonlinear model recovers > 0, ``nonlinear_signal_present`` is flagged.

Covers
------
- ``PredictionResult`` carries ``nonlinear`` / ``nonlinear_signal_present`` /
  ``nonlinear_note`` and round-trips through to_dict / from_dict
- both prediction entry points populate the nonlinear summary with the
  expected model keys and per-model structure
- a linear signal yields ``nonlinear_signal_present == False`` and a clear note
"""
from __future__ import annotations

import numpy as np
import pytest

from multisync.prediction import (
    PredictionResult,
    cross_modal_prediction,
    rolling_origin_cv,
)

EXPECTED_MODELS = ("linear_logreg", "random_forest", "svm_rbf")


def _slow_square_wave(n: int = 1200, hz: float = 1.0, period: float = 120.0) -> np.ndarray:
    """A slow square wave (0.2 / 0.8) at the given period.

    With ``threshold=0.5`` the future-horizon labels alternate 0/1, giving
    a balanced, learnable (and linear-friendly) structure for the
    prediction labels.  The series is long enough that several CV folds
    retain both classes in train and test.
    """
    t = np.arange(n) / hz
    return np.where((t // (period / 2)) % 2 == 0, 0.8, 0.2).astype(float)


def test_prediction_result_has_nonlinear_fields():
    r = PredictionResult(source_pair="a", target_pair="a", mode="intra")
    assert r.nonlinear == {}
    assert r.nonlinear_signal_present is False
    assert isinstance(r.nonlinear_note, str)


def test_rolling_origin_cv_populates_nonlinear_summary():
    wcc = _slow_square_wave()
    res = rolling_origin_cv(
        wcc, window_size=60, hz=1.0, n_splits=5, gap=2, max_iter=300,
        threshold=0.5, pair_name="p",
    )
    assert isinstance(res.nonlinear, dict)
    for m in EXPECTED_MODELS:
        assert m in res.nonlinear, f"missing nonlinear model {m}"
        d = res.nonlinear[m]
        assert set(d.keys()) >= {"mean_auc", "mean_delta_auc", "delta_auc_ci", "n_folds"}
        assert isinstance(d["n_folds"], int)
        assert d["n_folds"] >= 1
    # When a clear linear signal exists, both linear and nonlinear agree.
    assert res.nonlinear_signal_present is False
    assert isinstance(res.nonlinear_note, str) and len(res.nonlinear_note) > 0


def test_cross_modal_prediction_populates_nonlinear_summary():
    src = _slow_square_wave(period=120.0)
    tgt = _slow_square_wave(period=120.0)
    res = cross_modal_prediction(
        src, tgt, window_size=60, hz=1.0, n_splits=5, gap=2, max_iter=300,
        threshold=0.5, source_name="src", target_name="tgt",
    )
    assert isinstance(res.nonlinear, dict)
    for m in EXPECTED_MODELS:
        assert m in res.nonlinear, f"missing nonlinear model {m}"
    assert res.nonlinear_signal_present is False
    assert isinstance(res.nonlinear_note, str) and len(res.nonlinear_note) > 0


def test_nonlinear_fields_round_trip_through_dict():
    wcc = _slow_square_wave()
    res = rolling_origin_cv(
        wcc, window_size=60, hz=1.0, n_splits=5, gap=2, max_iter=300,
        threshold=0.5, pair_name="p",
    )
    d = res.to_dict()
    assert "nonlinear" in d
    assert "nonlinear_signal_present" in d
    assert "nonlinear_note" in d
    r2 = PredictionResult.from_dict(d)
    assert r2.nonlinear == res.nonlinear
    assert r2.nonlinear_signal_present == res.nonlinear_signal_present
    assert r2.nonlinear_note == res.nonlinear_note

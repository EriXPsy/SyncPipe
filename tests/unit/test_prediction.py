from __future__ import annotations

# === source: test_prediction_failed_fold.py ===
"""Regression tests for BRM-2026-07-13 fix #2: prediction fake AUC=0.5.

Before the fix, when the *primary* (joint) model raised inside a CV fold,
the code did ``except: pass`` and left ``joint_auc = 0.5``. The fold was
still appended to ``folds`` with that fabricated 0.5, and the mean was
computed over it -- silently pretending a failed model produced a
"random-chance" result.

After the fix the joint model follows the same discipline as intra-mode:
on failure the fold is *skipped* (``continue``), counted in
``n_failed_folds``, and never enters the mean. A failed baseline
(restricted / naive) still falls back to 0.5 because that is a genuine
"no information" reference, but a failed *primary* model is never
fabricated.
"""

import numpy as np
import pytest

import multisync.prediction as pred_mod
from sklearn.linear_model import LogisticRegression as _RealLR


def _make_signals_fold(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    src = np.sin(t) + rng.normal(0, 0.1, n)
    tgt = np.cos(t) + rng.normal(0, 0.1, n)
    return src, tgt


class _AlwaysFailFit(_RealLR):
    """LogisticRegression stand-in whose ``fit`` always raises."""

    def fit(self, X, y):
        raise ValueError("forced primary-model failure")


class _FailOnSecondCall(_RealLR):
    """First ``fit`` call succeeds; second call (first fold's joint) raises.

    Within each fold the order of ``fit`` calls is
    [restricted, joint, ablation], so call #2 is the first fold's joint
    model. Raising there drops exactly that one fold.
    """

    n_calls = 0

    def fit(self, X, y):
        _FailOnSecondCall.n_calls += 1
        if _FailOnSecondCall.n_calls == 2:
            raise ValueError("forced joint failure on first fold")
        return super().fit(X, y)


def test_cross_modal_all_joint_failures_dropped_not_fabricated(monkeypatch):
    """Every joint model fails -> all folds dropped, no fabricated 0.5."""
    monkeypatch.setattr(pred_mod, "LogisticRegression", _AlwaysFailFit)
    src, tgt = _make_signals_fold()
    res = pred_mod.cross_modal_prediction(
        src, tgt, window_size=30, n_splits=5,
        source_name="s", target_name="t",
    )
    # OLD code would have appended every fold with dynamic_auc=0.5 and
    # reported a "successful" mean near 0.5. NEW code reports no valid
    # folds and drops everything.
    assert res.warning == "no_valid_folds"
    assert res.n_failed_folds >= 1
    assert len(res.folds) == 0
    # No fold was silently recorded as a real 0.5 AUC result.
    assert all(f.dynamic_auc != 0.5 for f in res.folds)


def test_cross_modal_partial_joint_failure_excluded_from_means(monkeypatch):
    """One joint model fails -> that fold is excluded, n_failed_folds=1."""
    _FailOnSecondCall.n_calls = 0
    monkeypatch.setattr(pred_mod, "LogisticRegression", _FailOnSecondCall)
    src, tgt = _make_signals_fold()
    res = pred_mod.cross_modal_prediction(
        src, tgt, window_size=30, n_splits=5,
        source_name="s", target_name="t",
    )
    # Exactly one fold (the first) had its joint model fail and was dropped.
    assert res.n_failed_folds == 1, res.n_failed_folds
    assert len(res.folds) == 4, len(res.folds)  # 5 attempted, 1 dropped
    # Crucially the dropped fold is NOT present in the reported folds, so
    # its fabricated 0.5 cannot leak into the mean.
    assert all(not np.isnan(f.dynamic_auc) for f in res.folds)


def test_cross_modal_success_path_reports_zero_failed(monkeypatch):
    """Unpatched happy path: nothing is dropped, n_failed_folds == 0."""
    src, tgt = _make_signals_fold()
    res = pred_mod.cross_modal_prediction(
        src, tgt, window_size=30, n_splits=5,
        source_name="s", target_name="t",
    )
    assert res.n_failed_folds == 0
    assert len(res.folds) >= 1

# === source: test_prediction_finding17.py ===
"""Regression test for Finding 17: restricted / AR baseline AUC=0.5 fabrication.

Before the fix, when the *restricted* (target-only + AR) baseline in
``cross_modal_prediction`` failed to fit, the code set ``restricted_auc = 0.5``
and the aggregate ``mean_restricted`` was computed with a plain ``np.mean``
that had NO NaN filter. A fabricated 0.5 therefore silently entered the mean
and inflated the joint model's apparent incremental value
(``delta_auc = joint_auc - max(baseline, restricted)``).

After the fix (Finding 17): a failed restricted baseline leaves
``restricted_auc = nan``; the fold's ``ar_baseline_auc`` is stored as nan; and
the aggregate uses ``np.nanmean`` so the failed fold is excluded instead of
poisoning the mean.  The intra-mode AR baseline (``rolling_origin_cv``) follows
the identical discipline via the same commit.

This is the symmetric counterpart to the joint-model fix already covered by
``test_prediction_failed_fold.py`` -- it guards against the exact "fixed one
model, forgot the adjacent baseline" regression class that Finding 10 / 17
call out.
"""

import math

import numpy as np
import pytest

import multisync.prediction as pred_mod
from sklearn.metrics import roc_auc_score as _real_roc_auc


def _make_signals_find17(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    src = np.sin(t) + rng.normal(0, 0.1, n)
    tgt = np.cos(t) + rng.normal(0, 0.1, n)
    return src, tgt


class _FailFirstRocCall:
    """Make ``roc_auc_score`` raise on its very first call.

    Inside ``cross_modal_prediction`` the first ``roc_auc_score`` invocation of
    each run is the *restricted* baseline of fold 0 (line ~1451), so this
    exercises the restricted-failure path while joint / ablation / naive
    baselines still fit and score normally.
    """

    _state = {"n": 0}

    def __call__(self, y_true, y_score, **kw):
        _FailFirstRocCall._state["n"] += 1
        if _FailFirstRocCall._state["n"] == 1:
            raise ValueError("forced restricted-baseline AUC failure")
        return _real_roc_auc(y_true, y_score, **kw)


def test_cross_modal_restricted_failure_not_fabricated(monkeypatch):
    """A failed restricted baseline must stay NaN and be excluded by nanmean."""
    _FailFirstRocCall._state["n"] = 0
    monkeypatch.setattr(pred_mod, "roc_auc_score", _FailFirstRocCall())
    src, tgt = _make_signals_find17()
    res = pred_mod.cross_modal_prediction(
        src, tgt, window_size=30, n_splits=4,
        source_name="s", target_name="t",
    )
    # Joint model succeeded -> fold 0 was NOT dropped (only joint failure drops).
    assert res.n_failed_folds == 0
    # Fold 0's restricted baseline failed -> stored as NaN, never 0.5.
    assert math.isnan(res.folds[0].ar_baseline_auc)
    assert res.folds[0].ar_baseline_auc != 0.5
    # Aggregate must use nanmean: the NaN fold is excluded, not counted as 0.5.
    expected = float(np.nanmean([f.ar_baseline_auc for f in res.folds]))
    assert res.mean_ar_baseline_auc == pytest.approx(expected)
    # And the reported mean must differ from what a fabricated 0.5 would yield.
    real = [f.ar_baseline_auc for f in res.folds if not math.isnan(f.ar_baseline_auc)]
    if real:
        fabricated = float(np.mean([0.5] + real))
        assert res.mean_ar_baseline_auc != pytest.approx(fabricated)

# === source: test_prediction_seed.py ===
"""Regression tests for BUG-7 (seed not propagated to prediction models).

``rolling_origin_cv`` and ``cross_modal_prediction`` hard-coded
``random_state=42`` inside every estimator and never received the user
``seed``.  The fix threads ``seed`` from the caller (``SyncPipe.seed``)
through to all LogisticRegression / RandomForest / SVC instances.

Karpathy discipline: same seed -> identical results (Rule 9: tests encode
the intent — reproducibility is the contract).
"""

import numpy as np
import pytest

from multisync import dynamic_features as df
from multisync import prediction as pred_mod


def _make_signals_seed(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.6 * x + rng.normal(0, 0.5, n)
    return x, y


def test_rolling_origin_cv_is_seed_reproducible():
    """Two runs with the same seed must yield byte-identical results."""
    x, y = _make_signals_seed(n=3000, seed=3)
    wcc = df.sliding_window_wcc(x, y, window_size=30, hz=1.0)
    r1 = pred_mod.rolling_origin_cv(wcc, window_size=30, n_splits=5, seed=123)
    r2 = pred_mod.rolling_origin_cv(wcc, window_size=30, n_splits=5, seed=123)
    assert r1.to_dict() == r2.to_dict()


def test_cross_modal_prediction_accepts_seed_kwarg():
    """``cross_modal_prediction`` must accept ``seed`` without error and be
    deterministic for a fixed seed."""
    xs, _ = _make_signals_seed(n=3000, seed=5)
    xt, _ = _make_signals_seed(n=3000, seed=6)
    src_wcc = df.sliding_window_wcc(xs, _make_signals_seed(n=3000, seed=7)[0], window_size=30, hz=1.0)
    tgt_wcc = df.sliding_window_wcc(xt, _make_signals_seed(n=3000, seed=8)[0], window_size=30, hz=1.0)
    r1 = pred_mod.cross_modal_prediction(
        src_wcc, tgt_wcc, window_size=30, n_splits=5, seed=7
    )
    r2 = pred_mod.cross_modal_prediction(
        src_wcc, tgt_wcc, window_size=30, n_splits=5, seed=7
    )
    assert r1.to_dict() == r2.to_dict()

# === source: test_nonlinear_baseline.py ===
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


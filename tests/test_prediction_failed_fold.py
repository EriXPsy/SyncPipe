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


def _make_signals(n=3000, seed=0):
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
    src, tgt = _make_signals()
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
    src, tgt = _make_signals()
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
    src, tgt = _make_signals()
    res = pred_mod.cross_modal_prediction(
        src, tgt, window_size=30, n_splits=5,
        source_name="s", target_name="t",
    )
    assert res.n_failed_folds == 0
    assert len(res.folds) >= 1

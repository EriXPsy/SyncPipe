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


def _make_signals(n=3000, seed=0):
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
    src, tgt = _make_signals()
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

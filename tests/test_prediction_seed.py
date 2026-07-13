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


def _make_signals(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.6 * x + rng.normal(0, 0.5, n)
    return x, y


def test_rolling_origin_cv_is_seed_reproducible():
    """Two runs with the same seed must yield byte-identical results."""
    x, y = _make_signals(n=3000, seed=3)
    wcc = df.sliding_window_wcc(x, y, window_size=30, hz=1.0)
    r1 = pred_mod.rolling_origin_cv(wcc, window_size=30, n_splits=5, seed=123)
    r2 = pred_mod.rolling_origin_cv(wcc, window_size=30, n_splits=5, seed=123)
    assert r1.to_dict() == r2.to_dict()


def test_cross_modal_prediction_accepts_seed_kwarg():
    """``cross_modal_prediction`` must accept ``seed`` without error and be
    deterministic for a fixed seed."""
    xs, _ = _make_signals(n=3000, seed=5)
    xt, _ = _make_signals(n=3000, seed=6)
    src_wcc = df.sliding_window_wcc(xs, _make_signals(n=3000, seed=7)[0], window_size=30, hz=1.0)
    tgt_wcc = df.sliding_window_wcc(xt, _make_signals(n=3000, seed=8)[0], window_size=30, hz=1.0)
    r1 = pred_mod.cross_modal_prediction(
        src_wcc, tgt_wcc, window_size=30, n_splits=5, seed=7
    )
    r2 = pred_mod.cross_modal_prediction(
        src_wcc, tgt_wcc, window_size=30, n_splits=5, seed=7
    )
    assert r1.to_dict() == r2.to_dict()

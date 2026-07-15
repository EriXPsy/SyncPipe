"""Regression tests for the #17 dual-API per-pair entry point.

`compute_pair_pipeline` is the single canonical entry for per-pair feature
extraction.  It accepts EITHER raw signals (wcc=None) OR a pre-computed WCC
(wcc=<array>).  The core invariant this suite locks: **a given WCC always
yields identical features**, regardless of which input mode produced it.
`quick_compute` / `batch_compute` are thin wrappers that must stay
behaviorally identical to the prior implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from multisync.computation_pipeline import (
    PairResult,
    batch_compute,
    compute_pair_pipeline,
    quick_compute,
)


def _make_signals(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.standard_normal(n)
    b = np.sin(2 * np.pi * 0.05 * t + 0.3) + 0.1 * rng.standard_normal(n)
    return a, b


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def test_signals_path_returns_wcc_and_features():
    sa, sb = _make_signals()
    res = compute_pair_pipeline(sa, sb, hz=4.0, window_size=40)
    assert isinstance(res, PairResult)
    assert res.wcc is not None
    assert len(res.wcc) == 200 - 40 + 1
    fdict = res.features_dict
    assert isinstance(fdict, dict)
    assert "peak_amplitude" in fdict
    assert isinstance(res.to_dataframe(), pd.DataFrame)


def test_wcc_path_matches_signals_path():
    """Given the SAME WCC, the wcc-given path must produce features
    identical to the signals path (the dual-API equivalence contract)."""
    sa, sb = _make_signals()
    via_signals = compute_pair_pipeline(sa, sb, hz=4.0, window_size=40)
    wcc = via_signals.wcc  # exact array the signals path produced
    via_wcc = compute_pair_pipeline(sa, sb, hz=4.0, window_size=40, wcc=wcc)

    fa, fb = via_signals.features_dict, via_wcc.features_dict
    mism = [
        k for k in fa
        if _isnum(fa[k]) and _isnum(fb[k]) and abs(fa[k] - fb[k]) > 1e-9
    ]
    assert mism == [], f"feature mismatch on shared WCC: {mism}"
    # The supplied WCC is used verbatim (no recomputation).
    np.testing.assert_array_equal(via_wcc.wcc, wcc)


def test_quick_compute_delegates_and_matches():
    sa, sb = _make_signals()
    df_q = quick_compute(sa, sb, hz=4.0, window_size=40)
    res = compute_pair_pipeline(sa, sb, hz=4.0, window_size=40)
    df_r = res.to_dataframe()
    # Same feature columns (label may be absent in both).
    assert "peak_amplitude" in df_q.columns
    assert "peak_amplitude" in df_r.columns
    assert abs(
        float(df_q["peak_amplitude"].iloc[0]) - float(df_r["peak_amplitude"].iloc[0])
    ) < 1e-9


def test_batch_compute_preserves_dyad_id_metadata():
    sa, sb = _make_signals()
    sc, sd = _make_signals(seed=1)
    df = batch_compute([(sa, sb), (sc, sd)], hz=4.0, window_size=40)
    assert len(df) == 2
    assert "dyad_id" in df.columns
    assert list(df["dyad_id"]) == [0, 1]

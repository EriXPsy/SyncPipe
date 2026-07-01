"""
Tests for session-level pooled surrogate thresholding.
"""
from __future__ import annotations

import numpy as np
import pytest

from multisync.session_threshold import (
    compute_session_pooled_threshold,
    compute_condition_pooled_thresholds,
)
from multisync.synthetic import generate_ground_truth_dyad


def _make_dyad_signals(coupling=0.6, seed=0, duration_sec=60, hz=1.0):
    ds = generate_ground_truth_dyad(
        coupling=coupling,
        noise_ratio=0.3,
        duration_sec=duration_sec,
        hz=hz,
        seed=seed,
    )
    ds.align(target_hz=hz)
    ds, _ = ds.zscore()
    a = ds.get_aligned_array("behavior", "person_a")
    b = ds.get_aligned_array("behavior", "person_b")
    return a, b


def test_session_pooled_threshold_single_dyad():
    a, b = _make_dyad_signals()
    threshold, meta = compute_session_pooled_threshold(
        [(a, b)],
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        percentile=95.0,
        seed=42,
    )
    assert np.isfinite(threshold)
    assert meta["mode"] == "session_pooled"
    assert meta["n_dyads"] == 1
    assert meta["total_replicates"] == 50
    assert not meta["fallback_used"]


def test_session_pooled_threshold_across_dyads_shared():
    signals = [_make_dyad_signals(seed=i) for i in range(3)]
    threshold, meta = compute_session_pooled_threshold(
        signals,
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        percentile=95.0,
        seed=42,
    )
    assert meta["n_dyads"] == 3
    assert meta["total_replicates"] == 150


def test_condition_pooled_thresholds():
    cond_a = [_make_dyad_signals(seed=i) for i in range(3)]
    cond_b = [_make_dyad_signals(seed=i + 10) for i in range(3)]
    results = compute_condition_pooled_thresholds(
        {"A": cond_a, "B": cond_b},
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        seed=42,
    )
    assert set(results.keys()) == {"A", "B"}
    for cond, (thr, meta) in results.items():
        assert meta["mode"] == "condition_pooled"
        assert meta["condition"] == cond
        assert not meta["fallback_used"]


def test_session_pooled_threshold_fallback_on_empty():
    threshold, meta = compute_session_pooled_threshold(
        [],
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
    )
    assert meta["fallback_used"]
    assert meta["reason"] == "empty dyad_signals"

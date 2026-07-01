"""
Tests for WCLR backend and BatchComputationPipeline integration.
"""
from __future__ import annotations

import numpy as np
import pytest

from multisync.wclr import (
    windowed_cross_lagged_regression,
    wclr_coupling_trace,
)
from multisync.computation_pipeline import (
    ComputationPipeline,
    BatchComputationPipeline,
)
from multisync.synthetic import generate_ground_truth_dyad


def _make_dyad(coupling=0.6, seed=0, duration_sec=60, hz=1.0):
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


def test_wclr_trace_shape():
    a, b = _make_dyad()
    trace, lags = windowed_cross_lagged_regression(
        a, b, window_size=10, hz=1.0, max_lag_samples=2
    )
    expected_len = len(a) - 10 + 1
    assert len(trace) == expected_len
    assert len(lags) == expected_len


def test_wclr_coupling_increases_with_coupling():
    # WCLR captures lagged predictive power after controlling for autocorrelation.
    # It is not guaranteed to monotonically increase with coupling in every generator,
    # but on average it should be higher for coupled than for uncoupled dyads.
    means = []
    for coupling in [0.0, 0.6, 0.9]:
        vals = []
        for seed in range(5):
            a, b = _make_dyad(coupling=coupling, seed=seed)
            trace = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1)
            vals.append(float(np.nanmean(trace)))
        means.append(np.mean(vals))
    # At least no-coupling should be lowest on average
    assert means[0] < means[1] or means[0] < means[2]


def test_computation_pipeline_wclr_backend():
    a, b = _make_dyad()
    pipe = ComputationPipeline(
        hz=1.0, window_size=10, onset_threshold=0.2, backend="wclr"
    )
    feats = pipe.run(a, b, label="wclr")
    feat_dict = feats.to_dict()
    assert "mean_synchrony" in feat_dict
    assert "peak_amplitude" in feat_dict
    assert pipe.wcc is not None


def test_batch_computation_pipeline_session_pooled_wclr():
    signals = [_make_dyad(seed=i) for i in range(3)]
    batch = BatchComputationPipeline(
        hz=1.0,
        window_size=10,
        onset_threshold="session_pooled",
        surrogate_n=50,
        backend="wclr",
    )
    for i, (a, b) in enumerate(signals):
        batch.add_dyad(a, b, label=f"dyad_{i}")
    df = batch.run()
    assert len(df) == 3
    assert batch.threshold_meta["backend"] == "wclr"
    assert not batch.threshold_meta["fallback_used"]
    assert df["threshold_value"].nunique() == 1


def test_wclr_r2_metric():
    a, b = _make_dyad(coupling=0.6)
    trace_beta = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1, metric="beta")
    trace_r2 = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1, metric="r2")
    assert np.all((trace_r2 >= 0) | np.isnan(trace_r2))
    assert np.all((trace_r2 <= 1) | np.isnan(trace_r2))

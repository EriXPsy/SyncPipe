from __future__ import annotations

import numpy as np

from syncpipe.computation_pipeline import ComputationPipeline
from syncpipe.coupling_pipeline import compute_coupling_trace


def _signals(n=120):
    t = np.arange(n, dtype=float)
    return np.sin(t / 7), np.cos(t / 11)


def _trace(backend="wcc", window_type="rect", method="cumsum"):
    a, b = _signals()
    return compute_coupling_trace(
        a, b, 4.0, 20, backend, method, True, window_type, 2, "beta"
    )


def test_wcc_helper_matches_stateful_pipeline():
    a, b = _signals()
    pipe = ComputationPipeline(hz=4.0, window_size=20)
    pipe.load_signals(a, b)
    expected = pipe.compute_wcc()
    np.testing.assert_array_equal(_trace(), expected)


def test_tapered_wcc_helper_matches_stateful_pipeline():
    a, b = _signals()
    pipe = ComputationPipeline(hz=4.0, window_size=20, window_type="hann")
    pipe.load_signals(a, b)
    expected = pipe.compute_wcc()
    np.testing.assert_array_equal(_trace(window_type="hann", method="stride"), expected)


def test_wclr_helper_matches_stateful_pipeline():
    a, b = _signals()
    pipe = ComputationPipeline(hz=4.0, window_size=20, backend="wclr")
    pipe.load_signals(a, b)
    expected = pipe.compute_wcc()
    np.testing.assert_array_equal(_trace(backend="wclr"), expected)

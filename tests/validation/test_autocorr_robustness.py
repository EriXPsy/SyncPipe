"""Unit tests for the autocorrelation-robustness validation module.

These pin two invariants:

1. ``generate_signals(noise_model="ar1")`` produces signals whose lag-1
   autocorrelation is close to ``ar_phi`` (the independent component is now
   persistent, not white), while ``noise_model="white"`` (the default) stays
   backward compatible with the original generator.
2. The existence-audit FPR measurement runs and returns a well-formed result
   in both regimes (no calibration bound asserted here — that belongs to the
   larger scripted run, not a fast unit test).
"""
from __future__ import annotations

import numpy as np

from syncpipe.simulation.shared_signal_model import (
    generate_signals,
    constant_coupling,
)
from syncpipe.validation.autocorr_robustness import (
    measure_fpr,
    measure_power,
)


def _lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def test_ar1_noise_is_autocorrelated():
    r = generate_signals(
        constant_coupling(0.15), duration_sec=800, hz=1.0,
        noise_sigma=0.3, seed=0, noise_model="ar1", ar_phi=0.9,
    )
    # At c=0.15 the signal is ~85% AR(1) independent component, so lag-1
    # autocorrelation should be well above the white-noise floor (~0).
    ac = _lag1(r.x_A)
    assert ac > 0.5, f"expected persistent AR(1) noise, got lag-1 ac={ac:.3f}"
    assert r.params["noise_model"] == "ar1"
    assert r.params["ar_phi"] == 0.9


def test_white_noise_stays_backward_compatible():
    r_new = generate_signals(
        constant_coupling(0.15), duration_sec=400, hz=1.0,
        noise_sigma=0.3, seed=7, noise_model="white",
    )
    r_default = generate_signals(
        constant_coupling(0.15), duration_sec=400, hz=1.0,
        noise_sigma=0.3, seed=7,
    )
    # Default is "white" and byte-identical to the explicit white path.
    assert r_new.params.get("noise_model", "white") == "white"
    assert np.array_equal(r_new.x_A, r_default.x_A)


def test_fpr_measurement_returns_wellformed_result():
    res = measure_fpr("ar1", n_dyads=6, n_samples=200, window=30,
                      surrogate_n=39, phi=0.9, seed=3)
    assert res["noise_model"] == "ar1"
    assert res["n_dyads"] == 6.0
    assert 0.0 <= res["fpr"] <= 1.0


def test_power_measurement_runs():
    res = measure_power("ar1", coupling=0.6, n_dyads=6, n_samples=300,
                        window=30, surrogate_n=39, phi=0.9, seed=3)
    assert res["noise_model"] == "ar1"
    assert 0.0 <= res["power"] <= 1.0

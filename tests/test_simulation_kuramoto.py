"""Tests for the simulation ground-truth generators (Blind Spot B fuses).

The Kuramoto simulator and the shared-signal model are what *produce* the
validation numbers cited in the paper.  If a coupling parameter is
mis-wired or a noise term is broken, the output still looks plausible but
the ground truth underneath is wrong.  These tests pin the core invariant:
synchrony must increase with coupling strength.
"""

from __future__ import annotations

import numpy as np

from multisync.simulation import constant_coupling, generate_signals
from multisync.simulation.kuramoto import mean_sync_from_K, solve_phase_difference


def test_solve_phase_difference_in_unit_range():
    r = solve_phase_difference(0.5, 0.7, 0.0, 60.0)
    assert r.min() >= 0.0
    assert r.max() <= 1.0
    assert r.shape[0] == 2000


def test_kuramoto_sync_monotonic_in_coupling():
    """Larger constant coupling -> larger time-averaged synchrony."""
    Ks = [0.0, 0.2, 0.5, 1.0, 2.0]
    syncs = np.array([mean_sync_from_K(K) for K in Ks])
    # Strictly increasing across the grid.
    assert np.all(np.diff(syncs) > 1e-3), f"non-monotonic: {syncs}"


def test_shared_signal_coupling_monotonic():
    """Higher preset coupling -> higher observed cross-person correlation."""
    c_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    cors = []
    for c in c_vals:
        res = generate_signals(
            constant_coupling(c),
            duration_sec=120.0,
            hz=2.0,
            noise_sigma=0.2,
            seed=0,
        )
        cors.append(abs(np.corrcoef(res.x_A, res.x_B)[0, 1]))
    cors = np.array(cors)
    assert np.all(np.diff(cors) > 0), f"non-monotonic coupling->corr: {cors}"

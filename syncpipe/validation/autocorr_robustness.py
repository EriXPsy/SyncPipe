"""Autocorrelation-robustness validation for the signal-level existence audit.

SyncPipe's synchrony-existence null is **signal-level IAAFT**: it randomises
each signal while preserving its autocorrelation (and amplitude distribution),
so that a "significant" result means the two signals share coupling *beyond*
what two independent, but self-persistent, processes could produce.

The critical question for that null is whether it holds on **realistic**
signals — low-frequency physiological envelopes (EDA / HRV / respiration) whose
lag-1 autocorrelation is ~0.9 — rather than on the white-noise signals the
earlier GT validation used. On white noise, IAAFT degenerates toward FT
(phase randomisation), because there is no autocorrelation to preserve; GT
validation on white noise therefore never actually exercises the
autocorrelation-preservation machinery that real data depends on.

This module closes that gap by measuring, under two noise regimes
(``white`` vs ``ar1``, AR(1) coefficient ``phi``):

1. **False-positive rate (FPR)** — independent dyads with *no* coupling:
   the fraction of dyads declared significant on ``peak_amplitude`` should be
   ≈ ``alpha`` (the existence audit must not manufacture coupling from
   self-persistent noise).
2. **Power** — coupled dyads: the fraction declared significant should be
   well above ``alpha``.

These are NOT confirmatory results — they are calibration checks that the
null behaves as advertised on the statistical structure of real data.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..design_controls import synchrony_existence_audit
from ..simulation.shared_signal_model import generate_signals, constant_coupling


# ---------------------------------------------------------------------------
# Local noise helpers (small; avoids depending on private simulation names)
# ---------------------------------------------------------------------------

def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Stationary unit-variance AR(1) noise, lag-1 autocorrelation ``phi``."""
    x = np.empty(n, dtype=float)
    x[0] = rng.normal(0.0, 1.0)
    eps = rng.normal(0.0, 1.0, n)
    scale = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + scale * eps[i]
    return x


def _noise_fn(noise_model: str, phi: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    if noise_model == "ar1":
        return lambda rng, n: _ar1(rng, n, phi)
    return lambda rng, n: rng.normal(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------

def _run_audit(sig_a: np.ndarray, sig_b: np.ndarray, window: int,
               surrogate_n: int, seed: int) -> bool:
    """Run the existence audit for one pair; return peak_amplitude significance."""
    r = synchrony_existence_audit(
        sig_a, sig_b, hz=1.0, window_size=window,
        surrogate_n=surrogate_n, seed=seed,
    )
    return bool(r.get("per_feature_significant", {}).get("peak_amplitude", False))


def measure_fpr(
    noise_model: str,
    *,
    n_dyads: int,
    n_samples: int,
    window: int,
    surrogate_n: int,
    phi: float = 0.9,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Fraction of *independent* dyads declared significant (the null case).

    Each dyad is two independent draws of the given noise model — there is no
    shared signal, so a correctly-calibrated existence audit should reject at
    a rate ≈ ``alpha``.
    """
    fn = _noise_fn(noise_model, phi)
    rng = np.random.default_rng(seed)
    n_sig = 0
    for d in range(n_dyads):
        a = fn(rng, n_samples)
        b = fn(rng, n_samples)
        n_sig += int(_run_audit(a, b, window, surrogate_n, seed=seed * 1000 + d))
    return {
        "noise_model": noise_model,
        "phi": phi if noise_model == "ar1" else 0.0,
        "n_dyads": float(n_dyads),
        "n_significant": float(n_sig),
        "fpr": n_sig / n_dyads,
        "alpha": alpha,
    }


def measure_power(
    noise_model: str,
    *,
    coupling: float,
    n_dyads: int,
    n_samples: int,
    window: int,
    surrogate_n: int,
    phi: float = 0.9,
    seed: int = 0,
) -> Dict[str, float]:
    """Fraction of *coupled* dyads declared significant (the alternative case).

    Dyads are generated with :func:`generate_signals` at constant coupling
    ``coupling`` under the given noise model; the existence audit should reject
    the null at a rate well above ``alpha``.
    """
    n_sig = 0
    for d in range(n_dyads):
        r = generate_signals(
            constant_coupling(coupling),
            duration_sec=n_samples, hz=1.0, noise_sigma=0.3,
            seed=seed + d, noise_model=noise_model, ar_phi=phi,
        )
        n_sig += int(_run_audit(r.x_A, r.x_B, window, surrogate_n, seed=seed * 1000 + d))
    return {
        "noise_model": noise_model,
        "phi": phi if noise_model == "ar1" else 0.0,
        "coupling": coupling,
        "n_dyads": float(n_dyads),
        "n_significant": float(n_sig),
        "power": n_sig / n_dyads,
    }


# ---------------------------------------------------------------------------
# Top-level calibration run
# ---------------------------------------------------------------------------

def run_autocorr_robustness(
    *,
    noise_models: tuple = ("white", "ar1"),
    phi: float = 0.9,
    coupling: float = 0.6,
    n_dyads: int = 40,
    n_samples: int = 600,
    window: int = 30,
    surrogate_n: int = 99,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the full FPR + power calibration under each noise regime.

    Returns a dict with ``fpr`` and ``power`` sub-dicts keyed by noise model,
    plus the parameters used.
    """
    fpr = {}
    power = {}
    for nm in noise_models:
        fpr[nm] = measure_fpr(
            nm, n_dyads=n_dyads, n_samples=n_samples, window=window,
            surrogate_n=surrogate_n, phi=phi, seed=seed,
        )
        power[nm] = measure_power(
            nm, coupling=coupling, n_dyads=n_dyads, n_samples=n_samples,
            window=window, surrogate_n=surrogate_n, phi=phi, seed=seed,
        )
    return {
        "parameters": {
            "noise_models": list(noise_models),
            "phi": phi,
            "coupling": coupling,
            "n_dyads": n_dyads,
            "n_samples": n_samples,
            "window": window,
            "surrogate_n": surrogate_n,
            "seed": seed,
        },
        "fpr": fpr,
        "power": power,
    }

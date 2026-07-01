"""
Unified signal generator for all PGT scenarios.

Signal model: person = c(t)*shared + (1-c(t))*independent + noise.
Math derivation: see docs/signal_model.md.

Responsible for: generate PGT signals, ground-truth WCC, coupling builders.
MUST NOT import SyncPipe feature extraction (circular dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared rhythmic signal
# ---------------------------------------------------------------------------

def _shared_rhythm(
    t: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Generate a multi-sine shared rhythm in the interpersonal synchrony band.

    Frequencies span 0.08-0.35 Hz, corresponding to the typical range of
    spontaneous dyadic coordination (movement, respiration, social gaze).
    The composite avoids a single pure sinusoid, which would be trivially
    detectable and artificially inflate feature recovery metrics.

    Parameters
    ----------
    t : np.ndarray
        Time vector (seconds).
    seed : int
        RNG seed for reproducible phase offsets.

    Returns
    -------
    np.ndarray
        Normalised (mean ≈ 0, std ≈ 1) shared rhythm.
    """
    rng = np.random.default_rng(seed + 9999)
    n_components = 5
    freqs = np.linspace(0.08, 0.35, n_components)
    amps = np.linspace(1.0, 0.4, n_components)  # decreasing amplitude with freq
    phases = rng.uniform(0, 2 * np.pi, n_components)

    signal = np.zeros_like(t, dtype=float)
    for f, a, phi in zip(freqs, amps, phases):
        signal += a * np.sin(2 * np.pi * f * t + phi)

    # Normalise to unit variance for consistent signal-to-noise scaling
    sd = float(np.std(signal))
    if sd > 1e-12:
        signal /= sd
    return signal


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PGTResult:
    """Container for a PGT-generated dyadic time series.

    Attributes
    ----------
    x_A, x_B : np.ndarray
        Person A and B observed signals (includes noise).
    c_t : np.ndarray
        Ground-truth coupling trajectory c(t) — the preset function
        that determines the shared-vs-independent mix at each time point.
    t : np.ndarray
        Time vector (seconds).
    hz : float
        Sampling rate.
    noise_sigma : float
        Noise standard deviation (σ in the signal model).
    params : dict
        Scenario-specific parameters for traceability.
    """
    x_A: np.ndarray
    x_B: np.ndarray
    c_t: np.ndarray
    t: np.ndarray
    hz: float
    noise_sigma: float
    params: dict

    @property
    def duration_sec(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def n_samples(self) -> int:
        return len(self.t)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_signals(
    c_t: np.ndarray | Callable[[np.ndarray], np.ndarray],
    duration_sec: float = 300.0,
    hz: float = 1.0,
    noise_sigma: float = 0.3,
    micro_lag_sec: float = 0.0,
    seed: int = 42,
    scenario_params: Optional[dict] = None,
) -> PGTResult:
    """Generate dyadic signals under time-varying coupling c(t).

    This is the single generative entry point for all PGT scenarios.
    The coupling function ``c(t)`` is the only thing that distinguishes
    PGT-1 (constant), PGT-2 (alternating), and PGT-3 (trapezoidal).

    Parameters
    ----------
    c_t : np.ndarray or callable
        Coupling strength at each time point.  If callable, it must
        accept a time vector ``t`` and return a same-shape array.
        Values must be in [0, 1].
    duration_sec : float
        Total duration in seconds.
    hz : float
        Sampling rate (Hz).
    noise_sigma : float
        Standard deviation of the independent noise component.
        Scaled relative to the unit-variance shared rhythm, so
        σ = 0.3 means noise is 30% of signal amplitude.
    micro_lag_sec : float
        Tiny phase lag for Person B (seconds).  Non-zero values
        simulate physiological conduction delay without creating a
        meaningful lead-lag structure.  Default 0.0.
    seed : int
        RNG seed for reproducible noise and shared rhythm phases.
    scenario_params : dict, optional
        Arbitrary metadata for traceability (e.g. coupling grid value).

    Returns
    -------
    PGTResult
        Generated signals with ground-truth coupling trajectory.
    """
    rng = np.random.default_rng(seed)

    n = int(duration_sec * hz)
    t = np.arange(n) / hz

    # --- Coupling function ---
    if callable(c_t):
        c_arr = np.asarray(c_t(t), dtype=float)
    else:
        c_arr = np.asarray(c_t, dtype=float)

    if c_arr.shape != t.shape:
        raise ValueError(
            f"c_t must produce shape {t.shape}, got {c_arr.shape}"
        )
    if np.any(c_arr < 0) or np.any(c_arr > 1):
        raise ValueError("c_t values must be in [0, 1]")

    # --- Shared rhythm ---
    s = _shared_rhythm(t, seed=seed)

    # --- Person B with micro-lag ---
    lag_samples = int(round(micro_lag_sec * hz))
    if lag_samples > 0:
        s_b = np.zeros_like(s)
        s_b[lag_samples:] = s[:-lag_samples]
        s_b[:lag_samples] = s[:lag_samples]  # no wrap-around; use same values
    else:
        s_b = s

    # --- Independent noise (same variance as shared rhythm = 1.0) ---
    n_a = rng.normal(0, 1.0, n)
    n_b = rng.normal(0, 1.0, n)

    # --- Mix ---
    x_a = c_arr * s + (1 - c_arr) * n_a + noise_sigma * rng.normal(0, 1.0, n)
    x_b = c_arr * s_b + (1 - c_arr) * n_b + noise_sigma * rng.normal(0, 1.0, n)

    return PGTResult(
        x_A=x_a,
        x_B=x_b,
        c_t=c_arr,
        t=t,
        hz=hz,
        noise_sigma=noise_sigma,
        params=scenario_params or {},
    )


# ---------------------------------------------------------------------------
# Coupling function builders — one per PGT scenario
# ---------------------------------------------------------------------------

def constant_coupling(c: float) -> Callable[[np.ndarray], np.ndarray]:
    """PGT-1: time-invariant coupling.

    Parameters
    ----------
    c : float in [0, 1]
        Constant coupling strength.

    Returns
    -------
    callable
        f(t) -> array of shape t.shape, all equal to c.
    """
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"c must be in [0, 1], got {c}")
    c_val = float(c)

    def _c(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, c_val, dtype=float)

    _c.__name__ = f"constant_coupling_c={c_val}"
    return _c


def alternating_coupling(
    c_high: float = 0.80,
    c_low: float = 0.15,
    epoch_duration: float = 30.0,
    n_epochs: int = 4,
) -> Callable[[np.ndarray], np.ndarray]:
    """PGT-2: alternating high/low coupling epochs.

    Produces a square-wave c(t) with equal-duration high and low phases::

        c(t) = c_high  for t ∈ [t₀ + 2k·T, t₀ + (2k+1)·T)
        c(t) = c_low   for t ∈ [t₀ + (2k+1)·T, t₀ + (2k+2)·T)

    where T = epoch_duration and k = 0, 1, ..., n_epochs-1.

    The total episode count is 2 * n_epochs (n_epochs high + n_epochs low),
    and the first epoch is always HIGH.

    Parameters
    ----------
    c_high : float
        Coupling strength during high-sync epochs.
    c_low : float
        Coupling strength during low-sync epochs.
    epoch_duration : float
        Duration (seconds) of each individual high or low phase.
    n_epochs : int
        Number of high-low pairs.

    Returns
    -------
    callable
        f(t) -> array with alternating c_high / c_low.
    """
    def _c(t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t, dtype=float)
        total_duration = 2 * n_epochs * epoch_duration
        for k in range(n_epochs):
            # High phase
            t_start_high = k * 2 * epoch_duration
            t_end_high = (k * 2 + 1) * epoch_duration
            mask_high = (t >= t_start_high) & (t < t_end_high)
            out[mask_high] = c_high

            # Low phase
            t_start_low = (k * 2 + 1) * epoch_duration
            t_end_low = (k * 2 + 2) * epoch_duration
            mask_low = (t >= t_start_low) & (t < t_end_low)
            out[mask_low] = c_low

        # After last epoch: hold at c_low
        out[t >= total_duration] = c_low
        return out

    _c.__name__ = (
        f"alternating_coupling_ch={c_high}_cl={c_low}"
        f"_T={epoch_duration}_n={n_epochs}"
    )
    return _c


def trapezoidal_coupling(
    onset_delay: float = 30.0,
    rise_duration: float = 15.0,
    plateau_duration: float = 60.0,
    decay_duration: float = 30.0,
    c_baseline: float = 0.15,
    c_peak: float = 0.85,
) -> Callable[[np.ndarray], np.ndarray]:
    """PGT-3 Core: single trapezoidal episode.

    Produces a c(t) with linear onset → rise → plateau → decay → baseline::

                       c_peak ┌─────────────────────┐
                              ╱                       ╲
                             ╱                         ╲
        c_baseline ────────╱                           ╲────────
                   |--onset--|--rise--|--plateau--|--decay--|

    The onset delay (t₀) is the time from t=0 to the start of the rise.
    This is the most conservative test of temporal feature recovery:
    all phase boundaries are unambiguously defined, so any error is
    attributable to the feature extraction algorithm, not to ground-truth
    ambiguity.

    Parameters
    ----------
    onset_delay : float
        Time (seconds) from t=0 to the start of the rise phase.
        The "true onset latency" that onset_latency should recover.
    rise_duration : float
        Duration of the linear rise from c_baseline to c_peak.
    plateau_duration : float
        Duration at c_peak (fixed — not varied in the grid).
    decay_duration : float
        Duration of the linear decay from c_peak back to c_baseline.
    c_baseline : float
        Coupling during baseline (before rise and after decay).
    c_peak : float
        Coupling during plateau.

    Returns
    -------
    callable
        f(t) -> trapezoidal array.
    """
    def _c(t: np.ndarray) -> np.ndarray:
        out = np.full_like(t, c_baseline, dtype=float)

        t_rise_start = onset_delay
        t_rise_end = t_rise_start + rise_duration
        t_plateau_end = t_rise_end + plateau_duration
        t_decay_end = t_plateau_end + decay_duration

        # Rise: linear ramp from baseline to peak
        mask_rise = (t >= t_rise_start) & (t < t_rise_end)
        frac_rise = (t[mask_rise] - t_rise_start) / max(rise_duration, 1e-9)
        out[mask_rise] = c_baseline + (c_peak - c_baseline) * frac_rise

        # Plateau: constant at peak
        mask_plat = (t >= t_rise_end) & (t < t_plateau_end)
        out[mask_plat] = c_peak

        # Decay: linear ramp from peak back to baseline
        mask_decay = (t >= t_plateau_end) & (t < t_decay_end)
        frac_decay = (t[mask_decay] - t_plateau_end) / max(decay_duration, 1e-9)
        out[mask_decay] = c_peak - (c_peak - c_baseline) * frac_decay

        return out

    _c.__name__ = (
        f"trapezoidal_coupling_t0={onset_delay}"
        f"_rise={rise_duration}_plat={plateau_duration}_decay={decay_duration}"
    )
    return _c


def smooth_trapezoidal_coupling(
    onset_delay: float = 30.0,
    rise_duration: float = 15.0,
    plateau_duration: float = 60.0,
    decay_duration: float = 30.0,
    c_baseline: float = 0.15,
    c_peak: float = 0.85,
    sigmoid_width: float = 0.0,
    rise_decay_ratio: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """PGT-3 Extended: trapezoidal with shape robustness parameters.

    Extends :func:`trapezoidal_coupling` with two additional knobs for
    the shape robustness diagnostic:

    - **sigmoid_width**: if > 0, replaces the sharp linear corners with
      smooth sigmoid transitions.  Larger values produce increasingly
      ambiguous phase boundaries, testing onset detection under boundary
      ambiguity.

    - **rise_decay_ratio**: asymmetry factor.  ratio=1.0 is symmetric
      (rise = decay duration); ratio=0.2 is fast-rise/slow-decay;
      ratio=5.0 is slow-rise/fast-decay.  The *mean* of rise and decay
      durations is held constant, so total episode duration is preserved.

    Parameters
    ----------
    onset_delay, plateau_duration, c_baseline, c_peak : float
        As in :func:`trapezoidal_coupling`.
    rise_duration : float
        Mean rise/decay duration (the actual rise and decay durations
        will be adjusted by rise_decay_ratio).
    decay_duration : float
        Ignored when rise_decay_ratio ≠ 1.0; computed from rise_duration
        and rise_decay_ratio.
    sigmoid_width : float
        Width (seconds) of the sigmoid transition at each corner.
        0 = sharp linear transition (identical to trapezoidal_coupling).
    rise_decay_ratio : float
        Ratio of rise duration to decay duration.  Values ≠ 1.0 create
        asymmetric episodes.

    Returns
    -------
    callable
        f(t) -> smooth/asymmetric trapezoidal array.
    """
    # Adjust rise and decay durations for asymmetry
    if rise_decay_ratio != 1.0:
        # Hold the mean constant while adjusting ratio
        mean_dur = 0.5 * (rise_duration + decay_duration)
        # rise/decay = ratio  =>  rise = ratio * decay
        # mean = (rise + decay) / 2 = (ratio*decay + decay) / 2 = decay*(ratio+1)/2
        actual_decay = 2 * mean_dur / (rise_decay_ratio + 1)
        actual_rise = rise_decay_ratio * actual_decay
    else:
        actual_rise = rise_duration
        actual_decay = decay_duration

    def _c(t: np.ndarray) -> np.ndarray:
        out = np.full_like(t, c_baseline, dtype=float)
        amplitude = c_peak - c_baseline

        t_rise_start = onset_delay
        t_rise_end = t_rise_start + actual_rise
        t_plateau_end = t_rise_end + plateau_duration
        t_decay_end = t_plateau_end + actual_decay

        if sigmoid_width <= 0:
            # Sharp piecewise-linear (identical to trapezoidal_coupling)
            mask_rise = (t >= t_rise_start) & (t < t_rise_end)
            frac = (t[mask_rise] - t_rise_start) / max(actual_rise, 1e-9)
            out[mask_rise] = c_baseline + amplitude * frac

            mask_plat = (t >= t_rise_end) & (t < t_plateau_end)
            out[mask_plat] = c_peak

            mask_decay = (t >= t_plateau_end) & (t < t_decay_end)
            frac = (t[mask_decay] - t_plateau_end) / max(actual_decay, 1e-9)
            out[mask_decay] = c_peak - amplitude * frac
        else:
            # Smooth sigmoid transitions
            w = sigmoid_width
            for i, ti in enumerate(t):
                if ti < t_rise_start - w:
                    continue
                elif ti < t_rise_start + w:
                    # baseline → rise transition
                    frac = _sigmoid((ti - t_rise_start) / w)
                    out[i] = c_baseline + amplitude * frac * 0.0  # at very start
                    # Actually: smoothly rise from baseline toward rise slope
                    frac = _sigmoid((ti - (t_rise_start - w)) / (2 * w))
                    rise_frac = max(0.0, min(1.0, (ti - t_rise_start) / max(actual_rise, 1e-9)))
                    out[i] = c_baseline + amplitude * frac * rise_frac
                elif ti < t_rise_end - w:
                    # Pure rise
                    frac = (ti - t_rise_start) / max(actual_rise, 1e-9)
                    out[i] = c_baseline + amplitude * frac
                elif ti < t_rise_end + w:
                    # rise → plateau transition
                    rise_end_frac = _sigmoid((ti - t_rise_end + w) / (2 * w))
                    out[i] = c_baseline + amplitude * rise_end_frac
                elif ti < t_plateau_end - w:
                    # Pure plateau
                    out[i] = c_peak
                elif ti < t_plateau_end + w:
                    # plateau → decay transition
                    plat_end_frac = _sigmoid((ti - t_plateau_end + w) / (2 * w))
                    out[i] = c_peak - amplitude * (1 - plat_end_frac)
                elif ti < t_decay_end - w:
                    # Pure decay
                    frac = (ti - t_plateau_end) / max(actual_decay, 1e-9)
                    out[i] = c_peak - amplitude * frac
                elif ti < t_decay_end + w:
                    # decay → baseline transition
                    decay_end_frac = _sigmoid((ti - t_decay_end + w) / (2 * w))
                    out[i] = c_baseline + amplitude * (1 - decay_end_frac)

        return out

    _c.__name__ = (
        f"smooth_trapezoidal_t0={onset_delay}"
        f"_rise={actual_rise:.1f}_decay={actual_decay:.1f}"
        f"_sigmoid={sigmoid_width}_ratio={rise_decay_ratio}"
    )
    return _c


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid: 1 / (1 + exp(-x))."""
    # Clip to avoid overflow
    x_clipped = max(-50.0, min(50.0, x))
    return float(1.0 / (1.0 + np.exp(-x_clipped)))

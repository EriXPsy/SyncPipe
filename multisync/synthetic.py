"""
Synthetic data generator for Ground Truth validation.

Morphology: "identical" (same shape) or "divergent" (burst vs decay).
Usage: from multisync.synthetic import generate_ground_truth_dyad
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import SynchronyDataset

# Deterministic phase offsets for generate_multimodal_dyad.
# hash(name) is NOT reproducible across processes (Python 3.3+ hash randomization),
# so we use a fixed lookup table instead.
PHASE_OFFSETS: Dict[str, float] = {
    "neural": 0.0,
    "behavior": 1.2,
    "bio": 2.4,
    "psycho": 3.6,
}


def generate_ground_truth_dyad(
    lead_modality: str = "behavior",
    lag_modality: str = "neural",
    true_lag_sec: float = 12.0,
    noise_ratio: float = 0.3,
    duration_sec: float = 300.0,
    hz: float = 1.0,
    seed: int = 42,
    n_bursts: int = 5,
    burst_sigma: float = 3.0,
    gap_prob: float = 0.02,
    morphology: Literal["identical", "divergent"] = "identical",
    coupling: float = 0.7,
) -> SynchronyDataset:
    """Generate synthetic dyad with controllable lead-lag + coupling.

    Signal model: person = coupling*shared + (1-coupling)*idiosyncratic + noise.
    Theoretical rho_AB ≈ c²/(c²+(1-c)²+noise²).

    Parameters
    ----------
    coupling : float in [0,1]
        Inter-person coupling. 0.7 = moderate realistic value.
    Other params: see signature.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError(f"coupling must be in [0, 1], got {coupling}")

    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    lag_samples = int(true_lag_sec * hz)

    # Shared event anchors (the "true" episode times)
    burst_times = rng.uniform(20, duration_sec - 20 - true_lag_sec, size=n_bursts)

    if morphology == "identical":
        base_lead = np.zeros(n)
        for bt in burst_times:
            base_lead += np.exp(-0.5 * ((t - bt) / burst_sigma) ** 2)
        base_lead += 0.3 * np.sin(2 * np.pi * t / 60.0)
        signal_std = float(np.std(base_lead))

        # Idiosyncratic components (mean-zero, same std as base_lead so
        # the variance of person = coupling*shared + (1-c)*indep stays
        # comparable across coupling values).
        indep_a_lead = rng.normal(0, signal_std, size=n)
        indep_b_lead = rng.normal(0, signal_std, size=n)
        noise_a_lead = rng.normal(0, noise_ratio * signal_std, size=n)
        noise_b_lead = rng.normal(0, noise_ratio * signal_std, size=n)

        lead_a = coupling * base_lead + (1 - coupling) * indep_a_lead + noise_a_lead
        lead_b = coupling * base_lead + (1 - coupling) * indep_b_lead + noise_b_lead

        # Lag modality: shifted shared signal (or copy when lag=0)
        if lag_samples == 0:
            base_lag = base_lead.copy()
        else:
            base_lag = np.zeros(n)
            base_lag[lag_samples:] = base_lead[:-lag_samples]
        indep_a_lag = rng.normal(0, signal_std, size=n)
        indep_b_lag = rng.normal(0, signal_std, size=n)
        noise_a_lag = rng.normal(0, noise_ratio * signal_std, size=n)
        noise_b_lag = rng.normal(0, noise_ratio * signal_std, size=n)

        lag_a = coupling * base_lag + (1 - coupling) * indep_a_lag + noise_a_lag
        lag_b = coupling * base_lag + (1 - coupling) * indep_b_lag + noise_b_lag

    elif morphology == "divergent":
        lead_signal = np.zeros(n)
        for bt in burst_times:
            lead_signal += np.exp(-0.5 * ((t - bt) / burst_sigma) ** 2)
        lead_signal += 0.3 * np.sin(2 * np.pi * t / 60.0)

        lag_signal = np.zeros(n)
        tau_rise = burst_sigma * 0.5
        tau_decay = burst_sigma * 2.5
        for bt in burst_times:
            dt = t - bt - true_lag_sec * 0.3
            lag_signal += np.where(
                dt >= 0,
                np.exp(-dt / tau_decay),
                np.exp(dt / tau_rise),
            )
        lag_signal += 0.2 * np.sin(2 * np.pi * t / 45.0 + 1.5)

        std_lead = float(np.std(lead_signal))
        std_lag = float(np.std(lag_signal))

        indep_a_lead = rng.normal(0, std_lead, size=n)
        indep_b_lead = rng.normal(0, std_lead, size=n)
        indep_a_lag = rng.normal(0, std_lag, size=n)
        indep_b_lag = rng.normal(0, std_lag, size=n)
        noise_a_lead = rng.normal(0, noise_ratio * std_lead, size=n)
        noise_b_lead = rng.normal(0, noise_ratio * std_lead, size=n)
        noise_a_lag = rng.normal(0, noise_ratio * std_lag, size=n)
        noise_b_lag = rng.normal(0, noise_ratio * std_lag, size=n)

        lead_a = coupling * lead_signal + (1 - coupling) * indep_a_lead + noise_a_lead
        lead_b = coupling * lead_signal + (1 - coupling) * indep_b_lead + noise_b_lead
        lag_a = coupling * lag_signal + (1 - coupling) * indep_a_lag + noise_a_lag
        lag_b = coupling * lag_signal + (1 - coupling) * indep_b_lag + noise_b_lag

    else:
        raise ValueError(f"Unknown morphology: {morphology}. Use 'identical' or 'divergent'.")

    # Optional NaN gaps (device dropout)
    if gap_prob > 0:
        for signal in [lead_a, lead_b, lag_a, lag_b]:
            nan_mask = rng.random(n) < gap_prob
            signal[nan_mask] = np.nan

    df_lead = pd.DataFrame({"time": t, "person_a": lead_a, "person_b": lead_b})
    df_lag = pd.DataFrame({"time": t, "person_a": lag_a, "person_b": lag_b})

    ds = SynchronyDataset(
        dyad_id=(
            f"synthetic_lag{true_lag_sec}s_noise{noise_ratio}_"
            f"coup{coupling}_{morphology}"
        ),
        modalities={lead_modality: df_lead, lag_modality: df_lag},
    )

    ds._ground_truth = {
        "lead": lead_modality,
        "lag": lag_modality,
        "true_lag_sec": true_lag_sec,
        "actual_lag_sec": true_lag_sec * 0.3 if morphology == "divergent" else true_lag_sec,
        "noise_ratio": noise_ratio,
        "n_bursts": n_bursts,
        "burst_times": burst_times.tolist(),
        "burst_sigma": burst_sigma,
        "coupling": coupling,
        "morphology": morphology,
    }
    return ds


def generate_shared_stim_null_dyad(
    modality: str = "behavior",
    shared_drive_strength: float = 0.5,
    noise_ratio: float = 0.5,
    duration_sec: float = 180.0,
    hz: float = 1.0,
    seed: int = 42,
    n_bursts: int = 5,
    burst_sigma: float = 3.0,
) -> SynchronyDataset:
    """Generate shared-stimulus null dyad (zero inter-person coupling).

    Both persons driven by same external signal + independent noise.
    Canonical shared-stimulus confound (Burgess 2013).
    Use to quantify false positive rate.

    Parameters
    ----------
    shared_drive_strength : float in [0,1]
        Weight of common external signal in each person.
    Other params: see signature.
    """
    if not 0.0 <= shared_drive_strength <= 1.0:
        raise ValueError(
            f"shared_drive_strength must be in [0, 1], got {shared_drive_strength}"
        )
    if not 0.0 <= noise_ratio:
        raise ValueError(f"noise_ratio must be >= 0, got {noise_ratio}")

    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz

    burst_times = rng.uniform(20, duration_sec - 20, size=n_bursts)
    external_drive = np.zeros(n)
    for bt in burst_times:
        external_drive += np.exp(-0.5 * ((t - bt) / burst_sigma) ** 2)
    external_drive += 0.3 * np.sin(2 * np.pi * t / 60.0)
    sig_std = float(np.std(external_drive))
    if sig_std == 0.0:
        sig_std = 1.0

    indep_a = rng.normal(0, sig_std, size=n)
    indep_b = rng.normal(0, sig_std, size=n)
    noise_a = rng.normal(0, noise_ratio * sig_std, size=n)
    noise_b = rng.normal(0, noise_ratio * sig_std, size=n)

    s = shared_drive_strength
    person_a = s * external_drive + (1.0 - s) * indep_a + noise_a
    person_b = s * external_drive + (1.0 - s) * indep_b + noise_b

    df = pd.DataFrame({"time": t, "person_a": person_a, "person_b": person_b})

    ds = SynchronyDataset(
        dyad_id=(
            f"shared_stim_null_drive{shared_drive_strength}_"
            f"noise{noise_ratio}_dur{int(duration_sec)}_seed{seed}"
        ),
        modalities={modality: df},
    )
    ds._ground_truth = {
        "design": "shared_stim_null",
        "true_coupling": 0.0,
        "shared_drive_strength": shared_drive_strength,
        "noise_ratio": noise_ratio,
        "duration_sec": duration_sec,
        "hz": hz,
        "n_bursts": n_bursts,
        "burst_times": burst_times.tolist(),
        "burst_sigma": burst_sigma,
        "external_drive_std": sig_std,
        "expected_between_person_r": (
            (s * s) / (s * s + (1.0 - s) ** 2 + noise_ratio ** 2)
        ),
    }
    return ds


def generate_multimodal_dyad(
    duration_sec: float = 300.0,
    hz: float = 1.0,
    seed: int = 42,
    modalities: Optional[Dict[str, float]] = None,
    noise_ratio: float = 0.3,
) -> SynchronyDataset:
    """
    Generate a synthetic dyad with 3-4 modalities at different Hz.

    Parameters
    ----------
    modalities : dict or None
        {modality_name: original_hz}.  Default:
        {"neural": 1.0, "behavior": 10.0, "bio": 4.0}
    """
    if modalities is None:
        modalities = {"neural": 1.0, "behavior": 10.0, "bio": 4.0}

    rng = np.random.default_rng(seed)

    # Shared burst times — all modalities use the SAME temporal anchors
    # so that cross-modality synchrony is genuinely present in Ground Truth.
    # Each modality then applies a fixed offset to simulate lead-lag.
    shared_bursts = rng.uniform(20, duration_sec - 40, size=5)

    dataframes = {}
    for name, orig_hz in modalities.items():
        n = int(duration_sec * orig_hz)
        t = np.arange(n) / orig_hz

        signal = np.zeros(n)
        for bt in shared_bursts:
            # Offset each modality
            offset = {"neural": 0, "behavior": -5, "bio": -3}.get(name, 0)
            signal += np.exp(-0.5 * ((t - bt - offset) / 3.0) ** 2)

        # Deterministic phase offset (reproducible across Python processes;
        # hash(name) is randomized in Python 3.3+ and breaks reproducibility).
        _phase = PHASE_OFFSETS.get(
            name, float(sum(ord(c) for c in name) % 10)
        )
        signal += 0.2 * np.sin(2 * np.pi * t / 45.0 + _phase)
        signal += rng.normal(0, noise_ratio * np.std(signal), size=n)

        dataframes[name] = pd.DataFrame({"time": t, "value": signal})

    return SynchronyDataset(
        dyad_id="synthetic_multimodal",
        modalities=dataframes,
    )

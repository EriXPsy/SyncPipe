"""Adversarial discriminant-validity benchmark for ``peak_amplitude``.

The benchmark asks whether the existing descriptor and evidence-chain controls
separate reciprocal interaction from rival data-generating processes. It does
not add a descriptor and does not treat simulation as construct validation.

Several negative controls are expected to pass the signal-level IAAFT audit:
IAAFT removes independent dynamics, not shared stimulus, common drift, or shared
artifact. Such detections are *construct false positives*, not implementation
errors. Pseudo-pair and time-shift results show which rival explanations remain.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from ..design_controls import design_control_audit, synchrony_existence_audit


SCENARIO_METADATA: Dict[str, Dict[str, object]] = {
    "independent_ar1": {
        "positive_control": False,
        "rival": "independent autocorrelated dynamics",
    },
    "shared_stimulus": {
        "positive_control": False,
        "rival": "common event-locked input without interpersonal coupling",
    },
    "common_drift": {
        "positive_control": False,
        "rival": "shared low-frequency drift without interpersonal coupling",
    },
    "shared_context": {
        "positive_control": False,
        "rival": "aligned nonstationary context/variance without cross-person coupling",
    },
    "shared_artifact": {
        "positive_control": False,
        "rival": "simultaneous measurement artifact",
    },
    "reciprocal_var": {
        "positive_control": True,
        "rival": "bidirectional lagged interaction (positive control)",
    },
}


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    return (x - np.mean(x)) / sd if sd > 1e-12 else x - np.mean(x)


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie in (-1, 1)")
    x = np.empty(n, dtype=float)
    x[0] = rng.normal()
    eps = rng.normal(size=n)
    scale = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + scale * eps[i]
    return x


def generate_discriminant_pair(
    scenario: str,
    *,
    n_samples: int = 600,
    phi: float = 0.9,
    seed: int = 42,
    interaction_strength: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one pair under a declared rival or reciprocal control model.

    ``interaction_strength`` is used only by ``reciprocal_var`` and controls
    the cross-person lagged coefficient. It permits known between-dyad
    heterogeneity in dependability experiments.
    """
    if scenario not in SCENARIO_METADATA:
        raise ValueError(
            f"unknown scenario {scenario!r}; choose from {sorted(SCENARIO_METADATA)}"
        )
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")
    rng = np.random.default_rng(seed)
    a_noise = _ar1(rng, n_samples, phi)
    b_noise = _ar1(rng, n_samples, phi)
    t = np.arange(n_samples, dtype=float)

    if scenario == "independent_ar1":
        a, b = a_noise, b_noise

    elif scenario == "shared_stimulus":
        drive = np.zeros(n_samples, dtype=float)
        centers = np.linspace(80, n_samples - 80, 5)
        for center in centers:
            drive += np.exp(-0.5 * ((t - center) / 8.0) ** 2)
        drive = _standardize(drive)
        a = 0.75 * drive + 0.65 * a_noise
        b = 0.75 * drive + 0.65 * b_noise

    elif scenario == "common_drift":
        drift = np.cumsum(rng.normal(size=n_samples))
        # Smooth without importing SyncPipe measurement code.
        kernel = np.ones(41, dtype=float) / 41.0
        drift = _standardize(np.convolve(drift, kernel, mode="same"))
        a = 0.8 * drift + 0.6 * a_noise
        b = 0.8 * drift + 0.6 * b_noise

    elif scenario == "shared_context":
        # Both people enter the same high/low arousal schedule, but their
        # innovations remain independent. This is only an operational proxy
        # for aligned context, not a complete model of psychological co-presence.
        envelope = 0.45 + 0.55 * (np.sin(2.0 * np.pi * t / 120.0) > 0)
        a = envelope * a_noise + 0.15 * _ar1(rng, n_samples, 0.5)
        b = envelope * b_noise + 0.15 * _ar1(rng, n_samples, 0.5)

    elif scenario == "shared_artifact":
        a, b = a_noise.copy(), b_noise.copy()
        for center in np.linspace(100, n_samples - 100, 3).astype(int):
            width = min(5, n_samples - center)
            pulse = np.linspace(6.0, 2.0, width)
            a[center:center + width] += pulse
            b[center:center + width] += pulse

    else:  # reciprocal_var
        # Stable symmetric VAR(1): each person depends on both their own and
        # the partner's preceding state. This is a positive control for
        # interaction-contingent dependence, not a physiological mechanism.
        own = 0.62
        cross = float(interaction_strength)
        if not 0.0 <= cross < 1.0 - own:
            raise ValueError(
                f"interaction_strength must lie in [0, {1.0 - own:.2f}) "
                "for the reciprocal VAR to remain stable"
            )
        a = np.empty(n_samples, dtype=float)
        b = np.empty(n_samples, dtype=float)
        a[0], b[0] = rng.normal(size=2)
        eps_a = rng.normal(size=n_samples)
        eps_b = rng.normal(size=n_samples)
        for i in range(1, n_samples):
            a[i] = own * a[i - 1] + cross * b[i - 1] + eps_a[i]
            b[i] = own * b[i - 1] + cross * a[i - 1] + eps_b[i]

    return _standardize(a), _standardize(b)


def run_discriminant_benchmark(
    *,
    scenarios: Sequence[str] = tuple(SCENARIO_METADATA),
    n_replicates: int = 20,
    n_samples: int = 600,
    window_size: int = 30,
    surrogate_n: int = 99,
    phi: float = 0.9,
    seed: int = 42,
    shift_lags_sec: Sequence[float] = (-120.0, -60.0, 60.0, 120.0),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run L0 and design controls across adversarial data-generating models.

    Returns
    -------
    replicate_table, scenario_summary, design_control_summary
        Per-replicate L0 outcomes, aggregated detection rates, and cohort-level
        pseudo-pair/time-shift outcomes for each scenario.
    """
    scenarios = tuple(str(s) for s in scenarios)
    unknown = sorted(set(scenarios) - set(SCENARIO_METADATA))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    if n_replicates < 2:
        raise ValueError("n_replicates must be >= 2")
    if n_samples <= window_size + max(abs(float(x)) for x in shift_lags_sec):
        raise ValueError("n_samples is too short for window_size and requested shifts")

    rows = []
    design_rows = []
    for scenario_index, scenario in enumerate(scenarios):
        pairs = {}
        for replicate in range(int(n_replicates)):
            pair_seed = int(seed + scenario_index * 100_000 + replicate)
            a, b = generate_discriminant_pair(
                scenario, n_samples=n_samples, phi=phi, seed=pair_seed
            )
            pair_id = f"{scenario}_{replicate:04d}"
            pairs[pair_id] = (a, b)
            audit = synchrony_existence_audit(
                a, b, hz=1.0, window_size=window_size,
                surrogate_n=surrogate_n, seed=pair_seed + 50_000,
            )
            rows.append({
                "scenario": scenario,
                "replicate": replicate,
                "positive_control": bool(
                    SCENARIO_METADATA[scenario]["positive_control"]
                ),
                "rival": str(SCENARIO_METADATA[scenario]["rival"]),
                "status": audit.get("status"),
                "peak_amplitude": audit.get("observed", {}).get(
                    "peak_amplitude", np.nan
                ),
                "p_peak_amplitude": audit.get("p_values", {}).get(
                    "peak_amplitude", np.nan
                ),
                "l0_significant": bool(
                    audit.get("per_feature_significant", {}).get(
                        "peak_amplitude", False
                    )
                ),
            })

        design = design_control_audit(
            pairs,
            hz=1.0,
            window_size=window_size,
            feature_names=("peak_amplitude",),
            n_pseudo_per_dyad=min(10, max(1, n_replicates - 1)),
            shift_lags_sec=shift_lags_sec,
            seed=seed + scenario_index,
        )
        summary = design["feature_summary"]["peak_amplitude"]
        design_rows.append({
            "scenario": scenario,
            "positive_control": bool(
                SCENARIO_METADATA[scenario]["positive_control"]
            ),
            "real_median": summary["real_median"],
            "pseudo_pair_median": summary["pseudo_pair_median"],
            "time_shift_median": summary["time_shift_median"],
            "real_minus_pseudo_mean": summary["real_minus_pseudo_mean"],
            "real_minus_time_shift_mean": summary["real_minus_time_shift_mean"],
            "p_real_gt_pseudo": summary["p_real_gt_pseudo"],
            "p_real_gt_time_shift": summary["p_real_gt_time_shift"],
        })

    values = pd.DataFrame(rows)
    scenario_summary = (
        values.groupby(
            ["scenario", "positive_control", "rival"], as_index=False,
            dropna=False,
        )
        .agg(
            n_replicates=("replicate", "nunique"),
            median_peak=("peak_amplitude", "median"),
            median_p_l0=("p_peak_amplitude", "median"),
            l0_detection_rate=("l0_significant", "mean"),
        )
    )
    scenario_summary["construct_false_positive_rate"] = np.where(
        scenario_summary["positive_control"],
        np.nan,
        scenario_summary["l0_detection_rate"],
    )
    return values, scenario_summary, pd.DataFrame(design_rows)


__all__ = [
    "SCENARIO_METADATA",
    "generate_discriminant_pair",
    "run_discriminant_benchmark",
]

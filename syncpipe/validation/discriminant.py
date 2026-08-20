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
from scipy.stats import beta

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
            n_l0_detected=("l0_significant", "sum"),
            l0_detection_rate=("l0_significant", "mean"),
        )
    )
    scenario_summary["construct_false_positive_rate"] = np.where(
        scenario_summary["positive_control"],
        np.nan,
        scenario_summary["l0_detection_rate"],
    )
    return values, scenario_summary, pd.DataFrame(design_rows)


def exact_binomial_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> Tuple[float, float]:
    """Clopper-Pearson interval for a detection or false-positive rate."""
    successes = int(successes)
    trials = int(trials)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def evaluate_discriminant_acceptance(
    scenario_summary: pd.DataFrame,
    design_controls: pd.DataFrame,
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
    minimum_positive_power: float = 0.80,
    maximum_construct_fpr: float = 0.10,
) -> pd.DataFrame:
    """Evaluate pre-declared calibration and discriminant-validity criteria.

    Criteria are intentionally strict and may fail. A failed criterion is not
    rewritten after observing the benchmark; it identifies either inadequate
    method behavior or insufficient replication precision.
    """
    required_summary = {
        "scenario", "n_replicates", "n_l0_detected", "l0_detection_rate",
        "positive_control",
    }
    required_controls = {
        "scenario", "p_real_gt_pseudo", "p_real_gt_time_shift",
        "positive_control",
    }
    if not required_summary.issubset(scenario_summary.columns):
        raise ValueError("scenario_summary lacks required benchmark columns")
    if not required_controls.issubset(design_controls.columns):
        raise ValueError("design_controls lacks required benchmark columns")

    rows = []
    summaries = scenario_summary.set_index("scenario")
    controls = design_controls.set_index("scenario")

    def _add(scenario, criterion, observed, lower, upper, threshold, passed, note):
        rows.append({
            "scenario": scenario,
            "criterion": criterion,
            "observed": float(observed),
            "ci_lower": float(lower) if np.isfinite(lower) else np.nan,
            "ci_upper": float(upper) if np.isfinite(upper) else np.nan,
            "threshold": str(threshold),
            "passed": bool(passed),
            "interpretation": note,
        })

    for scenario, item in summaries.iterrows():
        n = int(item["n_replicates"])
        k = int(item["n_l0_detected"])
        rate = float(item["l0_detection_rate"])
        lower, upper = exact_binomial_interval(k, n, confidence=confidence)
        positive = bool(item["positive_control"])
        if scenario == "independent_ar1":
            _add(
                scenario, "independent_null_calibration", rate, lower, upper,
                f"CI contains alpha={alpha}", lower <= alpha <= upper,
                "The exact FPR interval must include the nominal alpha.",
            )
        elif positive:
            _add(
                scenario, "positive_control_power", rate, lower, upper,
                f"lower CI >= {minimum_positive_power}",
                lower >= minimum_positive_power,
                "Power must clear the frozen lower-confidence bound.",
            )
        else:
            _add(
                scenario, "construct_false_positive_rate", rate, lower, upper,
                f"upper CI <= {maximum_construct_fpr}",
                upper <= maximum_construct_fpr,
                "Negative-control detections quantify construct-level false positives.",
            )

        if scenario in controls.index:
            control = controls.loc[scenario]
            p_pseudo = float(control["p_real_gt_pseudo"])
            p_shift = float(control["p_real_gt_time_shift"])
            if positive:
                pseudo_pass = np.isfinite(p_pseudo) and p_pseudo < alpha
                shift_pass = np.isfinite(p_shift) and p_shift < alpha
                pseudo_note = "The reciprocal positive control should exceed pseudo-pairs."
                shift_note = "The reciprocal positive control should exceed shifted pairs."
                threshold = f"p < {alpha}"
            else:
                pseudo_pass = np.isfinite(p_pseudo) and p_pseudo >= alpha
                shift_pass = np.isfinite(p_shift) and p_shift >= alpha
                pseudo_note = "A negative control must not acquire false partner specificity."
                shift_note = "A negative control must not acquire false alignment specificity."
                threshold = f"p >= {alpha}"
            _add(
                scenario, "pseudo_pair_specificity", p_pseudo,
                np.nan, np.nan, threshold, pseudo_pass, pseudo_note,
            )
            _add(
                scenario, "time_shift_specificity", p_shift,
                np.nan, np.nan, threshold, shift_pass, shift_note,
            )

    result = pd.DataFrame(rows)
    result.attrs["all_passed"] = bool(result["passed"].all()) if not result.empty else False
    result.attrs["criteria"] = {
        "alpha": float(alpha),
        "confidence": float(confidence),
        "minimum_positive_power": float(minimum_positive_power),
        "maximum_construct_fpr": float(maximum_construct_fpr),
    }
    return result


__all__ = [
    "SCENARIO_METADATA",
    "generate_discriminant_pair",
    "run_discriminant_benchmark",
    "exact_binomial_interval",
    "evaluate_discriminant_acceptance",
]

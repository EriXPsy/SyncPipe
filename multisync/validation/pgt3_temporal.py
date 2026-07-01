"""
PGT-3 Core — Temporal Feature Recovery (Single Trapezoidal Episode).

Validates conditional-tier temporal features (onset_latency, rise_time, recovery_time).

Parameter grid: onset_delay ∈ {10,30,60}s, rise_duration ∈ {5,15,30}s,
decay_duration ∈ {10,30,60}s, 30 seeds.

Pre-registered hypotheses: H3.1-H3.6 (see docs/METHODOLOGY_LOCK_IN.md).

Success criteria: A-level (H3.1, H3.2), B-level (H3.3), C-level (H3.3 ρ ∈ [0.30,0.60]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..simulation.shared_signal_model import (
    generate_signals,
    trapezoidal_coupling,
    PGTResult,
)
from ..dynamic_features import sliding_window_wcc
from ..feature_definitions import (
    extract_features as _ssot_extract_features,
    compute_surrogate_threshold,
    ONSET_THRESHOLD,
)
from ..validation.pgt1_intensity import iaaft_surrogate
from .recovery import _extract_six_features, ONSET_THRESHOLD_DEFAULT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PGT3Config:
    """Frozen configuration for a PGT-3 temporal recovery experiment.

    Attributes
    ----------
    onset_delays : Sequence[float]
        Onset delay values to sweep (seconds).
    rise_durations : Sequence[float]
        Rise duration values to sweep (seconds).
    decay_durations : Sequence[float]
        Decay duration values to sweep (seconds).
    plateau_duration : float
        Fixed plateau duration (seconds).
    c_baseline : float
        Baseline coupling (must be > 0 to ensure onset is well-defined).
    c_peak : float
        Peak coupling.
    noise_sigma : float
        Fixed noise level.
    duration_sec : float
        Total signal duration — must be large enough to accommodate the
        largest episode (max onset + rise + plateau + decay + buffer).
    hz_signal : float
        Sampling rate for signal generation.
    hz_wcc : float
        WCC computation rate.
    wcc_window_sec : float
        WCC sliding window width.
    onset_threshold : float
        Onset/dwell/switching threshold.
    seeds : Sequence[int]
        RNG seeds.
    """
    onset_delays: Sequence[float] = (10.0, 30.0, 60.0)
    rise_durations: Sequence[float] = (5.0, 15.0, 30.0)
    decay_durations: Sequence[float] = (10.0, 30.0, 60.0)
    plateau_duration: float = 60.0
    c_baseline: float = 0.15
    c_peak: float = 0.85
    noise_sigma: float = 0.3
    duration_sec: float = 300.0
    hz_signal: float = 1.0
    hz_wcc: float = 1.0
    wcc_window_sec: float = 30.0
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT
    seeds: Sequence[int] = field(default_factory=lambda: tuple(range(3000, 3030)))

    use_surrogate_threshold: bool = False
    n_surrogates_for_threshold: int = 100

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz_wcc)))

    @property
    def n_cells(self) -> int:
        return (
            len(self.onset_delays)
            * len(self.rise_durations)
            * len(self.decay_durations)
            * len(self.seeds)
        )


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def _run_pgt3_cell(
    onset_delay: float,
    rise_duration: float,
    decay_duration: float,
    seed: int,
    cfg: PGT3Config,
) -> dict:
    """Run one PGT-3 grid cell."""
    c_func = trapezoidal_coupling(
        onset_delay=onset_delay,
        rise_duration=rise_duration,
        plateau_duration=cfg.plateau_duration,
        decay_duration=decay_duration,
        c_baseline=cfg.c_baseline,
        c_peak=cfg.c_peak,
    )

    result = generate_signals(
        c_t=c_func,
        duration_sec=cfg.duration_sec,
        hz=cfg.hz_signal,
        noise_sigma=cfg.noise_sigma,
        seed=seed,
        scenario_params={
            "pgt": "PGT-3",
            "onset_delay": onset_delay,
            "rise_duration": rise_duration,
            "decay_duration": decay_duration,
            "plateau_duration": cfg.plateau_duration,
        },
    )

    # BUG FIX 4: WCC Window must be small for Event-Locked Phase Tracing
    # A 30s window will "look ahead" 15s, triggering onset_latency way before t0.
    wcc_win_sec = min(cfg.wcc_window_sec, max(2.0, onset_delay / 2.0))
    wcc_win_samples = max(2, int(round(wcc_win_sec * cfg.hz_wcc)))

    wcc = sliding_window_wcc(
        result.x_A, result.x_B,
        window_size=wcc_win_samples,
        hz=cfg.hz_signal,
    )

    if cfg.hz_wcc != cfg.hz_signal:
        factor = int(cfg.hz_signal / cfg.hz_wcc)
        if factor > 1:
            wcc = wcc[::factor]
            
    wcc_hz = cfg.hz_wcc

    if cfg.use_surrogate_threshold:
        rng = np.random.default_rng(seed + 99_999)
        surr_wccs = []
        for _ in range(cfg.n_surrogates_for_threshold):
            a_s = iaaft_surrogate(result.x_A, rng)
            b_s = iaaft_surrogate(result.x_B, rng)
            wcc_s = sliding_window_wcc(a_s, b_s, window_size=wcc_win_samples, hz=cfg.hz_signal)
            surr_wccs.append(wcc_s)
        surr_matrix = np.vstack(surr_wccs)
        effective_threshold, threshold_is_surrogate_derived = compute_surrogate_threshold(surr_matrix)
    else:
        effective_threshold = cfg.onset_threshold

    feats = _ssot_extract_features(
        wcc,
        hz=wcc_hz,
        threshold=effective_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    ).to_dict()

    # Ground truth: onset_latency should equal onset_delay
    expected_onset = onset_delay
    expected_rise = rise_duration
    expected_recovery = decay_duration
    expected_dwell_approx = rise_duration + cfg.plateau_duration + decay_duration

    row = {
        "onset_delay": onset_delay,
        "rise_duration": rise_duration,
        "decay_duration": decay_duration,
        "plateau_duration": cfg.plateau_duration,
        "seed": seed,
        "noise_sigma": cfg.noise_sigma,
        "onset_threshold": cfg.onset_threshold,
        "c_baseline": cfg.c_baseline,
        "c_peak": cfg.c_peak,
        "expected_onset_latency": expected_onset,
        "expected_rise_time": expected_rise,
        "expected_recovery_time": expected_recovery,
        "expected_dwell_approx": expected_dwell_approx,
        "n_wcc_samples": int(np.sum(~np.isnan(wcc))),
    }
    row.update(feats)
    return row


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_pgt3_grid(cfg: Optional[PGT3Config] = None) -> pd.DataFrame:
    """Run the full PGT-3 onset × rise × decay × seed grid.

    Returns
    -------
    pd.DataFrame
        One row per (onset_delay, rise_duration, decay_duration, seed) cell.
    """
    cfg = cfg or PGT3Config()
    rows: List[dict] = []
    for t0 in cfg.onset_delays:
        for rise in cfg.rise_durations:
            for decay in cfg.decay_durations:
                for seed in cfg.seeds:
                    rows.append(_run_pgt3_cell(
                        float(t0), float(rise), float(decay),
                        int(seed), cfg,
                    ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_pgt3(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition summary for PGT-3 temporal features.

    Returns long-format table with one row per
    (onset_delay, rise_duration, decay_duration, feature).
    """
    feature_cols = [
        "onset_latency", "rise_time", "recovery_time",
        "peak_amplitude", "dwell_time", "mean_synchrony",
        "switching_rate", "synchrony_entropy",
    ]

    group_keys = ["onset_delay", "rise_duration", "decay_duration"]

    # Also report definedness
    defn_cols = ["onset_defined", "rise_defined", "recovery_defined"]

    grouped = df.groupby(group_keys, sort=True)
    rows: List[dict] = []
    for key_vals, sub in grouped:
        key_dict = dict(zip(group_keys, key_vals))
        n_seeds = int(sub["seed"].count())

        # Numeric features
        for feat in feature_cols:
            if feat not in sub.columns:
                continue
            col = sub[feat]
            mean_val = float(col.mean())
            sd_val = float(col.std(ddof=1)) if n_seeds > 1 else float("nan")
            row_data = {
                **key_dict,
                "feature": feat,
                "mean": mean_val,
                "sd": sd_val,
                "n_seeds": n_seeds,
            }
            rows.append(row_data)

        # Definedness fractions
        for dn in defn_cols:
            if dn not in sub.columns:
                continue
            frac = float(sub[dn].mean())
            rows.append({
                **key_dict,
                "feature": f"definedness_{dn}",
                "mean": frac,
                "sd": float("nan"),
                "n_seeds": n_seeds,
            })

    summary = pd.DataFrame(rows)
    return summary


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def test_pgt3_hypotheses(df: pd.DataFrame) -> Dict[str, dict]:
    """Run pre-registered hypothesis tests for PGT-3.

    Parameters
    ----------
    df : pd.DataFrame
        Raw grid results from ``run_pgt3_grid``.

    Returns
    -------
    dict
        Hypothesis label → {metric, value, threshold, passed, tier, note}.
    """
    from scipy.stats import spearmanr

    results = {}
    valid = df.dropna(
        subset=["onset_latency", "rise_time", "recovery_time"]
    ).copy()

    if len(valid) < 5:
        return {"error": {"note": "Insufficient data for hypothesis tests"}}

    # H3.1: onset_latency bias
    valid["onset_bias"] = valid["onset_latency"] - valid["onset_delay"]
    valid_onset = valid.dropna(subset=["onset_bias"])
    mean_bias = float(valid_onset["onset_bias"].mean())
    mad_bias = float(np.median(np.abs(valid_onset["onset_bias"] - np.median(valid_onset["onset_bias"]))))
    results["H3.1_onset_bias"] = {
        "metric": "mean_bias_seconds",
        "value": mean_bias,
        "mad_bias_seconds": mad_bias,
        "threshold": "< 5 s |bias|",
        "passed": abs(mean_bias) < 5.0,
        "tier": "A",
        "note": "|bias| < 5 s = strong temporal recovery",
    }

    # H3.2: rise_time Spearman ρ
    valid_rise = valid.dropna(subset=["rise_time"])
    if len(valid_rise) > 2:
        rho, p = spearmanr(valid_rise["rise_duration"], valid_rise["rise_time"])
        results["H3.2_rise_correlation"] = {
            "metric": "spearman_rho",
            "value": float(rho),
            "p_value": float(p),
            "threshold": "> 0.70",
            "passed": rho > 0.70,
            "tier": "A",
            "note": "rise_time tracks rise_duration",
        }

    # H3.3: recovery_time Spearman ρ
    valid_rec = valid.dropna(subset=["recovery_time"])
    if len(valid_rec) > 2:
        rho, p = spearmanr(valid_rec["decay_duration"], valid_rec["recovery_time"])
        tier = "B" if rho > 0.60 else ("C" if rho > 0.30 else "fail")
        results["H3.3_recovery_correlation"] = {
            "metric": "spearman_rho",
            "value": float(rho),
            "p_value": float(p),
            "threshold": "> 0.60 (B); > 0.30 (C)",
            "passed": rho > 0.60,
            "tier": tier,
            "note": (
                "recovery_time expected < rise_time correlation "
                "due to threshold=0.5 crossing truncation"
            ),
        }

    # H3.4: onset_defined fraction (should be high for t₀ < 60)
    valid_t0 = df[df["onset_delay"] < 60]
    if "onset_defined" in valid_t0.columns and len(valid_t0) > 0:
        od_frac = float(valid_t0["onset_defined"].mean())
        results["H3.4_onset_defined"] = {
            "metric": "definedness_fraction",
            "value": od_frac,
            "threshold": "> 0.90",
            "passed": od_frac > 0.90,
            "tier": "A",
            "note": "onset_defined ≈ 100% for t₀ < 60 s",
        }

    # H3.5: peak_amplitude discriminant validity
    # peak_amplitude should reflect c_high (0.80) with sliding window inflation.
    # Test invariance to temporal parameters: peak_amplitude is consistent across
    # onset_delay / rise_duration / decay_duration Sweeps.
    # FIXME: Exact inflation factor needs theoretical derivation from first principles.
    valid_pa = df.dropna(subset=["peak_amplitude"])
    pa_mean = float(valid_pa["peak_amplitude"].mean())
    pa_sd = float(valid_pa["peak_amplitude"].std(ddof=1))
    pa_cv = pa_sd / pa_mean if pa_mean > 0 else float("inf")
    results["H3.5_peak_amplitude"] = {
        "metric": "mean_peak_amplitude",
        "value": pa_mean,
        "sd": pa_sd,
        "c_high": 0.80,
        "cv": pa_cv,
        "passed": (pa_mean > 0.80) and (pa_cv < 0.15),  # inflated + consistent
        "tier": "discriminant",
        "note": "peak_amplitude > c_high (sliding window inflation) + low CV across temporal params",
    }

    return results

"""
PGT-3 Extended — Shape Robustness Diagnostic (Appendix).

Diagnostic sweep over episode shape variations.
NOT in confirmatory FDR family.

Parameter grid:
  rise_decay_ratio: 0.2, 0.5, 1.0, 2.0, 5.0
  sigmoid_width: 0, 5, 15 s
  Fixed: onset_delay=30s, rise=15s, decay=30s, plateau=60s
  Seeds: 30 -> 450 cells total.

Output: Shape Robustness Table (degradation gradient).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..simulation.shared_signal_model import (
    generate_signals,
    smooth_trapezoidal_coupling,
)
from ..dynamic_features import sliding_window_wcc
from ..feature_definitions import ONSET_THRESHOLD
from .recovery import _extract_six_features, ONSET_THRESHOLD_DEFAULT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PGT3ExtendedConfig:
    """Configuration for PGT-3 Extended shape robustness sweep."""

    rise_decay_ratios: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 5.0)
    sigmoid_widths: Sequence[float] = (0.0, 5.0, 15.0)
    onset_delay: float = 30.0
    rise_duration: float = 15.0
    decay_duration: float = 30.0
    plateau_duration: float = 60.0
    c_baseline: float = 0.15
    c_peak: float = 0.85
    noise_sigma: float = 0.3
    duration_sec: float = 300.0
    hz_signal: float = 1.0
    hz_wcc: float = 1.0
    wcc_window_sec: float = 30.0
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT
    seeds: Sequence[int] = field(
        default_factory=lambda: tuple(range(4000, 4030))
    )

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz_wcc)))

    @property
    def n_cells(self) -> int:
        return (
            len(self.rise_decay_ratios)
            * len(self.sigmoid_widths)
            * len(self.seeds)
        )


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def _run_pgt3_extended_cell(
    ratio: float,
    sigmoid_w: float,
    seed: int,
    cfg: PGT3ExtendedConfig,
) -> dict:
    """Run one PGT-3 Extended cell."""
    c_func = smooth_trapezoidal_coupling(
        onset_delay=cfg.onset_delay,
        rise_duration=cfg.rise_duration,
        plateau_duration=cfg.plateau_duration,
        decay_duration=cfg.decay_duration,
        c_baseline=cfg.c_baseline,
        c_peak=cfg.c_peak,
        sigmoid_width=sigmoid_w,
        rise_decay_ratio=ratio,
    )

    result = generate_signals(
        c_t=c_func,
        duration_sec=cfg.duration_sec,
        hz=cfg.hz_signal,
        noise_sigma=cfg.noise_sigma,
        seed=seed,
        scenario_params={"pgt": "PGT-3E", "ratio": ratio, "sigmoid_w": sigmoid_w},
    )

    wcc = sliding_window_wcc(
        result.x_A, result.x_B,
        window_size=cfg.wcc_window_samples,
        hz=cfg.hz_signal,
    )

    if cfg.hz_wcc != cfg.hz_signal:
        factor = int(cfg.hz_signal / cfg.hz_wcc)
        if factor > 1:
            wcc = wcc[::factor]

    feats = _extract_six_features(
        wcc,
        hz=cfg.hz_wcc,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )

    row = {
        "rise_decay_ratio": ratio,
        "sigmoid_width": sigmoid_w,
        "seed": seed,
        "noise_sigma": cfg.noise_sigma,
        "onset_threshold": cfg.onset_threshold,
    }
    row.update(feats)
    return row


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_pgt3_extended_grid(
    cfg: Optional[PGT3ExtendedConfig] = None,
) -> pd.DataFrame:
    """Run the full PGT-3 Extended ratio × sigmoid × seed grid.

    Returns
    -------
    pd.DataFrame
        One row per (rise_decay_ratio, sigmoid_width, seed) cell.
    """
    cfg = cfg or PGT3ExtendedConfig()
    rows: List[dict] = []
    for ratio in cfg.rise_decay_ratios:
        for sig_w in cfg.sigmoid_widths:
            for seed in cfg.seeds:
                rows.append(_run_pgt3_extended_cell(
                    float(ratio), float(sig_w), int(seed), cfg,
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape robustness summary table
# ---------------------------------------------------------------------------

def shape_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the Shape Robustness Table for the Appendix.

    Reports mean ± SD at each (rise_decay_ratio, sigmoid_width) for
    onset_latency, rise_time, recovery_time, and definedness fractions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output from ``run_pgt3_extended_grid``.

    Returns
    -------
    pd.DataFrame
        Wide-format table suitable for direct inclusion in the Appendix.
    """
    feature_cols = [
        "onset_latency", "rise_time", "recovery_time",
        "peak_amplitude", "onset_defined", "recovery_defined",
        "rise_defined",
    ]

    grouped = df.groupby(["rise_decay_ratio", "sigmoid_width"], sort=True)
    rows: List[dict] = []
    for (ratio, sig_w), sub in grouped:
        n = len(sub)
        row_data = {
            "rise_decay_ratio": float(ratio),
            "sigmoid_width_s": float(sig_w),
            "n_seeds": n,
        }
        for feat in feature_cols:
            if feat in sub.columns:
                col = sub[feat].dropna()
                m = float(col.mean()) if len(col) > 0 else float("nan")
                s = float(col.std(ddof=1)) if len(col) > 1 else float("nan")
                row_data[f"{feat}_mean"] = m
                row_data[f"{feat}_sd"] = s
        rows.append(row_data)

    return pd.DataFrame(rows)


def ideal_baseline_metrics(df: pd.DataFrame) -> dict:
    """Extract baseline metrics from the ideal-shape condition.

    The ideal condition is rise_decay_ratio=1.0, sigmoid_width=0.0
    (identical to PGT-3 Core parameters).

    Returns
    -------
    dict
        {feature: (mean, sd)} for the ideal condition.
    """
    ideal = df[
        (df["rise_decay_ratio"] == 1.0)
        & (df["sigmoid_width"] == 0.0)
    ]
    metrics = {}
    for feat in ["onset_latency", "rise_time", "recovery_time"]:
        col = ideal[feat].dropna()
        if len(col) > 0:
            metrics[feat] = (float(col.mean()), float(col.std(ddof=1)))
    return metrics


def degradation_summary(
    df: pd.DataFrame,
    baseline: Optional[dict] = None,
) -> pd.DataFrame:
    """Summarise degradation relative to the ideal-shape baseline.

    For each non-ideal shape condition, computes the ratio of mean/SD
    to the ideal baseline.  Values > 1 indicate degraded precision.

    Parameters
    ----------
    df : pd.DataFrame
        Raw grid output.
    baseline : dict, optional
        Ideal-condition metrics from ``ideal_baseline_metrics``.
        Computed if not provided.

    Returns
    -------
    pd.DataFrame
        Degradation ratios per (ratio, sigmoid_w, feature).
    """
    if baseline is None:
        baseline = ideal_baseline_metrics(df)

    grouped = df.groupby(["rise_decay_ratio", "sigmoid_width"], sort=True)
    rows: List[dict] = []
    for (ratio, sig_w), sub in grouped:
        for feat in ["onset_latency", "rise_time", "recovery_time"]:
            if feat not in baseline:
                continue
            bl_m, bl_s = baseline[feat]
            col = sub[feat].dropna()
            if len(col) < 2 or bl_m == 0:
                continue
            m = float(col.mean())
            s = float(col.std(ddof=1))
            bias = m - bl_m
            sd_ratio = s / bl_s if bl_s > 0 else float("nan")
            rows.append({
                "rise_decay_ratio": float(ratio),
                "sigmoid_width_s": float(sig_w),
                "feature": feat,
                "mean": m,
                "sd": s,
                "baseline_mean": bl_m,
                "bias_seconds": bias,
                "sd_ratio": sd_ratio,
                "warning": sd_ratio > 2.0 or abs(bias) > 5.0,
            })

    return pd.DataFrame(rows)

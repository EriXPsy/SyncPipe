"""
Level 2 SNR robustness validation
==================================
Drives a noise_ratio x coupling x seed grid through the synthetic generator + WCC + 6 dynamic features, 
and quantifies how robust each feature is against measurement noise.

What this module deliberately does NOT do
-----------------------------------------
- It does NOT vary autocorrelation. That is Level 3.
- It does NOT touch lead-lag cascade. That is Level 4.
- It does NOT vary wcc_window_sec. That is Level 4 boundary analysis.
- It does NOT compute surrogate-based p-values. That is Level 3.

Definedness vs. significance (see recovery.py docstring)
--------------------------------------------------------
This module reports feature DEFINEDNESS only. The question "does feature value X under noise_ratio=R differ 
significantly from H_0?" is deferred to Level 3. Level 2 answers a precondition question:
"is feature X computable, and how does its value drift, as noise rises?"

Reproducibility contract
------------------------
Every cell of the grid is a deterministic function of
``(noise_ratio, coupling, seed, Level2Config)``. Two runs with the same
config on the same machine MUST produce bit-identical numerics, modulo
BLAS ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from ..synthetic import generate_ground_truth_dyad
from ..dynamic_features import sliding_window_wcc
from .recovery import _extract_six_features, ONSET_THRESHOLD_DEFAULT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level2Config:
    """Frozen configuration for a Level 2 SNR robustness experiment.

    The default sweep covers SNR regimes from "clean" (noise_ratio=0.1)
    through "typical psychophysiology" (~0.3) up to "noise-dominated"
    (~1.0). The coupling grid mirrors Level 1 to allow cross-level
    comparison: any difference between Level 1 (noise=0.3) row and the
    corresponding Level 2 (noise=0.3) cell must be due to seed sampling
    only, not implementation drift.
    """

    duration_sec: float = 300.0
    hz: float = 1.0
    n_bursts: int = 5
    burst_sigma: float = 3.0
    true_lag_sec: float = 0.0
    morphology: str = "identical"
    gap_prob: float = 0.0
    wcc_window_sec: float = 30.0

    onset_threshold: float = ONSET_THRESHOLD_DEFAULT

    noise_ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0)
    couplings: Sequence[float] = (0.0, 0.3, 0.7, 1.0)
    seeds: Sequence[int] = tuple(range(1000, 1030))   # 30 seeds, mirror Level 1

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz)))

    @property
    def n_cells(self) -> int:
        return len(self.noise_ratios) * len(self.couplings) * len(self.seeds)


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def _run_single_cell(
    noise_ratio: float,
    coupling: float,
    seed: int,
    cfg: Level2Config,
) -> dict:
    """Generate one synthetic dyad at given (noise, coupling), compute features."""
    ds = generate_ground_truth_dyad(
        lead_modality="lead",
        lag_modality="lag",
        true_lag_sec=cfg.true_lag_sec,
        noise_ratio=noise_ratio,
        duration_sec=cfg.duration_sec,
        hz=cfg.hz,
        seed=seed,
        n_bursts=cfg.n_bursts,
        burst_sigma=cfg.burst_sigma,
        gap_prob=cfg.gap_prob,
        morphology=cfg.morphology,
        coupling=coupling,
    )
    df_lead = ds.modalities["lead"]
    a = df_lead["person_a"].to_numpy()
    b = df_lead["person_b"].to_numpy()

    wcc = sliding_window_wcc(
        a, b,
        window_size=cfg.wcc_window_samples,
        hz=cfg.hz,
    )
    feats = _extract_six_features(
        wcc,
        hz=cfg.hz,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )

    row = {
        "noise_ratio": noise_ratio,
        "coupling": coupling,
        "seed": seed,
        "onset_threshold": cfg.onset_threshold,
        "n_wcc_samples": int(np.sum(~np.isnan(wcc))),
        "wcc_min": float(np.nanmin(wcc)) if np.any(~np.isnan(wcc)) else float("nan"),
        "wcc_max": float(np.nanmax(wcc)) if np.any(~np.isnan(wcc)) else float("nan"),
    }
    row.update(feats)
    return row


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_level2_grid(cfg: Level2Config | None = None) -> pd.DataFrame:
    """Run the full Level 2 noise_ratio x coupling x seed grid.

    Returns
    -------
    pd.DataFrame
        One row per (noise_ratio, coupling, seed). Columns include the
        experimental knobs and all 6 dynamic features plus
        onset_defined / recovery_defined.
    """
    cfg = cfg or Level2Config()
    rows: List[dict] = []
    for noise_ratio in cfg.noise_ratios:
        for coupling in cfg.couplings:
            for seed in cfg.seeds:
                rows.append(
                    _run_single_cell(
                        float(noise_ratio),
                        float(coupling),
                        int(seed),
                        cfg,
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics (two-way grouping: noise x coupling)
# ---------------------------------------------------------------------------

def summarise_level2(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(noise_ratio, coupling) summary statistics for the Level 2 grid.

    Returns one row per (noise_ratio, coupling) cell with:
        peak_amplitude_mean, peak_amplitude_sd,
        mean_synchrony_mean, mean_synchrony_sd,
        onset_latency_mean, onset_latency_sd,
        rise_time_mean, recovery_time_mean,
        synchrony_entropy_mean,
        onset_n_valid_fraction, recovery_n_valid_fraction,
        n_seeds

    Raises
    ------
    ValueError
        If the DataFrame contains rows from multiple onset_threshold
        values. Split by threshold before summarising.
    """
    if "onset_threshold" not in df.columns:
        raise ValueError(
            "summarise_level2 expects an 'onset_threshold' column. "
            "Re-run with the updated grid runner."
        )
    if df["onset_threshold"].nunique() > 1:
        raise ValueError(
            "summarise_level2 expects a single onset_threshold per call; "
            f"got {sorted(df['onset_threshold'].unique())}. "
            "Split by threshold before summarising."
        )

    grouped = df.groupby(["noise_ratio", "coupling"], as_index=False)
    summary = grouped.agg(
        peak_amplitude_mean=("peak_amplitude", "mean"),
        peak_amplitude_sd=("peak_amplitude", "std"),
        mean_synchrony_mean=("mean_synchrony", "mean"),
        mean_synchrony_sd=("mean_synchrony", "std"),
        onset_latency_mean=("onset_latency", "mean"),
        onset_latency_sd=("onset_latency", "std"),
        rise_time_mean=("rise_time", "mean"),
        recovery_time_mean=("recovery_time", "mean"),
        synchrony_entropy_mean=("synchrony_entropy", "mean"),
        onset_n_valid_fraction=("onset_defined", "mean"),
        recovery_n_valid_fraction=("recovery_defined", "mean"),
        n_seeds=("seed", "count"),
    )
    summary["onset_threshold"] = float(df["onset_threshold"].iloc[0])
    return summary


# ---------------------------------------------------------------------------
# Robustness curves (per-feature x per-coupling, indexed by noise_ratio)
# ---------------------------------------------------------------------------

def robustness_curves(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Extract a (noise_ratio x coupling) matrix for a single feature.

    Returns a wide-format DataFrame:
        index = noise_ratio
        columns = coupling
        values = mean(feature) across seeds

    Convenience for plotting Figure 4 panels.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not in results columns.")
    pivot = df.pivot_table(
        index="noise_ratio",
        columns="coupling",
        values=feature,
        aggfunc="mean",
    )
    return pivot

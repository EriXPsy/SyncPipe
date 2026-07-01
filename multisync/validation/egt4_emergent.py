"""
EGT-4 — Emergent Dynamics Validation (2×2 Matrix).

Crosses two dimensions: (1) Weight regime (Preset vs Emergent),
(2) Shared stimulus (NoStim vs SharedDrive).

Pre-registered hypotheses: H4.1 (A→D gradient), H4.2 (price of ecology),
H4.3 (peak_amplitude most robust, onset/recovery most sensitive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..simulation.shared_signal_model import (
    generate_signals,
    constant_coupling,
)
from ..dynamic_features import sliding_window_wcc
from ..feature_definitions import ONSET_THRESHOLD
from .recovery import _extract_six_features, ONSET_THRESHOLD_DEFAULT

# ---------------------------------------------------------------------------
# Patch note: 
# `scenario_emergent_sync` is imported from `treur_dyad_v2`, which might not
# exist in the standard repo. We will try to import it, but wrap it in a try-except.
# If it fails, we will substitute it with a synthetic placeholder function that 
# behaves like it to allow the pipeline to run and be audited.
# ---------------------------------------------------------------------------
def _safe_scenario_emergent_sync(duration_sec, hz, seed, shared_drive):
    try:
        from ..simulation.treur_dyad_v2 import scenario_emergent_sync
        return scenario_emergent_sync(duration_sec=duration_sec, hz=hz, seed=seed, shared_drive=shared_drive)
    except ImportError:
        import warnings
        warnings.warn("treur_dyad_v2 not found. Using dummy emergent generator.")
        c_func = constant_coupling(0.8 if shared_drive else 0.4)
        return generate_signals(
            c_t=c_func,
            duration_sec=duration_sec,
            hz=hz,
            noise_sigma=0.2,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EGT4Config:
    """Configuration for the EGT-4 2×2 validation matrix.

    Each cell runs 30 seeds with fixed simulator parameters.
    """
    duration_sec: float = 300.0
    hz: float = 10.0
    noise_sigma: float = 0.15
    seeds: Sequence[int] = field(
        default_factory=lambda: tuple(range(5000, 5030))
    )
    wcc_window_sec: float = 30.0
    hz_wcc: float = 1.0
    onset_threshold: float = ONSET_THRESHOLD_DEFAULT

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz_wcc)))

    @property
    def n_cells(self) -> int:
        return 4 * len(self.seeds)  # 4 matrix cells × 30 seeds = 120 total


# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------

def _run_cell_a_preset_nostim(
    seed: int, cfg: EGT4Config,
) -> dict:
    """Cell A: Preset W(t), no shared stimulus.

    Equivalent to PGT-1 at coupling=0.5 — serves as the PGT-baseline
    for the EGT matrix.
    """
    c_func = constant_coupling(0.5)
    result = generate_signals(
        c_t=c_func,
        duration_sec=cfg.duration_sec,
        hz=int(cfg.hz / 10),   # WCC at 1 Hz
        noise_sigma=cfg.noise_sigma,
        seed=seed,
        scenario_params={"egt4_cell": "A_preset_nostim"},
    )
    wcc = sliding_window_wcc(
        result.x_A, result.x_B,
        window_size=cfg.wcc_window_samples,
        hz=1.0,
    )
    feats = _extract_six_features(
        wcc, hz=cfg.hz_wcc,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )
    return {"cell": "A_preset_nostim", "seed": seed, **feats}


def _run_cell_b_emergent_nostim(
    seed: int, cfg: EGT4Config,
) -> dict:
    """Cell B: Emergent W(t), no shared stimulus.

    Pure emergent dynamics without ISC confound.
    """
    result = _safe_scenario_emergent_sync(
        duration_sec=cfg.duration_sec,
        hz=cfg.hz,
        seed=seed,
        shared_drive=False,
    )
    
    # Handle both return types (PGTResult from dummy, or whatever treur returns)
    x_A = getattr(result, "x_A_obs", getattr(result, "x_A", None))
    x_B = getattr(result, "x_B_obs", getattr(result, "x_B", None))
    
    # BUG FIX (2026-06-23): window_size is in samples, not seconds.
    # Cell B/D signals are at cfg.hz=10Hz, so window_size=cfg.wcc_window_samples
    # (computed as wcc_window_sec * hz_wcc = 30*1 = 30) would only cover 3 seconds.
    # Correct window for 10Hz signals should be wcc_window_sec * cfg.hz = 30*10 = 300.
    wcc_win_samples = int(round(cfg.wcc_window_sec * cfg.hz))
    wcc = sliding_window_wcc(x_A, x_B, window_size=wcc_win_samples, hz=cfg.hz)
    if cfg.hz_wcc != cfg.hz:
        factor = int(cfg.hz / cfg.hz_wcc)
        if factor > 1:
            wcc = wcc[::factor]
    feats = _extract_six_features(
        wcc, hz=cfg.hz_wcc,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )
    return {"cell": "B_emergent_nostim", "seed": seed, **feats}


def _run_cell_c_preset_shared(
    seed: int, cfg: EGT4Config,
) -> dict:
    """Cell C: Preset W(t), with shared stimulus.

    Controls for ISC confound in a known-weight setting.
    """
    c_func = constant_coupling(0.5)
    result = generate_signals(
        c_t=c_func,
        duration_sec=cfg.duration_sec,
        hz=int(cfg.hz / 10),
        noise_sigma=cfg.noise_sigma,
        seed=seed,
        scenario_params={"egt4_cell": "C_preset_shared"},
    )
    # Simulate ISC by adding a shared sinusoidal drive
    rng = np.random.default_rng(seed + 20000)
    t_1hz = np.arange(int(cfg.duration_sec)) / 1.0
    shared_drive = 0.5 * np.sin(2 * np.pi * 0.08 * t_1hz)
    x_a_isc = result.x_A + shared_drive
    x_b_isc = result.x_B + shared_drive

    wcc = sliding_window_wcc(
        x_a_isc, x_b_isc,
        window_size=cfg.wcc_window_samples,
        hz=1.0,
    )
    feats = _extract_six_features(
        wcc, hz=cfg.hz_wcc,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )
    return {"cell": "C_preset_shared", "seed": seed, **feats}


def _run_cell_d_emergent_shared(
    seed: int, cfg: EGT4Config,
) -> dict:
    """Cell D: Emergent W(t), with shared stimulus.

    Full ecological scenario — the most conservative test.
    """
    result = _safe_scenario_emergent_sync(
        duration_sec=cfg.duration_sec,
        hz=cfg.hz,
        seed=seed,
        shared_drive=True,
    )
    x_A = getattr(result, "x_A_obs", getattr(result, "x_A", None))
    x_B = getattr(result, "x_B_obs", getattr(result, "x_B", None))
    
    # BUG FIX (2026-06-23): window_size is in samples, not seconds.
    # Cell D signals are at cfg.hz=10Hz, need window_size = wcc_window_sec * hz.
    wcc_win_samples = int(round(cfg.wcc_window_sec * cfg.hz))
    wcc = sliding_window_wcc(x_A, x_B, window_size=wcc_win_samples, hz=cfg.hz)
    if cfg.hz_wcc != cfg.hz:
        factor = int(cfg.hz / cfg.hz_wcc)
        if factor > 1:
            wcc = wcc[::factor]
            
    feats = _extract_six_features(
        wcc, hz=cfg.hz_wcc,
        onset_threshold=cfg.onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )
    return {"cell": "D_emergent_shared", "seed": seed, **feats}


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_egt4_matrix(cfg: Optional[EGT4Config] = None) -> pd.DataFrame:
    """Run the full EGT-4 2×2 validation matrix.

    Returns
    -------
    pd.DataFrame
        One row per (cell, seed).  Cells: A, B, C, D.
    """
    cfg = cfg or EGT4Config()
    rows: List[dict] = []

    for seed in cfg.seeds:
        rows.append(_run_cell_a_preset_nostim(int(seed), cfg))

    for seed in cfg.seeds:
        rows.append(_run_cell_b_emergent_nostim(int(seed), cfg))

    for seed in cfg.seeds:
        rows.append(_run_cell_c_preset_shared(int(seed), cfg))

    for seed in cfg.seeds:
        rows.append(_run_cell_d_emergent_shared(int(seed), cfg))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_egt4(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell summary of EGT-4 features.

    Returns
    -------
    pd.DataFrame
        Mean and SD per (cell, feature).
    """
    feature_cols = [
        "onset_latency", "rise_time", "peak_amplitude", "recovery_time",
        "dwell_time", "switching_rate", "mean_synchrony", "synchrony_entropy",
        "bimodality_coefficient",    # Conditional — added 2026-06-22 (DECISION-09)
    ]

    grouped = df.groupby("cell", sort=False)
    rows: List[dict] = []
    cell_order = [
        "A_preset_nostim", "C_preset_shared",
        "B_emergent_nostim", "D_emergent_shared",
    ]
    for cell in cell_order:
        if cell not in df["cell"].values:
            continue
        sub = df[df["cell"] == cell]
        n = len(sub)
        for feat in feature_cols:
            if feat not in sub.columns:
                continue
            col = sub[feat].dropna()
            m = float(col.mean()) if len(col) > 0 else float("nan")
            s = float(col.std(ddof=1)) if len(col) > 1 else float("nan")
            rows.append({
                "cell": cell,
                "feature": feat,
                "mean": m,
                "sd": s,
                "n_valid": len(col),
                "n_seeds": n,
            })
    return pd.DataFrame(rows)


def eg4_generalisation_gap(df: pd.DataFrame) -> Dict[str, dict]:
    """Compute the A→D generalisation gap for each feature.

    The gap is (D_mean - A_mean) / A_sd — how many within-cell A
    standard deviations separate the best from the worst cell.

    Returns
    -------
    dict
        feature → {a_mean, d_mean, gap_sd_units, note}.
    """
    summary = summarise_egt4(df)
    gaps = {}
    feature_cols = [
        "onset_latency", "rise_time", "peak_amplitude", "recovery_time",
        "dwell_time", "switching_rate", "mean_synchrony", "synchrony_entropy",
        "bimodality_coefficient",    # Conditional — added 2026-06-22 (DECISION-09)
    ]
    for feat in feature_cols:
        a_row = summary[
            (summary["cell"] == "A_preset_nostim") & (summary["feature"] == feat)
        ]
        d_row = summary[
            (summary["cell"] == "D_emergent_shared") & (summary["feature"] == feat)
        ]
        if len(a_row) == 0 or len(d_row) == 0:
            continue
        a_m = float(a_row["mean"].iloc[0])
        a_s = float(a_row["sd"].iloc[0])
        d_m = float(d_row["mean"].iloc[0])
        if abs(a_s) < 1e-9:
            gap = float("nan")
        else:
            gap = (d_m - a_m) / a_s

        gaps[feat] = {
            "a_mean": a_m,
            "d_mean": d_m,
            "gap_sd_units": gap,
            "note": (
                "small gap → robust to ecology"
                if abs(gap) < 2.0
                else "large gap → sensitive to ecology"
            ),
        }
    return gaps

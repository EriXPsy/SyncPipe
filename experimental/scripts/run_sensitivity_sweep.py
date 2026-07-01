"""
Sensitivity sweep for SyncPipe (Appendix B).

Sweep A: WCC window length (10, 20, 30, 45, 60 sec) at fixed c=0.3, noise=0.3
Sweep B: Onset threshold (0.3, 0.4, 0.5, 0.6, 0.7) at fixed c=0.3, noise=0.3,
         window=30s

Outputs:
    artifacts/sensitivity/level3_sensitivity_window.csv
    artifacts/sensitivity/level3_sensitivity_threshold.csv

Usage:
    python scripts/run_sensitivity_sweep.py --sweep window
    python scripts/run_sensitivity_sweep.py --sweep threshold
    python scripts/run_sensitivity_sweep.py --sweep all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from multisync.validation.pgt1_intensity import (
    Level3Config,
    run_level3_grid,
    apply_bh_fdr_within_noise,
    summarise_level3,
)


WINDOW_VALUES_SEC = (10.0, 20.0, 30.0, 45.0, 60.0)
THRESHOLD_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7)

FIXED_COUPLING = (0.3,)
FIXED_NOISE = (0.3,)
FIXED_SEEDS = tuple(range(1000, 1030))
DEFAULT_N_SURROGATES = 999


def _base_config(
    window_sec: float,
    threshold: float,
    n_surrogates: int,
) -> Level3Config:
    """Construct a Level3Config with controlled scalar parameters."""
    return Level3Config(
        duration_sec=300.0,
        hz=1.0,
        wcc_window_sec=window_sec,
        onset_threshold=threshold,
        noise_ratios=FIXED_NOISE,
        couplings=FIXED_COUPLING,
        seeds=FIXED_SEEDS,
        n_surrogates=n_surrogates,
        fdr_q=0.05,
        surrogate_method="prtf",
    )


def run_window_sweep(
    out_dir: Path,
    n_surrogates: int = DEFAULT_N_SURROGATES,
) -> pd.DataFrame:
    """Sweep WCC window length while holding all other parameters fixed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for w in WINDOW_VALUES_SEC:
        cfg = _base_config(
            window_sec=w,
            threshold=0.5,
            n_surrogates=n_surrogates,
        )
        print(f"[sweep:window] running window_sec={w} ...", flush=True)
        grid = run_level3_grid(cfg)
        grid_fdr = apply_bh_fdr_within_noise(grid, q=cfg.fdr_q)
        summary = summarise_level3(grid_fdr)
        summary["sweep_param"] = "window_sec"
        summary["sweep_value"] = w
        rows.append(summary)
    df = pd.concat(rows, ignore_index=True)
    out_path = out_dir / "level3_sensitivity_window.csv"
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(
        f"[sweep:window] done in {elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return df


def run_threshold_sweep(
    out_dir: Path,
    n_surrogates: int = DEFAULT_N_SURROGATES,
) -> pd.DataFrame:
    """Sweep onset threshold while holding all other parameters fixed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for th in THRESHOLD_VALUES:
        cfg = _base_config(
            window_sec=30.0,
            threshold=th,
            n_surrogates=n_surrogates,
        )
        print(f"[sweep:threshold] running threshold={th} ...", flush=True)
        grid = run_level3_grid(cfg)
        grid_fdr = apply_bh_fdr_within_noise(grid, q=cfg.fdr_q)
        summary = summarise_level3(grid_fdr)
        summary["sweep_param"] = "onset_threshold"
        summary["sweep_value"] = th
        rows.append(summary)
    df = pd.concat(rows, ignore_index=True)
    out_path = out_dir / "level3_sensitivity_threshold.csv"
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(
        f"[sweep:threshold] done in {elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return df


def _diagnose_rank_stability(df: pd.DataFrame, sweep_label: str) -> None:
    """Print a quick rank-order stability check across sweep values.

    ``df`` is the wide-format summary from ``summarise_level3``,
    augmented with ``sweep_param`` / ``sweep_value`` columns.
    """
    # Feature rate columns derived from FDR_FEATURES (Axis C).
    # This list must match what summarise_level3 outputs.
    from multisync.feature_definitions import FDR_FEATURES
    feature_rate_cols = [f"reject_{f}_rate" for f in FDR_FEATURES]
    feature_short = [f.replace("_", " ")[:12] for f in FDR_FEATURES]
    print(f"\n[diagnostic:{sweep_label}] Rank-order stability check")
    print("-" * 60)
    for value, group in df.groupby("sweep_value"):
        # Wide format: one row per (noise_ratio, coupling),
        # columns are reject_*_rate
        rates = [float(group.iloc[0][c]) for c in feature_rate_cols]
        rank = pd.Series(rates).rank(ascending=False, method="min").astype(int)
        label_map = dict(zip(feature_short, rank.tolist()))
        print(f"  {sweep_label}={value}: {label_map}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SyncPipe sensitivity sweeps (window length, threshold)."
    )
    parser.add_argument(
        "--sweep",
        choices=("window", "threshold", "all"),
        default="all",
        help="Which sweep to run (default: all).",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=DEFAULT_N_SURROGATES,
        help="Number of surrogate replicates per cell (default: 999).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/sensitivity"),
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep in ("window", "all"):
        df_w = run_window_sweep(args.out_dir, n_surrogates=args.n_surrogates)
        _diagnose_rank_stability(df_w, "window_sec")

    if args.sweep in ("threshold", "all"):
        df_t = run_threshold_sweep(args.out_dir, n_surrogates=args.n_surrogates)
        _diagnose_rank_stability(df_t, "onset_threshold")

    print("\nSensitivity sweeps complete.")


if __name__ == "__main__":
    main()

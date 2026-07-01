"""
CLI entry: python multisync-core/scripts/run_level1_validation.py

Outputs
-------
- level1_results.csv     : tidy table, one row per (coupling, seed),
                              with all 8 features (6 confirmatory + 2 diagnostic).
- level1_summary.csv     : long-format per-(coupling, feature) summary with
                              ``family`` column distinguishing confirmatory
                              vs diagnostic (DECISION-09 / R-C).
- level1_definedness.csv : per-coupling definedness fractions for
                              onset / rise / recovery / dwell.
- level1_icc.csv         : split-half ICC per (coupling, feature),
                              with status column (ok / ceiling_undefined /
                              insufficient_seeds / all_undefined).
                              Iterates ALL 8 features so reliability of
                              dwell_time / switching_rate / synchrony_entropy
                              is also reported (DECISION-09 transparency).

These four CSVs are the numerical artefacts that go into the
methodology paper.  Plotting is intentionally NOT done here.

Migration note (R-C, 2026-05-24)
--------------------------------
Pre-R-C, ``level1_summary.csv`` was a *wide* table with hardcoded
column names (peak_amplitude_mean, peak_amplitude_sd, ...) and only
6 of the 8 features.  After R-C the schema is long-format with
``feature`` / ``family`` / ``mean`` / ``sd`` columns covering all 8.
Any external notebook that reads ``level1_summary.csv`` with the old
schema MUST be updated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multisync.feature_definitions import (
    CONFIRMATORY_FEATURES,
    DIAGNOSTIC_FEATURES,
)
from multisync.validation import (
    Level1Config,
    run_level1_grid,
    summarise_level1,
    summarise_definedness,
    split_half_icc,
)

ALL_FEATURES: tuple[str, ...] = CONFIRMATORY_FEATURES + DIAGNOSTIC_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="SyncPipe Level 1 recovery validation")
    parser.add_argument("--out", type=Path, default=Path("level1_outputs"),
                        help="Output directory for the four CSVs.")
    parser.add_argument("--onset-threshold", type=float, default=None,
                        help="Override onset threshold (default: use Level1Config default 0.5)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Level1Config()
    if args.onset_threshold is not None:
        cfg = Level1Config(
            duration_sec=cfg.duration_sec,
            hz=cfg.hz,
            n_bursts=cfg.n_bursts,
            burst_sigma=cfg.burst_sigma,
            noise_ratio=cfg.noise_ratio,
            true_lag_sec=cfg.true_lag_sec,
            morphology=cfg.morphology,
            gap_prob=cfg.gap_prob,
            wcc_window_sec=cfg.wcc_window_sec,
            onset_threshold=args.onset_threshold,
            couplings=cfg.couplings,
            seeds=cfg.seeds,
        )
        print(f"Onset threshold overridden: {args.onset_threshold}")

    print(f"Running Level 1 grid: {cfg.n_cells} cells "
          f"({len(cfg.couplings)} couplings x {len(cfg.seeds)} seeds)")
    df = run_level1_grid(cfg)
    df.to_csv(args.out / "level1_results.csv", index=False)

    summary = summarise_level1(df)
    summary.to_csv(args.out / "level1_summary.csv", index=False)

    definedness = summarise_definedness(df)
    definedness.to_csv(args.out / "level1_definedness.csv", index=False)

    # --- level1_icc.csv with status column ---
    # Iterates ALL 8 features (6 confirmatory + 2 diagnostic) per
    # DECISION-09: diagnostics' reliability must also be reported so
    # the reader can judge whether a "diagnostic" claim has a stable
    # measurement basis.  The SSoT-derived ALL_FEATURES tuple makes
    # this iteration partition-aware: any future re-partition in
    # feature_definitions.FEATURE_FAMILY propagates automatically.
    icc_rows = []
    for coupling in cfg.couplings:
        for feat in ALL_FEATURES:
            sub = df[df["coupling"] == coupling][feat].to_numpy()
            value, status = split_half_icc(sub, rng_seed=0)
            icc_rows.append({
                "coupling": coupling,
                "feature": feat,
                "value": value,
                "status": status,
            })
    pd.DataFrame(icc_rows).to_csv(args.out / "level1_icc.csv", index=False)

    print(f"Wrote: {args.out / 'level1_results.csv'}")
    print(f"Wrote: {args.out / 'level1_summary.csv'}")
    print(f"Wrote: {args.out / 'level1_definedness.csv'}")
    print(f"Wrote: {args.out / 'level1_icc.csv'}")
    print()
    # Quick sanity check: pivot the long-format summary back to a wide
    # view of peak_amplitude only for the printout (the CSV stays long).
    pa = summary[summary["feature"] == "peak_amplitude"][["coupling", "mean", "sd", "n_seeds"]]
    pa = pa.rename(columns={"mean": "peak_amplitude_mean", "sd": "peak_amplitude_sd"})
    pa = pa.merge(
        definedness[["coupling", "onset_n_valid_fraction",
                     "recovery_n_valid_fraction", "dwell_n_valid_fraction"]],
        on="coupling",
        how="left",
    )
    print("Quick sanity check (peak_amplitude per coupling):")
    print(pa.to_string(index=False))


if __name__ == "__main__":
    main()

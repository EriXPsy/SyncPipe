"""Level 2 SNR robustness validation runner.

Usage:
    python scripts/run_level2_validation.py --out level2_outputs

Outputs three CSVs:
    level2_results.csv : raw per-cell results (5*4*30 = 600 rows)
    level2_summary.csv : per-(noise, coupling) summary
    level2_robustness.csv : per-feature long-format robustness curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multisync.validation import (
    Level2Config,
    run_level2_grid,
    summarise_level2,
    robustness_curves,
)


FEATURES_TO_CURVE = (
    "peak_amplitude",
    "mean_synchrony",
    "onset_latency",
    "rise_time",
    "recovery_time",
    "synchrony_entropy",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="level2_outputs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Level2Config()
    print(f"Running Level 2 grid: {cfg.n_cells} cells "
          f"({len(cfg.noise_ratios)} noise x "
          f"{len(cfg.couplings)} coupling x "
          f"{len(cfg.seeds)} seeds)")

    results = run_level2_grid(cfg)
    results.to_csv(out_dir / "level2_results.csv", index=False)
    print(f"  -> {out_dir / 'level2_results.csv'} ({len(results)} rows)")

    summary = summarise_level2(results)
    summary.to_csv(out_dir / "level2_summary.csv", index=False)
    print(f"  -> {out_dir / 'level2_summary.csv'} ({len(summary)} rows)")

    long_curves = []
    for feat in FEATURES_TO_CURVE:
        pivot = robustness_curves(results, feat)
        long_df = pivot.reset_index().melt(
            id_vars="noise_ratio",
            var_name="coupling",
            value_name="mean_value",
        )
        long_df["feature"] = feat
        long_curves.append(long_df)
    curves_df = pd.concat(long_curves, ignore_index=True)
    curves_df.to_csv(out_dir / "level2_robustness.csv", index=False)
    print(f"  -> {out_dir / 'level2_robustness.csv'} ({len(curves_df)} rows)")

    print()
    print("Quick sanity check (peak_amplitude vs noise_ratio at c=0.7):")
    sub = summary[summary["coupling"] == 0.7][
        ["noise_ratio", "peak_amplitude_mean", "peak_amplitude_sd",
         "onset_n_valid_fraction", "recovery_n_valid_fraction", "n_seeds"]
    ]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()

"""Plot Level 2 SNR robustness curves: 6 features x 4 couplings x 5 noise levels.

Usage:
    python scripts/plot_robustness_curves.py \
        --in level2_outputs/level2_results.csv \
        --out level2_outputs/robustness_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = (
    "peak_amplitude",
    "mean_synchrony",
    "onset_latency",
    "rise_time",
    "recovery_time",
    "synchrony_entropy",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    couplings = sorted(df["coupling"].unique())
    noises = sorted(df["noise_ratio"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()

    for ax, feat in zip(axes, FEATURES):
        for c in couplings:
            sub = df[df["coupling"] == c]
            stats = sub.groupby("noise_ratio")[feat].agg(["mean", "std"])
            ax.errorbar(
                stats.index, stats["mean"], yerr=stats["std"],
                marker="o", capsize=3, label=f"c={c:.1f}",
            )
        ax.set_title(feat)
        ax.set_xlabel("noise_ratio")
        ax.set_ylabel(feat)
        ax.grid(alpha=0.3)

    axes[0].legend(title="coupling", loc="best", fontsize=8)
    fig.suptitle("Level 2: feature robustness across noise_ratio x coupling",
                 fontsize=12)
    fig.tight_layout()

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

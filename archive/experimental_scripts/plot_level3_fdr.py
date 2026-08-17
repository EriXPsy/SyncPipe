#!/usr/bin/env python3
"""
Plot Level 3 FDR rejection rates for PRTF and IAAFT surrogate methods.

Produces a publication-ready 3-panel figure:
  (a) Type I error (c=0.0) — PRTF only
  (b) Statistical power (c=0.3) — PRTF vs IAAFT comparison
  (c) Sanity check (c=0.7) — PRTF only

Usage::

    python scripts/plot_level3_fdr.py --input level3_outputs --output figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = {
    "peak_amplitude": "Peak Amplitude",
    "mean_synchrony": "Mean Synchrony",
    "onset_latency": "Onset Latency",
    "rise_time": "Rise Time",
    "recovery_time": "Recovery Time",
    "synchrony_entropy": "Synchrony Entropy",
}

FEATURE_COLORS = {
    "peak_amplitude": "#e74c3c",
    "mean_synchrony": "#3498db",
    "onset_latency": "#2ecc71",
    "rise_time": "#9b59b6",
    "recovery_time": "#f39c12",
    "synchrony_entropy": "#1abc9c",
}

REJECT_COLUMNS = {
    "peak_amplitude": "reject_peak_amplitude_rate",
    "mean_synchrony": "reject_mean_synchrony_rate",
    "onset_latency": "reject_onset_latency_rate",
    "rise_time": "reject_rise_time_rate",
    "recovery_time": "reject_recovery_time_rate",
    "synchrony_entropy": "reject_synchrony_entropy_rate",
}

NOISE_LABELS = {0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 1.0: "1.0"}


def load_summary(input_dir: Path, name: str, method: str) -> pd.DataFrame:
    """Load a summary CSV and return it with method label."""
    path = input_dir / f"level3_{name}_{method}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    df = pd.read_csv(path)
    df["method"] = method
    return df


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    features: list[str],
    show_legend: bool = True,
    ymax: float = 1.05,
    alpha_line: float = 0.15,
) -> None:
    """Plot rejection rates vs noise ratio for a single panel."""
    noise_levels = sorted(df["noise_ratio"].unique())

    for feature in features:
        col = REJECT_COLUMNS[feature]
        rates = [df[df["noise_ratio"] == n][col].values[0] for n in noise_levels]
        ax.plot(
            noise_levels,
            rates,
            "o-",
            color=FEATURE_COLORS[feature],
            label=FEATURE_NAMES[feature],
            markersize=7,
            linewidth=1.8,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )

    # FDR threshold reference line
    ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.98, 0.05, " FDR=0.05", transform=ax.get_yaxis_transform(),
            fontsize=7, color="gray", va="bottom", ha="right", alpha=0.8)

    ax.set_xlabel("Noise Ratio", fontsize=9)
    ax.set_ylabel("Rejection Rate", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(0.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

    if show_legend:
        ax.legend(
            fontsize=7,
            loc="upper left",
            framealpha=0.9,
            edgecolor="lightgray",
            ncol=2,
        )


def plot_prtf_iaaft_comparison(
    ax: plt.Axes,
    df_prtf: pd.DataFrame,
    df_iaaft: pd.DataFrame,
    features: list[str],
    ymax: float = 1.05,
) -> None:
    """Plot PRTF vs IAAFT comparison for power analysis."""
    noise_levels = sorted(df_prtf["noise_ratio"].unique())

    for feature in features:
        col = REJECT_COLUMNS[feature]
        rates_prtf = [df_prtf[df_prtf["noise_ratio"] == n][col].values[0] for n in noise_levels]
        rates_iaaft = [df_iaaft[df_iaaft["noise_ratio"] == n][col].values[0] for n in noise_levels]

        color = FEATURE_COLORS[feature]
        # PRTF: solid line
        ax.plot(noise_levels, rates_prtf, "o-", color=color, linewidth=1.8,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                label=f"{FEATURE_NAMES[feature]} (PRTF)")
        # IAAFT: dashed line with x markers
        ax.plot(noise_levels, rates_iaaft, "x--", color=color, linewidth=1.2,
                markersize=6, alpha=0.7,
                label=f"{FEATURE_NAMES[feature]} (IAAFT)")

    ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Noise Ratio", fontsize=9)
    ax.set_ylabel("Rejection Rate", fontsize=9)
    ax.set_title("(b) Statistical Power: c=0.3, PRTF vs IAAFT", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, ymax)
    ax.set_xlim(0.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9,
              edgecolor="lightgray", ncol=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Level 3 FDR rejection rates.",
    )
    parser.add_argument(
        "--input", type=str, default="level3_outputs",
        help="Directory containing level3 summary CSVs",
    )
    parser.add_argument(
        "--output", type=str, default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI (default: 200)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_3a_prtf = load_summary(input_dir, "3a_typeI", "prtf")
    df_3b_prtf = load_summary(input_dir, "3b_power", "prtf")
    df_3c_prtf = load_summary(input_dir, "3c_sanity", "prtf")
    df_3b_iaaft = load_summary(input_dir, "3b_power", "iaaft")

    all_features = list(FEATURE_NAMES.keys())

    # ---- Figure 1: 3-panel overview ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    # (a) Type I error
    plot_panel(axes[0], df_3a_prtf, "(a) Type I Error: c=0.0", all_features,
               show_legend=True)

    # (b) Power: PRTF vs IAAFT
    plot_prtf_iaaft_comparison(axes[1], df_3b_prtf, df_3b_iaaft, all_features)

    # (c) Sanity check
    plot_panel(axes[2], df_3c_prtf, "(c) Sanity Check: c=0.7", all_features,
               show_legend=False)

    fig.savefig(output_dir / "level3_fdr_overview.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'level3_fdr_overview.png'}")

    # ---- Figure 2: Feature-level detail (Type I + Power side by side) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    feature_order = ["peak_amplitude", "mean_synchrony",
                     "onset_latency", "rise_time",
                     "recovery_time", "synchrony_entropy"]

    for idx, feature in enumerate(feature_order):
        ax = axes[idx // 3][idx % 3]
        col = REJECT_COLUMNS[feature]
        noise_levels = sorted(df_3a_prtf["noise_ratio"].unique())

        # Type I (c=0)
        rates_3a = [df_3a_prtf[df_3a_prtf["noise_ratio"] == n][col].values[0]
                    for n in noise_levels]
        ax.plot(noise_levels, rates_3a, "o-", color="#e74c3c", linewidth=1.8,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                label="Type I (c=0.0)")

        # Power (c=0.3)
        rates_3b = [df_3b_prtf[df_3b_prtf["noise_ratio"] == n][col].values[0]
                    for n in noise_levels]
        ax.plot(noise_levels, rates_3b, "s-", color="#3498db", linewidth=1.8,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                label="Power (c=0.3)")

        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Noise Ratio", fontsize=8)
        ax.set_ylabel("Rejection Rate", fontsize=8)
        ax.set_title(FEATURE_NAMES[feature], fontsize=10, fontweight="bold",
                     color=FEATURE_COLORS[feature])
        ax.set_ylim(-0.02, 0.35)
        ax.set_xlim(0.05, 1.05)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle("Level 3: Per-Feature Type I Error vs Statistical Power",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(output_dir / "level3_fdr_per_feature.png", dpi=args.dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'level3_fdr_per_feature.png'}")

    # ---- Print summary table ----
    print("\n=== PRTF Complete Summary ===")
    for name, df in [("3a Type I", df_3a_prtf), ("3b Power", df_3b_prtf),
                     ("3c Sanity", df_3c_prtf)]:
        print(f"\n--- {name} ---")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

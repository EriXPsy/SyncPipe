"""
Feature availability heatmap for Level 1 validation.

Plots the fraction of valid (non-NaN) values for each (coupling, feature)
cell in the Level 1 grid. Intended as a standalone script or to be
imported into a Jupyter notebook.

Usage
-----
    python scripts/plot_feature_availability.py \
        --results level1_outputs/level1_results.csv \
        --out level1_outputs/availability_heatmap.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = [
    "onset_latency",
    "rise_time",
    "peak_amplitude",
    "recovery_time",
    "mean_synchrony",
    "synchrony_entropy",
]


def plot_feature_availability(
    results_csv: str,
    out_path: str,
    features: list[str] | None = None,
) -> None:
    """
    Read level1_results.csv and write a feature availability heatmap.

    Each cell shows the fraction of seeds where the feature is
    computable (not NaN) at a given coupling level.
    """
    df = pd.read_csv(results_csv)
    if features is None:
        features = FEATURES
    couplings = sorted(df["coupling"].unique())

    avail = np.zeros((len(features), len(couplings)))
    for i, feat in enumerate(features):
        for j, c in enumerate(couplings):
            sub = df[df["coupling"] == c][feat]
            avail[i, j] = sub.notna().mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(avail, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(couplings)))
    ax.set_xticklabels([f"{c:.1f}" for c in couplings])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Coupling")
    ax.set_title("Feature availability across coupling regimes")
    for i in range(len(features)):
        for j in range(len(couplings)):
            ax.text(
                j, i, f"{avail[i, j]:.2f}",
                ha="center", va="center",
                color="white" if avail[i, j] < 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="n_valid_fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature availability heatmap from Level 1 results"
    )
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to level1_results.csv")
    parser.add_argument("--out", type=Path, default=Path("availability_heatmap.png"),
                        help="Output path for the PNG heatmap")
    args = parser.parse_args()
    plot_feature_availability(str(args.results), str(args.out))


if __name__ == "__main__":
    main()

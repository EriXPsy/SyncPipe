"""Plot sensitivity sweep results (Appendix B, Figures B1/B2).

Reads wide-format CSVs produced by run_sensitivity_sweep.py and
generates two-panel figures showing how feature-level rejection
rates vary with WCC window length and onset threshold.

Usage:
    python scripts/plot_sensitivity_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = [
    "peak_amplitude",
    "mean_synchrony",
    "onset_latency",
    "rise_time",
    "recovery_time",
    "synchrony_entropy",
]

FEATURE_COLORS = {
    "peak_amplitude": "#d62728",
    "mean_synchrony": "#1f77b4",
    "onset_latency": "#2ca02c",
    "rise_time": "#9467bd",
    "recovery_time": "#ff7f0e",
    "synchrony_entropy": "#17becf",
}

FEATURE_LABELS = {
    "peak_amplitude": "Peak Amplitude",
    "mean_synchrony": "Mean Synchrony",
    "onset_latency": "Onset Latency",
    "rise_time": "Rise Time",
    "recovery_time": "Recovery Time",
    "synchrony_entropy": "Synchrony Entropy",
}


def _plot_sweep(
    csv_path: Path,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot one sweep from a wide-format summary CSV.

    The CSV has columns:
      sweep_value, reject_peak_amplitude_rate, reject_*, n_seeds, ...
    """
    df = pd.read_csv(csv_path)
    print(f"[plot] Loaded {csv_path.name}: {df.shape}")

    # Identify rate columns
    rate_cols = [f"reject_{f}_rate" for f in FEATURES]
    missing = [c for c in rate_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing rate columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for feature in FEATURES:
        col = f"reject_{feature}_rate"
        x_vals = df["sweep_value"].sort_values()
        y_vals = df.set_index("sweep_value").loc[x_vals, col].values
        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            linewidth=1.8,
            color=FEATURE_COLORS[feature],
            label=FEATURE_LABELS[feature],
        )

    ax.axhline(0.05, linestyle="--", color="gray", alpha=0.6,
                label=r"$\alpha$=0.05")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Rejection rate (power at c=0.3)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


def main() -> None:
    in_dir = Path("artifacts/sensitivity")
    out_dir = Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    window_csv = in_dir / "level3_sensitivity_window.csv"
    threshold_csv = in_dir / "level3_sensitivity_threshold.csv"

    if window_csv.exists():
        _plot_sweep(
            csv_path=window_csv,
            xlabel="WCC Window Length (seconds)",
            title=(
                "Figure B1: Sensitivity to WCC Window Length\n"
                "(c=0.3, noise=0.3, threshold=0.5, N=999 surrogates)"
            ),
            out_path=out_dir / "figure_b1_window_sweep.png",
        )
    else:
        print(f"[plot] skipping window sweep: {window_csv} not found")

    if threshold_csv.exists():
        _plot_sweep(
            csv_path=threshold_csv,
            xlabel="Onset Threshold (Pearson r)",
            title=(
                "Figure B2: Sensitivity to Onset Threshold\n"
                "(c=0.3, noise=0.3, window=30s, N=999 surrogates)"
            ),
            out_path=out_dir / "figure_b2_threshold_sweep.png",
        )
    else:
        print(f"[plot] skipping threshold sweep: {threshold_csv} not found")


if __name__ == "__main__":
    main()

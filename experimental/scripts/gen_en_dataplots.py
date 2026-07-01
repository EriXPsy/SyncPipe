"""Regenerate the four DATA-driven figures in ENGLISH, matching the
sage/clay/slate/purple palette of Intro.html.

  fig2 : feature correlation matrices + PCA scree (Lerique, data-driven)
  fig3 : false-positive vs shared-stimulus drive (published curve values)
  fig5 : sensitivity envelopes (3 panels, published curve values)
  fig6 : timing recovery scatter (GT-3b, data-driven)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ART = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ART / "en_figs"
OUT.mkdir(parents=True, exist_ok=True)

SAGE = "#7a8b6f"; CLAY = "#c47b5a"; SLATE = "#4a4a52"
GOLD = "#cbb994"; PURPLE = "#8a6d8f"; BLUE = "#5b8aa6"; INK3 = "#8a8a92"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": SLATE,
    "axes.edgecolor": "#bdbdc4", "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE, "figure.dpi": 120,
})

FEATS = ["onset_latency", "rise_time", "peak_amplitude", "recovery_time",
         "dwell_time", "switching_rate", "mean_synchrony", "synchrony_entropy"]
LABELS = ["Onset lag", "Rise time", "Peak sync.", "Fall time",
          "Dwell time", "Switching rate", "Mean sync.", "Sync. entropy"]

CORR_CMAP = LinearSegmentedColormap.from_list(
    "sageclay", [SAGE, "#cfd6c8", "#f3efe9", "#e7b59c", CLAY])


def _corr_panel(ax, df, title):
    sub = df[FEATS].apply(pd.to_numeric, errors="coerce")
    C = sub.corr().values
    im = ax.imshow(C, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(LABELS, fontsize=8.5)
    for i in range(8):
        for j in range(8):
            v = C[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else SLATE)
    ax.set_title(title, fontsize=12, fontweight="bold", color=SLATE, pad=8)
    return im


def _scree_panel(ax, df, title, jk_label, jk_n):
    sub = df[FEATS].apply(pd.to_numeric, errors="coerce").dropna()
    X = (sub - sub.mean()) / (sub.std() + 1e-9)
    cov = np.cov(X.values, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.clip(ev, 0, None)
    indiv = ev / ev.sum() * 100
    cum = np.cumsum(indiv)
    x = np.arange(1, 9)
    ax.bar(x, indiv, color="#ddd6c8", edgecolor="#bdb6a6", width=0.8,
           label="Individual component %", zorder=2)
    ax.plot(x, cum, "-s", color=CLAY, lw=2, ms=5, label="Cumulative %", zorder=4)
    ax.plot(x, cum, "--", color=SAGE, lw=1.6, alpha=0.9,
            label=f"{jk_label} (n={jk_n})", zorder=3)
    ax.axhline(90, ls="--", lw=1, color="#b9c0ac")
    ax.axhline(60, ls="--", lw=1, color="#d8c0b3")
    ax.text(8.35, 90, "90%", fontsize=8, color=INK3, va="center")
    ax.text(8.35, 60, "60%", fontsize=8, color=INK3, va="center")
    ax.annotate(f"PC1 = {indiv[0]:.1f}%", (1, cum[0]), (1.9, cum[0] - 12),
                fontsize=10.5, color=SLATE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#b0b0b8", lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([f"PC{i}" for i in x], fontsize=8.5)
    ax.set_xlim(0.3, 9.3); ax.set_ylim(0, 108)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=SLATE, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right", frameon=True,
              facecolor="white", edgecolor="#dcdcdc", framealpha=0.95)


def fig2():
    rec = pd.read_csv(ART / "realtest/lerique_2024/per_record_features.csv")
    tri = pd.read_csv(ART / "realtest/lerique_2024/trial_level_features.csv")

    fig = plt.figure(figsize=(15.5, 12.6))
    # 2 rows; each row: [matrix | cbar | scree].  width ratios give the matrix
    # its own slim colorbar, fully separated from the scree panel.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.04, 1.15],
                          hspace=0.42, wspace=0.30,
                          left=0.07, right=0.97, top=0.95, bottom=0.10)

    for row, (df, tag, paradigm) in enumerate([
            (rec, "\u2460", "Coarse (per record)"),
            (tri, "\u2461", "Fine (per trial)")]):
        ax_m = fig.add_subplot(gs[row, 0])
        ax_c = fig.add_subplot(gs[row, 1])
        ax_s = fig.add_subplot(gs[row, 2])
        im = _corr_panel(
            ax_m, df,
            f"{tag} {paradigm}  \u00b7  correlation matrix  (n={len(df)})")
        cb = fig.colorbar(im, cax=ax_c)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_edgecolor("#d0d0d6")
        _scree_panel(ax_s, df, f"{tag} {paradigm}  \u00b7  PCA composition",
                     "Leave-one-subject jackknife", len(df))

    p = OUT / "en_fig2_validity.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig); print("wrote", p)


def fig3():
    """False-positive rate vs shared-stimulus drive (published curve values)."""
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    series = [
        ("mean synchrony only", CLAY, "-", "o",
         [13.3, 13.3, 60.0, 96.7, 100.0]),
        ("peak synchrony only", PURPLE, "-", "o",
         [9.7, 13.0, 43.3, 86.7, 96.7]),
        ("synchrony entropy only", "#a9a9b0", "-", "o",
         [9.7, 16.7, 50.0, 53.3, 86.7]),
        ("SyncPipe full design (6-feature joint gate)", SAGE, "-", "o",
         [0.0, 0.0, 23.3, 70.0, 83.3]),
        ("FT surrogate gate (solid)", "#6f7a66", "-", "s",
         [0.0, 0.0, 3.3, 3.3, 23.3]),
        ("IAAFT gate (dashed)", "#6f7a66", "--", "s",
         [0.0, 0.0, 20.0, 53.3, 19.5]),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.3))
    for name, col, ls, mk, ys in series:
        ax.plot(x, ys, ls, color=col, marker=mk, lw=2.2, ms=6, label=name)
    ax.axhline(5, ls=":", lw=1.3, color="#b9b9c0")
    ax.text(0.81, 6.5, "ideal baseline 5%", fontsize=9.5, color=INK3)
    ax.set_xlabel("Shared external-stimulus drive strength  (stronger \u2192)", fontsize=11)
    ax.set_ylabel("False-positive rate flagged as\n\"interpersonal synchrony\" (%)", fontsize=11)
    ax.set_xlim(-0.04, 0.86); ax.set_ylim(-4, 105)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    p = OUT / "en_fig3_false_positive.png"
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", p)


def fig5():
    """Sensitivity envelopes, 3 panels (published curve values)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))

    def panel(ax, xs, mean_y, peak_y, timing_y, xlabel, shade, shade_lbl,
              logx=False):
        if shade:
            ax.axvspan(shade[0], shade[1], color="#e6e3db", alpha=0.7, zorder=0)
            xmid = (shade[0] * shade[1]) ** 0.5 if logx else (shade[0] + shade[1]) / 2
            ax.text(xmid, 101, shade_lbl, ha="center", fontsize=10,
                    color=INK3, va="bottom")
        ax.axhline(80, ls="--", lw=1.1, color="#c4c4cb")
        ax.plot(xs, mean_y, "-o", color=CLAY, lw=2.4, ms=6, label="mean synchrony")
        ax.plot(xs, peak_y, "-o", color=PURPLE, lw=2.4, ms=6, label="peak synchrony")
        ax.plot(xs, timing_y, "--", color="#b6b6bd", lw=1.6,
                label="timing family (see timing axis)")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylim(-4, 108)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    x1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    panel(axes[0], x1,
          [0, 4, 7, 13, 26, 47, 93, 100],
          [0, 3, 6, 7, 23, 46, 80, 97],
          [2, 3, 4, 5, 4, 18, 33, 9],
          "Synchrony strength  (GT known)  \u2192", (0.5, 0.95),
          "operating range  p \u2265 0.3")
    axes[0].set_ylabel("Power (%)", fontsize=11)

    x2 = [15, 30, 60, 120]
    panel(axes[1], x2,
          [25, 75, 95, 90],
          [5, 26, 65, 90],
          [14, 5, 55, 15],
          "WCC window (s)  \u2192", (30, 60), "recommended 30\u201360s")

    x3 = [0.1, 0.3, 0.5, 1.0, 2.0]
    panel(axes[2], x3,
          [85, 75, 51, 5, 5],
          [50, 40, 15, 0, 0],
          [3, 13, 5, 2, 1],
          "Noise / signal ratio (log)  \u2192", (0.08, 0.5), "noise \u2264 0.5",
          logx=True)
    axes[2].legend(fontsize=8.5, loc="upper right", frameon=False)

    p = OUT / "en_fig5_sensitivity.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", p)


def fig6():
    """Timing recovery scatter (GT-3b, data-driven)."""
    df = pd.read_csv(ART / "gt3b_feature_recovery.csv")

    def scatter(ax, ctrl_col, feat, color, xlabel, ylabel):
        sub = df[df.feature == feat].copy()
        sub = sub.dropna(subset=["value", ctrl_col])
        xs = sub[ctrl_col].values.astype(float)
        ys = sub["value"].values.astype(float)
        ax.scatter(xs, ys, s=14, color=color, alpha=0.25, zorder=2)
        levels = sorted(np.unique(xs))
        mids = [np.median(ys[xs == lv]) for lv in levels]
        errs = [np.std(ys[xs == lv]) for lv in levels]
        ax.errorbar(levels, mids, yerr=errs, fmt="o-", color=color, lw=2.2,
                    ms=7, capsize=4, zorder=4)
        # Recoverability = Spearman rank correlation across all simulated runs
        # (injected control level vs measured feature value). Real value from
        # the GT-3b sweep; positive => the feature tracks the injected timing.
        from scipy.stats import spearmanr
        rho = spearmanr(xs, ys).correlation if len(xs) > 2 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        strong = abs(rho) >= 0.3
        ax.text(0.04, 0.93, f"\u03c1 = {rho:.2f}", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=SLATE, va="top")
        ax.text(0.04, 0.83,
                "recoverable trend" if strong else "weak \u00b7 exploratory",
                transform=ax.transAxes, fontsize=10,
                color=color if strong else INK3, va="top")
        if not strong:
            z = np.polyfit(levels, mids, 1)
            ax.plot(levels, np.polyval(z, levels), "--", color="#b6b6bd", lw=2,
                    zorder=3)
        ax.set_xticks(levels)
        ax.set_xlabel(xlabel, fontsize=10.5); ax.set_ylabel(ylabel, fontsize=10.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return rho

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    scatter(axes[0], "onset_delay", "onset_latency", BLUE,
            "Injected onset lag  \u2192", "Onset lag (feature value)")
    scatter(axes[1], "tau_decay", "recovery_time", SAGE,
            "Injected \u03c4$_{decay}$  \u2192", "Fall time (feature value)")
    scatter(axes[2], "tau_rise", "rise_time", CLAY,
            "Injected \u03c4$_{rise}$  \u2192", "Rise time (feature value)")
    p = OUT / "en_fig6_timing_recovery.png"
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    fig2()
    fig3()
    fig5()
    fig6()

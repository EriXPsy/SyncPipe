"""Regenerate the two hand-authored schematic figures (GT 3-axis & surrogate
hourglass) in ENGLISH, matching the sage/clay/slate/purple palette used in
Intro.html. Outputs PNG to artifacts/en_figs/.

These are layout-only schematics (no underlying data), so they are drawn
directly with matplotlib primitives.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "en_figs"
OUT.mkdir(parents=True, exist_ok=True)

SAGE = "#7a8b6f"
CLAY = "#c47b5a"
SLATE = "#4a4a52"
GOLD = "#cbb994"
PURPLE = "#8a6d8f"
BLUE = "#5b8aa6"
INK3 = "#6f6f78"
BG = "#f7f5f1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": SLATE,
    "axes.edgecolor": SLATE,
    "figure.dpi": 120,
})


def _gt_panel(fig, rect, color, title, subtitle, plot_fn):
    """Draw one GT panel: rounded border + pill header (outer transparent axes),
    an inner plot with L-shaped arrowed axes, and a subtitle below the panel."""
    # Outer axes spanning the panel rectangle (for border, header pill, subtitle)
    outer = fig.add_axes(rect)
    outer.set_xlim(0, 1); outer.set_ylim(0, 1)
    outer.axis("off")
    # rounded border
    border = FancyBboxPatch((0.02, 0.10), 0.96, 0.86,
                            boxstyle="round,pad=0,rounding_size=0.04",
                            linewidth=2.2, edgecolor=color, facecolor="white",
                            mutation_aspect=0.5, transform=outer.transAxes,
                            zorder=1)
    outer.add_patch(border)
    # pill header
    pill = FancyBboxPatch((0.18, 0.84), 0.64, 0.085,
                          boxstyle="round,pad=0,rounding_size=0.04",
                          linewidth=0, facecolor=color, mutation_aspect=0.5,
                          transform=outer.transAxes, zorder=3)
    outer.add_patch(pill)
    outer.text(0.5, 0.882, title, ha="center", va="center", color="white",
               fontsize=15, fontweight="bold", zorder=4)
    # subtitle below the panel
    outer.text(0.5, 0.015, subtitle, ha="center", va="center", color=color,
               fontsize=13, fontweight="bold", zorder=4)

    # Inner plot axes, inset within the panel
    x0, y0, w, h = rect
    inset = fig.add_axes([x0 + 0.055 * w, y0 + 0.20 * h,
                          w * 0.86, h * 0.52])
    inset.set_facecolor("none")
    plot_fn(inset)
    inset.set_xticks([]); inset.set_yticks([])
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        inset.spines[s].set_visible(True)
        inset.spines[s].set_color(SLATE)
        inset.spines[s].set_linewidth(1.4)
    return inset


def fig1_gt_axes():
    fig = plt.figure(figsize=(15.5, 5.6))
    t = np.linspace(0, 1, 240)

    # ---- Panel 1: GT-1 Intensity axis ----
    def p1(ax):
        xs = np.array([0.12, 0.5, 0.88]); ys = np.array([0.18, 0.5, 0.82])
        ax.plot([0.08, 1.0], [0.12, 0.93], color=BLUE, lw=2.0, ls="--", alpha=0.6)
        ax.plot(xs, ys, color=BLUE, lw=2.6)
        ax.errorbar(xs, ys, yerr=0.07, fmt="o", color=BLUE, capsize=3, ms=7)
        ax.set_xlim(0, 1.08); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Synchrony strength  \u2192", fontsize=11, color=SLATE)
        ax.set_ylabel("Feature value  \u2191", fontsize=11, color=SLATE)
    _gt_panel(fig, [0.015, 0.0, 0.30, 1.0], BLUE, "GT-1  Intensity axis",
              "Distinguish strong vs. weak synchrony", p1)

    # ---- Panel 2: GT-2 Temporal axis ----
    def p2(ax):
        env = np.piecewise(t, [t < 0.18, (t >= 0.18) & (t < 0.42),
                               (t >= 0.42) & (t < 0.62), t >= 0.62],
                           [0.06, lambda z: 0.06 + (z - 0.18) / 0.24 * 0.84,
                            0.9, lambda z: 0.9 * np.exp(-(z - 0.62) * 5.5) + 0.06])
        ax.plot(t, env, color=PURPLE, lw=2.8)
        ax.axhline(0.06, color=GOLD, lw=1.4, ls=":")
        for xx, lab in [(0.20, "onset"), (0.34, "rise"), (0.5, "dwell"), (0.68, "fall")]:
            ax.text(xx, 0.18, lab, color=PURPLE, fontsize=10, fontweight="bold",
                    ha="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time  \u2192", fontsize=11, color=SLATE)
        ax.set_ylabel("Synchrony strength  \u2191", fontsize=11, color=SLATE)
    _gt_panel(fig, [0.35, 0.0, 0.30, 1.0], PURPLE, "GT-2  Temporal axis",
              "Characterise the rise\u2013fall dynamics", p2)

    # ---- Panel 3: GT-3 Paradigm axis ----
    def p3(ax):
        structured = 0.45 + 0.46 * np.exp(-((t - 0.26) ** 2) / 0.0035) - 0.05 * t
        sq = (np.sin(2 * np.pi * t * 4.5) > 0).astype(float)
        free = 0.30 + 0.14 * sq
        ax.plot(t, structured, color=CLAY, lw=2.6)
        ax.plot(t, free, color=SAGE, lw=2.6)
        ax.text(0.62, 0.95, "Structured", color=CLAY, fontsize=11, fontweight="bold")
        ax.text(0.62, 0.34, "Free-form", color=SAGE, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
        ax.set_xlabel("Time  \u2192", fontsize=11, color=SLATE)
        ax.set_ylabel("Synchrony strength  \u2191", fontsize=11, color=SLATE)
    _gt_panel(fig, [0.685, 0.0, 0.30, 1.0], CLAY, "GT-3  Paradigm axis",
              "Select features by paradigm", p3)

    # arrows between panels (figure coords)
    for xa in (0.327, 0.662):
        a = FancyArrowPatch((xa, 0.52), (xa + 0.02, 0.52), arrowstyle="-|>",
                            mutation_scale=24, color=SLATE, lw=2.2,
                            transform=fig.transFigure)
        fig.patches.append(a)

    p = OUT / "en_fig1_gt_axes.png"
    fig.savefig(p, dpi=240, facecolor="white")
    plt.close(fig)
    print("wrote", p)


def _funnel(ax, cx, cy, color, dots_seed):
    """A trapezoid 'sieve' with scattered dots above and one dot falling below."""
    rng = np.random.default_rng(dots_seed)
    w_top, w_bot, h = 0.95, 0.42, 0.85
    poly = Polygon([(cx - w_top / 2, cy + h / 2), (cx + w_top / 2, cy + h / 2),
                    (cx + w_bot / 2, cy - h / 2), (cx - w_bot / 2, cy - h / 2)],
                   closed=True, facecolor="#efece6", edgecolor=color, linewidth=2.2)
    ax.add_patch(poly)
    for _ in range(7):
        ax.plot(cx + rng.uniform(-0.42, 0.42), cy + h / 2 + rng.uniform(0.08, 0.34),
                "o", color=color, ms=5, alpha=0.85)
    ax.plot(cx, cy - h / 2 - 0.28, "o", color=color, ms=8)


def _box(ax, x, y, w, h, color, title, fontsize=15, text_color="white", rounding=0.04):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                         linewidth=0, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold")


def _frame(ax, x, y, w, h, color, title):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.04",
                         linewidth=2.0, edgecolor=color, facecolor="white")
    ax.add_patch(box)
    hh = 0.62
    hdr = FancyBboxPatch((x, y + h - hh), w, hh,
                         boxstyle="round,pad=0.0,rounding_size=0.04",
                         linewidth=0, facecolor=color)
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")


def _arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=22, color=SLATE, lw=2))


def fig4_hourglass():
    fig, ax = plt.subplots(figsize=(7.5, 8.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.9)
    ax.axis("off")

    _box(ax, 3.0, 10.9, 4.0, 0.85, GOLD, "Observed synchrony",
         fontsize=16, text_color=SLATE)
    _arrow(ax, 5.0, 10.85, 10.25)

    _frame(ax, 0.6, 6.0, 8.8, 4.2, PURPLE,
           "Paradigm pre-filters  \u00b7  enabled on demand")
    funnels = [(2.3, "pseudo-pair", "mismatched pairing", BLUE, 1),
               (5.0, "time-shift", "temporal misalignment", SAGE, 2),
               (7.7, "across-stim", "shared stimulus", PURPLE, 3)]
    for cx, name, sub, col, sd in funnels:
        _funnel(ax, cx, 8.3, col, sd)
        ax.text(cx, 7.05, name, ha="center", fontsize=13, fontweight="bold", color=SLATE)
        ax.text(cx, 6.62, sub, ha="center", fontsize=9.5, color=INK3)
    _arrow(ax, 5.0, 5.95, 5.15)

    _frame(ax, 0.6, 1.2, 8.8, 3.8, CLAY,
           "Two core judges  \u00b7  unified final ruling")
    _box(ax, 1.8, 3.35, 6.4, 0.72, CLAY,
         "IAAFT  \u00b7  first pass (primary test)", fontsize=13)
    _arrow(ax, 5.0, 3.30, 2.95)
    _box(ax, 1.8, 2.05, 6.4, 0.72, PURPLE,
         "FT surrogate  \u00b7  review (robustness)", fontsize=13)

    _box(ax, 0.6, 0.25, 8.8, 0.85, SAGE,
         "Genuine interpersonal synchrony", fontsize=16)

    p = OUT / "en_fig4_hourglass.png"
    fig.savefig(p, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig1_gt_axes()
    fig4_hourglass()

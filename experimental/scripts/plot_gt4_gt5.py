"""Render GT-4 and GT-5 result figures (PNG, 300 dpi)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GT4_JSON = HERE / "treur_validation_out" / "treur_monte_carlo_results.json"
GT5_JSON = HERE / "gt5_out" / "gt5_v2_calibrated_results.json"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette (matches Intro.html: sage / clay / slate)
SAGE = "#7a8b6f"
CLAY = "#c47b5a"
SLATE = "#4a4a52"
GOLD = "#cbb994"
RED = "#b5564a"
GREY = "#9a9aa2"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": SLATE,
    "axes.labelcolor": SLATE,
    "text.color": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "figure.dpi": 120,
})

import sys
sys.path.insert(0, str(HERE.parent))
from multisync.simulation.treur_dyad_v2 import scenario_emergent_sync
from multisync.dynamic_features import sliding_window_wcc
from multisync.transition_detection import detect_transitions

HZ = 10.0
WIN = int(30 * HZ)
STEP = int(5 * HZ)


def _example_traces(seed, shared_drive):
    """Reproduce one episode for the example panels."""
    rng = np.random.default_rng(seed)
    dur = 200.0
    s1 = float(rng.uniform(0.20, 0.40)) * dur
    s2 = float(rng.uniform(0.55, 0.75)) * dur
    switch_times = [s1, s2]
    switch_alphas = [
        (float(rng.uniform(0.2, 0.4)), float(rng.uniform(0.8, 1.0))),
        (float(rng.uniform(0.7, 0.9)), float(rng.uniform(0.1, 0.3))),
    ]
    res = scenario_emergent_sync(
        duration_sec=dur, hz=HZ, seed=seed,
        switch_times=switch_times, switch_alphas=switch_alphas,
        shared_drive=shared_drive,
    )
    w = res.synchrony_ground_truth
    wcc = sliding_window_wcc(res.x_A_obs, res.x_B_obs,
                             window_size=WIN, hz=HZ, step_samples=STEP)
    wcc_abs = np.abs(wcc)
    wcc_hz = HZ / STEP
    t_w = np.arange(len(w)) / HZ
    t_c = np.arange(len(wcc_abs)) / wcc_hz
    half = max(2, int((30 * wcc_hz) / 2))
    tr = detect_transitions(wcc_abs, half_window=half, method="average")
    bnd = (tr.boundary_indices / wcc_hz)
    return t_w, w, t_c, wcc_abs, bnd, switch_times


def _paired_w_wcc(seed, shared_drive):
    """Return paired (W_AB, |WCC|) samples on the WCC grid for one seed."""
    t_w, w, t_c, wcc, _, _ = _example_traces(seed, shared_drive)
    # resample ground-truth W_AB onto the (coarser) WCC time grid
    w_on_c = np.interp(t_c, t_w, w)
    finite = np.isfinite(wcc)
    return w_on_c[finite], wcc[finite]



def fig_gt4():
    data = json.loads(GT4_JSON.read_text(encoding="utf-8"))
    clean = data["arms"]["clean"]["per_seed"]
    shared = data["arms"]["shared_drive"]["per_seed"]
    # representative seed for the shared-drive example panel (B)
    seed = min(shared, key=lambda r: abs(r["err_dwell_fixed"] - 109))["seed"]

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.suptitle("GT-4  Emergent synchrony: recoverable without common input, "
                 "confounded with it", fontsize=14, fontweight="bold", color=SLATE)

    # -- (A) WCC-vs-truth calibration across all clean seeds --
    a = ax[0, 0]
    all_w, all_c = [], []
    for r in clean:
        wv, cv = _paired_w_wcc(r["seed"], shared_drive=False)
        all_w.append(wv); all_c.append(cv)
    all_w = np.concatenate(all_w); all_c = np.concatenate(all_c)
    a.scatter(all_w, all_c, s=3, color=SAGE, alpha=0.10, edgecolors="none")
    # binned median +/- IQR
    edges = np.linspace(0, 1, 11)
    centers, med, q1, q3 = [], [], [], []
    for i in range(len(edges) - 1):
        m = (all_w >= edges[i]) & (all_w < edges[i + 1])
        if m.sum() >= 20:
            centers.append((edges[i] + edges[i + 1]) / 2)
            med.append(np.median(all_c[m]))
            q1.append(np.percentile(all_c[m], 25))
            q3.append(np.percentile(all_c[m], 75))
    centers = np.array(centers)
    a.fill_between(centers, q1, q3, color=CLAY, alpha=0.25, label="IQR")
    a.plot(centers, med, color=CLAY, lw=2.4, marker="o", ms=4, label="binned median $|WCC|$")
    a.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1.0, alpha=0.6, label="identity")
    a.set_title("(A) Independent drives \u2014 $|WCC|$ tracks true $W_{AB}$",
                fontsize=11, fontweight="bold")
    a.set_xlabel("ground-truth $W_{AB}(t)$"); a.set_ylabel("recovered $|WCC|$")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.legend(loc="upper left", fontsize=8, framealpha=0.9)
    a.text(0.98, 0.04, "100 seeds pooled", transform=a.transAxes,
           ha="right", va="bottom", fontsize=8, color=GREY)

    # -- (B) shared-drive arm example --
    t_w, w, t_c, wcc, bnd, sw = _example_traces(seed, shared_drive=True)
    b = ax[0, 1]
    b.plot(t_w, w, color=SAGE, lw=2.4, label="ground-truth $W_{AB}(t)$")
    b.plot(t_c, wcc, color=CLAY, lw=1.5, alpha=0.9, label="recovered $|WCC|$")
    lo, hi = sw[0], sw[1]
    b.axvspan(lo, hi, color=RED, alpha=0.10)
    b.text((lo + hi) / 2, 0.30, "true de-sync\n(invisible to WCC)",
           ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold")
    b.set_title("(B) Shared exogenous drive \u2014 ISC confound", fontsize=11, fontweight="bold")
    b.set_xlabel("time (s)"); b.set_ylabel("synchrony")
    b.set_ylim(-0.05, 1.05); b.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # -- (C) error distributions (violin) --
    c = ax[1, 0]
    metrics = ["err_dwell_fixed", "err_switch_fixed", "err_peak"]
    labels = ["dwell err (s)", "switch err (/min)", "peak err"]
    positions = []
    pos = 1
    for mi, m in enumerate(metrics):
        for ai, (arm, col) in enumerate([(clean, SAGE), (shared, CLAY)]):
            vals = np.array([r[m] for r in arm if np.isfinite(r[m])])
            vp = c.violinplot([vals], positions=[pos], widths=0.7, showmedians=True)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_alpha(0.6); body.set_edgecolor(SLATE)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                vp[key].set_color(SLATE); vp[key].set_linewidth(1.0)
            positions.append(pos); pos += 1
        pos += 1
    c.axhline(0, color=SLATE, lw=1.0, ls="-", alpha=0.5)
    c.set_xticks([1.5, 4.5, 7.5]); c.set_xticklabels(labels, fontsize=9)
    c.set_title("(C) Recovered-vs-truth error (100 seeds)", fontsize=11, fontweight="bold")
    c.set_ylabel("error (recovered \u2212 truth)")
    c.plot([], [], color=SAGE, lw=6, alpha=0.6, label="clean")
    c.plot([], [], color=CLAY, lw=6, alpha=0.6, label="shared drive")
    c.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # -- (D) onset-localisation error histogram --
    d = ax[1, 1]
    on_clean = [r["onset_err_switch1"] for r in clean if r["onset_err_switch1"] is not None
                and np.isfinite(r["onset_err_switch1"])]
    on_shared = [r["onset_err_switch1"] for r in shared if r["onset_err_switch1"] is not None
                 and np.isfinite(r["onset_err_switch1"])]
    bins = np.linspace(0, 30, 16)
    d.hist(on_clean, bins=bins, color=SAGE, alpha=0.65, label="clean", edgecolor=SLATE)
    d.hist(on_shared, bins=bins, color=CLAY, alpha=0.55, label="shared drive", edgecolor=SLATE)
    d.axvline(np.median(on_clean), color=SAGE, ls="--", lw=1.6)
    d.set_title("(D) Transition onset-localisation error (switch 1)",
                fontsize=11, fontweight="bold")
    d.set_xlabel("|detected \u2212 true switch| (s)"); d.set_ylabel("count")
    d.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for row in ax:
        for axx in row:
            axx.spines["top"].set_visible(False)
            axx.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "gt4_monte_carlo.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


CONDS = ["random_baseline", "sync_high_seg_high", "sync_high_seg_low",
         "sync_low_seg_high", "sync_low_seg_low"]
CLABS = ["random\nbaseline", "Cond1\nhi-sync\nhi-seg", "Cond2\nhi-sync\nlo-seg",
         "Cond3\nlo-sync\nhi-seg", "Cond4\nlo-sync\nlo-seg"]


def fig_gt5():
    data = json.loads(GT5_JSON.read_text(encoding="utf-8"))
    beh = data["behavioral_sync"]
    feats = data["features"]
    ibi = data["ibi_corr"]
    eda = data["eda_corr"]
    conds_meta = {c["name"]: c for c in data["conditions"]}

    colors = [GREY, SAGE, SAGE, CLAY, CLAY]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("GT-5  Gordon (2025)-calibrated dyads vs decoupled random baseline",
                 fontsize=14, fontweight="bold", color=SLATE)

    base_corr = float(np.mean(beh["random_baseline"]))

    # -- (A) behavioral synchrony vs chance floor --
    a = ax[0]
    means = [float(np.mean(beh[c])) for c in CONDS]
    sds = [float(np.std(beh[c])) for c in CONDS]
    a.bar(range(5), means, yerr=sds, color=colors, edgecolor=SLATE,
          alpha=0.85, capsize=4)
    a.axhspan(0, base_corr, color=RED, alpha=0.12)
    a.axhline(base_corr, color=RED, ls="--", lw=1.4,
              label=f"chance floor = {base_corr:.3f}")
    a.set_xticks(range(5)); a.set_xticklabels(CLABS, fontsize=8)
    a.set_ylabel("behavioral synchrony  corr(A,B)")
    a.set_title("(A) All conditions sit above chance", fontsize=11, fontweight="bold")
    a.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # -- (B) dwell-time distribution per condition --
    b = ax[1]
    dwell_data = []
    for cn in CONDS:
        vals = [f["dwell_time"] for f in feats[cn]
                if f["dwell_time"] is not None and np.isfinite(f["dwell_time"])]
        dwell_data.append(vals if vals else [0.0])
    bp = b.boxplot(dwell_data, positions=range(5), widths=0.6,
                   patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(SLATE)
    for key in ("whiskers", "caps", "medians"):
        for ln in bp[key]:
            ln.set_color(SLATE)
    b.set_xticks(range(5)); b.set_xticklabels(CLABS, fontsize=8)
    b.set_ylabel("synchrony-episode dwell time (s)")
    b.set_title("(B) Dwell time: stable conditions hold longer episodes",
                fontsize=11, fontweight="bold")
    b.annotate("segregation pull\n\u2192 shorter episodes",
               xy=(3, np.median(dwell_data[3])), xytext=(1.4, max(np.median(d) for d in dwell_data) * 0.55),
               fontsize=8.5, color=CLAY, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=CLAY))

    # -- (C) IBI sim vs empirical target --
    c = ax[2]
    for ci, cn in enumerate(CONDS):
        sim = float(np.mean(ibi[cn]))
        tgt = conds_meta[cn].get("target_ibi_sync", 0.0)
        c.scatter(tgt, sim, s=90, color=colors[ci], edgecolor=SLATE, zorder=3,
                  label=CLABS[ci].replace("\n", " "))
    lim = 0.22
    c.plot([0, lim], [0, lim], color=SLATE, ls="--", lw=1.0, alpha=0.6)
    c.set_xlim(-0.01, lim); c.set_ylim(-0.01, lim)
    c.set_xlabel("empirical target  (Gordon 2025)")
    c.set_ylabel("simulated IBI synchrony")
    c.set_title("(C) IBI calibration fidelity", fontsize=11, fontweight="bold")
    c.legend(loc="upper left", fontsize=7, framealpha=0.9)

    for axx in ax:
        axx.spines["top"].set_visible(False)
        axx.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "gt5_conditions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_gt4()
    fig_gt5()
    print("done ->", OUT)



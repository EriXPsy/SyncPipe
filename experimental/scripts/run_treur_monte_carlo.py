"""
run_treur_monte_carlo.py
========================

GT-4 Monte Carlo validation.

Runs the *emergent* Treur dyad scenario (``scenario_emergent_sync``) over many
random seeds.  For each run the coupling weight W_AB(t) emerges from the
adaptive-network dynamics (Hebbian-with-saturation rule) and is the ground
truth.  SyncPipe recovers synchrony features from the noisy observed signals;
this script aggregates the *error distribution* of the recovered features
against the ground-truth W_AB(t) across seeds, plus the onset-localisation
error of the Treur transition detector against the known control switches.

Run:  python scripts/run_treur_monte_carlo.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multisync.simulation.treur_dyad_v2 import scenario_emergent_sync
from multisync.epoch_detection import dual_stream_epoch_analysis
from multisync.dynamic_features import sliding_window_wcc
from multisync.adaptive_threshold import calibrate_threshold_from_signals
from multisync.transition_detection import detect_transitions


# -- Config ---------------------------------------------------------------
N_SEEDS = 100
DURATION_SEC = 200.0
HZ = 10.0
WCC_WINDOW_SEC = 30.0
WCC_STEP_SEC = 5.0
GT_THRESHOLD = 0.5          # threshold on ground-truth W_AB(t)
OUTDIR = Path(__file__).resolve().parent / "treur_validation_out"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _segment_metrics(curve, hz, threshold):
    """Dwell (mean episode length, s) and switching rate (entries/min)."""
    mask = curve >= threshold
    if mask.size == 0:
        return float("nan"), float("nan"), 0
    # episode lengths
    lengths = []
    run = 0
    n_entries = 0
    prev = False
    for m in mask:
        if m:
            if not prev:
                n_entries += 1
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
        prev = bool(m)
    if run > 0:
        lengths.append(run)
    dwell = (np.mean(lengths) / hz) if lengths else 0.0
    minutes = (len(mask) / hz) / 60.0
    switching = (n_entries / minutes) if minutes > 0 else 0.0
    return float(dwell), float(switching), n_entries


def run_one(seed, shared_drive=True):
    """One emergent-sync episode -> recovered vs ground-truth metrics.

    The control switches (timing and target alphas) are *resampled per seed*
    so that the ground-truth W_AB(t) trajectory itself varies across the Monte
    Carlo ensemble, not just the observation noise.  This turns the ensemble
    into a distribution over plausible in-and-out-of-sync episodes rather than
    repeated noise draws on one fixed trajectory.

    shared_drive : bool
        True  -> both agents receive a common exogenous stimulus (ISC
                 confound arm): observed WCC reflects W_AB(t) + shared drive.
        False -> no common input (clean arm): observed WCC reflects W_AB(t).
    """
    rng = np.random.default_rng(seed)

    # two control switches at randomised times within the middle of the run
    s1 = float(rng.uniform(0.20, 0.40)) * DURATION_SEC
    s2 = float(rng.uniform(0.55, 0.75)) * DURATION_SEC
    switch_times = [s1, s2]
    # de-sync episode in the middle (independence pull), then back in sync
    a_sync_lo = float(rng.uniform(0.2, 0.4))
    a_indep_hi = float(rng.uniform(0.8, 1.0))
    a_sync_hi = float(rng.uniform(0.7, 0.9))
    a_indep_lo = float(rng.uniform(0.1, 0.3))
    switch_alphas = [(a_sync_lo, a_indep_hi), (a_sync_hi, a_indep_lo)]

    res = scenario_emergent_sync(
        duration_sec=DURATION_SEC, hz=HZ, seed=seed,
        switch_times=switch_times, switch_alphas=switch_alphas,
        shared_drive=shared_drive,
    )
    a, b = res.x_A_obs, res.x_B_obs
    w_gt = res.synchrony_ground_truth

    # -- Ground-truth segment metrics on W_AB(t) --------------------------
    gt_dwell, gt_switch, gt_n = _segment_metrics(w_gt, HZ, GT_THRESHOLD)
    gt_peak = float(np.max(w_gt))

    # -- Recovered WCC curve + features -----------------------------------
    win = int(WCC_WINDOW_SEC * HZ)
    step = int(WCC_STEP_SEC * HZ)
    wcc = sliding_window_wcc(a, b, window_size=win, hz=HZ, step_samples=step)
    wcc_hz = HZ / step
    wcc_abs = np.abs(wcc)

    # adaptive threshold from the raw signals (correct null calibration)
    theta_d = calibrate_threshold_from_signals(
        a, b, wcc_window_sec=WCC_WINDOW_SEC, wcc_step_sec=WCC_STEP_SEC,
        hz=HZ, seed=seed, n_surrogates=199,
    )

    rec_dwell_f, rec_switch_f, _ = _segment_metrics(wcc_abs, wcc_hz, 0.5)
    rec_dwell_a, rec_switch_a, _ = _segment_metrics(wcc_abs, wcc_hz, theta_d)
    rec_peak = float(np.nanmax(wcc_abs))

    # -- Transition-based onset localisation vs known switches ------------
    sw_true_sec = switch_times  # per-seed randomised switch schedule
    half_w = max(2, int((WCC_WINDOW_SEC * wcc_hz) / 2))
    tr = detect_transitions(wcc_abs, half_window=half_w, method="average")
    bnd_sec = (tr.boundary_indices / wcc_hz).tolist()

    onset_errs = []
    for st in sw_true_sec:
        if bnd_sec:
            nearest = min(bnd_sec, key=lambda x: abs(x - st))
            onset_errs.append(abs(nearest - st))
        else:
            onset_errs.append(float("nan"))

    return {
        "seed": seed,
        "theta_adaptive": round(theta_d, 4),
        "W_AB_std": round(float(np.std(w_gt)), 4),
        # ground truth
        "gt_dwell": gt_dwell, "gt_switch": gt_switch,
        "gt_n_episodes": gt_n, "gt_peak": gt_peak,
        # recovered (fixed 0.5)
        "rec_dwell_fixed": rec_dwell_f, "rec_switch_fixed": rec_switch_f,
        # recovered (adaptive)
        "rec_dwell_adapt": rec_dwell_a, "rec_switch_adapt": rec_switch_a,
        "rec_peak": rec_peak,
        # errors
        "err_dwell_fixed": rec_dwell_f - gt_dwell,
        "err_switch_fixed": rec_switch_f - gt_switch,
        "err_dwell_adapt": rec_dwell_a - gt_dwell,
        "err_switch_adapt": rec_switch_a - gt_switch,
        "err_peak": rec_peak - gt_peak,
        "n_boundaries": int(tr.boundary_indices.size),
        "onset_err_switch1": onset_errs[0],
        "onset_err_switch2": onset_errs[1],
    }


def _summ(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "median": round(float(np.median(a)), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "n": int(a.size),
    }


if __name__ == "__main__":
    t0 = time.time()
    print(f"GT-4 Monte Carlo: {N_SEEDS} seeds x scenario_emergent_sync (2 arms)")
    print(f"  duration={DURATION_SEC}s hz={HZ:.0f} wcc_window={WCC_WINDOW_SEC}s\n")

    keys = [
        "W_AB_std", "theta_adaptive", "n_boundaries",
        "gt_dwell", "rec_dwell_fixed", "rec_dwell_adapt",
        "gt_switch", "rec_switch_fixed", "rec_switch_adapt",
        "gt_peak", "rec_peak",
        "err_dwell_fixed", "err_dwell_adapt",
        "err_switch_fixed", "err_switch_adapt", "err_peak",
        "onset_err_switch1", "onset_err_switch2",
    ]

    arms = {"shared_drive": True, "clean": False}
    all_out = {"meta": {"n_seeds": N_SEEDS, "duration_sec": DURATION_SEC,
                        "hz": HZ, "gt_threshold": GT_THRESHOLD}, "arms": {}}

    for arm_name, sd in arms.items():
        print(f"\n--- arm: {arm_name} (shared_drive={sd}) ---")
        rows = []
        for i in range(N_SEEDS):
            rows.append(run_one(seed=1000 + i, shared_drive=sd))
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{N_SEEDS} ({time.time() - t0:.0f}s)")
        summary = {k: _summ([r[k] for r in rows]) for k in keys}
        all_out["arms"][arm_name] = {"summary": summary, "per_seed": rows}

        print(f"\n  {'metric':<22s} {'mean':>10s} {'std':>10s} {'median':>10s} {'p05':>9s} {'p95':>9s}")
        print("  " + "-" * 74)
        for k in keys:
            s = summary[k]
            if s["mean"] is None:
                print(f"  {k:<22s} {'--':>10s}")
                continue
            print(f"  {k:<22s} {s['mean']:>10.4f} {s['std']:>10.4f} {s['median']:>10.4f} "
                  f"{s['p05']:>9.4f} {s['p95']:>9.4f}")

    out_json = OUTDIR / "treur_monte_carlo_results.json"
    out_json.write_text(json.dumps(all_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults saved to: {out_json}")
    print(f"Completed in {time.time() - t0:.1f}s")

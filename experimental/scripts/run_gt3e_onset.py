"""
GT-3e: Onset-explicit ground truth for onset_latency construct validity.
========================================================================
Motivation (2026-06-08)
-----------------------
BM1/GT-3 reported onset_latency recovery rho ~= 0 and concluded the
feature has "zero construct validity".  That conclusion is INVALID,
because the GT-2/GT-3 generator never manipulates an onset time: the
first synchrony event there is drawn from an exponential waiting time
(run_gt3_resolution.generate_temporal_gt, line ~70), i.e. it is random
NOISE, not a controlled parameter.  Correlating onset_latency against
switch_freq therefore cannot test whether onset_latency measures the
ONSET of synchrony -- there is no onset truth to recover.

This script supplies the missing ground truth: it EXPLICITLY manipulates
the moment the first sustained synchrony episode ignites (onset_true),
then asks whether onset_latency recovers it.  We also contrast the
locked fixed threshold (0.5) against a relative threshold
(baseline + k * SD) to separate two hypotheses:

  H_op   : the onset OPERATIONALIZATION (first sustained crossing) is
           fundamentally broken -> low recovery under BOTH thresholds.
  H_thr  : only the FIXED 0.5 threshold is the problem (it ignores
           per-dyad baseline) -> low recovery under fixed, high under
           relative.

Design
------
- onset_true in {30, 60, 90, 120} s (the controlled truth).
- coupling(t) = 0 before onset_true; after onset, it rises to a
  plateau `coupling_high` and stays on (a single sustained episode).
- baseline_sync in {0.0, 0.2} : the OFF-episode coupling floor, used to
  create datasets where the trace sits well below vs near the 0.5
  threshold (mimics Gordon low-baseline vs Andersen high-baseline).
- noise_sigma in {0.3} ; seeds = 20.

Output:
  artifacts/gt3e_onset_recovery.csv   # per-run onset_latency + onset_true
  console: Spearman rho(onset_latency, onset_true) per threshold mode
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features

HZ = 1.0
DURATION = 180          # seconds
N_SEEDS = 20
COUPLING_HIGH = 0.7     # plateau coupling once synchrony ignites
WCC_WINDOW_SEC = 30.0

ONSET_TRUES = [30, 60, 90, 120]     # the controlled onset truth (s)
BASELINES = [0.0, 0.2]              # off-episode coupling floor
NOISE = 0.3


def generate_onset_gt(onset_true, baseline, noise_sigma, seed):
    """Two channels coupled by a step-like envelope that turns ON at
    onset_true and stays on.  Returns (a, b, coupling_envelope)."""
    rng = np.random.default_rng(seed)
    n = int(DURATION * HZ)
    onset_idx = int(onset_true * HZ)

    coupling = np.full(n, baseline, dtype=float)
    # smooth ignition over ~5 s so the rise is finite, not a hard step
    ramp = int(5 * HZ)
    for i in range(n):
        if i >= onset_idx:
            frac = min(1.0, (i - onset_idx + 1) / max(1, ramp))
            coupling[i] = baseline + (COUPLING_HIGH - baseline) * frac

    shared = np.sin(np.linspace(0, DURATION * 2 * np.pi / 60, n))
    shared += 0.5 * np.sin(np.linspace(0, DURATION * 2 * np.pi / 20, n))
    shared /= np.std(shared) + 1e-10

    a = coupling * shared + noise_sigma * rng.normal(0, 1, n)
    b = coupling * shared + noise_sigma * rng.normal(0, 1, n)
    return a, b, coupling


def _relative_threshold(wcc, k=2.0, baseline_frac=0.20):
    """baseline + k*SD using the first `baseline_frac` of finite samples."""
    finite = wcc[np.isfinite(wcc)]
    if finite.size < 5:
        return 0.5
    n0 = max(3, int(baseline_frac * finite.size))
    base = finite[:n0]
    return float(np.mean(base) + k * np.std(base))


def main():
    rows = []
    for onset_true in ONSET_TRUES:
        for baseline in BASELINES:
            for seed in range(N_SEEDS):
                a, b, _ = generate_onset_gt(onset_true, baseline, NOISE, seed)
                wcc = sliding_window_wcc(a, b,
                                         window_size=int(WCC_WINDOW_SEC * HZ),
                                         hz=HZ, step_samples=1)
                trace_hz = HZ  # step_samples=1 -> trace hz == signal hz
                # fixed threshold (locked default 0.5)
                feat_fixed = extract_dynamic_features(
                    wcc, hz=HZ, wcc_window_sec=WCC_WINDOW_SEC,
                    onset_threshold=0.5)
                # relative threshold (per-trace baseline + k*SD)
                thr_rel = _relative_threshold(wcc)
                feat_rel = extract_dynamic_features(
                    wcc, hz=HZ, wcc_window_sec=WCC_WINDOW_SEC,
                    onset_threshold=thr_rel)
                onset_fixed = getattr(feat_fixed, "onset_latency", np.nan)
                onset_rel = getattr(feat_rel, "onset_latency", np.nan)
                rows.append({
                    "onset_true": onset_true,
                    "baseline": baseline,
                    "seed": seed,
                    "rel_threshold": thr_rel,
                    "onset_fixed": onset_fixed,
                    "onset_fixed_defined": int(np.isfinite(onset_fixed)),
                    "onset_rel": onset_rel,
                    "onset_rel_defined": int(np.isfinite(onset_rel)),
                })

    df = pd.DataFrame(rows)
    out_dir = Path(REPO) / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gt3e_onset_recovery.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} runs -> {out_path}")

    print("\n=== onset_latency construct validity (onset-explicit GT) ===")
    print(f"{'mode':12s} {'n_defined':>10s} {'rho_vs_onset_true':>18s} {'p':>10s}")
    for mode, col, defcol in [
        ("fixed_0.5", "onset_fixed", "onset_fixed_defined"),
        ("relative", "onset_rel", "onset_rel_defined"),
    ]:
        sub = df[df[defcol] == 1]
        n_def = len(sub)
        if n_def >= 5:
            rho, p = spearmanr(sub[col], sub["onset_true"])
        else:
            rho, p = float("nan"), float("nan")
        print(f"{mode:12s} {n_def:>10d} {rho:>18.3f} {p:>10.2e}")

    print("\n=== defined-rate by baseline (does fixed 0.5 fail at low base?) ===")
    g = df.groupby("baseline")[["onset_fixed_defined", "onset_rel_defined"]].mean()
    print(g.round(3).to_string())

    print("\n=== mean onset estimate by onset_true (fixed thr, defined only) ===")
    sub = df[df["onset_fixed_defined"] == 1]
    if len(sub):
        print(sub.groupby("onset_true")["onset_fixed"].mean().round(1).to_string())


if __name__ == "__main__":
    main()

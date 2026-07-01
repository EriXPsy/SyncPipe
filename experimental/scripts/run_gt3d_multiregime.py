"""
GT-3d: Multi-regime ground truth -- a FAIR trial for the HSMM segmenter.
========================================================================
GT-3c finding (honest): on the single-event convolutional GT (GT-3b/3c),
HSMM LOST to the locked-in WCC>0.5 threshold on every timing feature, and
BIC picked K=2 in 100% of runs.  But that GT has only TWO regimes
(event vs baseline), so it is an unfair trial for a multi-state tool --
and BIC correctly refused to hallucinate extra states.

GT-3d gives HSMM its home turf, matching the user's insight that synchrony
may be MULTI-state, not binary (the free-play / continuous paradigm):

    A 3-regime Markov/semi-Markov process over coupling levels
        regime 0: decoupled   (rho ~ 0.0)
        regime 1: weak sync    (rho ~ 0.4)
        regime 2: strong sync  (rho ~ 0.8)
    with a controllable mean dwell time (regime persistence).

Questions:
  Q-K2:  Does BIC now recover K=3 when the GT truly has 3 regimes?
         (If yes -> the K-selection is trustworthy, not biased to 2.)
  Q-dwell: Does HSMM recover dwell_time / switching_rate vs the GT
           mean_dwell BETTER than the WCC>0.5 binarization (which
           collapses 3 regimes into 2)?
  Q-richness: Does synchrony_entropy track the number of visited regimes
              better under HSMM segmentation?

This is the continuous-paradigm complement to GT-3b's event paradigm.

Output:
  artifacts/gt3d_path_comparison.csv
  artifacts/gt3d_k_distribution.csv
  artifacts/gt3d_summary.csv
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
from multisync.segmentation import hsmm_high_sync_trace

HZ = 1.0
DURATION = 600          # longer: continuous paradigm needs more regime visits
N_SEEDS = 15
WIN = 20
STEP = 2
TRACE_HZ = HZ / STEP

REGIME_RHOS = np.array([0.0, 0.4, 0.8])      # 3 coupling regimes
MEAN_DWELLS = [10.0, 25.0, 60.0]             # mean regime persistence (sec)
N_REGIMES_TRUE = 3

ALL_FEATS = ["onset_latency", "rise_time", "peak_amplitude",
             "recovery_time", "dwell_time", "switching_rate",
             "mean_synchrony", "synchrony_entropy"]
ART = Path(REPO) / "artifacts"


def generate_multiregime_gt(mean_dwell, duration, hz, seed):
    """3-regime semi-Markov coupling: each regime persists ~mean_dwell sec,
    then jumps to another regime.  P1/P2 coupled at the regime's rho."""
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    # build a regime label sequence with geometric-ish dwell
    p_stay = 1.0 - 1.0 / (mean_dwell * hz)
    regimes = np.zeros(n, dtype=int)
    cur = rng.integers(0, N_REGIMES_TRUE)
    for i in range(n):
        if rng.random() > p_stay:
            choices = [r for r in range(N_REGIMES_TRUE) if r != cur]
            cur = int(rng.choice(choices))
        regimes[i] = cur
    rho_t = REGIME_RHOS[regimes]
    # generate coupled signals at time-varying rho via shared+independent mix
    shared = rng.normal(0, 1, n)
    ea = rng.normal(0, 1, n)
    eb = rng.normal(0, 1, n)
    a = np.sqrt(rho_t) * shared + np.sqrt(1 - rho_t) * ea
    b = np.sqrt(rho_t) * shared + np.sqrt(1 - rho_t) * eb
    n_visited = len(np.unique(regimes))
    return a, b, rho_t, regimes, n_visited


def main():
    rows, krows = [], []
    total = len(MEAN_DWELLS) * N_SEEDS
    i = 0
    for md in MEAN_DWELLS:
        for seed in range(N_SEEDS):
            i += 1
            a, b, rho_t, regimes, n_vis = generate_multiregime_gt(md, DURATION, HZ, 4000 + seed)
            wcc = sliding_window_wcc(a, b, window_size=int(WIN * HZ),
                                     hz=HZ, step_samples=STEP)
            if wcc is None or len(wcc) < 10:
                continue
            wcc = np.nan_to_num(np.asarray(wcc, float), nan=0.0)

            f_wcc = extract_dynamic_features(wcc, hz=TRACE_HZ, wcc_window_sec=WIN)
            high, info = hsmm_high_sync_trace(
                wcc, k_grid=(2, 3, 4), min_dwell=3, high_frac=0.5, seed=seed)
            f_hsmm = extract_dynamic_features(high, hz=TRACE_HZ, wcc_window_sec=WIN)
            krows.append({"mean_dwell": md, "seed": seed,
                          "k_selected": info["k"], "n_visited_true": n_vis})

            for feat in ALL_FEATS:
                vw = getattr(f_wcc, feat, None)
                vh = getattr(f_hsmm, feat, None)
                if vw is not None and np.isfinite(vw):
                    rows.append({"mean_dwell": md, "seed": seed, "path": "WCC_thresh",
                                 "feature": feat, "value": float(vw)})
                if vh is not None and np.isfinite(vh):
                    rows.append({"mean_dwell": md, "seed": seed, "path": "HSMM_seg",
                                 "feature": feat, "value": float(vh)})
            if i % 10 == 0:
                print(f"[{i}/{total}]", flush=True)

    df = pd.DataFrame(rows)
    dk = pd.DataFrame(krows)
    df.to_csv(ART / "gt3d_path_comparison.csv", index=False)
    dk.to_csv(ART / "gt3d_k_distribution.csv", index=False)

    print("\n=== SELECTED K (GT truly has 3 regimes -> does BIC recover K=3?) ===")
    print(dk["k_selected"].value_counts().sort_index().to_string())
    print("mean K selected:", round(float(dk["k_selected"].mean()), 2))

    print("\n=== DWELL/SWITCHING RECOVERY vs GT mean_dwell (Spearman rho) ===")
    summ = []
    for feat in ["dwell_time", "switching_rate", "synchrony_entropy"]:
        line = {"feature": feat}
        for path in ["WCC_thresh", "HSMM_seg"]:
            sub = df[(df.path == path) & (df.feature == feat)]
            if len(sub) >= 10:
                rho = spearmanr(sub["mean_dwell"], sub.value)[0]
                line[path] = rho
                line[f"{path}_n"] = len(sub)
            else:
                line[path] = np.nan
                line[f"{path}_n"] = len(sub)
        summ.append(line)
    df_sum = pd.DataFrame(summ)
    df_sum.to_csv(ART / "gt3d_summary.csv", index=False)
    for _, r in df_sum.iterrows():
        print(f"  {r.feature:18s} vs mean_dwell  "
              f"WCC rho={r.WCC_thresh:+.3f} (n={int(r.WCC_thresh_n)})  |  "
              f"HSMM rho={r.HSMM_seg:+.3f} (n={int(r.HSMM_seg_n)})")

    print("\nSaved: gt3d_path_comparison.csv / gt3d_k_distribution.csv / gt3d_summary.csv")


if __name__ == "__main__":
    main()

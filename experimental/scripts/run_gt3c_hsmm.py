"""
GT-3c: HSMM segmenter vs WCC-threshold path on the convolutional event GT.
==========================================================================
Two questions in one experiment (user 2026-05-31):

  Q-HSMM: Does an HSMM state segmentation of the synchrony trace recover the
          timing family (onset/rise/recovery) BETTER than the locked-in
          WCC>0.5 threshold path?  Both paths feed the SAME
          extract_dynamic_features -> feature vectors are comparable.

  Q-rise: rise_time was the only weak timing feature in GT-3b (rho=0.29).
          Can a different SEGMENTER (HSMM posterior, which is smoother and
          state-aware) rescue rise_time, or is rise fundamentally
          resolution-limited regardless of segmenter?

  Q-K:    How many states does BIC pick?  Is synchrony binary (K=2) or
          multi-state (K>=3)?  (Answers the user's "多种切换方式" question.)

Reuses GT-3b's convolutional generator: coupling(t) = event (x) bi-exp kernel,
with controllable onset_delay / tau_rise / tau_decay.

Output:
  artifacts/gt3c_path_comparison.csv   # per-run feature values x path
  artifacts/gt3c_k_distribution.csv    # selected K per run
  artifacts/gt3c_summary.csv           # recovery rho per path x feature
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
from run_gt3b_event import generate_event_gt, ONSET_DELAYS, TAU_RISES, TAU_DECAYS

HZ = 1.0
DURATION = 300
N_SEEDS = 12
WIN = 20
STEP = 2
TRACE_HZ = HZ / STEP

PAIRS = {"onset_latency": "onset_delay",
         "rise_time": "tau_rise",
         "recovery_time": "tau_decay"}
ALL_FEATS = ["onset_latency", "rise_time", "peak_amplitude",
             "recovery_time", "dwell_time", "switching_rate",
             "mean_synchrony", "synchrony_entropy"]
ART = Path(REPO) / "artifacts"


def main():
    rows, krows = [], []
    grid = [(od, tr, td) for od in ONSET_DELAYS for tr in TAU_RISES for td in TAU_DECAYS]
    total = len(grid) * N_SEEDS
    i = 0
    for (od, tr, td) in grid:
        for seed in range(N_SEEDS):
            i += 1
            a, b, env, driver = generate_event_gt(od, tr, td, DURATION, HZ, 3000 + seed)
            wcc = sliding_window_wcc(a, b, window_size=int(WIN * HZ),
                                     hz=HZ, step_samples=STEP)
            if wcc is None or len(wcc) < 6:
                continue
            wcc = np.nan_to_num(np.asarray(wcc, float), nan=0.0)

            # --- Path 1: WCC threshold (locked-in) ---
            f_wcc = extract_dynamic_features(wcc, hz=TRACE_HZ, wcc_window_sec=WIN)

            # --- Path 2: HSMM segmentation -> high-sync posterior trace ---
            high, info = hsmm_high_sync_trace(
                wcc, k_grid=(2, 3, 4), min_dwell=3, high_frac=0.5, seed=seed)
            f_hsmm = extract_dynamic_features(high, hz=TRACE_HZ, wcc_window_sec=WIN)
            krows.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                          "seed": seed, "k": info["k"]})

            for feat in ALL_FEATS:
                vw = getattr(f_wcc, feat, None)
                vh = getattr(f_hsmm, feat, None)
                if vw is not None and np.isfinite(vw):
                    rows.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                 "seed": seed, "path": "WCC_thresh",
                                 "feature": feat, "value": float(vw)})
                if vh is not None and np.isfinite(vh):
                    rows.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                 "seed": seed, "path": "HSMM_seg",
                                 "feature": feat, "value": float(vh)})
            if i % 30 == 0:
                print(f"[{i}/{total}]", flush=True)

    df = pd.DataFrame(rows)
    dk = pd.DataFrame(krows)
    df.to_csv(ART / "gt3c_path_comparison.csv", index=False)
    dk.to_csv(ART / "gt3c_k_distribution.csv", index=False)

    print("\n=== SELECTED K DISTRIBUTION (synchrony: binary vs multi-state?) ===")
    print(dk["k"].value_counts().sort_index().to_string())

    print("\n=== TIMING RECOVERY: HSMM_seg vs WCC_thresh (Spearman rho) ===")
    summ = []
    for feat, knob in PAIRS.items():
        line = {"feature": feat, "knob": knob}
        for path in ["WCC_thresh", "HSMM_seg"]:
            sub = df[(df.path == path) & (df.feature == feat)]
            if len(sub) >= 10:
                rho = spearmanr(sub[knob], sub.value)[0]
                line[path] = rho
                line[f"{path}_n"] = len(sub)
            else:
                line[path] = np.nan
                line[f"{path}_n"] = len(sub)
        summ.append(line)
    df_sum = pd.DataFrame(summ)
    df_sum.to_csv(ART / "gt3c_summary.csv", index=False)
    for _, r in df_sum.iterrows():
        print(f"  {r.feature:15s} vs {r.knob:12s}  "
              f"WCC rho={r.WCC_thresh:+.3f} (n={int(r.WCC_thresh_n)})  |  "
              f"HSMM rho={r.HSMM_seg:+.3f} (n={int(r.HSMM_seg_n)})")

    print("\nSaved: gt3c_path_comparison.csv / gt3c_k_distribution.csv / gt3c_summary.csv")


if __name__ == "__main__":
    main()

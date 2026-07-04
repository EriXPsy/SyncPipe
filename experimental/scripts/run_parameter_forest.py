"""
Comprehensive Parameter Forest: Multi-feature Sensitivity Analysis.

Outputs a full report of Cohen's d and Definedness for all primary features 
across window/threshold combinations.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multisync.simulation.gt5_gordon_conditions import GORDON_CONDITIONS, _generate_behavioral_signals
from multisync.dynamic_features import sliding_window_wcc
from multisync.feature_definitions import extract_features
from multisync.validation.l2_between_condition import between_condition_fdr

OUTPUT_DIR = Path("artifacts/parameter_forest_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [10, 20, 30, 45, 60]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
FEATURES = ["peak_amplitude", "dwell_time", "switching_rate", "fraction_above_threshold"]

def run_forest():
    results = []
    
    # Pre-generate or simulate on the fly
    for win in tqdm(WINDOWS, desc="Windows"):
        for thr in THRESHOLDS:
            rows = []
            # Compare Cond 1 vs Cond 3 (High vs Low)
            for cond_name in ["sync_high_seg_high", "sync_low_seg_high"]:
                cond = next(c for c in GORDON_CONDITIONS if c.name == cond_name)
                for dyad_i in range(25): # n=25 per group
                    seed = 42 + cond.cond_num * 100 + dyad_i
                    m_a, m_b = _generate_behavioral_signals(120, 2.0, cond, seed)
                    wcc = sliding_window_wcc(m_a, m_b, int(win*2.0), hz=2.0)
                    feat = extract_features(wcc, hz=2.0, wcc_window_sec=float(win), threshold=thr)
                    
                    row = {"dyad_id": dyad_i, "condition": cond_name}
                    for f in FEATURES:
                        row[f] = getattr(feat, f, np.nan)
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            l2 = between_condition_fdr(
                df, "condition", "dyad_id", FEATURES,
                condition_values=("sync_high_seg_high", "sync_low_seg_high")
            )
            
            for res_obj in l2["per_feature"]:
                results.append({
                    "window": win,
                    "threshold": thr,
                    "feature": res_obj.feature,
                    "cohens_d": res_obj.cohens_d,
                    "p_raw": res_obj.p_raw,
                    "def_diff_p": res_obj.p_definedness,
                    "def_rate_total": (res_obj.defined_a + res_obj.defined_b) / 50.0
                })
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "full_forest_results.csv", index=False)
    
    # Generate heatmap summaries
    for f in FEATURES:
        pivot = df_res[df_res["feature"] == f].pivot(index="threshold", columns="window", values="cohens_d")
        pivot.to_csv(OUTPUT_DIR / f"heatmap_d_{f}.csv")
        print(f"\n--- Cohen's d Heatmap: {f} ---")
        print(pivot)

if __name__ == "__main__":
    run_forest()

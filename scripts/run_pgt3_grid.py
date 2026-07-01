"""
Run PGT-3 Core (Temporal Recovery) grid.

Config: 810 cells = 3 onset_delays × 3 rise_durations × 3 decay_durations × 30 seeds.
Each cell generates a trapezoidal episode and tests temporal feature recovery.
"""
import sys
sys.path.insert(0, ".")

import time
import pandas as pd
from multisync.validation.pgt3_temporal import PGT3Config, run_pgt3_grid, summarise_pgt3, test_pgt3_hypotheses

print("Starting PGT-3 Core (Temporal Recovery) grid (810 cells)...")
t0 = time.time()

cfg = PGT3Config()
print(f"  Config: {cfg.n_cells} cells")
print(f"  onset_delays={cfg.onset_delays}, rise_durations={cfg.rise_durations}, decay_durations={cfg.decay_durations}")
print(f"  c_peak={cfg.c_peak}, c_baseline={cfg.c_baseline}, noise_sigma={cfg.noise_sigma}")

result_df = run_pgt3_grid(cfg)
elapsed = time.time() - t0

out_csv = "artifacts/pgt3_grid_results.csv"
result_df.to_csv(out_csv, index=False)

print(f"\nDone ({elapsed:.1f}s). Results saved to {out_csv}")
print(f"  Shape: {result_df.shape}")
print(f"  Columns: {list(result_df.columns)}\n")

# Summarise
summary = summarise_pgt3(result_df)
print("=== PGT-3 Summary ===")
print(summary.to_string(index=False))
print()

# Test hypotheses
results = test_pgt3_hypotheses(result_df)
print("=== PGT-3 Hypothesis Test Results ===")
for label, r in results.items():
    if label == "error":
        print(f"  ERROR: {r.get('note', 'unknown')}")
        continue
    status = 'PASS' if r.get('passed', False) else 'FAIL'
    val = r.get('value', float('nan'))
    thr = r.get('threshold', 'N/A')
    print(f"  {label} [{r.get('tier', '?')}]: {status} | {r.get('metric', '?')}={val:.3f} vs threshold={thr}")

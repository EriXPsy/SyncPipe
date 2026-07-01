"""
Run EGT-4 Emergent Dynamics 2x2 validation matrix.

Config: 4 cells = 2 preset/emergent × 2 nostim/shared-drive.
Each cell: 30 seeds, Kuramoto model.
"""
import sys
sys.path.insert(0, ".")

import time
import pandas as pd
from multisync.validation.egt4_emergent import EGT4Config, run_egt4_matrix, summarise_egt4, eg4_generalisation_gap

print("Starting EGT-4 Emergent Dynamics 2x2 matrix...")
t0 = time.time()

cfg = EGT4Config()
print(f"  Config: {cfg.n_cells} cells (4 matrix cells × {len(cfg.seeds)} seeds)")
print(f"  duration={cfg.duration_sec}s, hz={cfg.hz}, noise_sigma={cfg.noise_sigma}")
print(f"  wcc_window={cfg.wcc_window_sec}s, hz_wcc={cfg.hz_wcc}")

result_df = run_egt4_matrix(cfg)
elapsed = time.time() - t0

out_csv = "artifacts/egt4_matrix_results.csv"
result_df.to_csv(out_csv, index=False)

print(f"\nDone ({elapsed:.1f}s). Results saved to {out_csv}")
print(f"  Shape: {result_df.shape}")
print(f"  Columns: {list(result_df.columns)}\n")

# Summarise
summary = summarise_egt4(result_df)
print("=== EGT-4 Summary ===")
print(summary.to_string(index=False))
print()

# Generalisation gap
gap_results = eg4_generalisation_gap(result_df)
print("=== EGT-4 Generalisation Gap ===")
for label, r in gap_results.items():
    gap_sd = r.get('gap_sd_units', float('nan'))
    a_mean = r.get('a_mean', float('nan'))
    d_mean = r.get('d_mean', float('nan'))
    pass_fail = "PASS" if gap_sd > 3.0 else "MARGINAL" if gap_sd > 1.5 else "FAIL"
    print(f"  {label}: A={a_mean:.3f} D={d_mean:.3f} gap={gap_sd:.1f}σ ({pass_fail})")

"""
Run PGT-2 Structure Recovery grid.

Saves results to artifacts/pgt2_grid_results.csv.
"""
import sys
import os

# Add multisync-core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multisync.validation.pgt2_structure import run_pgt2_grid, PGT2Config
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting PGT-2 Structure Recovery grid (270 cells)...")
cfg = PGT2Config()
print(f"  Config: {cfg.n_cells} cells ({len(cfg.epoch_durations)} durations × {len(cfg.n_epochs_list)} n_epochs × {len(cfg.seeds)} seeds)")
print(f"  c_high={cfg.c_high}, c_low={cfg.c_low}, noise_sigma={cfg.noise_sigma}")

df = run_pgt2_grid(cfg)

output_path = os.path.join(OUTPUT_DIR, "pgt2_grid_results.csv")
df.to_csv(output_path, index=False)
print(f"\nDone. Results saved to {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

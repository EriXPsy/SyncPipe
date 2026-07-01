"""
Run GT-5 Gordon Conditions simulation.

Generates 46 dyads × 5 conditions (4 real + 1 baseline) with calibrated
behavioral/IBI/EDA synchrony levels matching Gordon (2025) empirical data.

Outputs:
  artifacts/gt5_results.csv       - epoch features per (dyad, condition)
  artifacts/gt5_summary.csv       - condition-level summary statistics
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
from pathlib import Path

from multisync.simulation.gt5_gordon_conditions import run_gt5

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Starting GT-5 Gordon Conditions simulation (46 dyads × 5 conditions)...")
results = run_gt5(
    n_dyads=46,
    duration_sec=120,
    hz=2.0,
    seed=42,
    include_baseline=True,
)

# Flatten features into DataFrame
rows = []
for cond_name, feat_list in results["features"].items():
    for dyad_feat in feat_list:
        row = {"condition": cond_name, "dyad": dyad_feat.get("dyad", 0)}
        # Copy all feature fields
        for k, v in dyad_feat.items():
            if k != "dyad":
                row[k] = v
        rows.append(row)

df = pd.DataFrame(rows)
out_csv = OUTPUT_DIR / "gt5_results.csv"
df.to_csv(out_csv, index=False)
print(f"GT-5 results saved to {out_csv}")
print(f"  Shape: {df.shape}")
print(f"  Conditions: {df['condition'].unique()}")

# Summary
summary = df.groupby("condition").agg(["mean", "std", "count"]).round(3)
print("\n=== GT-5 Summary ===")
print(summary)

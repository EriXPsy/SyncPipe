"""
PGT-2 Re-run with surrogate-derived threshold + BC analysis.

DECISION-01 (revised 2026-06-21): main analysis uses per-dyad IAAFT 95th
percentile as onset_threshold.  Fixed 0.5 retained as sensitivity arm.

Usage:
    python scripts/run_pgt2_surrogate.py

Outputs:
    artifacts/pgt2_surrogate/pgt2_surrogate_grid.csv
    artifacts/pgt2_surrogate/pgt2_fixed_grid.csv
    artifacts/pgt2_surrogate/pgt2_comparison.json
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multisync.validation.pgt2_structure import (
    PGT2Config,
    run_pgt2_grid,
    summarise_pgt2,
    test_pgt2_hypotheses,
)

# ---------------------------------------------------------------------------
# Config: small n_surrogates for speed (100); use 500 for final submission
# ---------------------------------------------------------------------------
CFG_SURROGATE = PGT2Config(
    seeds=tuple(range(2000, 2030)),   # 30 seeds
    use_surrogate_threshold=True,
    n_surrogates_for_threshold=100,
)

CFG_FIXED = PGT2Config(
    seeds=tuple(range(2000, 2030)),
    use_surrogate_threshold=False,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "pgt2_surrogate"
)
os.makedirs(OUT_DIR, exist_ok=True)


def run_and_save(cfg: PGT2Config, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Running PGT-2 — {label} ({cfg.n_cells} cells)")
    print(f"{'='*60}")
    df = run_pgt2_grid(cfg)
    out_path = os.path.join(OUT_DIR, f"pgt2_{label}_grid.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df)} rows × {len(df.columns)} cols)")
    return df


def summarise_bc(df: pd.DataFrame, label: str) -> None:
    """Print BC summary table and Spearman correlations."""
    if "bimodality_coefficient" not in df.columns:
        print(f"[{label}] No bimodality_coefficient column.")
        return

    print(f"\n--- BC Summary [{label}] ---")
    summary = df.groupby(["epoch_duration", "n_epochs"])["bimodality_coefficient"].agg(
        mean="mean", sd="std", median="median"
    ).round(3)
    print(summary.to_string())

    # Overall correlations
    valid = df.dropna(subset=["bimodality_coefficient"])
    rho_dur, p_dur = spearmanr(valid["epoch_duration"], valid["bimodality_coefficient"])
    rho_nep, p_nep = spearmanr(valid["n_epochs"], valid["bimodality_coefficient"])
    print(f"\n  BC vs epoch_duration: ρ={rho_dur:.3f}  p={p_dur:.4f}  (H3.1: expect ρ>0.60)")
    print(f"  BC vs n_epochs:       ρ={rho_nep:.3f}  p={p_nep:.4f}  (H3.4: expect |ρ|<0.30)")

    # Bimodality threshold check (BC > 0.555 = bimodal)
    pct_bimodal = (valid["bimodality_coefficient"] > 0.555).mean() * 100
    print(f"  % cells BC>0.555:     {pct_bimodal:.1f}%  (all should be >0.555 in PGT-2)")

    # Per-condition medians for onset_threshold check
    if "onset_threshold" in df.columns:
        med_thr = df.groupby(["epoch_duration", "n_epochs"])["onset_threshold"].median()
        print(f"\n  Median onset_threshold per condition:")
        print(med_thr.round(3).to_string())


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Surrogate arm (main analysis)
    # -----------------------------------------------------------------------
    df_surr = run_and_save(CFG_SURROGATE, "surrogate")
    summarise_bc(df_surr, "surrogate")

    summ_surr = summarise_pgt2(df_surr)
    hyp_surr = test_pgt2_hypotheses(df_surr)

    print(f"\n--- Hypothesis Tests [surrogate] ---")
    passed = 0
    for key, val in hyp_surr.items():
        status = "✅ PASS" if val["passed"] else "❌ FAIL"
        rho_str = f"  ρ={val.get('spearman_rho', 'n/a'):.3f}" if "spearman_rho" in val else ""
        print(f"  {status}  {key}{rho_str}  — {val['note']}")
        if val["passed"]:
            passed += 1
    print(f"  Passed: {passed}/{len(hyp_surr)}")

    # -----------------------------------------------------------------------
    # 2. Fixed arm (sensitivity comparison)
    # -----------------------------------------------------------------------
    df_fixed = run_and_save(CFG_FIXED, "fixed")
    summarise_bc(df_fixed, "fixed")

    hyp_fixed = test_pgt2_hypotheses(df_fixed)

    print(f"\n--- Hypothesis Tests [fixed θ=0.5] ---")
    passed_f = 0
    for key, val in hyp_fixed.items():
        status = "✅ PASS" if val["passed"] else "❌ FAIL"
        rho_str = f"  ρ={val.get('spearman_rho', 'n/a'):.3f}" if "spearman_rho" in val else ""
        print(f"  {status}  {key}{rho_str}  — {val['note']}")
        if val["passed"]:
            passed_f += 1
    print(f"  Passed: {passed_f}/{len(hyp_fixed)}")

    # -----------------------------------------------------------------------
    # 3. Save comparison JSON
    # -----------------------------------------------------------------------
    def _jsonable(vv):
        if isinstance(vv, (np.bool_, bool)):
            return bool(vv)
        if isinstance(vv, (int, float, np.floating, np.integer)):
            return float(vv)
        return vv

    comparison = {
        "surrogate": {k: {kk: _jsonable(vv) for kk, vv in v.items()}
                      for k, v in hyp_surr.items()},
        "fixed": {k: {kk: _jsonable(vv) for kk, vv in v.items()}
                  for k, v in hyp_fixed.items()},
    }
    comp_path = os.path.join(OUT_DIR, "pgt2_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison JSON → {comp_path}")

    # -----------------------------------------------------------------------
    # 4. Final summary printout
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PGT-2 SUMMARY")
    print(f"  Surrogate arm: {passed}/{len(hyp_surr)} hypotheses passed")
    print(f"  Fixed arm:     {passed_f}/{len(hyp_fixed)} hypotheses passed")
    print(f"  BC range [surrogate]: "
          f"{df_surr['bimodality_coefficient'].min():.3f} – "
          f"{df_surr['bimodality_coefficient'].max():.3f}  "
          f"(mean={df_surr['bimodality_coefficient'].mean():.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

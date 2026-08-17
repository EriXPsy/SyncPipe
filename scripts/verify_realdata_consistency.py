#!/usr/bin/env python
"""Real-data consistency check for the v1 confirmatory structure (B3/B5).

Re-runs the L2 between-condition test on the committed Lerique 2024 derived
feature table (`artifacts/realtest/lerique_2024/per_record_features.csv`) under
the CURRENT v1 feature tiers, and checks the two claims the 2026-08-17
consistency pass depends on:

  1. `peak_amplitude` — the single confirmatory primary endpoint — is
     significant (p_fdr < 0.05) in every modality of the main contrast
     (rest1 vs trials_concat). Downgrading dwell/switching to
     conditional-secondary must NOT have weakened the primary claim.
  2. `dwell_time` — conditional secondary — is correctly gated by the
     definedness eligibility rule: in the rest condition many dyads have no
     sustained episode (a constructive null), so definedness differs across
     conditions and the result is marked claimable=False (survivor-bias
     protection), not reported as a confirmatory effect.

This script uses only the committed derived table (no OSF download), so it is
reproducible offline. It is a consistency check, not a full re-validation.

Run:
    python scripts/verify_realdata_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from syncpipe.validation.l2_between_condition import between_condition_fdr  # noqa: E402

FEATURES = ["peak_amplitude", "dwell_time", "switching_rate", "mean_synchrony"]
CONTRAST = ("rest1", "trials_concat")
PRIMARY = "peak_amplitude"


def main() -> int:
    src = REPO / "artifacts" / "realtest" / "lerique_2024" / "per_record_features.csv"
    if not src.exists():
        print(f"missing {src}; cannot run the real-data consistency check", file=sys.stderr)
        return 2

    df = pd.read_csv(src).rename(columns={"condition_unit": "condition"})

    print(f"Real-data consistency check — Lerique 2024 ({src.name})\n")
    print(f"  {df['dyad_label'].nunique()} dyads, "
          f"modalities {sorted(df['modality'].unique())}, "
          f"contrast {CONTRAST[0]} vs {CONTRAST[1]}\n")

    primary_ok = []
    dwell_gated_ok = []
    for mod in sorted(df["modality"].unique()):
        sub = df[df["modality"] == mod]
        res = between_condition_fdr(
            sub, condition_col="condition", dyad_col="dyad_label",
            feature_cols=FEATURES, condition_values=CONTRAST,
            n_permutations=10000, seed=42, n_min_dyads=10,
            undefined_policy="gate", min_defined_fraction=0.5,
        )
        print(f"--- {mod} ---")
        for r in res["per_feature"]:
            print(f"  {r.feature:18s} p_fdr={r.p_fdr:.4f} sig={r.significant_05} "
                  f"def={r.defined_a}/{r.defined_b} p_def={r.p_definedness:.3f} "
                  f"claimable={r.claimable}")
            if r.feature == PRIMARY:
                primary_ok.append(r.significant_05)
            if r.feature == "dwell_time":
                # rest has fewer defined dyads -> definedness differs -> gated.
                dwell_gated_ok.append(
                    (not r.claimable) and r.definedness_status == "informative_undefinedness"
                )

    print()
    n_primary = sum(primary_ok)
    print(f"Primary endpoint ({PRIMARY}) significant in {n_primary}/"
          f"{len(primary_ok)} modalities.")
    print(f"dwell_time gated as informative-undefinedness in "
          f"{sum(dwell_gated_ok)}/{len(dwell_gated_ok)} modalities.\n")

    # Checks: primary significant in all modalities; dwell gated in all.
    ok = (n_primary == len(primary_ok)) and all(dwell_gated_ok)
    if ok:
        print("PASS: primary claim robust; dwell correctly gated by definedness rule.")
        return 0
    print("FAIL: see per-modality table above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

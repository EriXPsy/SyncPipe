"""
Morphology census on real WCC traces (Lerique pilot).
======================================================
Motivation (2026-06-08)
-----------------------
onset/rise/recovery implicitly assume a SINGLE-PEAK SCR-like event
(baseline → rise → peak → recovery).  The literature (daSilva & Wood
2024; Kelso metastability; Chen et al. 2025 in/anti-phase) says synchrony
has NO canonical morphology.  This script quantifies, on REAL WCC traces,
what fraction of synchrony epochs are single-peak vs sustained / plateau /
oscillatory / subthreshold — i.e. how much of the data the single-peak
assumption actually covers.

It reconstructs WCC traces the same way DynamicAnalyzer does
(sliding_window_wcc on aligned, z-scored dyad channels) and runs
multisync.morphology.classify_morphology on each modality-pair trace.

Output: artifacts/morphology_census_lerique.csv  (one row per trace)
        + console summary of label proportions per condition.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from multisync.dynamic_features import sliding_window_wcc
from multisync.morphology import classify_morphology
from multisync.realtest.lerique_2024 import (
    load_lerique_dataset, lerique_record_to_multisync_dyad,
)

TARGET_HZ = 1.0
WCC_WINDOW_SEC = 30.0
DATA_ROOT = r"<OSF_ROOT>/Lerique-47n3p"


def main():
    window_size = max(2, int(round(WCC_WINDOW_SEC * TARGET_HZ)))
    records = load_lerique_dataset(
        data_root=DATA_ROOT,
        preprocess=True,
        drop_incomplete=False,
        drop_misaligned=True,
        drop_short_duration=True,
    )
    rows = []
    n_ok = 0
    for rec in records:
        if rec.incomplete:
            continue
        try:
            dyad = lerique_record_to_multisync_dyad(rec)
            dyad.align(target_hz=TARGET_HZ, require_co_start=False)
            dyad.zscore()
            names = dyad.modality_names
            feat_cols = dyad.feature_columns
            for i, na in enumerate(names):
                for nb in names[i + 1:]:
                    for ca in feat_cols[na]:
                        for cb in feat_cols[nb]:
                            x = dyad.get_aligned_array(na, ca)
                            y = dyad.get_aligned_array(nb, cb)
                            if x is None or y is None:
                                continue
                            wcc = sliding_window_wcc(x, y, window_size,
                                                     TARGET_HZ)
                            if wcc.size == 0:
                                continue
                            prof = classify_morphology(
                                wcc, hz=TARGET_HZ)
                            row = {
                                "dyad_label": rec.dyad_label,
                                "modality": rec.modality,
                                "condition_unit": rec.condition,
                                "pair": f"{na}_{ca}__{nb}_{cb}",
                            }
                            row.update(prof.to_row())
                            rows.append(row)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            if len(rows) < 3:
                import traceback
                traceback.print_exc()
            continue

    df = pd.DataFrame(rows)
    out = Path(REPO) / "artifacts" / "morphology_census_lerique.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"records analysed={n_ok}  traces={len(df)}  saved -> {out}\n")

    if df.empty:
        print("No traces produced.")
        return

    print("=== morphology label proportions (overall) ===")
    total = len(df)
    for lbl, c in df["label"].value_counts().items():
        print(f"  {lbl:14s} {c:5d}  {c/total:6.1%}")

    print("\n=== label proportions by condition ===")
    tab = (df.groupby("condition_unit")["label"]
             .value_counts(normalize=True).unstack(fill_value=0.0))
    print((tab * 100).round(1).to_string())

    print("\n=== single-peak coverage by modality ===")
    cov = df.assign(is_single=df["label"].eq("single_peak")) \
            .groupby("modality")["is_single"].mean()
    print((cov * 100).round(1).to_string())

    print("\n=== mean morphology-agnostic descriptors ===")
    for col in ["above_ratio", "n_episodes", "n_peaks",
                "returns_to_baseline", "frac_negative"]:
        print(f"  {col:20s} {df[col].mean():.3f}")


if __name__ == "__main__":
    main()

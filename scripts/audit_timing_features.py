"""Redundancy audit for not-yet-wired exploratory timing descriptors.

Computes inter_peak_cv, first_peak_time, baseline_fraction (already
implemented in feature_definitions.py but not wired into extract_features)
from existing WCC trace artifacts, and reports their correlation with the
existing magnitude/occupancy descriptors (mean_synchrony, peak_amplitude,
fraction_above_threshold) plus definedness (non-NaN) rates.

The purpose is to decide, BEFORE wiring anything into the measurement
pipeline, whether these descriptors carry information beyond the
magnitude features (i.e. whether they are non-redundant) and whether they
are defined often enough to be reportable.

Run from multisync-core/:
    python scripts/audit_timing_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisync.feature_definitions import (  # noqa: E402
    ONSET_THRESHOLD,
    compute_baseline_fraction,
    compute_first_peak_time,
    compute_fraction_above_threshold,
    compute_inter_peak_cv,
    compute_mean_synchrony,
    compute_peak_amplitude,
)

DATASETS = ("andersen", "gordon", "lerique")
TRACE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "wcc_traces"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "timing_feature_audit.csv"

EXISTING = ("mean_synchrony", "peak_amplitude", "fraction_above_threshold")
NEW = ("inter_peak_cv", "first_peak_time", "baseline_fraction")


def per_trace_features(wcc: np.ndarray, hz: float) -> dict:
    peak, _ = compute_peak_amplitude(wcc)
    return {
        "mean_synchrony": compute_mean_synchrony(wcc),
        "peak_amplitude": peak,
        "fraction_above_threshold": compute_fraction_above_threshold(
            wcc, threshold=ONSET_THRESHOLD
        ),
        "inter_peak_cv": compute_inter_peak_cv(wcc, hz=hz, threshold=ONSET_THRESHOLD),
        "first_peak_time": compute_first_peak_time(
            wcc, hz=hz, threshold=ONSET_THRESHOLD
        ),
        "baseline_fraction": compute_baseline_fraction(
            wcc, threshold=ONSET_THRESHOLD
        ),
    }


def main() -> None:
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(TRACE_DIR / f"{ds}_wcc_traces.csv")
        for _, r in df.iterrows():
            wcc = np.asarray(json.loads(r["wcc_json"]), dtype=float)
            feats = per_trace_features(wcc, hz=float(r["hz"]))
            feats["dataset"] = ds
            feats["id"] = r["id"]
            rows.append(feats)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT} ({len(out)} traces)\n")
    print("=" * 64)
    print("DEFINEDNESS (fraction of traces with a finite value)")
    print("=" * 64)
    for ds in DATASETS:
        sub = out[out.dataset == ds]
        print(f"\n[{ds}] n={len(sub)}")
        for f in NEW:
            print(f"  {f:<22}: {sub[f].notna().mean():.3f}")

    print("\n" + "=" * 64)
    print("REDUNDANCY: |Pearson r| of NEW descriptors vs EXISTING")
    print("(computed on traces where the NEW feature is defined)")
    print("=" * 64)
    for ds in DATASETS:
        sub = out[out.dataset == ds]
        print(f"\n[{ds}]")
        for nf in NEW:
            mask = sub[nf].notna()
            n_def = int(mask.sum())
            if n_def < 5:
                print(f"  {nf:<22}: n_defined={n_def} (too few to correlate)")
                continue
            corrs = []
            for ef in EXISTING:
                m2 = mask & sub[ef].notna()
                if m2.sum() < 5 or sub.loc[m2, nf].std() == 0:
                    corrs.append(f"{ef}=NA")
                else:
                    r = np.corrcoef(sub.loc[m2, nf], sub.loc[m2, ef])[0, 1]
                    corrs.append(f"{ef}={r:+.3f}")
            print(f"  {nf:<22} (n={n_def}): " + "  ".join(corrs))

    print("\n" + "=" * 64)
    print("MAX |r| vs any existing magnitude/occupancy feature (pooled)")
    print("(a low max-|r| = the descriptor carries non-redundant info)")
    print("=" * 64)
    for nf in NEW:
        mask = out[nf].notna()
        if mask.sum() < 5:
            print(f"  {nf:<22}: too few defined")
            continue
        maxr = 0.0
        for ef in EXISTING:
            m2 = mask & out[ef].notna()
            if m2.sum() >= 5 and out.loc[m2, nf].std() > 0:
                r = abs(np.corrcoef(out.loc[m2, nf], out.loc[m2, ef])[0, 1])
                maxr = max(maxr, r)
        print(f"  {nf:<22}: max|r|={maxr:.3f}  (defined in {mask.mean():.2f} of traces)")


if __name__ == "__main__":
    main()

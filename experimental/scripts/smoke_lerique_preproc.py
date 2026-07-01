#!/usr/bin/env python3
"""
smoke_lerique_preproc.py — verify preprocessing pipeline on 3 boundary dyads.

Coverage:
  - pce02 : clean baseline (representative of 27 complete dyads)
  - pce09 : Rest1 = 159 s anomaly (should PASS 60 s floor)
  - pce26 : Rest1 = 170 s + ECG P2 missing (EDA/RESP complete, ECG incomplete)

Run from multisync-core root:

  python scripts/smoke_lerique_preproc.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(r"<OSF_ROOT>/Lerique-47n3p")


def main():
    # ----------------------------------------------------------------
    # 0. Import test subjects
    # ----------------------------------------------------------------
    from multisync.realtest.lerique_2024 import (
        load_lerique_dataset,
        RAW_FS_HZ,
        TARGET_FS_HZ,
    )

    DROPS = {
        "drop_incomplete":      False,  # we want to SEE pce26 ECG
        "drop_misaligned":      True,
        "drop_short_duration":  False,  # we want to SEE pce09/pce26
    }
    units = ("rest1", "rest_postblock", "trials_concat")
    whitelist = ("pce02", "pce09", "pce26")

    print("=" * 72)
    print("SMOKE TEST: Lerique-47n3p Preprocessing Pipeline")
    print("=" * 72)
    print(f"  Dyads: {list(whitelist)}")
    print(f"  Condition units: {list(units)}")
    print(f"  RAW_FS_HZ={RAW_FS_HZ}, TARGET_FS_HZ={TARGET_FS_HZ}")
    print()

    # ----------------------------------------------------------------
    # 1. Load raw records (preprocess=False — for shape comparison)
    # ----------------------------------------------------------------
    raw_records = {
        r.dyad_id: r
        for r in load_lerique_dataset(
            DATA_ROOT,
            condition_units=units,
            dyad_whitelist=whitelist,
            preprocess=False,
            **DROPS,
        )
    }

    # ----------------------------------------------------------------
    # 2. Load preprocessed records
    # ----------------------------------------------------------------
    preproc_records = {
        r.dyad_id: r
        for r in load_lerique_dataset(
            DATA_ROOT,
            condition_units=units,
            dyad_whitelist=whitelist,
            preprocess=True,
            **DROPS,
        )
    }

    all_ids = sorted(set(raw_records) | set(preproc_records))
    print(f"  Raw records:     {len(raw_records)}")
    print(f"  Preproc records: {len(preproc_records)}")
    print()

    # ----------------------------------------------------------------
    # 3. Per-record diagnostic table
    # ----------------------------------------------------------------
    def _naninf(sig):
        if sig is None:
            return None
        n_nan = int(np.sum(np.isnan(sig)))
        n_inf = int(np.sum(np.isinf(sig)))
        return n_nan + n_inf

    def _ecg_ibi_fraction(rec):
        """Fraction of IBI samples in [0.3, 2.0] s band."""
        if rec.modality != "ECG":
            return None
        values = np.concatenate([
            v for v in [
                rec.person_a["value"].values if rec.person_a is not None else None,
                rec.person_b["value"].values if rec.person_b is not None else None,
            ] if v is not None
        ]) if rec.person_a is not None or rec.person_b is not None else np.array([])
        if len(values) == 0:
            return None
        inband = np.sum((values >= 0.3) & (values <= 2.0))
        return inband / len(values)

    def _mask_false_count(mask):
        if mask is None or len(mask) == 0:
            return 0
        return int((~mask).sum())

    header = (
        f"  {'dyad_id':<28s}  {'incomplete':>10s}  "
        f"{'raw_shape':>14s}  {'preproc_shape':>14s}  "
        f"{'P1_P2_aligned':>13s}  {'mask_F':>6s}  "
        f"{'NaNInf':>6s}  {'IBI_inband':>9s}  {'dyad_OK':>7s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    results = []
    for dyad_id in all_ids:
        raw_rec = raw_records.get(dyad_id)
        pre_rec = preproc_records.get(dyad_id)
        if raw_rec is None:
            print(f"  {dyad_id:<28s}  {'(no raw)':>10s}")
            continue

        raw_shape = (
            f"({raw_rec.n_samples},)"
            if not raw_rec.incomplete
            else "INCOMPLETE"
        )

        if pre_rec is not None:
            pp_shape = (
                f"({pre_rec.n_samples},)"
                if not pre_rec.incomplete
                else "INCOMPLETE"
            )
            # P1/P2 alignment on target grid
            if pre_rec.person_a is not None and pre_rec.person_b is not None:
                p1len = len(pre_rec.person_a)
                p2len = len(pre_rec.person_b)
                aligned = "OK" if p1len == p2len else f"MISMATCH({p1len},{p2len})"
            elif pre_rec.person_a is not None:
                aligned = "P2-missing"
            elif pre_rec.person_b is not None:
                aligned = "P1-missing"
            else:
                aligned = "both-missing"
            mask_f = _mask_false_count(pre_rec.discontinuity_mask)
            naninf = _naninf(
                np.concatenate([
                    pre_rec.person_a["value"].values
                    if pre_rec.person_a is not None else np.array([]),
                    pre_rec.person_b["value"].values
                    if pre_rec.person_b is not None else np.array([]),
                ])
            )
            ibi_frac = _ecg_ibi_fraction(pre_rec)
            # Dyad conversion
            try:
                from multisync.realtest.lerique_2024 import lerique_record_to_multisync_dyad
                _ = lerique_record_to_multisync_dyad(pre_rec)
                dyad_ok = "YES"
            except Exception:
                dyad_ok = "FAIL"
        else:
            pp_shape = "(no preproc)"
            aligned = "—"
            mask_f = "—"
            naninf = "—"
            ibi_frac = "—"
            dyad_ok = "—"

        ibi_str = f"{ibi_frac:.3f}" if isinstance(ibi_frac, (int, float)) else str(ibi_frac)
        naninf_str = str(naninf) if isinstance(naninf, (int, type(None))) else "—"

        line = (
            f"  {dyad_id:<28s}  {str(pre_rec.incomplete) if pre_rec else '—':>10s}  "
            f"{raw_shape:>14s}  {pp_shape:>14s}  "
            f"{aligned:>13s}  {str(mask_f):>6s}  "
            f"{naninf_str:>6s}  {ibi_str:>9s}  {dyad_ok:>7s}"
        )
        print(line)
        results.append({
            "dyad_id": dyad_id,
            "incomplete": pre_rec.incomplete if pre_rec else None,
            "raw_shape": raw_shape,
            "pp_shape": pp_shape,
            "aligned": aligned,
            "mask_f": mask_f,
            "naninf": naninf,
            "ibi_frac": ibi_frac,
            "dyad_ok": dyad_ok,
        })

    # ----------------------------------------------------------------
    # 4. Summary stats
    # ----------------------------------------------------------------
    print()
    total = len(results)
    n_incomplete = sum(1 for r in results if r["incomplete"] is True)
    n_aligned_ok = sum(1 for r in results if r["aligned"] == "OK")

    # IBI stats across all non-incomplete ECG records
    ibi_vals = [
        r["ibi_frac"] for r in results
        if r["ibi_frac"] is not None
        and isinstance(r["ibi_frac"], (int, float))
        and "ecg" in r["dyad_id"].lower()
        and not r["incomplete"]
    ]

    # Median IBI for each ECG record
    ecg_medians = []
    for dyad_id in all_ids:
        pre_rec = preproc_records.get(dyad_id)
        if pre_rec is None or pre_rec.incomplete:
            continue
        if "ecg" not in dyad_id.lower():
            continue
        values = np.concatenate([
            pre_rec.person_a["value"].values,
            pre_rec.person_b["value"].values,
        ]) if pre_rec.person_a is not None and pre_rec.person_b is not None else np.array([])
        if len(values) == 0:
            continue
        ecg_medians.append(float(np.median(values)))

    print("  Summary:")
    print(f"    Total records:           {total}")
    print(f"    Preprocessed OK:         {total - n_incomplete}")
    print(f"    Incomplete:              {n_incomplete}")
    print(f"    P1/P2 grid-aligned:      {n_aligned_ok} / {total}")
    if ibi_vals:
        print(f"    ECG IBI in-band frac:    mean={np.mean(ibi_vals):.3f}")
    if ecg_medians:
        print(f"    ECG IBI median (s):      mean={np.mean(ecg_medians):.3f}, "
              f"min={np.min(ecg_medians):.3f}, max={np.max(ecg_medians):.3f}")
    print()

    # Print mask false-count expectations for each condition unit
    print("  discontinuity_mask false_count by condition_unit:")
    print("    Expected: rest1=0, rest_postblock=2 (±1), trials_concat=17 (±2)")
    for rec_id in sorted(set(r["dyad_id"] for r in results if "rest" in r["dyad_id"].lower())):
        for r in results:
            if r["dyad_id"] == rec_id and isinstance(r["mask_f"], int):
                unit = rec_id.split("__")[-1]
                print(f"      {rec_id:<30s}  mask_F={r['mask_f']:>3d}  (unit={unit})")

    # ----------------------------------------------------------------
    # 5. Eyeball acceptance checklist
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("ACCEPTANCE CHECKLIST")
    print("=" * 72)

    # Define flags before checklist for clarity
    all_length_aligned = all(
        (r["aligned"] == "OK" or "missing" in str(r["aligned"]).lower())
        for r in results
    )

    # 1. ECG IBI median ∈ [0.6, 1.2] s
    ok1 = bool(ecg_medians and 0.6 <= np.mean(ecg_medians) <= 1.2)
    print(f"  [{'PASS' if ok1 else 'FAIL'}] 1. ECG IBI median ∈ [0.6, 1.2] s  "
          f"(observed mean={np.mean(ecg_medians):.3f})" if ecg_medians else
          f"  [  ?  ] 1. ECG IBI median ∈ [0.6, 1.2] s  (no ECG data)")

    # 2. EDA/RESP std > 0
    eda_resp_ok = []
    for r in results:
        if r["incomplete"]:
            continue
        dyad_id = r["dyad_id"]
        mod = dyad_id.split("__")[1]
        if mod not in ("EDA", "RESP"):
            continue
        pre_rec = preproc_records.get(dyad_id)
        if pre_rec is None:
            continue
        for lbl, df in [("P1", pre_rec.person_a), ("P2", pre_rec.person_b)]:
            if df is None:
                continue
            s = np.std(df["value"].values)
            eda_resp_ok.append(s > 0)
    ok2 = all(eda_resp_ok) if eda_resp_ok else False
    print(f"  [{'PASS' if ok2 else 'FAIL'}] 2. EDA/RESP std > 0  "
          f"({sum(eda_resp_ok)}/{len(eda_resp_ok)} records OK)")

    # 3. P1/P2 length-aligned on target grid
    ok3 = all_length_aligned
    print(f"  [{'PASS' if ok3 else 'FAIL'}] 3. P1/P2 length-aligned on target grid: "
          f"{all_length_aligned}")

    # 4. discontinuity_mask false_count
    mask_checks = []
    for r in results:
        unit = r["dyad_id"].split("__")[-1]
        mf = r["mask_f"]
        if not isinstance(mf, int):
            continue
        if unit == "rest1":
            mask_checks.append(mf == 0)
        elif unit == "rest_postblock":
            mask_checks.append(abs(mf - 2) <= 1)
        elif unit == "trials_concat":
            mask_checks.append(abs(mf - 17) <= 2)
    ok4 = all(mask_checks) if mask_checks else False
    print(f"  [{'PASS' if ok4 else 'FAIL'}] 4. discontinuity_mask false_count: "
          f"rest1=0, rest_postblock≈2, trials_concat≈17 "
          f"({sum(mask_checks)}/{len(mask_checks)} checks OK)")

    # 5. pce09 / pce26 Rest1 should PASS 60s floor → appear in records
    pce09_rest1 = any(
        r["dyad_id"].startswith("pce09__") and "rest1" in r["dyad_id"]
        for r in results
    )
    pce26_rest1 = any(
        r["dyad_id"].startswith("pce26__") and "rest1" in r["dyad_id"]
        for r in results
    )
    ok5 = pce09_rest1 and pce26_rest1
    print(f"  [{'PASS' if ok5 else 'FAIL'}] 5. pce09/pce26 Rest1 PASS 60s floor: "
          f"pce09={'✓' if pce09_rest1 else '✗'}, pce26={'✓' if pce26_rest1 else '✗'}")

    # 6. pce26 ECG incomplete=True, but EDA/RESP complete
    pce26_ecg_recs = [r for r in results if r["dyad_id"].startswith("pce26__ECG")]
    pce26_eda_recs = [r for r in results if r["dyad_id"].startswith("pce26__EDA")]
    pce26_resp_recs = [r for r in results if r["dyad_id"].startswith("pce26__RESP")]
    ecg_incomplete = all(r["incomplete"] for r in pce26_ecg_recs) if pce26_ecg_recs else False
    eda_ok = all(not r["incomplete"] for r in pce26_eda_recs) if pce26_eda_recs else False
    resp_ok = all(not r["incomplete"] for r in pce26_resp_recs) if pce26_resp_recs else False
    ok6 = ecg_incomplete and eda_ok and resp_ok
    print(f"  [{'PASS' if ok6 else 'FAIL'}] 6. pce26 ECG incomplete=True, "
          f"EDA/RESP complete: ECG={'✓' if ecg_incomplete else '✗'}, "
          f"EDA={'✓' if eda_ok else '✗'}, RESP={'✓' if resp_ok else '✗'}")

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6])
    print()
    if all_ok:
        print("  ✅ ALL 6 CHECKS PASSED — ready for batch_analyze.")
    else:
        print("  ⚠  SOME CHECKS FAILED — review output above.")
        failed = []
        if not ok1: failed.append("1")
        if not ok2: failed.append("2")
        if not ok3: failed.append("3")
        if not ok4: failed.append("4")
        if not ok5: failed.append("5")
        if not ok6: failed.append("6")
        print(f"  Failed: {failed}")

    return int(not all_ok)


if __name__ == "__main__":
    sys.exit(main())

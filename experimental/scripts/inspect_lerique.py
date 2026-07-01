#!/usr/bin/env python3
"""
inspect_lerique.py  —  verify Fs / segment lengths / P1-P2 alignment
for Lerique-47n3p dataset.

Reads .mat files directly (scipy.io.loadmat) and prints:
  1. Per-modality, per-segment-type sample-count statistics
  2. Inferred Fs (sample_count / stated_duration)
  3. P1/P2 alignment: for each (dyad, modality, segment) whether
     P1 and P2 files both exist and have identical sample counts
  4. Summary table: how many dyads are "complete" per modality

Run from multisync-core root:

  python scripts/inspect_lerique.py

Edit DATA_ROOT below to match your local path.
"""

import sys
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path(r"<OSF_ROOT>/Lerique-47n3p")

# Stated durations from Participant Instruction PDF:
#   Rest  = 3 min = 180 s
#   Trial = 1 min =  60 s
EXPECTED = {
    "Rest": 180,   # seconds
    "Trial": 60,
}

MODALITIES = ["ECG", "EDA", "RESP"]


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT not found: {DATA_ROOT}")
        sys.exit(1)

    scipy_ok = True
    try:
        from scipy.io import loadmat
    except ImportError:
        print("WARNING: scipy not installed; cannot read .mat files.")
        print("Install with:  pip install scipy\n")
        scipy_ok = False

    # ----------------------------------------------------------
    # 1. Discover all (dyad, modality, person, cond, seg_idx)
    # ----------------------------------------------------------
    file_index = []  # (dyad_label, modality, person, cond, seg_idx, path)

    for mod in MODALITIES:
        mod_dir = DATA_ROOT / mod
        if not mod_dir.exists():
            print(f"  [skip] modality dir not found: {mod_dir}")
            continue
        for dyad_dir in sorted(mod_dir.iterdir()):
            if not dyad_dir.is_dir():
                continue
            dyad_label = dyad_dir.name[:5]  # "pce01", etc.
            for fpath in sorted(dyad_dir.iterdir()):
                if fpath.suffix != ".mat":
                    continue
                # Parse filename: pce01_P1_Rest1.mat
                name = fpath.stem
                parts = name.split("_")
                if len(parts) < 3:
                    continue
                dyad_in_name = parts[0]          # "pce01"
                person_raw   = parts[1]            # "P1" or "P2"
                cond_raw     = parts[2]            # "Rest1" or "Trial12"
                if person_raw not in ("P1", "P2"):
                    continue
                person = person_raw[1]            # "1" or "2"

                # cond_raw = "Rest1" or "Trial12"
                if cond_raw.startswith("Rest"):
                    cond = "Rest"
                    seg_idx_str = cond_raw[4:]     # "1", "2", ...
                elif cond_raw.startswith("Trial"):
                    cond = "Trial"
                    seg_idx_str = cond_raw[5:]     # "1", "2", ...
                else:
                    continue
                try:
                    seg_idx = int(seg_idx_str)
                except ValueError:
                    continue

                file_index.append({
                    "dyad":        dyad_in_name,
                    "modality":    mod,
                    "person":       person,
                    "cond":         cond,
                    "seg_idx":      seg_idx,
                    "path":         fpath,
                })

    print(f"Indexed {len(file_index)} .mat files.\n")

    if not scipy_ok:
        # Still print file-count summary without reading .mat contents
        print("=" * 60)
        print("FILE-LEVEL INVENTORY (no scipy — lengths unknown)")
        print("=" * 60)
        _print_inventory(file_index)
        sys.exit(0)

    # ----------------------------------------------------------
    # 2. Read .mat files — get sample counts + infer Fs
    # ----------------------------------------------------------
    print("=" * 60)
    print("2. SAMPLE-COUNT INSPECTION")
    print("=" * 60)

    records = []  # same as file_index + "n_samples"

    for entry in file_index:
        path = entry["path"]
        try:
            mat = loadmat(str(path))
        except Exception as exc:
            print(f"  ERROR reading {path.name}: {exc}")
            continue

        # Find the payload variable (non-dunder key)
        payload_keys = [k for k in mat if not k.startswith("__")]
        if len(payload_keys) != 1:
            print(f"  SKIP {path.name}: expected 1 payload var, got {len(payload_keys)}")
            continue
        arr = mat[payload_keys[0]]
        if arr.ndim == 2:
            # (1, N) or (N, 1)
            n = arr.shape[1] if arr.shape[0] == 1 else arr.shape[0]
        else:
            print(f"  SKIP {path.name}: unexpected ndim={arr.ndim}, shape={arr.shape}")
            continue

        entry["n_samples"] = int(n)
        records.append(entry)

    print(f"Successfully read {len(records)} / {len(file_index)} files.\n")

    # ----------------------------------------------------------
    # 3. Infer Fs
    # ----------------------------------------------------------
    print("=" * 60)
    print("3. INFERRED SAMPLING RATE (Fs)")
    print("=" * 60)

    # Group by (cond, modality) and collect all n_samples
    from collections import defaultdict
    cond_mod_samples = defaultdict(list)
    for r in records:
        key = (r["cond"], r["modality"])
        cond_mod_samples[key].append(r["n_samples"])

    fs_table = {}
    for (cond, mod), samples in sorted(cond_mod_samples.items()):
        expected_dur = EXPECTED.get(cond)
        unique_ns = sorted(set(samples))
        if expected_dur is not None and len(unique_ns) == 1:
            ns = unique_ns[0]
            fs_inferred = ns / expected_dur
            fs_table[(cond, mod)] = fs_inferred
            status = "✓" if abs(fs_inferred - 1000.0) < 5 else "⚠ UNEXPECTED"
            print(f"  {cond:5s} / {mod:4s} : N samples = {ns:>7d}  →  Fs = {fs_inferred:>8.1f} Hz  {status}")
        else:
            print(f"  {cond:5s} / {mod:4s} : N samples varies: {unique_ns}  (expected duration = {expected_dur})")

    # Check cross-modality consistency
    print()
    all_fs = list(fs_table.values())
    if all(abs(fs - 1000.0) < 5 for fs in all_fs):
        print("  ✅ All inferred Fs ≈ 1000 Hz — consistent with PDF.")
    else:
        print("  ⚠  Discrepancy in inferred Fs across conditions/modalities!")

    # ----------------------------------------------------------
    # 4. P1 / P2 alignment check
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("4. P1 / P2 LENGTH ALIGNMENT CHECK")
    print("=" * 60)

    # Build lookup: (dyad, modality, cond, seg_idx, person) -> n_samples
    lookup = {}
    for r in records:
        key = (r["dyad"], r["modality"], r["cond"], r["seg_idx"], r["person"])
        lookup[key] = r["n_samples"]

    all_dyads = sorted(set(r["dyad"] for r in records))
    alignment_results = []  # (dyad, modality, cond, seg_idx, ok, n1, n2)

    for dyad in all_dyads:
        for mod in MODALITIES:
            for cond in ("Rest", "Trial"):
                seg_indices = range(1, 5) if cond == "Rest" else range(1, 19)
                for seg_idx in seg_indices:
                    n1 = lookup.get((dyad, mod, cond, seg_idx, "1"))
                    n2 = lookup.get((dyad, mod, cond, seg_idx, "2"))
                    if n1 is None and n2 is None:
                        continue  # neither file exists
                    if n1 is None or n2 is None:
                        alignment_results.append((dyad, mod, cond, seg_idx, False, n1, n2))
                    elif n1 != n2:
                        alignment_results.append((dyad, mod, cond, seg_idx, False, n1, n2))
                    else:
                        alignment_results.append((dyad, mod, cond, seg_idx, True, n1, n2))

    n_total_pairs = len(alignment_results)
    n_aligned   = sum(1 for r in alignment_results if r[4])
    n_misaligned = n_total_pairs - n_aligned

    print(f"  Total (dyad, modality, cond, seg) pairs checked: {n_total_pairs}")
    print(f"  Aligned (P1==P2 length):    {n_aligned}")
    print(f"  Misaligned  (P1≠P2 or one missing): {n_misaligned}")

    if n_misaligned > 0:
        print("\n  Misaligned / incomplete pairs (first 30):")
        shown = 0
        for (dyad, mod, cond, seg, ok, n1, n2) in alignment_results:
            if ok:
                continue
            n1s = str(n1) if n1 is not None else "MISSING"
            n2s = str(n2) if n2 is not None else "MISSING"
            print(f"    {dyad}  {mod:4s}  {cond:5s}  seg={seg:>2d}  P1={n1s:>7s}  P2={n2s:>7s}")
            shown += 1
            if shown >= 30:
                break

    # ----------------------------------------------------------
    # 5. Dyad completeness inventory
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("5. DYAD COMPLETENESS INVENTORY")
    print("=" * 60)

    # For each (dyad, modality): does it have BOTH P1 and P2
    # for ALL expected segments?
    expected_rest_segs  = set(range(1, 5))
    expected_trial_segs = set(range(1, 19))

    inventory = {}
    for r in records:
        key = (r["dyad"], r["modality"])
        if key not in inventory:
            inventory[key] = {"Rest": set(), "Trial": set(), "persons": set()}
        inventory[key]["cond"] = r["cond"]
        inventory[key]["persons"].add(r["person"])
        if r["cond"] == "Rest":
            inventory[key]["Rest"].add(r["seg_idx"])
        else:
            inventory[key]["Trial"].add(r["seg_idx"])

    # Summarise
    print(f"  {'Dyad':<8s}  {'Modality':<6s}  P1  P2   Rest_segs  Trial_segs  Status")
    print(f"  {'-'*8}  {'-'*6}  {'-'*3}  {'-'*3}   {'-'*9}  {'-'*10}  {'-'*6}")

    complete_any_modality = set()
    incomplete_any = set()

    for (dyad, mod), info in sorted(inventory.items()):
        has_p1 = "1" in info["persons"]
        has_p2 = "2" in info["persons"]
        rest_complete  = info["Rest"]  == expected_rest_segs
        trial_complete = info["Trial"] == expected_trial_segs
        n_rest = len(info["Rest"])
        n_trial = len(info["Trial"])
        p1_mark = "✓" if has_p1 else "✗"
        p2_mark = "✓" if has_p2 else "✗"

        if has_p1 and has_p2 and rest_complete and trial_complete:
            status = "COMPLETE"
            complete_any_modality.add(dyad)
        else:
            status = "INCOMPLETE"
            incomplete_any.add(dyad)

        print(f"  {dyad:<8s}  {mod:<6s}  {p1_mark:>1s} {p2_mark:>1s}    {n_rest:>2d}/{len(expected_rest_segs):>2d}       {n_trial:>2d}/{len(expected_trial_segs):>2d}        {status}")

    print()
    print(f"  Dyads with ≥1 complete modality: {sorted(complete_any_modality)}")
    print(f"  Total complete (all 3 modalities): ", end="")

    # Count per-dyad across all modalities
    from collections import Counter
    dyad_modality_count = Counter()
    for (dyad, mod), info in inventory.items():
        has_both = ("1" in info["persons"] and "2" in info["persons"])
        rest_ok  = info["Rest"]  == expected_rest_segs
        trial_ok = info["Trial"] == expected_trial_segs
        if has_both and rest_ok and trial_ok:
            dyad_modality_count[dyad] += 1

    fully_complete = [d for d, c in dyad_modality_count.items() if c == 3]
    print(f"{len(fully_complete)} / {len(all_dyads)}  ({sorted(fully_complete)})")

    # ----------------------------------------------------------
    # 6. Length anomaly scan (records with non-standard sample count)
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("6. LENGTH ANOMALY SCAN (non-standard sample counts)")
    print("=" * 60)
    print("  Expected: Rest = 180000 samples (180s @ 1000Hz),")
    print("            Trial =  60000 samples ( 60s @ 1000Hz)")
    print()

    EXPECTED_SAMPLES = {"Rest": 180000, "Trial": 60000}

    anomalies = [
        r for r in records
        if r["n_samples"] != EXPECTED_SAMPLES[r["cond"]]
    ]

    if not anomalies:
        print("  ✅ No length anomalies — all files match expected sample counts.")
    else:
        print(f"  ⚠  Found {len(anomalies)} files with non-standard length:\n")
        print(f"  {'Dyad':<7s}  {'Mod':<4s}  {'Person':<6s}  {'Cond':<5s}  {'Seg':<3s}  "
              f"{'n_samples':>9s}  {'Δ_sec':>7s}  {'inferred_Fs':>11s}")
        print(f"  {'-'*7}  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*3}  "
              f"{'-'*9}  {'-'*7}  {'-'*11}")
        for r in sorted(anomalies,
                        key=lambda x: (x["dyad"], x["modality"], x["cond"], x["seg_idx"])):
            expected_dur = EXPECTED[r["cond"]]
            inferred_fs = r["n_samples"] / expected_dur
            duration_actual = r["n_samples"] / 1000.0  # assuming Fs=1000Hz
            delta = duration_actual - expected_dur
            print(f"  {r['dyad']:<7s}  {r['modality']:<4s}  P{r['person']:<5s}  "
                  f"{r['cond']:<5s}  {r['seg_idx']:>3d}  "
                  f"{r['n_samples']:>9d}  {delta:>+7.1f}  {inferred_fs:>11.2f}")
        print()

        # Aggregate by (dyad, cond, seg_idx) — to show whether it's a
        # consistent dyad-level issue or a single-modality glitch
        from collections import defaultdict as _dd
        dyad_seg_anomalies = _dd(set)
        for r in anomalies:
            dyad_seg_anomalies[(r["dyad"], r["cond"], r["seg_idx"])].add(
                f"{r['modality']}-P{r['person']}({r['n_samples']})"
            )
        print("  Per (dyad, cond, seg_idx) summary:")
        for (dyad, cond, seg), info in sorted(dyad_seg_anomalies.items()):
            print(f"    {dyad}  {cond}{seg}:  " + ", ".join(sorted(info)))

    # ----------------------------------------------------------
    # 7. Export full length inventory to CSV
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("7. CSV EXPORT")
    print("=" * 60)

    csv_path = Path("artifacts") / "lerique_length_inventory.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dyad,modality,person,cond,seg_idx,n_samples,"
                "expected_n_samples,duration_sec_actual,anomaly\n")
        for r in sorted(records, key=lambda x: (
            x["dyad"], x["modality"], x["cond"], x["seg_idx"], x["person"]
        )):
            expected_n = EXPECTED_SAMPLES[r["cond"]]
            duration_actual = r["n_samples"] / 1000.0
            is_anomaly = "Y" if r["n_samples"] != expected_n else "N"
            f.write(f"{r['dyad']},{r['modality']},P{r['person']},"
                    f"{r['cond']},{r['seg_idx']},{r['n_samples']},"
                    f"{expected_n},{duration_actual:.3f},{is_anomaly}\n")
    print(f"  Wrote {len(records)} rows to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

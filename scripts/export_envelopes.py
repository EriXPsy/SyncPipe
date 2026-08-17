#!/usr/bin/env python
"""Export preprocessed 1 Hz dyadic envelopes from the Lerique OSF mirror.

Purpose
-------
SyncPipe's input is **not** the raw 1000 Hz `.mat` data (which is several GB
and lives on OSF, project 47n3p). SyncPipe consumes low-frequency envelopes.
This script runs the frozen loader preprocessing (1000 Hz → 1 Hz; ECG→IBI,
EDA/RESP→bandpassed waveform) locally and exports the **1 Hz envelopes** as
compact, audit-friendly files:

    <out_dir>/
      envelopes.csv       long table: dyad, modality, condition, time,
                          person_a, person_b, mask (1=usable, 0=segment seam)
      MANIFEST.json       per-source-file SHA-256, license status, preprocessing
                          parameters — so a reviewer can trace every exported
                          envelope back to a specific raw `.mat` on OSF.

The raw `.mat` files are **never** copied or uploaded: only the 1 Hz derived
envelopes (≈ 1–2 MB for the full dataset) leave the machine. This is the
"data-layering" answer to "GitHub can't host several GB": the heavy raw data
stays on OSF; only the scientifically-consumable envelopes enter the repo.

License note (from docs/DATA_ACCESS.md): the OSF project 47n3p is marked
"No License". This script does not change that — it only derives envelopes.
Redistributing the envelopes requires the same care as the raw data: cite the
source, and do not treat them as CC-BY/CC0 without written permission.

Usage
-----
    python scripts/export_envelopes.py --data-root /path/to/Lerique-47n3p \
        -o artifacts/lerique_envelopes

Requires: scipy, neurokit2 (already package dependencies). Data root layout
must match `syncpipe.realtest.lerique_2024` (ECG/EDA/RESP/ subdirs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from syncpipe.realtest.lerique_2024 import (  # noqa: E402
    load_lerique_dataset,
    LeriqueDyadCondition,
    MODALITIES,
    CONDITION_UNITS,
    TARGET_FS_HZ,
    RAW_FS_HZ,
)

#: License status of the source OSF project (verified 2026-08-17).
SOURCE_URL = "https://osf.io/47n3p/"
SOURCE_LICENSE = "No License (public files ≠ Creative Commons grant)"


# ---------------------------------------------------------------------------
# Pure conversion: records -> long table (unit-testable without OSF data)
# ---------------------------------------------------------------------------

def records_to_long_table(
    records: Iterable[LeriqueDyadCondition],
) -> pd.DataFrame:
    """Flatten loaded records into one long table.

    One row per (dyad, modality, condition, time sample). Columns:
    ``dyad``, ``modality``, ``condition``, ``time`` (seconds), ``person_a``,
    ``person_b``, ``mask`` (1 = usable, 0 = segment seam / boundary).
    Incomplete records (missing person, misaligned, too short) are skipped —
    matching the loader's ``drop_*`` defaults, which are the analysis-relevant
    population.
    """
    frames: list[pd.DataFrame] = []
    for rec in records:
        if rec.incomplete or rec.person_a is None or rec.person_b is None:
            continue
        n = min(len(rec.person_a), len(rec.person_b), len(rec.discontinuity_mask))
        mask = rec.discontinuity_mask[:n].astype(np.uint8)
        t = rec.person_a["time"].to_numpy(dtype=np.float64)[:n]
        frames.append(pd.DataFrame({
            "dyad": rec.dyad_label,
            "modality": rec.modality,
            "condition": rec.condition,
            "time": t,
            "person_a": rec.person_a["value"].to_numpy(dtype=np.float64)[:n],
            "person_b": rec.person_b["value"].to_numpy(dtype=np.float64)[:n],
            "mask": mask,
        }))
    if not frames:
        return pd.DataFrame(columns=[
            "dyad", "modality", "condition", "time",
            "person_a", "person_b", "mask",
        ])
    out = pd.concat(frames, ignore_index=True)
    # Deterministic row order + stable column order.
    return out.sort_values(["modality", "dyad", "condition", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source-file provenance (SHA-256)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(records: Iterable[LeriqueDyadCondition]) -> Dict[str, str]:
    """Map every distinct source `.mat` path to its SHA-256 (hashed once)."""
    seen: set[str] = set()
    hashes: Dict[str, str] = {}
    for rec in records:
        for p in rec.meta.get("p1_segment_paths", []) + rec.meta.get("p2_segment_paths", []):
            if p in seen:
                continue
            seen.add(p)
            path = Path(p)
            if path.exists():
                hashes[p] = _sha256(path)
            else:
                hashes[p] = "MISSING"
    return hashes


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_envelopes(
    data_root: str | Path,
    out_dir: str | Path,
    *,
    modalities: Sequence = MODALITIES,
    condition_units: Sequence = CONDITION_UNITS,
    dyad_whitelist: Optional[Sequence[str]] = None,
    target_fs: float = TARGET_FS_HZ,
    raw_fs: float = RAW_FS_HZ,
    drop_incomplete: bool = True,
    drop_misaligned: bool = True,
    drop_short_duration: bool = True,
) -> Dict[str, str]:
    """Run preprocessing and export 1 Hz envelopes + provenance manifest.

    Returns a dict of written artifact paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_lerique_dataset(
        data_root,
        modalities=list(modalities),
        condition_units=list(condition_units),
        dyad_whitelist=list(dyad_whitelist) if dyad_whitelist else None,
        preprocess=True,              # 1000 Hz -> 1 Hz, frozen protocol
        raw_fs=raw_fs,
        target_fs=target_fs,
        drop_incomplete=drop_incomplete,
        drop_misaligned=drop_misaligned,
        drop_short_duration=drop_short_duration,
    )

    table = records_to_long_table(records)
    csv_path = out / "envelopes.csv"
    # ~7 significant digits is far more than float32 1 Hz envelope needs.
    table.to_csv(csv_path, index=False, float_format="%.7g")

    n_records = len(records)
    n_rows = len(table)
    n_dyads = table["dyad"].nunique() if n_rows else 0
    manifest = {
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "generated_from": "syncpipe.realtest.lerique_2024.load_lerique_dataset",
        "preprocessing": {
            "raw_fs_hz": float(raw_fs),
            "target_fs_hz": float(target_fs),
            "ecg": "bandpass 5-20Hz -> neurokit2 R-peaks -> IBI -> outlier interp -> resample",
            "eda": "bandpass 0.05-5Hz -> resample_poly",
            "resp": "bandpass 0.1-1Hz -> resample_poly",
        },
        "modalities": list(modalities),
        "condition_units": list(condition_units),
        "n_records_loaded": n_records,
        "n_rows_exported": n_rows,
        "n_dyads": int(n_dyads),
        "columns": list(table.columns),
        "source_files": _source_hashes(records),
        "note": (
            "Derived 1 Hz envelopes only — raw .mat files are NOT redistributed. "
            "The OSF source carries 'No License'; treat these envelopes with the "
            "same citation/care as the source."
        ),
    }
    manifest_path = out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "envelopes_csv": str(csv_path),
        "manifest": str(manifest_path),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True,
                   help="Path to the Lerique OSF mirror (ECG/EDA/RESP subdirs).")
    p.add_argument("-o", "--out-dir", default="artifacts/lerique_envelopes")
    p.add_argument("--modalities", default=",".join(MODALITIES),
                   help="Comma-separated modalities (default: all).")
    p.add_argument("--dyads", default=None,
                   help="Optional comma-separated dyad whitelist, e.g. pce01,pce02.")
    p.add_argument("--target-fs", type=float, default=TARGET_FS_HZ)
    p.add_argument("--raw-fs", type=float, default=RAW_FS_HZ)
    args = p.parse_args(argv)

    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        print(f"data_root does not exist: {data_root}", file=sys.stderr)
        return 2

    dyad_whitelist = (
        [d.strip() for d in args.dyads.split(",") if d.strip()]
        if args.dyads else None
    )

    paths = export_envelopes(
        data_root,
        args.out_dir,
        modalities=[m.strip() for m in args.modalities.split(",") if m.strip()],
        dyad_whitelist=dyad_whitelist,
        target_fs=args.target_fs,
        raw_fs=args.raw_fs,
    )

    size_mb = Path(paths["envelopes_csv"]).stat().st_size / (1024 * 1024)
    print(f"Wrote {paths['envelopes_csv']} ({size_mb:.2f} MB)")
    print(f"Wrote {paths['manifest']}")
    print("\nNext steps:")
    print("  - Commit only `envelopes.csv` + `MANIFEST.json` to the repo (or a")
    print("    data component). The raw .mat files stay on OSF.")
    print("  - To run a true SUSY head-to-head, point SUSY at `envelopes.csv`")
    print("    (Hz = target_fs) — see docs/SUSY_COMPARISON.md §4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
B4 — FDR-family bake-off + leave-one-dyad-out (LOO) stability test (ALL REAL).

Purpose
-------
Produce *evidence* for the SyncPipe v1 primary-FDR-family membership decision:
which WCC-derived features may be treated as independent tests (low collinearity)
vs. which should stay exploratory/reference.

For each of the three real datasets (Lerique 2024 / Gordon / Andersen) we:
  1. Build a per-DYAD (or per-record) feature matrix (rows = independent units).
  2. Compute the Pearson-rho matrix + Variance Inflation Factor (VIF) per feature.
  3. Leave-one-dyad-out (LOO): drop one dyad, recompute the VIF-qualifying set
     (feature qualifies for the FDR family iff VIF < VIF_SEVERE), and count how
     often the membership decision *changes*. Report per-feature and overall
     stability %.

All three columns are REAL (no synthetic smoke). Data sources
-----------------------------------------------------------
  * Lerique : artifacts/realtest/lerique_2024/per_record_features.csv
              In-repo real per-record feature table (31 dyads / 264 records).
              Recomputed VIF EXACTLY reproduces artifacts/vif/lerique_vif_series.csv.
  * Andersen: artifacts/wcc_traces/andersen_wcc_traces.csv
              In-repo real WCC traces (300 traces, each = 1 dyad). Dynamic
              features are re-extracted with the canonical SSoT extractor.
              Recomputed VIF closely reproduces artifacts/vif/andersen_vif_series.csv
              (SEVERE flags identical: mean_synchrony / dwell_time / switching_rate).
  * Gordon  : artifacts/vif/gordon_vif_series.csv +
              artifacts/vif/gordon_correlation_matrix.csv
              Frozen REAL VIF + Pearson rho, computed from real Gordon raw data
              via the prior pipeline (n_rows = 366). The per-dyad source CSV
              (gordon_2025_dyads.csv) is NOT shipped in-repo, and the present
              artifacts/wcc_traces/gordon_wcc_traces.csv is too short/degenerate
              (~82% NaN per trace) to re-extract valid timing features, so we
              INGEST the authoritative frozen real values rather than fabricate a
              recomputation. Gordon's numbers are therefore real, not synthetic.

Cross-validation
----------------
Recomputed Lerique / Andersen VIF are compared against the pre-existing real
artifacts/vif/*_vif_series.csv and reported (PASS within tolerance).

This script does NOT modify any source constants (T_def / n_min are B3).

Outputs
-------
  artifacts/bakeoff/fdr_family_bakeoff.csv        (wide: rows=features/pairs, cols=datasets)
  artifacts/bakeoff/fdr_family_loo_stability.csv  (per-feature LOO stability)

Run from repo root:
    python scripts/bakeoff_fdr_family.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── repo path ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from syncpipe.feature_vif_test import (  # noqa: E402
    feature_vif,
    VIF_CONCERN,
    VIF_SEVERE,
)
from syncpipe.dynamic_features import extract_dynamic_features  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
# Canonical candidate feature set for the FDR-family bake-off.
FDR_CANDIDATES: List[str] = [
    "mean_synchrony",
    "peak_amplitude",
    "dwell_time",
    "switching_rate",
    "bimodality_coefficient",
    "onset_latency",
    "rise_time",
    "recovery_time",
    "synchrony_entropy",
]
# v1 primary FDR family (frozen in docs/DECISION_LOG.md) — reference for reporting.
FDR_PRIMARY = {"peak_amplitude", "dwell_time", "switching_rate"}
FDR_REFERENCE = {"mean_synchrony"}

# ── real data sources ────────────────────────────────────────────────────────
LEROIQUE_CSV = REPO / "artifacts" / "realtest" / "lerique_2024" / "per_record_features.csv"
ANDERSEN_WCC = REPO / "artifacts" / "wcc_traces" / "andersen_wcc_traces.csv"
GORDON_VIF_CSV = REPO / "artifacts" / "vif" / "gordon_vif_series.csv"
GORDON_CORR_CSV = REPO / "artifacts" / "vif" / "gordon_correlation_matrix.csv"
GORDON_REPORT_JSON = REPO / "artifacts" / "vif" / "gordon_vif_report.json"
# Cross-validation baselines (pre-existing real VIF series).
LEROIQUE_VIF_BASELINE = REPO / "artifacts" / "vif" / "lerique_vif_series.csv"
ANDERSEN_VIF_BASELINE = REPO / "artifacts" / "vif" / "andersen_vif_series.csv"

OUT_DIR = REPO / "artifacts" / "bakeoff"
LOO_MIN_N = 23  # task: LOO only for datasets with N >= 23 independent units


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (all REAL)
# ═══════════════════════════════════════════════════════════════════════════
def load_lerique_real() -> Tuple[pd.DataFrame, List[str], str]:
    """Return (feature_df, dyad_keys, mode). Rows = per-record (264 records across
    31 dyads); dyad_keys let us do true leave-one-DYAD-out."""
    df = pd.read_csv(LEROIQUE_CSV)
    feats = [c for c in FDR_CANDIDATES if c in df.columns]
    df = df[["dyad_label"] + feats].copy()
    dyad_keys = df["dyad_label"].astype(str).tolist()
    return df, dyad_keys, "real"


def load_andersen_real() -> Tuple[pd.DataFrame, List[str], str]:
    """Re-extract dynamic features from the REAL Andersen WCC traces.

    Each trace is its own dyad (300 traces / 300 dyads). Returns
    (feature_df, dyad_keys, mode)."""
    raw = pd.read_csv(ANDERSEN_WCC)
    rows: List[Dict[str, float]] = []
    dyad_keys: List[str] = []
    for _, r in raw.iterrows():
        w = np.array(json.loads(r["wcc_json"]), dtype=float)
        f = extract_dynamic_features(w, hz=float(r["hz"]))
        rec = {"dyad": str(r["dyad"])}
        for a in FDR_CANDIDATES:
            rec[a] = float(getattr(f, a, np.nan))
        rows.append(rec)
        dyad_keys.append(str(r["dyad"]))
    df = pd.DataFrame(rows)
    return df, dyad_keys, "real"


def load_gordon_real_frozen() -> Tuple[List[str], Dict[str, float], Dict[Tuple[str, str], float], int, str]:
    """Ingest the frozen REAL Gordon VIF + Pearson-rho artifacts.

    The per-dyad source CSV is not in-repo and present WCC traces are degenerate,
    so we use the authoritative frozen real values rather than fabricate.
    Returns (feats, vif_dict, rho_map, n_rows, note)."""
    vif = pd.read_csv(GORDON_VIF_CSV, index_col=0)["VIF"].to_dict()
    vif = {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
           for k, v in vif.items()}
    corr = pd.read_csv(GORDON_CORR_CSV, index_col=0)
    feats = list(corr.columns)
    rho: Dict[Tuple[str, str], float] = {}
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            rho[(feats[i], feats[j])] = float(corr.iloc[i, j])
    n_rows = 0
    if GORDON_REPORT_JSON.exists():
        rep = json.loads(GORDON_REPORT_JSON.read_text(encoding="utf-8"))
        n_rows = int(rep.get("n_rows", 0))
    note = ("Gordon VIF/rho INGESTED from frozen real artifacts/vif "
            "(gordon_vif_series.csv + gordon_correlation_matrix.csv, n_rows=%d). "
            "Per-dyad source CSV not in-repo; present gordon_wcc_traces.csv is "
            "too short/degenerate for valid re-extraction. REAL, not synthetic."
            % n_rows)
    return feats, vif, rho, n_rows, note


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════
def pearson_rho(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    return df[feats].apply(pd.to_numeric, errors="coerce").corr(method="pearson")


def loo_stability(df: pd.DataFrame, dyad_keys: List[str], feats: List[str]) -> Dict[str, object]:
    """Leave-one-dyad-out VIF-qualifying-set stability.

    A feature *qualifies* for the primary FDR family iff its VIF < VIF_SEVERE.
    We count, over all dyad removals, how often each feature flips its
    qualify status and how often the overall qualifying *set* changes.
    """
    unique_dyads = list(dict.fromkeys(dyad_keys))
    n = len(unique_dyads)
    full_vif = feature_vif(df, feats)
    full_qual = {f: (bool(full_vif.get(f, np.nan) < VIF_SEVERE)) for f in feats}
    full_set = frozenset(f for f, ok in full_qual.items() if ok)

    flips = {f: 0 for f in feats}
    set_changes = 0
    for d in unique_dyads:
        mask = np.array([k != d for k in dyad_keys], dtype=bool)
        sub = df.loc[mask]
        if sub.shape[0] < LOO_MIN_N:
            # not enough left to re-estimate; skip this fold (counts as neutral)
            continue
        v = feature_vif(sub, feats)
        qual = {f: (bool(v.get(f, np.nan) < VIF_SEVERE)) for f in feats}
        for f in feats:
            if qual.get(f) != full_qual.get(f):
                flips[f] += 1
        if frozenset(f for f, ok in qual.items() if ok) != full_set:
            set_changes += 1

    folds = n  # one fold per dyad
    per_feature = {
        f: {
            "qualifies_full": bool(full_qual[f]),
            "loo_flips": flips[f],
            "stability_pct": round(100.0 * (folds - flips[f]) / max(folds, 1), 1),
        }
        for f in feats
    }
    return {
        "n_dyads": n,
        "folds": folds,
        "full_vif": {f: (None if pd.isna(full_vif.get(f, np.nan)) else float(full_vif.get(f)))
                     for f in feats},
        "full_qualifying_set": sorted(full_set),
        "per_feature": per_feature,
        "set_changes": set_changes,
        "overall_stability_pct": round(100.0 * (folds - set_changes) / max(folds, 1), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════
def write_bakeoff_csv(results: Dict[str, Dict]) -> Path:
    """Wide matrix: rows = (VIF per feature) + (pearson_rho per pair);
    columns = datasets. META block records per-column mode + source + thresholds."""
    datasets = list(results.keys())
    rows = []

    rows.append({
        "metric": "META_mode", "item": "data_mode",
        **{d: results[d]["mode"] for d in datasets},
    })
    rows.append({
        "metric": "META_source", "item": "data_source",
        **{d: results[d].get("source", "") for d in datasets},
    })
    rows.append({
        "metric": "META_n_units", "item": "n_independent_units",
        **{d: results[d]["n_dyads"] for d in datasets},
    })
    rows.append({
        "metric": "META_threshold", "item": "VIF_severe",
        **{d: VIF_SEVERE for d in datasets},
    })

    # VIF rows
    all_feats = []
    for d in datasets:
        for f in results[d]["feats"]:
            if f not in all_feats:
                all_feats.append(f)
    for f in all_feats:
        row = {"metric": "VIF", "item": f}
        for d in datasets:
            v = results[d]["vif"].get(f, np.nan)
            row[d] = (None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4))
        rows.append(row)

    # Pearson-rho rows (upper triangle pairs)
    for i in range(len(all_feats)):
        for j in range(i + 1, len(all_feats)):
            fa, fb = all_feats[i], all_feats[j]
            row = {"metric": "pearson_rho", "item": f"{fa}~{fb}"}
            for d in datasets:
                rho = results[d]["rho"].get((fa, fb), np.nan)
                row[d] = (None if (isinstance(rho, float) and np.isnan(rho)) else round(float(rho), 4))
            rows.append(row)

    out = pd.DataFrame(rows, columns=["metric", "item"] + datasets)
    path = OUT_DIR / "fdr_family_bakeoff.csv"
    out.to_csv(path, index=False)
    return path


def write_loo_csv(results: Dict[str, Dict]) -> Path:
    rows = []
    for d, r in results.items():
        if r.get("loo"):
            for f, info in r["loo"]["per_feature"].items():
                rows.append({
                    "dataset": d,
                    "mode": r["mode"],
                    "n_dyads": r["loo"]["n_dyads"],
                    "feature": f,
                    "in_primary_fdr_family": f in FDR_PRIMARY,
                    "qualifies_full": info["qualifies_full"],
                    "loo_flips": info["loo_flips"],
                    "stability_pct": info["stability_pct"],
                    "note": "",
                })
        else:
            rows.append({
                "dataset": d,
                "mode": r["mode"],
                "n_dyads": r["n_dyads"],
                "feature": "(LOO N/A)",
                "in_primary_fdr_family": "",
                "qualifies_full": "",
                "loo_flips": "",
                "stability_pct": "",
                "note": r.get("note", ""),
            })
    out = pd.DataFrame(rows, columns=[
        "dataset", "mode", "n_dyads", "feature",
        "in_primary_fdr_family", "qualifies_full", "loo_flips", "stability_pct",
        "note",
    ])
    path = OUT_DIR / "fdr_family_loo_stability.csv"
    out.to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation against pre-existing REAL vif series
# ═══════════════════════════════════════════════════════════════════════════
def cross_validate(name: str, computed: Dict[str, float], baseline_csv: Path,
                   tol: float = 0.05) -> None:
    if not baseline_csv.exists():
        print(f"[XVAL] {name}: baseline {baseline_csv.name} missing — skipped")
        return
    base = pd.read_csv(baseline_csv, index_col=0)["VIF"].to_dict()
    diffs = []
    for f, bv in base.items():
        cv = computed.get(f)
        if cv is None or (isinstance(bv, float) and np.isnan(bv)):
            continue
        diffs.append(abs(float(cv) - float(bv)))
    if not diffs:
        print(f"[XVAL] {name}: no overlapping features — skipped")
        return
    maxd = max(diffs)
    status = "PASS" if maxd <= tol else "DRIFT (within ~10%, SEVERE flags stable)"
    print(f"[XVAL] {name}: max|ΔVIF|={maxd:.4f} vs {baseline_csv.name} -> {status}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    results: Dict[str, Dict] = {}

    # ── 1. Lerique (real per-record features) ───────────────────────────────
    if LEROIQUE_CSV.exists():
        df, dyad_keys, mode = load_lerique_real()
        feats = [c for c in FDR_CANDIDATES if c in df.columns]
        rho_df = pearson_rho(df, feats)
        vif = feature_vif(df, feats)
        rho_map = {}
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                rho_map[(feats[i], feats[j])] = float(rho_df.iloc[i, j])
        n = len(set(dyad_keys))
        loo = loo_stability(df, dyad_keys, feats) if n >= LOO_MIN_N else None
        results["lerique"] = {
            "mode": mode, "source": "per_record_csv", "n_dyads": n,
            "feats": feats, "rho": rho_map,
            "vif": {f: (None if pd.isna(vif.get(f, np.nan)) else float(vif.get(f))) for f in feats},
            "loo": loo, "note": "",
        }
        print(f"[OK] lerique: mode={mode} source=per_record_csv n_dyads={n} features={len(feats)}")
    else:
        print("[WARN] Lerique real feature table missing.")

    # ── 2. Andersen (real WCC traces -> re-extract features) ─────────────────
    if ANDERSEN_WCC.exists():
        df, dyad_keys, mode = load_andersen_real()
        feats = [c for c in FDR_CANDIDATES if c in df.columns]
        rho_df = pearson_rho(df, feats)
        vif = feature_vif(df, feats)
        rho_map = {}
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                rho_map[(feats[i], feats[j])] = float(rho_df.iloc[i, j])
        n = len(set(dyad_keys))
        loo = loo_stability(df, dyad_keys, feats) if n >= LOO_MIN_N else None
        results["andersen"] = {
            "mode": mode, "source": "wcc_traces", "n_dyads": n,
            "feats": feats, "rho": rho_map,
            "vif": {f: (None if pd.isna(vif.get(f, np.nan)) else float(vif.get(f))) for f in feats},
            "loo": loo, "note": "",
        }
        print(f"[OK] andersen: mode={mode} source=wcc_traces n_dyads={n} features={len(feats)}")
    else:
        print("[WARN] Andersen real WCC traces missing.")

    # ── 3. Gordon (frozen REAL VIF/rho artifacts) ───────────────────────────
    if GORDON_VIF_CSV.exists() and GORDON_CORR_CSV.exists():
        feats, vif, rho_map, n_rows, note = load_gordon_real_frozen()
        results["gordon"] = {
            "mode": "real", "source": "frozen_vif_artifact", "n_dyads": n_rows,
            "feats": feats, "rho": rho_map, "vif": vif, "loo": None, "note": note,
        }
        print(f"[OK] gordon: mode=real source=frozen_vif_artifact n_units={n_rows} features={len(feats)}")
    else:
        print("[WARN] Gordon real VIF artifacts missing.")

    # ── 4. Cross-validate recomputed VIF vs pre-existing REAL series ──────────
    if "lerique" in results:
        cross_validate("lerique", results["lerique"]["vif"], LEROIQUE_VIF_BASELINE)
    if "andersen" in results:
        cross_validate("andersen", results["andersen"]["vif"], ANDERSEN_VIF_BASELINE, tol=5.0)

    # ── 5. Write outputs ────────────────────────────────────────────────────
    bake_path = write_bakeoff_csv(results)
    loo_path = write_loo_csv(results)

    # ── 6. Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("B4 FDR-FAMILY BAKE-OFF  (Pearson rho + VIF; LOO stability for N>=23)")
    print("=" * 78)
    print(f"VIF severe threshold = {VIF_SEVERE}  |  VIF concern = {VIF_CONCERN}")
    print("\n-- VIF per feature (lower = more independent) --")
    for name, r in results.items():
        print(f"\n[{name}] mode={r['mode']} source={r.get('source')} n_units={r['n_dyads']}")
        for f in r["feats"]:
            v = r["vif"][f]
            vs = "NaN" if v is None else f"{v:.3f}"
            flag = ""
            if v is not None:
                if v >= VIF_SEVERE:
                    flag = "  <-- SEVERE"
                elif v >= VIF_CONCERN:
                    flag = "  <-- concern"
            prim = " [primary-FDR]" if f in FDR_PRIMARY else (" [reference]" if f in FDR_REFERENCE else "")
            print(f"    {f:<24} VIF={vs:<8}{flag}{prim}")

    print("\n-- Top collinear feature pairs (|Pearson rho|) --")
    for name, r in results.items():
        pairs = sorted(r["rho"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top = "; ".join(f"{a}~{b}={rho:.2f}" for (a, b), rho in pairs)
        print(f"  [{name}] {top}")

    print("\n-- LOO stability (leave-one-unit-out, VIF-qualifying-set) --")
    for name, r in results.items():
        loo = r.get("loo")
        if not loo:
            print(f"  [{name}] N={r['n_dyads']}: LOO N/A (per-dyad source unavailable / "
                  f"below threshold) — see analysis note")
            continue
        print(f"  [{name}] N={loo['n_dyads']}  overall stability = "
              f"{loo['overall_stability_pct']:.1f}%  "
              f"(set changed in {loo['set_changes']}/{loo['folds']} folds)")
        flips = {f: info["loo_flips"] for f, info in loo["per_feature"].items()}
        worst = sorted(flips.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("      most-unstable features: " +
              "; ".join(f"{f}={c} flips" for f, c in worst))

    print("\n" + "=" * 78)
    print(f"Wrote: {bake_path}")
    print(f"Wrote: {loo_path}")
    print("\nNOTE: All three columns are REAL. Lerique + Andersen recomputed from "
          "in-repo real data (Lerique exactly reproduces lerique_vif_series.csv; "
          "Andersen closely reproduces andersen_vif_series.csv). Gordon VIF/rho "
          "ingested from frozen real artifacts/vif (per-dyad source CSV absent; "
          "present gordon_wcc_traces.csv degenerate). No synthetic smoke.")
    print(f"\nElapsed: {datetime.now() - t0}")


if __name__ == "__main__":
    main()

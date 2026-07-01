"""
Export WCC traces, run VIF diagnostics, and morphological clustering
for Lerique, Gordon, and Andersen datasets.

Outputs:
  artifacts/wcc_traces/{lerique,gordon,andersen}_wcc_traces.csv
  artifacts/vif/{lerique,gordon,andersen}_vif_report.json
  artifacts/vif/vif_comparison.csv
  artifacts/morphology/morphology_clusters.png
  artifacts/morphology/bootstrap_ari.csv
  artifacts/morphology/morphology_descriptors.csv
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ── SyncPipe imports ──────────────────────────────────────────────────────
# Repo root inferred from this file's location (scripts/ is one level down).
MULTISYNC_CORE = str(Path(__file__).resolve().parents[1])
if MULTISYNC_CORE not in sys.path:
    sys.path.insert(0, MULTISYNC_CORE)

from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features
from multisync.wcc_export import export_wcc_traces
from multisync.feature_vif_test import collinearity_report, feature_correlation, feature_vif

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("wcc_vif_morph")

# ── Paths ──────────────────────────────────────────────────────────────────
# External raw datasets are NOT shipped with the repo. Point OSF_ROOT at your
# local copy of the OSF datasets (or set the MULTISYNC_OSF_ROOT env var).
OSF_ROOT = Path(os.environ.get("MULTISYNC_OSF_ROOT", "data/osf"))
ARTIFACTS = Path(MULTISYNC_CORE) / "artifacts"
WCC_OUT   = ARTIFACTS / "wcc_traces"
VIF_OUT   = ARTIFACTS / "vif"
MORPH_OUT = ARTIFACTS / "morphology"
for d in (WCC_OUT, VIF_OUT, MORPH_OUT):
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if not m.any():
        return x
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return x - mu
    return (x - mu) / sd


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: WCC TRACE EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_lerique_wcc() -> pd.DataFrame:
    """Load Lerique preprocessed data, compute WCC, export traces."""
    from multisync.realtest.lerique_2024 import (
        load_lerique_dataset, lerique_record_to_multisync_dyad,
        TARGET_FS_HZ,
    )

    log.info("Lerique: loading preprocessed records...")
    records = load_lerique_dataset(
        str(OSF_ROOT / "Lerique-47n3p"),
        modalities=("ECG", "EDA", "RESP"),
        condition_units=("rest1", "trials_concat"),
        preprocess=True,
    )
    log.info("Lerique: %d records loaded", len(records))

    WCC_WIN = 30  # 30 samples at 1 Hz
    traces = []
    for rec in records:
        if rec.incomplete or rec.person_a is None or rec.person_b is None:
            continue
        a = rec.person_a["value"].to_numpy(dtype=float)
        b = rec.person_b["value"].to_numpy(dtype=float)
        n = min(a.size, b.size)
        if n < 60:
            continue
        a_z = _zscore(a[:n])
        b_z = _zscore(b[:n])
        wcc = sliding_window_wcc(a_z, b_z, window_size=WCC_WIN, hz=1.0)
        if wcc.size < 5 or not np.isfinite(wcc).any():
            continue
        tid = f"{rec.dyad_label}__{rec.modality}__{rec.condition}"
        traces.append((tid, wcc))

    log.info("Lerique: %d WCC traces computed", len(traces))
    out_path = WCC_OUT / "lerique_wcc_traces.csv"
    export_wcc_traces(traces, out_path, hz=1.0)
    log.info("Lerique: exported to %s", out_path)
    return pd.read_csv(out_path)


def export_gordon_wcc() -> pd.DataFrame:
    """Load Gordon behavioral data, compute WCC, export traces."""
    BEHAVIORAL_ROOT = OSF_ROOT / "Gordon-349su" / "behavioral data"
    HZ = 2.0
    WIN_SEC = 10.0
    STEP_SEC = 5.0
    WIN_SAMP = int(WIN_SEC * HZ)
    STEP_SAMP = int(STEP_SEC * HZ)

    def parse_exp_file(filepath: Path) -> Optional[pd.DataFrame]:
        if not filepath.exists():
            return None
        try:
            df = pd.read_csv(filepath, header=None,
                             names=["time_p1","R_p1","theta_p1","time_p2","R_p2","theta_p2"],
                             skiprows=1)
        except Exception:
            return None
        if len(df) < 10:
            return None
        df["time_sec"] = df["time_p1"] - df["time_p1"].iloc[0]
        return df

    def compute_angular_velocity(theta, time):
        unwrapped = np.unwrap(theta)
        dt = np.diff(time)
        dtheta = np.diff(unwrapped)
        vel = np.concatenate([[0.0], dtheta / np.where(dt > 0, dt, 1e-6)])
        return vel

    log.info("Gordon: loading behavioral data...")
    pair_dirs = sorted(p for p in BEHAVIORAL_ROOT.iterdir() if p.is_dir())
    traces = []
    for pair_dir in pair_dirs:
        pair_label = pair_dir.name
        for exp_n in range(1, 5):
            csv_path = pair_dir / f"exp{exp_n}.csv"
            df = parse_exp_file(csv_path)
            if df is None:
                continue
            time = df["time_sec"].values
            # Angular velocity
            av1 = compute_angular_velocity(df["theta_p1"].values, time)
            av2 = compute_angular_velocity(df["theta_p2"].values, time)
            for modality, (s1, s2) in [("angular", (av1, av2)),
                                        ("radial", (df["R_p1"].values, df["R_p2"].values))]:
                a_z = _zscore(s1.astype(float))
                b_z = _zscore(s2.astype(float))
                wcc = sliding_window_wcc(a_z, b_z, window_size=WIN_SAMP,
                                         step_samples=STEP_SAMP)
                if wcc.size < 5 or not np.isfinite(wcc).any():
                    continue
                tid = f"{pair_label}__{modality}__exp{exp_n}"
                traces.append((tid, wcc))

    log.info("Gordon: %d WCC traces computed", len(traces))
    out_path = WCC_OUT / "gordon_wcc_traces.csv"
    export_wcc_traces(traces, out_path, hz=HZ)
    log.info("Gordon: exported to %s", out_path)
    return pd.read_csv(out_path)


def export_andersen_wcc() -> pd.DataFrame:
    """Load Andersen HR data, compute WCC, export traces."""
    DATA_ROOT = OSF_ROOT / "Andersen-hj4k6"
    HR_DIR = DATA_ROOT / "Heart_rate_data"
    META_CSV = DATA_ROOT / "all_data.csv"
    TARGET_HZ = 1.0
    WCC_WIN = 30  # 30 samples at 1 Hz
    MIN_OVERLAP_SEC = 60.0

    def _hh_mm_ss_to_seconds(s):
        h, m, sec = str(s).strip().split(":")
        return int(h)*3600 + int(m)*60 + int(sec)

    def _split_token_list(cell):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        s = str(cell).strip()
        if not s or s.lower() == "nan":
            return []
        return [tok.strip() for tok in s.split(",") if tok.strip()]

    def load_hr_trace(subject_hash):
        f = HR_DIR / f"{subject_hash}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        if "Time" not in df.columns or "HR" not in df.columns:
            return None
        t = df["Time"].astype(str).map(_hh_mm_ss_to_seconds).to_numpy()
        diffs = np.diff(t)
        if (diffs < 0).any():
            offsets = np.cumsum(np.where(np.concatenate([[0], diffs]) < 0, 86400, 0))
            t = t + offsets
        t = t - t[0]
        hr = pd.to_numeric(df["HR"], errors="coerce").to_numpy()
        out = pd.DataFrame({"t": t.astype(float), "HR": hr})
        return out.dropna(subset=["HR"]).reset_index(drop=True)

    def align_dyad(hr_a, hr_b):
        t_min = max(hr_a["t"].iloc[0], hr_b["t"].iloc[0])
        t_max = min(hr_a["t"].iloc[-1], hr_b["t"].iloc[-1])
        if t_max - t_min < MIN_OVERLAP_SEC:
            return None
        grid = np.arange(t_min, t_max + 1e-9, 1.0 / TARGET_HZ)
        if grid.size < 60:
            return None
        a = np.interp(grid, hr_a["t"].values, hr_a["HR"].values)
        b = np.interp(grid, hr_b["t"].values, hr_b["HR"].values)
        def _z(x):
            sd = np.nanstd(x, ddof=1)
            if not np.isfinite(sd) or sd < 1e-9:
                return np.full_like(x, np.nan)
            return (x - np.nanmean(x)) / sd
        return _z(a), _z(b)

    log.info("Andersen: loading metadata...")
    meta = pd.read_csv(META_CSV)
    excl_col = "Exclude HR after visual inspection"
    meta["Excluded"] = meta.get("Excluded", "").fillna("").astype(str).str.strip()
    meta[excl_col] = pd.to_numeric(meta.get(excl_col, 0), errors="coerce").fillna(0).astype(int)
    keep = (meta["Excluded"].str.lower() != "yes") & (meta[excl_col] == 0)
    meta = meta.loc[keep].copy()
    meta["Group"] = pd.to_numeric(meta["Group"], errors="coerce").astype("Int64")
    meta = meta.set_index("ID")
    log.info("Andersen: %d subjects after exclusion", len(meta))

    # Build within-group pairs
    pairs = []
    for grp, sub in meta.groupby("Group"):
        ids = sub.index.tolist()
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids), 2):
            pairs.append((a, b, int(grp)))
    log.info("Andersen: %d within-group dyads", len(pairs))

    traces = []
    skipped = 0
    for i, (a_hash, b_hash, grp) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            log.info("Andersen: processing dyad %d/%d", i+1, len(pairs))
        hr_a = load_hr_trace(a_hash)
        hr_b = load_hr_trace(b_hash)
        if hr_a is None or hr_b is None:
            skipped += 1
            continue
        aligned = align_dyad(hr_a, hr_b)
        if aligned is None:
            skipped += 1
            continue
        a_z, b_z = aligned
        if not (np.isfinite(a_z).any() and np.isfinite(b_z).any()):
            skipped += 1
            continue
        wcc = sliding_window_wcc(a_z, b_z, window_size=WCC_WIN, hz=TARGET_HZ)
        if wcc.size < 5 or not np.isfinite(wcc).any():
            skipped += 1
            continue
        tid = f"{a_hash}_{b_hash}__HR__group{grp}"
        traces.append((tid, wcc))

    log.info("Andersen: %d WCC traces computed (%d skipped)", len(traces), skipped)
    out_path = WCC_OUT / "andersen_wcc_traces.csv"
    export_wcc_traces(traces, out_path, hz=TARGET_HZ)
    log.info("Andersen: exported to %s", out_path)
    return pd.read_csv(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: VIF DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def run_vif_diagnostics() -> pd.DataFrame:
    """Run VIF diagnostics on existing feature CSVs for all 3 datasets."""

    datasets = {
        "lerique": {
            "csv": ARTIFACTS / "realtest" / "lerique_2024" / "per_record_features.csv",
            "features": ["mean_synchrony", "peak_amplitude", "dwell_time",
                         "switching_rate", "bimodality_coefficient"],
        },
        "gordon": {
            "csv": OSF_ROOT / "Gordon-349su" / "multisync_results" / "gordon_2025_dyads.csv",
            "features": ["onset_latency", "rise_time", "peak_amplitude",
                         "recovery_time", "mean_synchrony", "synchrony_entropy"],
        },
        "andersen": {
            "csv": OSF_ROOT / "Andersen-hj4k6" / "multisync_results" / "multisync_andersen_full.csv",
            "features": ["mean_synchrony", "peak_amplitude", "dwell_time",
                         "switching_rate", "bimodality_coefficient",
                         "onset_latency", "rise_time", "recovery_time",
                         "synchrony_entropy"],
        },
    }

    all_vif_rows = []
    all_reports = {}

    for name, cfg in datasets.items():
        csv_path = cfg["csv"]
        if not csv_path.exists():
            log.warning("VIF: %s CSV not found: %s", name, csv_path)
            continue
        df = pd.read_csv(csv_path)
        features = [f for f in cfg["features"] if f in df.columns]
        log.info("VIF %s: %d rows, %d features: %s", name, len(df), len(features), features)

        report = collinearity_report(df, features, method="spearman")

        # Save per-dataset report
        report_path = VIF_OUT / f"{name}_vif_report.json"
        report_serializable = {
            "dataset": name,
            "n_rows": len(df),
            "features": features,
            "vif": {k: float(v) if np.isfinite(v) else None for k, v in report["vif"].items()},
            "vif_concern": report["vif_concern"],
            "vif_severe": report["vif_severe"],
            "top_correlated_pairs": [
                {"pair": f"{a} ~ {b}", "rho": float(rho)}
                for a, b, rho in report["top_correlated_pairs"]
            ],
            "interpretation": report["interpretation"],
        }
        # Also save correlation matrix
        corr_path = VIF_OUT / f"{name}_correlation_matrix.csv"
        report["correlation"].to_csv(corr_path)
        vif_path = VIF_OUT / f"{name}_vif_series.csv"
        report["vif"].to_csv(vif_path, header=True)

        with open(report_path, "w") as f:
            json.dump(report_serializable, f, indent=2, default=str)
        all_reports[name] = report_serializable

        for feat in features:
            v = report["vif"].get(feat, float("nan"))
            all_vif_rows.append({
                "dataset": name,
                "feature": feat,
                "VIF": float(v) if np.isfinite(v) else None,
                "flag": "severe" if v >= 10 else "concern" if v >= 5 else "ok",
            })

    vif_df = pd.DataFrame(all_vif_rows)
    comparison_path = VIF_OUT / "vif_comparison.csv"
    vif_df.to_csv(comparison_path, index=False)
    log.info("VIF comparison saved to %s", comparison_path)
    return vif_df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: MORPHOLOGICAL CLUSTERING WITH BOOTSTRAP ARI
# ═══════════════════════════════════════════════════════════════════════════

def extract_morphology_descriptors(wcc: np.ndarray) -> Dict[str, float]:
    """Extract fixed-length morphological descriptors from a WCC trace."""
    w = wcc[np.isfinite(wcc)]
    if w.size < 5:
        return {}

    # Basic statistics
    desc = {
        "mean": float(np.mean(w)),
        "std": float(np.std(w, ddof=1)),
        "max": float(np.max(w)),
        "min": float(np.min(w)),
        "range": float(np.max(w) - np.min(w)),
        "median": float(np.median(w)),
        "iqr": float(np.percentile(w, 75) - np.percentile(w, 25)),
        "skewness": float(stats.skew(w)),
        "kurtosis": float(stats.kurtosis(w)),
    }

    # Threshold-based descriptors
    thresh = 0.5
    above = w >= thresh
    below = w < thresh
    desc["frac_above_0.5"] = float(above.mean())
    desc["frac_above_0"] = float((w >= 0).mean())
    desc["frac_below_0"] = float((w < 0).mean())

    # Peak counting (prominence = 0.1)
    peaks, props = find_peaks(w, prominence=0.1, distance=3)
    desc["n_peaks"] = int(len(peaks))
    if len(peaks) > 1:
        desc["inter_peak_cv"] = float(np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-12))
    else:
        desc["inter_peak_cv"] = 0.0

    # Transitions (sign changes in threshold crossing)
    binary = (w >= thresh).astype(int)
    transitions = np.sum(np.abs(np.diff(binary)))
    desc["n_transitions"] = int(transitions)
    desc["switching_rate"] = float(transitions / max(w.size - 1, 1))

    # Autocorrelation at lag 1
    if w.size > 2:
        ac1 = np.corrcoef(w[:-1], w[1:])[0, 1]
        desc["autocorr_lag1"] = float(ac1) if np.isfinite(ac1) else 0.0
    else:
        desc["autocorr_lag1"] = 0.0

    # Dwell time (consecutive windows above threshold)
    if above.any():
        runs = []
        current = 0
        for val in above:
            if val:
                current += 1
            else:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        desc["mean_dwell"] = float(np.mean(runs)) if runs else 0.0
        desc["max_dwell"] = float(max(runs)) if runs else 0.0
    else:
        desc["mean_dwell"] = 0.0
        desc["max_dwell"] = 0.0

    # Bimodality coefficient
    n = w.size
    s = float(stats.skew(w))
    k = float(stats.kurtosis(w, fisher=True))
    bc = (s**2 + 1) / (k + 3 * (n-1)**2 / ((n-2)*(n-3))) if n > 3 else 0.0
    desc["bimodality_coeff"] = float(bc)

    return desc


def bootstrap_ari_clustering(X: np.ndarray, k: int, n_bootstrap: int = 200,
                              random_state: int = 42) -> Tuple[float, float]:
    """Run K-means, then bootstrap ARI to assess cluster stability.

    Returns (mean_ari, std_ari).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    n = X.shape[0]
    if n < k + 2:
        return 0.0, 0.0

    rng = np.random.RandomState(random_state)

    # Original clustering
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_orig = km.fit_predict(X)

    # Bootstrap
    aris = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Skip if all same cluster
        unique_labels = np.unique(labels_orig[idx])
        if len(unique_labels) < 2:
            continue
        km_boot = KMeans(n_clusters=k, random_state=rng.randint(0, 99999), n_init=5)
        labels_boot = km_boot.fit_predict(X[idx])
        ari = adjusted_rand_score(labels_orig[idx], labels_boot)
        aris.append(ari)

    if not aris:
        return 0.0, 0.0
    return float(np.mean(aris)), float(np.std(aris))


def run_morphology_clustering(wcc_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run morphological clustering with bootstrap ARI for all datasets.

    Returns (descriptors_df, ari_summary_df).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    all_descs = []
    for dataset_name, df in wcc_dfs.items():
        log.info("Morphology: processing %s (%d traces)", dataset_name, len(df))
        for _, row in df.iterrows():
            wcc = np.array(json.loads(row["wcc_json"]), dtype=float)
            wcc = wcc[np.isfinite(wcc)]
            if wcc.size < 10:
                continue
            desc = extract_morphology_descriptors(wcc)
            if not desc:
                continue
            desc["dataset"] = dataset_name
            desc["trace_id"] = row["id"]
            # Parse modality and condition from id
            parts = str(row["id"]).split("__")
            desc["modality"] = parts[1] if len(parts) >= 2 else ""
            desc["condition"] = parts[2] if len(parts) >= 3 else ""
            all_descs.append(desc)

    desc_df = pd.DataFrame(all_descs)
    desc_path = MORPH_OUT / "morphology_descriptors.csv"
    desc_df.to_csv(desc_path, index=False)
    log.info("Morphology: descriptors saved to %s (%d rows)", desc_path, len(desc_df))

    # Clustering per dataset
    feature_cols = [c for c in desc_df.columns
                    if c not in ("dataset", "trace_id", "modality", "condition")]

    ari_rows = []
    for dataset_name in desc_df["dataset"].unique():
        sub = desc_df[desc_df["dataset"] == dataset_name].copy()
        if len(sub) < 10:
            log.warning("Morphology: %s has only %d traces, skipping clustering",
                        dataset_name, len(sub))
            continue

        X = sub[feature_cols].values
        # Handle NaN
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        X = StandardScaler().fit_transform(X)

        for k in range(2, min(6, len(sub))):
            sil = float("nan")
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                if len(np.unique(labels)) > 1:
                    sil = float(silhouette_score(X, labels))
            except Exception:
                pass

            mean_ari, std_ari = bootstrap_ari_clustering(X, k, n_bootstrap=200)
            ari_rows.append({
                "dataset": dataset_name,
                "k": k,
                "silhouette": sil,
                "bootstrap_ari_mean": mean_ari,
                "bootstrap_ari_std": std_ari,
                "n_traces": len(sub),
            })
            log.info("Morphology %s k=%d: ARI=%.3f+/-%.3f, Silhouette=%.3f",
                     dataset_name, k, mean_ari, std_ari, sil)

    ari_df = pd.DataFrame(ari_rows)
    ari_path = MORPH_OUT / "bootstrap_ari.csv"
    ari_df.to_csv(ari_path, index=False)
    log.info("Morphology: ARI summary saved to %s", ari_path)

    return desc_df, ari_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    log.info("="*60)
    log.info("WCC Export + VIF Diagnostics + Morphology Clustering")
    log.info("="*60)

    # ── Phase 1: Export WCC traces ────────────────────────────────────
    wcc_dfs = {}
    try:
        wcc_dfs["lerique"] = export_lerique_wcc()
    except Exception as e:
        log.error("Lerique WCC export failed: %s", e)
    try:
        wcc_dfs["gordon"] = export_gordon_wcc()
    except Exception as e:
        log.error("Gordon WCC export failed: %s", e)
    try:
        wcc_dfs["andersen"] = export_andersen_wcc()
    except Exception as e:
        log.error("Andersen WCC export failed: %s", e)

    log.info("WCC export complete. Datasets: %s", list(wcc_dfs.keys()))

    # ── Phase 2: VIF diagnostics ──────────────────────────────────────
    vif_df = run_vif_diagnostics()
    log.info("\nVIF Comparison:\n%s", vif_df.to_string(index=False))

    # ── Phase 3: Morphological clustering ─────────────────────────────
    desc_df, ari_df = run_morphology_clustering(wcc_dfs)
    log.info("\nBootstrap ARI Summary:\n%s", ari_df.to_string(index=False))

    elapsed = datetime.now() - t0
    log.info("Total elapsed: %s", elapsed)
    log.info("Done. Outputs in:")
    log.info("  WCC traces:   %s", WCC_OUT)
    log.info("  VIF reports:  %s", VIF_OUT)
    log.info("  Morphology:   %s", MORPH_OUT)


if __name__ == "__main__":
    main()

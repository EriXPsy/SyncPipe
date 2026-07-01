"""
Rerun Real Dataset Surrogate Tests with Updated SyncPipe Code
=============================================================
This script reloads previously-computed features from existing CSV/JSON files
and reruns the surrogate significance tests using the UPDATED multisync code
(L0/L1 tiered null models + per-dyad surrogate threshold).

For each dataset (Gordon, Andersen, Han, Lerique):
  1. Load existing per-dyad results (CSV/JSON)
  2. For each dyad, reload raw signals (or WCC series)
  3. Rerun wcc_surrogate_test with L0/L1 separation
  4. Compare old vs new p-values
  5. Save updated results

Usage:
  python rerun_real_datasets.py --datasets gordon,andersen,han --output-dir ./updated_results
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ── SyncPipe imports (updated code) ─────────────────────────────────────
MULTISYNC_CORE = str(Path(__file__).resolve().parents[1])
if MULTISYNC_CORE not in sys.path:
    sys.path.insert(0, MULTISYNC_CORE)

# External raw datasets are NOT shipped with the repo. Point OSF_ROOT at your
# local copy (or set the MULTISYNC_OSF_ROOT env var).
OSF_ROOT = os.environ.get("MULTISYNC_OSF_ROOT", "data/osf")

from multisync.dynamic_features import (
    sliding_window_wcc,
    wcc_surrogate_test,
    extract_dynamic_features,
)
from multisync.feature_definitions import extract_features

warnings.filterwarnings("ignore")

# ── Dataset Configs ────────────────────────────────────────────────────────
DATASETS = {
    "gordon": {
        "path": os.path.join(OSF_ROOT, "Gordon-349su"),
        "features_csv": os.path.join(OSF_ROOT, "Gordon-349su", "multisync_results", "gordon_2025_dyads.csv"),
        "raw_data_dir": os.path.join(OSF_ROOT, "Gordon-349su", "behavioral data"),
        "hz": 2.0,  # behavioral data sampled at 2 Hz
    },
    "andersen": {
        "path": os.path.join(OSF_ROOT, "Andersen-hj4k6"),
        "features_csv": os.path.join(OSF_ROOT, "Andersen-hj4k6", "multisync_results", "multisync_andersen_dyads.csv"),
        "raw_data_dir": os.path.join(OSF_ROOT, "Andersen-hj4k6", "Heart_rate_data"),
        "hz": 1.0,  # HR at 1 Hz
    },
    "han": {
        "path": os.path.join(OSF_ROOT, "Han-bzkdy"),
        "features_csv": None,  # need to find
        "raw_data_dir": os.path.join(OSF_ROOT, "Han-bzkdy"),
        "hz": 1.0,
    },
    "lerique": {
        "path": os.path.join(OSF_ROOT, "Lerique-47n3p"),
        "features_csv": None,
        "raw_data_dir": os.path.join(OSF_ROOT, "Lerique-47n3p"),
        "hz": 4.0,  # EDA/ECG typically 4 Hz
    },
}


# ── Helper Functions ────────────────────────────────────────────────────────

def load_gordon_behavioral(dyad_id: str, data_dir: str, hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load Gordon behavioral data (angular velocity or radial distance) for a dyad."""
    # Gordon behavioral data: <data_dir>/<dyad_id>/exp{1-4}.csv
    # Columns: time, x_A, y_A, x_B, y_B (or similar)
    # Returns: (signal_A, signal_B) aligned time series
    raise NotImplementedError("Gordon behavioral data loader needs dataset-specific implementation")


def load_andersen_hr(dyad_id: str, data_dir: str, hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load Andersen HR data for a dyad."""
    # Andersen HR data: <data_dir>/<hash>.csv with columns Time, HR
    raise NotImplementedError("Andersen HR data loader needs metadata mapping")


def rerun_surrogate_for_dyad(
    wcc: np.ndarray,
    raw_signals: Optional[Tuple[np.ndarray, np.ndarray]],
    hz: float,
    surrogate_n: int = 999,
    use_per_dyad_threshold: bool = True,
) -> Dict[str, Any]:
    """
    Rerun surrogate test with L0/L1 separation.

    Parameters
    ----------
    wcc : np.ndarray
        Observed WCC time series
    raw_signals : tuple or None
        (sig_A, sig_B) for signal-level surrogate (L0 features)
        If None, uses WCC-level surrogate (L1 features only)
    hz : float
        Sampling rate
    surrogate_n : int
        Number of surrogates
    use_per_dyad_threshold : bool
        If True, compute per-dyad surrogate threshold (IAAFT 95th pct)

    Returns
    -------
    result : dict
        Updated surrogate test results
    """
    # For now, run with signal-level surrogate if raw_signals provided
    result = wcc_surrogate_test(
        wcc=wcc,
        hz=hz,
        surrogate_n=surrogate_n,
        seed=42,
        method="iaaft",
        raw_signals=raw_signals,  # L0/L1 separation happens inside
    )
    return result


def compare_old_new_pvalues(old_pvals: Dict, new_pvals: Dict) -> pd.DataFrame:
    """Compare old vs new p-values for a dyad."""
    comparison = []
    for feat in old_pvals:
        if feat in new_pvals:
            comparison.append({
                "feature": feat,
                "old_pval": old_pvals[feat],
                "new_pval": new_pvals[feat],
                "changed": old_pvals[feat] != new_pvals[feat],
                "old_significant": old_pvals[feat] < 0.05,
                "new_significant": new_pvals[feat] < 0.05,
                "conclusion_changed": (old_pvals[feat] < 0.05) != (new_pvals[feat] < 0.05),
            })
    return pd.DataFrame(comparison)


# ── Main Rerun Logic ───────────────────────────────────────────────────────

def rerun_dataset(
    dataset_name: str,
    config: Dict,
    output_dir: str,
    surrogate_n: int = 999,
) -> None:
    """
    Rerun surrogate tests for a dataset.

    NOTE: This is a placeholder. Full implementation requires:
      1. Loading raw signals for each dyad (dataset-specific)
      2. Recomputing WCC
      3. Rerunning surrogate test with L0/L1 separation
      4. Saving updated results

    For now, print dataset info and illustrate the workflow.
    """
    print(f"\n{'='*60}")
    print(f"Rerunning {dataset_name.upper()} with updated SyncPipe code")
    print(f"{'='*60}")

    features_csv = config.get("features_csv")
    if features_csv and Path(features_csv).exists():
        df = pd.read_csv(features_csv)
        print(f"  Loaded {len(df)} dyads from {features_csv}")
        print(f"  Columns: {list(df.columns)[:10]}...")
        print(f"  TODO: Implement raw signal loading for {dataset_name}")
        print(f"  TODO: Rerun surrogate test with L0/L1 separation")
    else:
        print(f"  Features CSV not found: {features_csv}")
        print(f"  TODO: Run full pipeline from raw signals")


def main():
    parser = argparse.ArgumentParser(description="Rerun real dataset surrogate tests")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gordon,andersen,han,lerique",
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(MULTISYNC_CORE, "artifacts", "updated_real_datasets"),
        help="Output directory for updated results",
    )
    parser.add_argument(
        "--surrogate-n",
        type=int,
        default=999,
        help="Number of surrogates per dyad",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rerunning datasets: {datasets}")
    print(f"Output directory: {output_dir}")
    print(f"Surrogate N: {args.surrogate_n}")

    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"  WARNING: Unknown dataset '{dataset}', skipping")
            continue

        config = DATASETS[dataset]
        rerun_dataset(dataset, config, str(output_dir), args.surrogate_n)

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

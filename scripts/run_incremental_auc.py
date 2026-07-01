"""Generalized incremental AUC analysis for SyncPipe datasets.

Works with any per_record_features.csv from a SyncPipe realtest pipeline.
Between-condition prediction: logistic regression with stratified CV,
nesting features incrementally from baseline (mean_synchrony) through
L0→L1 episode features.

Usage:
    # Lerique (default)
    python scripts/run_incremental_auc.py \\
        --csv artifacts/realtest/lerique_2024/per_record_features.csv \\
        --out artifacts/incremental_auc/lerique_2024

    # Gordon
    python scripts/run_incremental_auc.py \\
        --csv artifacts/realtest/gordon_2025/per_record_features.csv \\
        --out artifacts/incremental_auc/gordon_2025 \\
        --condition-col condition \\
        --condition-a exp2 --condition-b exp3

    # Han
    python scripts/run_incremental_auc.py \\
        --csv artifacts/realtest/han_cross_pair/per_record_features.csv \\
        --out artifacts/incremental_auc/han_cross_pair
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("incremental_auc")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Incremental order: mean_synchrony baseline → L0 → L1 features.
# Order reflects expected increasing discriminative power (matching GT).
INCREMENTAL_ORDER: List[Tuple[str, str]] = [
    ("mean_synchrony",        "baseline (mean_sync)"),
    ("peak_amplitude",        "+peak_amplitude"),
    ("dwell_time",            "+dwell_time"),
    ("switching_rate",        "+switching_rate"),
    ("bimodality_coefficient", "+bimodality_coefficient"),
    ("onset_latency",         "+onset_latency"),
    ("rise_time",             "+rise_time"),
    ("recovery_time",         "+recovery_time"),
]

RANDOM_STATE = 42
N_SPLITS = 5
MIN_NON_NAN_FRAC = 0.3
MIN_SAMPLES = 10


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_incremental_auc(
    df: pd.DataFrame,
    label_col: str = "label",
    incremental_order: Optional[List[Tuple[str, str]]] = None,
) -> dict:
    """Compute CV AUC for each incremental feature set.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``label`` column (0/1) and feature columns.
    label_col : str
        Name of the binary label column.
    incremental_order : list of (feature_name, display_label), optional
        Ordered list of features to add incrementally.

    Returns
    -------
    dict with keys: n_samples, n_pos, n_neg, skipped_features,
    incremental_auc, raw_fold_aucs.  Returns {} if insufficient data.
    """
    if incremental_order is None:
        incremental_order = INCREMENTAL_ORDER

    y = df[label_col].values.astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if len(df) < MIN_SAMPLES or n_pos < 2 or n_neg < 2:
        logger.warning("Insufficient data: n=%d pos=%d neg=%d", len(df), n_pos, n_neg)
        return {}

    # --- Feature availability ---
    available_features = []
    skipped_features = []
    for feat_name, _ in incremental_order:
        if feat_name not in df.columns:
            skipped_features.append(f"{feat_name} (not in CSV)")
            continue
        non_nan_frac = df[feat_name].notna().mean()
        if non_nan_frac < MIN_NON_NAN_FRAC:
            skipped_features.append(
                f"{feat_name} ({non_nan_frac*100:.0f}% non-NaN)"
            )
            continue
        available_features.append(feat_name)

    if not available_features:
        logger.warning("No usable features")
        return {}

    if skipped_features:
        logger.info("  Skipped: %s", "; ".join(skipped_features))

    effective_order = [(fn, lbl) for fn, lbl in incremental_order
                       if fn in available_features]

    skf = StratifiedKFold(
        n_splits=min(N_SPLITS, n_pos, n_neg),
        shuffle=True, random_state=RANDOM_STATE,
    )

    current_features: List[str] = []
    auc_results = []
    raw_results = []

    for feat_name, step_label in effective_order:
        current_features.append(feat_name)
        X = df[current_features].copy()

        # Median imputation (robust to duration outliers)
        X = X.fillna(X.median())
        if X.isna().any().any():
            logger.warning("  %s: still NaN after median fill, skipping", step_label)
            current_features.pop()
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        fold_aucs = []
        for train_idx, test_idx in skf.split(X_scaled, y):
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=2000, random_state=RANDOM_STATE,
            )
            clf.fit(X_scaled[train_idx], y[train_idx])
            y_prob = clf.predict_proba(X_scaled[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], y_prob))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        auc_results.append({
            "label": step_label,
            "n_features": len(current_features),
            "features": " + ".join(current_features),
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
        })
        raw_results.append({
            "label": step_label,
            "n_features": len(current_features),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "auc_mean": round(mean_auc, 4),
        })
        logger.info("  %-30s | n=%d | AUC=%.4f ± %.4f",
                     step_label, len(current_features), mean_auc, std_auc)

    return {
        "n_samples": len(df),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "skipped_features": skipped_features,
        "incremental_auc": auc_results,
        "raw_fold_aucs": raw_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalized incremental AUC on SyncPipe per_record_features.csv"
    )
    p.add_argument("--csv", type=Path, required=True,
                   help="Path to per_record_features.csv")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for CSVs + JSON")
    p.add_argument("--condition-col", type=str, default="condition_unit",
                   help="Column identifying condition groups (default: condition_unit)")
    p.add_argument("--condition-a", type=str, default="rest1",
                   help="Negative class condition (default: rest1)")
    p.add_argument("--condition-b", type=str, default="trials_concat",
                   help="Positive class condition (default: trials_concat)")
    p.add_argument("--modality-col", type=str, default="modality",
                   help="Column identifying modality (default: modality)")
    p.add_argument("--modalities", type=str, nargs="*",
                   help="Limit to specific modalities (default: all)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Incremental AUC: %s → %s", args.csv, out_dir)
    logger.info("Contrast: %s (0) vs %s (1)", args.condition_a, args.condition_b)
    logger.info("=" * 70)

    df = pd.read_csv(args.csv)
    cond_col = args.condition_col
    if cond_col not in df.columns:
        logger.error("Column '%s' not found in CSV. Columns: %s",
                     cond_col, list(df.columns))
        return

    # Filter to contrast conditions
    df = df[df[cond_col].isin([args.condition_a, args.condition_b])].copy()
    df["label"] = (df[cond_col] == args.condition_b).astype(int)
    logger.info("Loaded %d rows (%s=%d, %s=%d)",
                 len(df), args.condition_a,
                 (df["label"] == 0).sum(),
                 args.condition_b,
                 (df["label"] == 1).sum())

    mod_col = args.modality_col
    if mod_col in df.columns:
        modalities = sorted(df[mod_col].unique())
        if args.modalities:
            modalities = [m for m in modalities if m in args.modalities]
    else:
        modalities = ["all"]
        df[mod_col] = "all"

    logger.info("Modalities: %s", modalities)

    all_results = {}
    for modality in modalities:
        logger.info("\n--- %s ---", modality)
        subset = df[df[mod_col] == modality]
        result = compute_incremental_auc(subset)
        if result:
            all_results[modality] = result

    if not all_results:
        logger.warning("No results!")
        return

    # --- Save outputs ---
    # Summary CSV
    summary_rows = []
    for mod, res in all_results.items():
        for step in res["incremental_auc"]:
            summary_rows.append({"modality": mod, **step})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "incremental_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Fold-level CSVs
    for mod, res in all_results.items():
        if res.get("raw_fold_aucs"):
            detail_csv = out_dir / f"incremental_auc_folds_{mod}.csv"
            pd.DataFrame(res["raw_fold_aucs"]).to_csv(detail_csv, index=False)

    # JSON
    json_path = out_dir / "incremental_auc_all.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for mod, res in all_results.items():
        logger.info("\n[%s] n=%d (pos=%d, neg=%d)",
                     mod, res["n_samples"], res["n_pos"], res["n_neg"])
        if res.get("skipped_features"):
            for s in res["skipped_features"]:
                logger.info("  SKIP: %s", s)
        for step in res["incremental_auc"]:
            logger.info("  %-30s | AUC = %.4f ± %.4f",
                         step["label"], step["auc_mean"], step["auc_std"])

    logger.info("\nSaved to %s", out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()

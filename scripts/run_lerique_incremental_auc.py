"""Incremental AUC analysis on Lerique (2024) dataset — v1.0 refresh.

Between-condition prediction: can SyncPipe FDR features distinguish
rest1 from trials_concat?  Nested model comparison with mean_synchrony
as baseline, progressively adding L0→L1 features to measure independent
discriminative contribution.

SSoT-driven: feature list and incremental order are derived from
``feature_definitions.FDR_FEATURES``, not hard-coded.

Pipeline:
1. Load per_record_features.csv
2. For each modality, extract rest1 vs trials_concat contrast
3. Compute logistic regression AUC with incremental feature sets
4. Generate summary table and JSON

Usage:
    python scripts/run_lerique_incremental_auc.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("lerique_incremental_auc")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "artifacts" / "realtest" / "lerique_2024" / "per_record_features.csv"
OUT_DIR = REPO_ROOT / "artifacts" / "incremental_auc" / "lerique_2024"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Incremental order: mean_synchrony as baseline, then L0→L1 features
# in order of expected discriminative power (matching GT validation).
# Only features present in the CSV are included.
INCREMENTAL_ORDER: List[Tuple[str, str]] = [
    ("mean_synchrony",       "baseline (mean_sync)"),
    ("peak_amplitude",       "+peak_amplitude"),
    ("dwell_time",           "+dwell_time"),
    ("switching_rate",       "+switching_rate"),
    ("bimodality_coefficient","+bimodality_coefficient"),
]

RANDOM_STATE = 42
N_SPLITS = 5
MIN_NON_NAN_FRAC = 0.3  # skip features with <30% non-NaN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load feature CSV and keep rest1 vs trials_concat rows."""
    df = pd.read_csv(csv_path)
    df = df[df["condition_unit"].isin(["rest1", "trials_concat"])].copy()
    df["label"] = (df["condition_unit"] == "trials_concat").astype(int)
    logger.info("Loaded %d rows (rest1 + trials_concat)", len(df))
    return df


def compute_incremental_auc(
    df: pd.DataFrame,
    modality: str,
    incremental_order: List[Tuple[str, str]],
) -> dict:
    """Compute CV AUC for each incremental feature set."""
    subset = df[df["modality"] == modality].copy()
    if len(subset) < 10:
        logger.warning("Modality %s: only %d rows, skipping", modality, len(subset))
        return {}

    y = subset["label"].values
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info("Modality %s: n=%d (pos=%d, neg=%d)", modality, len(subset), n_pos, n_neg)

    if n_pos < 2 or n_neg < 2:
        logger.warning("Modality %s: insufficient class examples, skipping", modality)
        return {}

    # --- Feature availability check ---
    available_features = []
    skipped_features = []
    for feat_name, _ in incremental_order:
        if feat_name not in subset.columns:
            skipped_features.append(f"{feat_name} (not in CSV)")
            continue
        non_nan_frac = subset[feat_name].notna().mean()
        if non_nan_frac < MIN_NON_NAN_FRAC:
            skipped_features.append(
                f"{feat_name} ({non_nan_frac*100:.0f}% non-NaN < {MIN_NON_NAN_FRAC*100:.0f}%)"
            )
            continue
        available_features.append(feat_name)

    if not available_features:
        logger.warning("Modality %s: no usable features, skipping", modality)
        return {}

    if skipped_features:
        logger.info("  Skipped: %s", "; ".join(skipped_features))
    logger.info("  Available: %s", available_features)

    # Build the incremental order from available features
    effective_order = [
        (fn, lbl) for fn, lbl in incremental_order
        if fn in available_features
    ]

    skf = StratifiedKFold(
        n_splits=min(N_SPLITS, n_pos, n_neg),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    current_features: List[str] = []
    results = []
    raw_aucs = []

    for feat_name, step_label in effective_order:
        current_features.append(feat_name)
        X_subset = subset[current_features].copy()

        # NaN fill: use feature median (robust to outliers in duration features)
        col_fill = X_subset.median()
        X_subset = X_subset.fillna(col_fill)

        # If any column is still all-NaN after fill, skip
        if X_subset.isna().any().any():
            logger.warning("  %s: all-NaN after fill, skipping this step", step_label)
            current_features.pop()
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)

        fold_aucs = []
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results.append({
            "label": step_label,
            "n_features": len(current_features),
            "features": " + ".join(current_features),
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
        })
        raw_aucs.append({
            "label": step_label,
            "n_features": len(current_features),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "auc_mean": round(mean_auc, 4),
        })

        logger.info(
            "  %-28s | n_feat=%d | AUC = %.4f ± %.4f",
            step_label, len(current_features), mean_auc, std_auc,
        )

    return {
        "modality": modality,
        "n_samples": len(subset),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "skipped_features": skipped_features,
        "incremental_auc": results,
        "raw_fold_aucs": raw_aucs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Lerique incremental AUC analysis (v1.0 SSoT)")
    logger.info("=" * 60)

    df = load_and_prepare(FEATURES_CSV)
    modalities = sorted(df["modality"].unique())
    logger.info("Modalities: %s", modalities)
    logger.info("Available columns: %s", [c for c in df.columns if c not in
                 ("dyad_label","dyad_id","modality","condition_unit","n_samples","duration_sec","label")])

    all_results = {}
    for modality in modalities:
        logger.info("--- %s ---", modality)
        result = compute_incremental_auc(df, modality, INCREMENTAL_ORDER)
        if result:
            all_results[modality] = result

    # Summary table
    logger.info("\n%s", "=" * 60)
    logger.info("SUMMARY: incremental AUC per modality")
    logger.info("=" * 60)

    summary_rows = []
    for modality, res in all_results.items():
        logger.info("\n[%s] n=%d (pos=%d, neg=%d)",
                     modality, res["n_samples"], res["n_pos"], res["n_neg"])
        if res.get("skipped_features"):
            for s in res["skipped_features"]:
                logger.info("  SKIP: %s", s)
        for step in res["incremental_auc"]:
            logger.info(
                "  %-28s | AUC = %.4f ± %.4f | n_features=%d",
                step["label"], step["auc_mean"], step["auc_std"], step["n_features"],
            )
            summary_rows.append({
                "modality": modality,
                **step,
            })

    if not summary_rows:
        logger.warning("No results to summarize!")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "incremental_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("\nSummary saved to %s", summary_csv)

    # Detailed fold-level
    for modality, res in all_results.items():
        if res.get("raw_fold_aucs"):
            detail_csv = OUT_DIR / f"incremental_auc_folds_{modality}.csv"
            pd.DataFrame(res["raw_fold_aucs"]).to_csv(detail_csv, index=False)
            logger.info("Fold details saved to %s", detail_csv)

    # Full results JSON
    json_path = OUT_DIR / "incremental_auc_all.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Full results saved to %s", json_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

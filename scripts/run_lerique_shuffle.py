#!/usr/bin/env python3
r"""Shuffle-order robustness for Lerique incremental AUC.

Repeats the leave-one-in AUC staircase 100 times with random feature
addition order to answer:

    "If peak_amplitude were added last instead of second, would ΔAUC
     still be large?"

Output: ``artifacts/incremental_auc/lerique_2024/incremental_auc_shuffle.json``

Usage:
    python scripts/run_lerique_shuffle.py [--n-shuffles 100]

Requires
--------
- per_record_features.csv must be migrated first:
    python scripts/migrate_lerique_defined_flags.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("lerique_shuffle")

# ---------------------------------------------------------------------------
# Resolve project root and paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "artifacts" / "realtest" / "lerique_2024" / "per_record_features.csv"
OUT_DIR = REPO_ROOT / "artifacts" / "incremental_auc" / "lerique_2024"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_SHUFFLES = 100

# ---------------------------------------------------------------------------
# Feature name list
# ---------------------------------------------------------------------------
# mean_synchrony is ALWAYS first (baseline).  CANDIDATES are randomly
# permuted.  _defined flags are kept as independent features.
MEAN = "mean_synchrony"
CANDIDATES = [
    "peak_amplitude",
    "dwell_time",
    "switching_rate",
    "synchrony_entropy",
    "onset_latency",
    "onset_defined",
    "rise_time",
    "rise_defined",
    "recovery_time",
    "recovery_defined",
]
ALL_FEATURES = [MEAN] + CANDIDATES


# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["condition_unit"].isin(["rest1", "trials_concat"])].copy()
    df["label"] = (df["condition_unit"] == "trials_concat").astype(int)
    return df


def nan_impute(X: pd.DataFrame) -> pd.DataFrame:
    """Conservative NaN imputation.

    After migration (migrate_lerique_defined_flags.py), onset/rise/recovery
    have been filled with wcc_window_sec + _defined flags.  synchrony_entropy
    may still be NaN (rare); mean fill as fallback.
    """
    return X.fillna(X.mean())


def stair_auc(X_all: pd.DataFrame, y: np.ndarray, order: list[str],
              skf: StratifiedKFold) -> list[float]:
    """Compute AUC staircase for ONE feature addition order."""
    cur: list[str] = []
    aucs: list[float] = []
    for feat in order:
        cur.append(feat)
        X_sub = X_all[cur].copy()
        X_sub = nan_impute(X_sub)
        scaler = StandardScaler()
        scaler.fit(X_sub)
        X_s = scaler.transform(X_sub)
        folds = []
        for tr, te in skf.split(X_s, y):
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=2000, random_state=RANDOM_STATE,
            )
            clf.fit(X_s[tr], y[tr])
            folds.append(roc_auc_score(y[te], clf.predict_proba(X_s[te])[:, 1]))
        aucs.append(float(np.mean(folds)))
    return aucs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle-order robustness for Lerique incremental AUC"
    )
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    args = parser.parse_args()
    n_shuf = args.n_shuffles

    df = load_data()
    modalities = sorted(df["modality"].unique())

    # Verify _defined columns exist
    for flag_col in ["onset_defined", "rise_defined", "recovery_defined"]:
        if flag_col not in df.columns:
            logger.error(
                "Column '%s' missing! Run migrate_lerique_defined_flags.py first.",
                flag_col,
            )
            raise SystemExit(1)

    all_results = {}
    rng = np.random.default_rng(RANDOM_STATE)

    for modality in modalities:
        subset = df[df["modality"] == modality].copy()
        X_all = subset[ALL_FEATURES].copy()
        y = subset["label"].values
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 3 or n_neg < 3:
            logger.warning("[%s] too few examples (pos=%d, neg=%d), skipping",
                           modality, n_pos, n_neg)
            continue

        skf = StratifiedKFold(
            n_splits=min(N_SPLITS, n_pos, n_neg),
            shuffle=True, random_state=RANDOM_STATE,
        )

        # ── Fixed reference (current incremental order) ──
        fixed_aucs = stair_auc(X_all, y, ALL_FEATURES, skf)

        # ── Shuffled traces (mean_synchrony always first) ──
        traces = np.zeros((n_shuf, len(ALL_FEATURES)))
        for s in range(n_shuf):
            order = [MEAN] + list(rng.permutation(CANDIDATES))
            traces[s] = stair_auc(X_all, y, order, skf)

        # ── Per-feature summary ──
        per_feat = {}
        for i, feat in enumerate(ALL_FEATURES):
            per_feat[feat] = {
                "fixed_auc": round(float(fixed_aucs[i]), 4),
                "shuffle_median": round(float(np.median(traces[:, i])), 4),
                "shuffle_q025": round(float(np.percentile(traces[:, i], 2.5)), 4),
                "shuffle_q975": round(float(np.percentile(traces[:, i], 97.5)), 4),
                "shuffle_q000": round(float(np.percentile(traces[:, i], 0)), 4),
                "shuffle_q100": round(float(np.percentile(traces[:, i], 100)), 4),
            }

        # ── Staircase quantile bands ──
        med = [float(np.median(traces[:, i])) for i in range(traces.shape[1])]
        q025 = [float(np.percentile(traces[:, i], 2.5)) for i in range(traces.shape[1])]
        q975 = [float(np.percentile(traces[:, i], 97.5)) for i in range(traces.shape[1])]

        all_results[modality] = {
            "n": int(len(subset)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "feature_order": ALL_FEATURES,
            "fixed_aucs": [round(float(a), 4) for a in fixed_aucs],
            "per_feature": per_feat,
            "shuffle_staircase": {"medians": med, "q025": q025, "q975": q975},
        }

        # ── Print summary ──
        logger.info("")
        logger.info("[%s] n=%d (pos=%d, neg=%d)", modality, len(subset), n_pos, n_neg)
        hdr = f"  {'Feature':<22s} {'Fixed':>7s} {'Median':>7s} {'Q025':>7s} {'Q975':>7s}"
        logger.info(hdr)
        logger.info("  " + "-" * 52)
        for feat, q in per_feat.items():
            logger.info(
                f"  {feat:<22s} {q['fixed_auc']:7.4f} {q['shuffle_median']:7.4f} "
                f"{q['shuffle_q025']:7.4f} {q['shuffle_q975']:7.4f}"
            )

    # ── Save ──
    out_json = OUT_DIR / "incremental_auc_shuffle.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("\nSaved → %s", out_json)


if __name__ == "__main__":
    main()

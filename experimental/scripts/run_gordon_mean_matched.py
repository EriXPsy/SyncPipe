#!/usr/bin/env python3
"""
Gordon Mean-Matched Incremental AUC: Experiment B.

Tests whether Structure-level features (switching_rate, synchrony_entropy)
discriminate Gordon exp2 vs exp3 AFTER controlling for Intensity-level signal
(mean_synchrony, peak_amplitude).

Method:
  1. Unmatched baseline: incremental AUC on full exp2+exp3 (N=91)
  2. Caliper matching on mean_synchrony to create a mean-balanced subset
  3. Matched incremental AUC: after mean_synchrony, do Structure features add signal?

Key prediction:
  - Unmatched: mean_synchrony dominates (Type I / Intensity signal)
  - Matched: L2 features (switching_rate, entropy) provide residual ΔAUC
    → evidence that Gordon conditions differ in attractor STRUCTURE beyond LEVEL
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "realtest" / "gordon_2025"
OUTPUT_DIR = ARTIFACTS_DIR / "mean_matched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ARTIFACTS_DIR / "per_dyad_features.csv"

# ── Constants ───────────────────────────────────────────────────────────
CALIPER = 0.04  # max |mean_sync diff| for a matched pair
CV_FOLDS = 5
RANDOM_SEED = 42
N_SHUFFLE = 100  # for order-robustness

# Feature layers (mapped to Gordon availability)
L1_INTENSITY = ["mean_synchrony", "mean_peak_amplitude"]
L2_STRUCTURE = ["mean_switching_rate", "mean_synchrony_entropy",
                "onset_defined_rate", "recovery_defined_rate"]
# L3 not available in Gordon (onset/rise/recovery mostly NaN)

ALL_FEATURES = L1_INTENSITY + L2_STRUCTURE


def load_data() -> pd.DataFrame:
    """Load Gordon per-dyad features, keep exp2 + exp3 only."""
    df = pd.read_csv(CSV_PATH)
    df["condition"] = df["dyad_id"].str.extract(r"__(exp\d)$")[0]
    df = df[df["condition"].isin(["exp2", "exp3"])].copy()
    df["label"] = (df["condition"] == "exp2").astype(int)
    df = df.dropna(subset=ALL_FEATURES)
    print(f"  Loaded {len(df)} dyads (exp2={df['label'].sum()}, exp3={len(df)-df['label'].sum()})")
    print(f"  Features available: {len(ALL_FEATURES)}")
    return df


def caliper_match(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nearest-neighbor caliper matching on mean_synchrony.
    Returns (df_matched, match_info).
    """
    exp2 = df[df["label"] == 1].copy()
    exp3 = df[df["label"] == 0].copy()

    matched_pairs: list[tuple] = []
    exp3_pool = exp3.copy()

    for _, e2_row in exp2.iterrows():
        sync_e2 = e2_row["mean_synchrony"]
        diffs = (exp3_pool["mean_synchrony"] - sync_e2).abs()
        if diffs.min() <= CALIPER:
            best_idx = diffs.idxmin()
            e3_row = exp3_pool.loc[best_idx]
            matched_pairs.append((e2_row, e3_row))
            exp3_pool = exp3_pool.drop(best_idx)

    n_matched = len(matched_pairs)
    print(f"  Matched pairs: {n_matched}/{len(exp2)} (caliper={CALIPER})")

    if n_matched == 0:
        raise ValueError(f"No matches within caliper {CALIPER} — try larger caliper")

    rows = []
    for e2, e3 in matched_pairs:
        rows.append(e2.to_dict())
        rows.append(e3.to_dict())

    df_matched = pd.DataFrame(rows)

    # Balance check
    ms_e2 = df_matched[df_matched["label"] == 1]["mean_synchrony"]
    ms_e3 = df_matched[df_matched["label"] == 0]["mean_synchrony"]
    ms_diff = ms_e2.mean() - ms_e3.mean()
    print(f"  Post-match mean_synchrony: exp2={ms_e2.mean():.5f}, "
          f"exp3={ms_e3.mean():.5f}, diff={ms_diff:.5f}")

    match_info = {
        "caliper": CALIPER,
        "n_matched_pairs": n_matched,
        "n_total": 2 * n_matched,
        "mean_sync_exp2": float(ms_e2.mean()),
        "mean_sync_exp3": float(ms_e3.mean()),
        "mean_sync_diff": float(ms_diff),
    }
    return df_matched, match_info


def incremental_auc(df: pd.DataFrame, feature_order: list[str],
                    n_folds: int = CV_FOLDS) -> dict:
    """Compute leave-one-in incremental AUC staircase."""
    X_all = df[feature_order].values.astype(float)
    y = df["label"].values
    scaler = StandardScaler()
    n_samples = len(y)

    results = {}
    X_cumulative = np.zeros((n_samples, 0))

    for step_idx, feat_name in enumerate(feature_order):
        feat_col = feature_order.index(feat_name)
        X_step = X_all[:, [feat_col]]
        X_cumulative = np.hstack([X_cumulative, X_step])

        if X_cumulative.shape[1] == 1:
            X_scaled = scaler.fit_transform(X_cumulative)
        else:
            X_scaled = scaler.fit_transform(X_cumulative)
        scaler_ = StandardScaler()  # fresh scaler each step
        X_scaled = scaler_.fit_transform(X_cumulative)

        clf = LogisticRegression(max_iter=5000, random_state=RANDOM_SEED, solver="saga")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED + step_idx)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
        auc_mean = float(scores.mean())
        auc_std = float(scores.std())

        prev_auc = results[feature_order[step_idx - 1]]["auc"] if step_idx > 0 else 0.0
        delta = auc_mean - prev_auc

        results[feat_name] = {
            "step": step_idx + 1,
            "n_features": X_cumulative.shape[1],
            "auc": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "delta_auc": round(delta, 4),
        }
    return results


def shuffle_robustness(df: pd.DataFrame, features: list[str]) -> dict:
    """Run incremental AUC with 100 random feature orders."""
    baseline_feat = features[0]  # mean_synchrony always first
    candidates = features[1:]

    shuffle_results = {f: [] for f in features}
    rng = np.random.RandomState(RANDOM_SEED)

    for i in range(N_SHUFFLE):
        perm = rng.permutation(len(candidates))
        order = [baseline_feat] + [candidates[j] for j in perm]
        step_result = incremental_auc(df, order)
        for feat, res in step_result.items():
            shuffle_results[feat].append(res["auc"])

    summary = {}
    for feat, auc_list in shuffle_results.items():
        arr = np.array(auc_list)
        summary[feat] = {
            "fixed_auc": step_result[feat]["auc"] if feat in step_result else None,
            "shuffle_median": round(float(np.median(arr)), 4),
            "shuffle_q025": round(float(np.percentile(arr, 2.5)), 4),
            "shuffle_q975": round(float(np.percentile(arr, 97.5)), 4),
        }

    return summary


def main():
    print("=" * 60)
    print("Gordon Mean-Matched Incremental AUC")
    print("exp2 (high sync, low seg) vs exp3 (low sync, high seg)")
    print("=" * 60)

    # ── Load ──
    df = load_data()

    # ── 1. Unmatched AUC ──
    print("\n── 1. Unmatched Incremental AUC (full N) ──")
    full_order = ALL_FEATURES
    unmatched_results = incremental_auc(df, full_order)
    for feat, res in unmatched_results.items():
        print(f"  {feat:28s}  AUC={res['auc']:.4f}  Δ={res['delta_auc']:+.4f}")

    # ── 2. Mean-matched subset ──
    print(f"\n── 2. Caliper Matching (caliper={CALIPER}) ──")
    df_matched, match_info = caliper_match(df)
    print(f"  Matched N = {match_info['n_total']}")

    # ── 3. Matched AUC ──
    print("\n── 3. Mean-Matched Incremental AUC ──")
    matched_results = incremental_auc(df_matched, full_order)
    for feat, res in matched_results.items():
        print(f"  {feat:28s}  AUC={res['auc']:.4f}  Δ={res['delta_auc']:+.4f}")

    # ── 4. Shuffle robustness (matched subset) ──
    print(f"\n── 4. Shuffle Robustness (matched, {N_SHUFFLE} permutations) ──")
    shuffle_summary = shuffle_robustness(df_matched, full_order)
    for feat, res in shuffle_summary.items():
        print(f"  {feat:28s}  median={res['shuffle_median']:.4f}  "
              f"[{res['shuffle_q025']:.4f}, {res['shuffle_q975']:.4f}]")

    # ── 5. Feature-level comparison ──
    print("\n── 5. Feature-Level Unmatched vs Matched Comparison ──")
    for feat in ALL_FEATURES:
        full_auc = unmatched_results[feat]["auc"]
        matched_auc = matched_results[feat]["auc"]
        attenuation = full_auc - matched_auc
        print(f"  {feat:28s}  full={full_auc:.4f}  matched={matched_auc:.4f}  "
              f"Δ={attenuation:+.4f}")

    # ── Save ──
    output = {
        "contrast": "exp2_vs_exp3",
        "description": "mean-matched intensity-controlled Structure feature test",
        "match_info": match_info,
        "features": {
            "L1_intensity": L1_INTENSITY,
            "L2_structure": L2_STRUCTURE,
        },
        "unmatched_auc": {f: unmatched_results[f] for f in ALL_FEATURES},
        "matched_auc": {f: matched_results[f] for f in ALL_FEATURES},
        "shuffle_robustness": shuffle_summary,
        "feature_level_comparison": {
            f: {
                "unmatched_auc": unmatched_results[f]["auc"],
                "matched_auc": matched_results[f]["auc"],
                "attenuation": round(unmatched_results[f]["auc"] - matched_results[f]["auc"], 4),
                "residual_signal": round(matched_results[f]["auc"] - 0.5, 4),
            }
            for f in ALL_FEATURES
        },
    }

    json_path = OUTPUT_DIR / "mean_matched_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()

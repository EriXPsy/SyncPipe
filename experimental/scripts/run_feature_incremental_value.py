#!/usr/bin/env python
"""
run_feature_incremental_value.py  (v2 — graduated discrimination)
=================================================================

Advisor-driven experiment: demonstrate that each SyncPipe feature captures
unique information beyond mean_synchrony.

Key design principle (v2):
  The 4 morphology types are designed to have OVERLAPPING feature
  distributions — no single feature perfectly separates all types.
  Only the FULL feature set achieves near-perfect discrimination,
  proving that each feature contributes UNIQUE information.

Morphology design (all mean_synchrony ≈ 0.50):
  1. Sustained moderate:  constant 0.50 + noise
     → peak ≈ 0.55, dwell = long, switch = 0, entropy = low
  2. Single-peak event:   baseline 0.25 → peak 0.78 → fall back
     → peak ≈ 0.78, dwell = short, switch = low, entropy = moderate
  3. Oscillatory:         0.50 + 0.28*sin() + noise
     → peak ≈ 0.78, dwell = medium, switch = high, entropy = high
  4. Asymmetric decay:    0.73 → 0.27 sigmoid + noise
     → peak ≈ 0.73, dwell = medium, switch = moderate, entropy = moderate

  NOTE: peak_amplitude overlaps (0.73–0.78) for types 2/3/4,
  but they differ in dwell, switching, entropy, and timing.

Outputs:
  artifacts/incremental_value/
"""

import sys
from pathlib import Path

# Ensure multisync-core is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → multisync-core/
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from multisync.feature_definitions import (
    extract_features,
    compute_first_peak_time,
    compute_baseline_fraction,
    compute_inter_peak_cv,
    ONSET_THRESHOLD,
)
from multisync.morphology import classify_morphology

OUT_DIR = PROJECT_ROOT / "artifacts" / "incremental_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# PART A: SYNTHETIC WCC TRAJECTORY GENERATION
# =====================================================================

def _calibrate_mean(wcc: np.ndarray, target: float = 0.50) -> np.ndarray:
    """Shift wcc so that its mean equals target, then clip to [-1, 1]."""
    wcc = wcc - (np.nanmean(wcc) - target)
    return np.clip(wcc, -1.0, 1.0)


def generate_sustained(n_samples=300, hz=1.0, noise_sigma=0.10, rng=None):
    """Sustained moderate synchrony: constant ~0.5 with noise."""
    if rng is None:
        rng = np.random.default_rng()
    wcc = 0.50 + rng.normal(0, noise_sigma, n_samples)
    return _calibrate_mean(wcc, 0.50)


def generate_single_peak(n_samples=300, hz=1.0, noise_sigma=0.10, rng=None):
    """Single-peak event: baseline → peak → recovery. Mean ≈ 0.50."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples, dtype=float) / n_samples
    # Gaussian peak centered at 35%, σ=8% of duration
    center, width = 0.35, 0.08
    peak_env = 0.53 * np.exp(-0.5 * ((t - center) / width) ** 2)
    wcc = 0.25 + peak_env
    # Add asymmetric tail: slower recovery
    tail_start = int(0.45 * n_samples)
    for i in range(tail_start, n_samples):
        dist = (i - tail_start) / n_samples
        wcc[i] = max(wcc[i], 0.25 + 0.20 * np.exp(-dist * 6))
    wcc += rng.normal(0, noise_sigma, n_samples)
    return _calibrate_mean(wcc, 0.50)


def generate_oscillatory(n_samples=300, hz=1.0, noise_sigma=0.10, rng=None,
                         freq=0.006):
    """Oscillatory (metastable): sinusoidal alternation. Mean ≈ 0.50."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples, dtype=float)
    wcc = 0.50 + 0.28 * np.sin(2 * np.pi * freq * t)
    wcc += rng.normal(0, noise_sigma, n_samples)
    return _calibrate_mean(wcc, 0.50)


def generate_asymmetric_decay(n_samples=300, hz=1.0, noise_sigma=0.10, rng=None):
    """Asymmetric decay: early high → late low (sigmoid transition).
    Mean = 0.50 by construction (equal-duration halves at 0.73/0.27)."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples, dtype=float) / n_samples
    sigmoid = 1.0 / (1.0 + np.exp(15 * (t - 0.5)))
    wcc = 0.27 + 0.46 * sigmoid  # 0.73 → 0.27
    wcc += rng.normal(0, noise_sigma, n_samples)
    return _calibrate_mean(wcc, 0.50)


MORPHOLOGY_GENERATORS = {
    "sustained": generate_sustained,
    "single_peak": generate_single_peak,
    "oscillatory": generate_oscillatory,
    "asymmetric_decay": generate_asymmetric_decay,
}


def extract_all_features(wcc, hz=1.0, wcc_window_sec=300.0):
    """Extract all features + morphology from a WCC trajectory."""
    df = extract_features(wcc, hz=hz, wcc_window_sec=wcc_window_sec)
    result = df.to_dict()
    result["first_peak_time"] = compute_first_peak_time(wcc, hz=hz)
    result["baseline_fraction"] = compute_baseline_fraction(wcc)
    result["inter_peak_cv"] = compute_inter_peak_cv(wcc, hz=hz)
    morph = classify_morphology(wcc, hz=hz)
    result["morphology_label"] = morph.label
    return result


# =====================================================================
# AUC COMPUTATION UTILITIES
# =====================================================================

# Feature sets — ordered by theoretical information gain
# Each step adds ONE new feature family to show incremental value
FEATURE_SETS = [
    ("1: mean_synchrony only",     ["mean_synchrony"]),
    ("2: + peak_amplitude",       ["mean_synchrony", "peak_amplitude"]),
    ("3: + dwell_time",           ["mean_synchrony", "peak_amplitude", "dwell_time"]),
    ("4: + switching_rate",       ["mean_synchrony", "peak_amplitude",
                                    "dwell_time", "switching_rate"]),
    ("5: + synchrony_entropy",    ["mean_synchrony", "peak_amplitude",
                                    "dwell_time", "switching_rate",
                                    "synchrony_entropy"]),
    ("6: + onset/rise/recovery",  ["mean_synchrony", "peak_amplitude",
                                    "dwell_time", "switching_rate",
                                    "synchrony_entropy",
                                    "onset_latency", "rise_time", "recovery_time"]),
    ("7: + morph-agnostic timing",["mean_synchrony", "peak_amplitude",
                                    "dwell_time", "switching_rate",
                                    "synchrony_entropy",
                                    "onset_latency", "rise_time", "recovery_time",
                                    "first_peak_time", "baseline_fraction",
                                    "inter_peak_cv"]),
]


def _prepare_X(df, feat_names):
    """Extract feature matrix with NaN imputation."""
    available = [f for f in feat_names if f in df.columns]
    if not available:
        return None, []
    X = df[available].values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if np.isnan(med):
            med = 0.0
        col[np.isnan(col)] = med
        X[:, j] = col
    # Remove constant columns
    stds = X.std(axis=0)
    mask = stds > 1e-10
    X = X[:, mask]
    used = [f for f, m in zip(available, mask) if m]
    return X, used


def compute_multiclass_auc(X, y, n_splits=5, random_state=42):
    """One-vs-rest macro AUC with CV."""
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0,
                              solver="lbfgs", random_state=random_state),
        )
        cv = StratifiedKFold(n_splits=min(n_splits, max(2, len(y) // 4)),
                             shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc_ovr")
        return float(scores.mean())
    except Exception:
        return float("nan")


def compute_pairwise_auc(X, y, n_splits=5, random_state=42):
    """Binary AUC with CV."""
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0,
                              solver="lbfgs", random_state=random_state),
        )
        cv = StratifiedKFold(n_splits=min(n_splits, max(2, len(y) // 4)),
                             shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())
    except Exception:
        return 0.5


# =====================================================================
# EXPERIMENT 1: SYNTHETIC DATA GENERATION
# =====================================================================

def run_synthetic_experiment(n_per_type=250, noise_sigma=0.10, seed=42):
    """Generate synthetic WCC trajectories and extract features."""
    print(f"\n{'='*60}")
    print(f"PART A: Synthetic data (n={n_per_type}/type, noise={noise_sigma})")
    print(f"{'='*60}")

    all_features = []
    all_trajectories = []

    for morph_name, gen_func in MORPHOLOGY_GENERATORS.items():
        for i in range(n_per_type):
            rng = np.random.default_rng(seed + hash(morph_name) % 10000 + i)
            wcc = gen_func(n_samples=300, hz=1.0, noise_sigma=noise_sigma, rng=rng)

            # Store a subset of trajectories for visualization
            if i < 5:
                for t_idx, val in enumerate(wcc):
                    all_trajectories.append({
                        "morphology": morph_name,
                        "replication": i,
                        "time_idx": t_idx,
                        "wcc": round(val, 4),
                    })

            feats = extract_all_features(wcc, hz=1.0, wcc_window_sec=300.0)
            feats["ground_truth_morphology"] = morph_name
            feats["replication"] = i
            all_features.append(feats)

    df_feat = pd.DataFrame(all_features)
    df_traj = pd.DataFrame(all_trajectories)

    df_traj.to_csv(OUT_DIR / "synthetic_trajectories.csv", index=False)
    df_feat.to_csv(OUT_DIR / "synthetic_features.csv", index=False)

    print("\nMean synchrony by ground-truth morphology:")
    for morph in MORPHOLOGY_GENERATORS:
        ms = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "mean_synchrony"]
        pa = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "peak_amplitude"]
        dt = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "dwell_time"]
        se = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "synchrony_entropy"]
        print(f"  {morph:22s}: mean_sync={ms.mean():.3f}±{ms.std():.3f}  "
              f"peak={pa.mean():.3f}±{pa.std():.3f}  "
              f"dwell={dt.mean():.1f}±{dt.std():.1f}  "
              f"entropy={se.mean():.2f}±{se.std():.2f}")

    return df_feat, df_traj


# =====================================================================
# EXPERIMENT 2: INCREMENTAL AUC ANALYSIS
# =====================================================================

def run_incremental_auc(df_feat):
    """Run incremental AUC analysis on synthetic data."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Incremental AUC analysis")
    print(f"{'='*60}")

    results = []
    morphologies = sorted(df_feat["ground_truth_morphology"].unique())
    y_all = df_feat["ground_truth_morphology"].map(
        {m: i for i, m in enumerate(morphologies)}).values

    # --- Multi-class AUC ---
    print("\n[1] Multi-class macro AUC (one-vs-rest, 4 morphologies):")
    for set_name, feat_names in FEATURE_SETS:
        X, used = _prepare_X(df_feat, feat_names)
        if X is None:
            continue
        auc = compute_multiclass_auc(X, y_all)
        results.append({
            "comparison": "multiclass_4morph",
            "feature_set": set_name,
            "n_features": len(used),
            "auc": auc,
        })
        print(f"  {set_name:40s} ({len(used):2d} feats): AUC = {auc:.4f}")

    # --- Key pairwise AUC ---
    key_pairs = [
        ("sustained", "asymmetric_decay"),
        ("sustained", "oscillatory"),
        ("sustained", "single_peak"),
        ("single_peak", "oscillatory"),
        ("asymmetric_decay", "oscillatory"),
    ]

    print("\n[2] Pairwise AUC for key morphology contrasts:")
    for morph_a, morph_b in key_pairs:
        df_a = df_feat[df_feat["ground_truth_morphology"] == morph_a]
        df_b = df_feat[df_feat["ground_truth_morphology"] == morph_b]
        df_pair = pd.concat([df_a, df_b], ignore_index=True)
        y_pair = np.array([0] * len(df_a) + [1] * len(df_b))

        print(f"\n  {morph_a} vs {morph_b}:")
        for set_name, feat_names in FEATURE_SETS:
            X, used = _prepare_X(df_pair, feat_names)
            if X is None:
                continue
            auc = compute_pairwise_auc(X, y_pair)
            results.append({
                "comparison": f"{morph_a}_vs_{morph_b}",
                "feature_set": set_name,
                "n_features": len(used),
                "auc": auc,
            })
            print(f"    {set_name:40s} ({len(used):2d} feats): AUC = {auc:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "incremental_auc.csv", index=False)
    return df_results


# =====================================================================
# EXPERIMENT 3: NOISE ROBUSTNESS
# =====================================================================

def run_noise_robustness(noise_levels=None, n_per_type=200, seed=42):
    """Test feature discrimination at increasing noise levels."""
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Noise robustness analysis")
    print(f"{'='*60}")

    results = []
    for noise_sigma in noise_levels:
        # Generate data at this noise level
        all_features = []
        for morph_name, gen_func in MORPHOLOGY_GENERATORS.items():
            for i in range(n_per_type):
                rng = np.random.default_rng(seed + hash(morph_name) % 10000 + i)
                wcc = gen_func(n_samples=300, hz=1.0, noise_sigma=noise_sigma, rng=rng)
                feats = extract_all_features(wcc, hz=1.0, wcc_window_sec=300.0)
                feats["ground_truth_morphology"] = morph_name
                all_features.append(feats)

        df = pd.DataFrame(all_features)
        morphologies = sorted(df["ground_truth_morphology"].unique())
        y = df["ground_truth_morphology"].map(
            {m: i for i, m in enumerate(morphologies)}).values

        for set_name, feat_names in FEATURE_SETS:
            X, used = _prepare_X(df, feat_names)
            if X is None:
                continue
            auc = compute_multiclass_auc(X, y)
            results.append({
                "noise_sigma": noise_sigma,
                "feature_set": set_name,
                "n_features": len(used),
                "auc": auc,
            })
        ms_auc = [r["auc"] for r in results if r["noise_sigma"] == noise_sigma
                   and r["feature_set"] == FEATURE_SETS[0][0]][0]
        full_auc = [r["auc"] for r in results if r["noise_sigma"] == noise_sigma
                    and r["feature_set"] == FEATURE_SETS[-1][0]][0]
        print(f"  noise={noise_sigma:.2f}: mean_sync AUC={ms_auc:.3f}, "
              f"full AUC={full_auc:.3f}")

    df_noise = pd.DataFrame(results)
    df_noise.to_csv(OUT_DIR / "noise_robustness.csv", index=False)
    return df_noise


# =====================================================================
# EXPERIMENT 4: REAL DATA MORPHOLOGY RESOLUTION
# =====================================================================

def run_real_data_analysis():
    """Show progressive morphology resolution on Lerique 2024 dataset."""
    print(f"\n{'='*60}")
    print("PART B: Real data morphology resolution (Lerique 2024)")
    print(f"{'='*60}")

    feat_path = PROJECT_ROOT / "artifacts" / "realtest" / "lerique_2024" / "per_record_features.csv"
    morph_path = PROJECT_ROOT / "artifacts" / "morphology_census_lerique.csv"

    if not feat_path.exists():
        print(f"  WARNING: {feat_path} not found, skipping")
        return None
    if not morph_path.exists():
        print(f"  WARNING: {morph_path} not found, skipping")
        return None

    df_feat = pd.read_csv(feat_path)
    df_morph = pd.read_csv(morph_path)

    # Merge on composite key
    df_feat["merge_key"] = (df_feat["dyad_label"] + "__" +
                            df_feat["modality"] + "__" +
                            df_feat["condition_unit"])
    df_morph["merge_key"] = (df_morph["dyad_label"] + "__" +
                             df_morph["modality"] + "__" +
                             df_morph["condition_unit"])
    df = df_feat.merge(df_morph[["merge_key", "label"]], on="merge_key", how="left")
    df = df.rename(columns={"label": "morphology_label"})
    df = df.drop(columns=["merge_key"])

    # Filter to major morphology types (≥5 records)
    counts = df["morphology_label"].value_counts()
    major = counts[counts >= 5].index.tolist()
    df = df[df["morphology_label"].isin(major)].copy()

    print(f"\n  Records: {len(df)}")
    print(f"  Morphology distribution:")
    for m in major:
        n = (df["morphology_label"] == m).sum()
        ms = df.loc[df["morphology_label"] == m, "mean_synchrony"]
        print(f"    {m:20s}: n={n:3d}  mean_sync={ms.mean():.3f}±{ms.std():.3f}")

    # Incremental AUC on real morphology
    results = []
    if len(major) >= 2:
        y = df["morphology_label"].map({m: i for i, m in enumerate(major)}).values

        real_feature_sets = [
            ("1: mean_synchrony only",  ["mean_synchrony"]),
            ("2: + peak_amplitude",     ["mean_synchrony", "peak_amplitude"]),
            ("3: + dwell_time",         ["mean_synchrony", "peak_amplitude", "dwell_time"]),
            ("4: + switching_rate",     ["mean_synchrony", "peak_amplitude",
                                         "dwell_time", "switching_rate"]),
            ("5: + synchrony_entropy",  ["mean_synchrony", "peak_amplitude",
                                         "dwell_time", "switching_rate",
                                         "synchrony_entropy"]),
            ("6: + onset/rise/recovery",["mean_synchrony", "peak_amplitude",
                                         "dwell_time", "switching_rate",
                                         "synchrony_entropy",
                                         "onset_latency", "rise_time", "recovery_time"]),
        ]

        print(f"\n  Multi-class morphology AUC ({len(major)} classes):")
        for set_name, feat_names in real_feature_sets:
            X, used = _prepare_X(df, feat_names)
            if X is None:
                continue
            auc = compute_multiclass_auc(X, y)
            results.append({
                "comparison": "real_morphology_multiclass",
                "feature_set": set_name,
                "n_features": len(used),
                "auc": auc,
            })
            print(f"    {set_name:40s} ({len(used):2d} feats): AUC = {auc:.4f}")

    df_real = pd.DataFrame(results)
    if len(df_real) > 0:
        df_real.to_csv(OUT_DIR / "real_morphology_auc.csv", index=False)
    df.to_csv(OUT_DIR / "real_morphology_features.csv", index=False)

    return df_real


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SyncPipe Feature Incremental Value Analysis (v2)")
    print("=" * 60)

    # Part A: Synthetic
    df_feat, df_traj = run_synthetic_experiment(
        n_per_type=250, noise_sigma=0.10, seed=42)

    # Part A2: Incremental AUC
    df_auc = run_incremental_auc(df_feat)

    # Part A3: Noise robustness
    df_noise = run_noise_robustness(
        noise_levels=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        n_per_type=200, seed=42)

    # Part B: Real data
    df_real = run_real_data_analysis()

    print("\n" + "=" * 60)
    print("DONE. Results saved to:", OUT_DIR)
    print("=" * 60)

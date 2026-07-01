#!/usr/bin/env python
"""
run_kuramoto_incremental_value.py — Kuramoto gray-box validation
================================================================

Validates SyncPipe features on synchrony trajectories that EMERGE from
coupled-oscillator dynamics rather than direct parametric formulas.

Theory:
  Two Kuramoto oscillators: dθ₁/dt = ω₁ + (K/2)·sin(θ₂−θ₁)
                            dθ₂/dt = ω₂ + (K/2)·sin(θ₁−θ₂)

  Phase difference:        d(Δθ)/dt = Δω − K·sin(Δθ)      [Adler equation]
  Order parameter:         r(t) = |cos(Δθ(t)/2)|           ∈ [0, 1]

  r(t) is the synchrony measure fed into the SyncPipe feature pipeline.
  It EMERGES from φ dynamics — no WCC(t)=f(t)+ε formula is hand-crafted.

Morphology → Kuramoto parameter regimes:
  1. Sustained   : K << |Δω|  →  fast drift  →  r oscillates rapidly (mean ≈ 2/π)
  2. Single-peak : K > |Δω|, θ₀ far from fixed point  →  transient toward locking
  3. Oscillatory : K ≈ |Δω| (just below)  →  bottleneck drift  →  slow-fast cycles
  4. Asym. decay : K(t) decays linearly  →  locked→drift transition

Defense:
  If SyncPipe features classify Kuramoto-emergent r(t) as effectively as
  parametric WCC, then the features capture REAL coordination dynamics,
  not generator-induced artifacts.

Outputs:
  artifacts/incremental_value/kuramoto_*.csv
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
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
)
from multisync.morphology import classify_morphology

OUT_DIR = PROJECT_ROOT / "artifacts" / "incremental_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# KURAMOTO SYNCHRONY GENERATOR
# =====================================================================

def solve_phase_difference(K_func, delta_omega, theta_0, T, n_fine=2000):
    """
    Solve d(Δθ)/dt = Δω − K(t)·sin(Δθ) using scipy solve_ivp.

    Parameters
    ----------
    K_func : float or callable(t)
        Coupling strength. If callable, must accept scalar t and return scalar K.
    delta_omega : float
        Natural frequency difference (Δω = ω₂ − ω₁).
    theta_0 : float
        Initial phase difference.
    T : float
        Total integration time.
    n_fine : int
        Number of output time points (fine grid).

    Returns
    -------
    t : np.ndarray  (n_fine,)
    r : np.ndarray  (n_fine,)  order parameter r(t) = |cos(Δθ(t)/2)|
    """
    if callable(K_func):
        def ode(t, y):
            K = K_func(t)
            dtheta = delta_omega - K * np.sin(y[0])
            return [dtheta]
    else:
        K = K_func
        def ode(t, y):
            dtheta = delta_omega - K * np.sin(y[0])
            return [dtheta]

    t_eval = np.linspace(0, T, n_fine)
    sol = solve_ivp(ode, [0, T], [theta_0], t_eval=t_eval,
                    method='RK45', rtol=1e-9, atol=1e-12)

    delta_theta = sol.y[0]
    r = np.abs(np.cos(delta_theta / 2.0))
    return sol.t, r


def sample_to_n(r_fine, n=300):
    """Downsample a fine-grid r(t) to n equally-spaced samples."""
    indices = np.linspace(0, len(r_fine) - 1, n, dtype=int)
    return r_fine[indices]


def add_synchrony_noise(r, noise_sigma, rng=None):
    """Add Gaussian noise to r(t) and clip to [0, 1]."""
    if rng is None:
        rng = np.random.default_rng()
    r_noisy = r + rng.normal(0, noise_sigma, len(r))
    return np.clip(r_noisy, 0.0, 1.0)


# =====================================================================
# 4 MORPHOLOGY GENERATORS FROM KURAMOTO
# =====================================================================

# These parameters were empirically tuned to produce clear morphologies
# with approximately moderate mean synchrony levels.

KURAMOTO_PARAMS = {
    "sustained": {
        "K": 0.05,           # Very weak coupling → fast uniform drift
        "delta_omega": 0.7,  # Moderate frequency mismatch
        "theta_0": 0.0,
        "T": 50,             # Long integration for many cycles
        "description": "K=0.05, Δω=0.7 → fast drift. r(t) oscillates rapidly "
                       "(analytical mean = 2/π ≈ 0.637). Empirical μ≈0.633."
    },
    "single_peak": {
        # K(t) = 0.05 + 1.5·exp(−((t−10)/2.5)²)  —  Gaussian coupling bump
        # Brief strong coupling event creates transient r(t) peak
        "K_bump_center": 10.0,
        "K_bump_width": 2.5,
        "K_bump_amplitude": 1.5,
        "K_baseline": 0.05,
        "delta_omega": 0.8,
        "theta_0": 2.5,      # Start far from equilibrium at r≈|cos(1.25)|≈0.315
        "T": 40,
        "description": "K(t)=0.05+1.5·exp(−((t−10)/2.5)²), Δω=0.8, θ₀=2.5. "
                       "Brief coupling bump → transient r peak near t=10. "
                       "Returns to drift after. μ≈0.652."
    },
    "oscillatory": {
        "K": 0.62,           # Near the locking boundary
        "delta_omega": 0.68, # Δω slightly > K → drift with bottleneck
        "theta_0": 0.0,
        "T": 80,             # Long enough to see multiple slow cycles
        "description": "K=0.62, Δω=0.68 → K/Δω=0.91, near-boundary drift. "
                       "Bottleneck at θ≈arcsin(0.62/0.68)≈1.14 creates stick-slip. "
                       "r(t) shows asymmetric slow-fast oscillations. μ≈0.690."
    },
    "asymmetric_decay": {
        "K_start": 1.2,      # Start modestly coupled
        "K_tau": 8.0,        # Exponential decay rate: K(t)=1.2·exp(−t/8)
        "delta_omega": 0.5,
        "theta_0": 0.0,
        "T": 30,
        "description": "K(t)=1.2·exp(−t/8), Δω=0.5. Exponential coupling decay. "
                       "Early: K>0.5 → locked (r≈0.94). Late: K<0.5 → drift (μ≈0.637). "
                       "Locked-to-drift transition at t≈7.0. Overall μ≈0.684."
    },
}


def generate_kuramoto_sustained(n_samples=300, noise_sigma=0.10, rng=None):
    """Sustained: K << Δω → fast drift, r oscillates rapidly. μ ≈ 2/π."""
    p = KURAMOTO_PARAMS["sustained"]
    _, r_fine = solve_phase_difference(p["K"], p["delta_omega"],
                                        p["theta_0"], p["T"], n_fine=2000)
    r = sample_to_n(r_fine, n_samples)
    return add_synchrony_noise(r, noise_sigma, rng)


def generate_kuramoto_single_peak(n_samples=300, noise_sigma=0.10, rng=None):
    """Single peak: Gaussian coupling bump K(t) → transient r peak."""
    p = KURAMOTO_PARAMS["single_peak"]
    c, w, A, K0 = p["K_bump_center"], p["K_bump_width"], p["K_bump_amplitude"], p["K_baseline"]
    K_func = lambda t: K0 + A * np.exp(-((t - c) / w) ** 2)
    _, r_fine = solve_phase_difference(K_func, p["delta_omega"],
                                        p["theta_0"], p["T"], n_fine=2000)
    r = sample_to_n(r_fine, n_samples)
    return add_synchrony_noise(r, noise_sigma, rng)


def generate_kuramoto_oscillatory(n_samples=300, noise_sigma=0.10, rng=None):
    """Oscillatory: near-boundary drift with bottleneck → asymmetric slow-fast cycles."""
    p = KURAMOTO_PARAMS["oscillatory"]
    _, r_fine = solve_phase_difference(p["K"], p["delta_omega"],
                                        p["theta_0"], p["T"], n_fine=2000)
    r = sample_to_n(r_fine, n_samples)
    return add_synchrony_noise(r, noise_sigma, rng)


def generate_kuramoto_asymmetric_decay(n_samples=300, noise_sigma=0.10, rng=None):
    """Asymmetric decay: exponential K(t) → locked→drift transition."""
    p = KURAMOTO_PARAMS["asymmetric_decay"]
    K_func = lambda t: p["K_start"] * np.exp(-t / p["K_tau"])
    _, r_fine = solve_phase_difference(K_func, p["delta_omega"],
                                        p["theta_0"], p["T"], n_fine=2000)
    r = sample_to_n(r_fine, n_samples)
    return add_synchrony_noise(r, noise_sigma, rng)


KURAMOTO_GENERATORS = {
    "sustained": generate_kuramoto_sustained,
    "single_peak": generate_kuramoto_single_peak,
    "oscillatory": generate_kuramoto_oscillatory,
    "asymmetric_decay": generate_kuramoto_asymmetric_decay,
}

# =====================================================================  
# FEATURE EXTRACTION (same as white-box pipeline)
# =====================================================================

def extract_all_features(sync, hz=1.0, wcc_window_sec=300.0):
    """Extract all features from a synchrony trajectory (WCC or Kuramoto r)."""
    df = extract_features(sync, hz=hz, wcc_window_sec=wcc_window_sec)
    result = df.to_dict()
    result["first_peak_time"] = compute_first_peak_time(sync, hz=hz)
    result["baseline_fraction"] = compute_baseline_fraction(sync)
    result["inter_peak_cv"] = compute_inter_peak_cv(sync, hz=hz)
    morph = classify_morphology(sync, hz=hz)
    result["morphology_label"] = morph.label
    return result


# =====================================================================
# CLASSIFICATION UTILITIES (same as white-box pipeline)
# =====================================================================

FEATURE_SETS = [
    ("1: mean_sync",     ["mean_synchrony"]),
    ("2: + peak_ampl",   ["mean_synchrony", "peak_amplitude"]),
    ("3: + dwell_time",   ["mean_synchrony", "peak_amplitude", "dwell_time"]),
    ("4: + switch_rate",  ["mean_synchrony", "peak_amplitude",
                            "dwell_time", "switching_rate"]),
    ("5: + sync_entropy", ["mean_synchrony", "peak_amplitude",
                            "dwell_time", "switching_rate",
                            "synchrony_entropy"]),
    ("6: + onset/rise/rec",["mean_synchrony", "peak_amplitude",
                            "dwell_time", "switching_rate",
                            "synchrony_entropy",
                            "onset_latency", "rise_time", "recovery_time"]),
    ("7: + timing",        ["mean_synchrony", "peak_amplitude",
                            "dwell_time", "switching_rate",
                            "synchrony_entropy",
                            "onset_latency", "rise_time", "recovery_time",
                            "first_peak_time", "baseline_fraction",
                            "inter_peak_cv"]),
]

FEATURE_LABELS_SHORT = [
    "mean\n_sync", "+ peak\n_ampl", "+ dwell\n_time", "+ switch\n_rate",
    "+ sync\n_entropy", "+ onset/rise\n/recovery", "+ timing"
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
    stds = X.std(axis=0)
    mask = stds > 1e-10
    X = X[:, mask]
    used = [f for f, m in zip(available, mask) if m]
    return X, used


def compute_multiclass_auc(X, y, n_splits=5, random_state=42):
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
# EXPERIMENT 1: KURAMOTO DATA GENERATION
# =====================================================================

def run_kuramoto_experiment(n_per_type=250, noise_sigma=0.10, seed=42):
    """Generate Kuramoto-emergent r(t) and extract features."""
    print(f"\n{'='*60}")
    print(f"KURAMOTO: Gray-box validation (n={n_per_type}/type, σ={noise_sigma})")
    print(f"{'='*60}")

    print("\nKuramoto parameter regimes:")
    for name, p in KURAMOTO_PARAMS.items():
        print(f"  {name:20s}: {p['description']}")

    all_features = []
    all_trajectories = []

    for morph_name, gen_func in KURAMOTO_GENERATORS.items():
        for i in range(n_per_type):
            rng = np.random.default_rng(seed + hash(morph_name) % 10000 + i)
            sync = gen_func(n_samples=300, noise_sigma=noise_sigma, rng=rng)

            # Store sample trajectories for visualization
            if i == 0:
                for t_idx, val in enumerate(sync):
                    all_trajectories.append({
                        "morphology": morph_name,
                        "time_idx": t_idx,
                        "sync": round(float(val), 4),
                    })

            feats = extract_all_features(sync, hz=1.0, wcc_window_sec=300.0)
            feats["ground_truth_morphology"] = morph_name
            feats["replication"] = i
            feats["source"] = "kuramoto"
            all_features.append(feats)

    df_feat = pd.DataFrame(all_features)
    df_traj = pd.DataFrame(all_trajectories)

    df_traj.to_csv(OUT_DIR / "kuramoto_trajectories.csv", index=False)
    df_feat.to_csv(OUT_DIR / "kuramoto_features.csv", index=False)

    print("\nKuramoto synchrony statistics by morphology:")
    for morph in KURAMOTO_GENERATORS:
        ms = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "mean_synchrony"]
        pa = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "peak_amplitude"]
        dt = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "dwell_time"]
        se = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "synchrony_entropy"]
        sr = df_feat.loc[df_feat["ground_truth_morphology"] == morph, "switching_rate"]
        print(f"  {morph:22s}: mean_sync={ms.mean():.3f}±{ms.std():.3f}  "
              f"peak={pa.mean():.3f}±{pa.std():.3f}  "
              f"dwell={dt.mean():.1f}±{dt.std():.1f}  "
              f"entropy={se.mean():.2f}±{se.std():.2f}  "
              f"switch={sr.mean():.2f}±{sr.std():.2f}")

    return df_feat, df_traj


# =====================================================================
# EXPERIMENT 2: INCREMENTAL AUC ON KURAMOTO DATA
# =====================================================================

def run_kuramoto_auc(df_feat):
    """Run incremental AUC analysis on Kuramoto data."""
    print(f"\n{'='*60}")
    print("KURAMOTO: Incremental AUC analysis")
    print(f"{'='*60}")

    results = []
    morphologies = sorted(df_feat["ground_truth_morphology"].unique())
    y_all = df_feat["ground_truth_morphology"].map(
        {m: i for i, m in enumerate(morphologies)}).values

    # Multi-class
    print("\n[1] Multi-class macro AUC (4 morphologies):")
    for set_name, feat_names in FEATURE_SETS:
        X, used = _prepare_X(df_feat, feat_names)
        if X is None:
            continue
        auc = compute_multiclass_auc(X, y_all)
        results.append({
            "comparison": "multiclass_4morph",
            "feature_set": set_name,
            "source": "kuramoto",
            "n_features": len(used),
            "auc": auc,
        })
        print(f"  {set_name:30s}: AUC = {auc:.4f}")

    # All 6 pairwise
    pairs = [
        ("sustained", "single_peak"),
        ("sustained", "oscillatory"),
        ("sustained", "asymmetric_decay"),
        ("single_peak", "oscillatory"),
        ("single_peak", "asymmetric_decay"),
        ("oscillatory", "asymmetric_decay"),
    ]

    print("\n[2] Pairwise AUC:")
    for morph_a, morph_b in pairs:
        df_a = df_feat[df_feat["ground_truth_morphology"] == morph_a]
        df_b = df_feat[df_feat["ground_truth_morphology"] == morph_b]
        df_pair = pd.concat([df_a, df_b], ignore_index=True)
        y_pair = np.array([0] * len(df_a) + [1] * len(df_b))

        pair_aucs = []
        for set_name, feat_names in FEATURE_SETS:
            X, used = _prepare_X(df_pair, feat_names)
            if X is None:
                continue
            auc = compute_pairwise_auc(X, y_pair)
            results.append({
                "comparison": f"{morph_a}_vs_{morph_b}",
                "feature_set": set_name,
                "source": "kuramoto",
                "n_features": len(used),
                "auc": auc,
            })
            pair_aucs.append(f"{auc:.3f}")
        print(f"  {morph_a:12s} vs {morph_b:20s}: {' → '.join(pair_aucs)}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "kuramoto_incremental_auc.csv", index=False)
    return df_results


# =====================================================================
# EXPERIMENT 3: NOISE ROBUSTNESS (KURAMOTO)
# =====================================================================

def run_kuramoto_noise_robustness(noise_levels=None, n_per_type=150, seed=42):
    """Test Kuramoto feature discrimination at increasing noise levels."""
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    print(f"\n{'='*60}")
    print("KURAMOTO: Noise robustness analysis")
    print(f"{'='*60}")

    results = []
    for noise_sigma in noise_levels:
        all_features = []
        for morph_name, gen_func in KURAMOTO_GENERATORS.items():
            for i in range(n_per_type):
                rng = np.random.default_rng(seed + hash(morph_name) % 10000 + i)
                sync = gen_func(n_samples=300, noise_sigma=noise_sigma, rng=rng)
                feats = extract_all_features(sync, hz=1.0, wcc_window_sec=300.0)
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
                "source": "kuramoto",
                "n_features": len(used),
                "auc": auc,
            })

        ms_auc = [r["auc"] for r in results if r["noise_sigma"] == noise_sigma
                   and r["feature_set"] == FEATURE_SETS[0][0]][0]
        full_auc = [r["auc"] for r in results if r["noise_sigma"] == noise_sigma
                    and r["feature_set"] == FEATURE_SETS[-1][0]][0]
        print(f"  σ={noise_sigma:.2f}: mean_sync={ms_auc:.3f}, full={full_auc:.3f}")

    df_noise = pd.DataFrame(results)
    df_noise.to_csv(OUT_DIR / "kuramoto_noise_robustness.csv", index=False)
    return df_noise


# =====================================================================
# EXPERIMENT 4: WHITE-BOX vs GRAY-BOX COMPARISON
# =====================================================================

def run_comparison_summary():
    """Load both parametric and Kuramoto results, compute comparison metrics."""
    print(f"\n{'='*60}")
    print("WHITE-BOX (parametric) vs GRAY-BOX (Kuramoto) COMPARISON")
    print(f"{'='*60}")

    wb_path = OUT_DIR / "incremental_auc.csv"
    kb_path = OUT_DIR / "kuramoto_incremental_auc.csv"

    if not wb_path.exists():
        print(f"  WARNING: white-box results not found at {wb_path}")
        return None
    if not kb_path.exists():
        print(f"  WARNING: kuramoto results not found at {kb_path}")
        return None

    df_wb = pd.read_csv(wb_path)
    df_kb = pd.read_csv(kb_path)

    # Multi-class comparison
    wb_mc = df_wb[df_wb["comparison"] == "multiclass_4morph"][["feature_set", "auc"]].copy()
    kb_mc = df_kb[df_kb["comparison"] == "multiclass_4morph"][["feature_set", "auc"]].copy()

    comparison = []
    print("\nMulti-class AUC comparison:")
    print(f"  {'Feature set':30s}  {'Parametric':>10s}  {'Kuramoto':>10s}  {'Δ':>8s}")
    print(f"  {'-'*30s}  {'-'*10s}  {'-'*10s}  {'-'*8s}")

    for _, row_w in wb_mc.iterrows():
        row_k = kb_mc[kb_mc["feature_set"] == row_w["feature_set"]]
        if len(row_k) == 0:
            continue
        auc_w = row_w["auc"]
        auc_k = row_k.iloc[0]["auc"]
        delta = auc_k - auc_w
        comparison.append({
            "feature_set": row_w["feature_set"],
            "auc_parametric": auc_w,
            "auc_kuramoto": auc_k,
            "delta": delta,
        })
        print(f"  {row_w['feature_set']:30s}  {auc_w:10.4f}  {auc_k:10.4f}  {delta:+8.4f}")

    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv(OUT_DIR / "kuramoto_vs_parametric_comparison.csv", index=False)
    return df_comp


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SyncPipe Kuramoto Gray-Box Validation")
    print("=" * 60)
    print("\nKey difference from white-box:")
    print("  White-box: WCC(t) = f(t) + ε  (direct parametric formula)")
    print("  Gray-box:  r(t) = |cos(Δθ(t)/2)|  (emergent from Kuramoto φ dynamics)")
    print()

    # Part 1: Generate Kuramoto data and extract features
    df_feat, df_traj = run_kuramoto_experiment(
        n_per_type=250, noise_sigma=0.10, seed=42)

    # Part 2: Incremental AUC
    df_auc = run_kuramoto_auc(df_feat)

    # Part 3: Noise robustness
    df_noise = run_kuramoto_noise_robustness(
        noise_levels=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        n_per_type=150, seed=42)

    # Part 4: White-box vs Gray-box comparison
    df_comp = run_comparison_summary()

    print("\n" + "=" * 60)
    print("DONE. Results saved to:", OUT_DIR)
    print("=" * 60)

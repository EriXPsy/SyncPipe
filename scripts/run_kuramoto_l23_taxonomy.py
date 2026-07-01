"""Kuramoto L2+L3 taxonomy validation — v3 (exact mean-matching).

Key fix vs v2:
    - Generate 600 samples/condition, then exact 1:1 match on mean_sync
      (caliper=0.005). Keeps N=60 matched pairs.
    - L3 Temporal: early_peak vs delayed_peak — naturally similar mean,
      but still apply matching for safety.
    - All conditions: T=60s, n_samples=300, hz=5.0 Hz.

Output
------
- artifacts/incremental_value/kuramoto_l23_v3_features.csv
- artifacts/incremental_value/kuramoto_l23_v3_auc.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "incremental_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from multisync.feature_definitions import extract_features

# =====================================================================
# SHARED PARAMS
# =====================================================================
T_SEC = 60.0
N_SAMPLES = 300
HZ = N_SAMPLES / T_SEC          # 5.0 Hz
WCC_WINDOW_SEC = T_SEC
NOISE_SIGMA = 0.10
N_GENERATE = 600                 # generate many, then match
N_KEEP = 60                    # matched pairs to keep
SEED = 42
CALIPER = 0.005                 # max mean_sync difference for a match

# =====================================================================
# Kuramoto solver (same as v2)
# =====================================================================

def solve_phase_difference(K_func, delta_omega, theta_0, T, n_fine=2000):
    def ode(t, y):
        K = K_func(t) if callable(K_func) else K_func
        return [delta_omega - K * np.sin(y[0])]
    t_eval = np.linspace(0, T, n_fine)
    sol = solve_ivp(ode, [0, T], [theta_0], t_eval=t_eval,
                    method='RK45', rtol=1e-9, atol=1e-12)
    delta_theta = sol.y[0]
    return np.abs(np.cos(delta_theta / 2.0))


def sample_to_n(r_fine, n=300):
    if len(r_fine) == n:
        return r_fine.copy()
    idx = np.linspace(0, len(r_fine) - 1, n).astype(int)
    return r_fine[idx]


def add_noise(r, sigma, rng):
    return np.clip(r + rng.normal(0, sigma, size=len(r)), 0.0, 1.0)


# =====================================================================
# Generators (all T=60s)
# =====================================================================

def gen_sustained(rng):
    K = 0.05; domega = 0.7; theta0 = 0.0
    r_fine = solve_phase_difference(K, domega, theta0, T_SEC, n_fine=2000)
    return add_noise(sample_to_n(r_fine, N_SAMPLES), NOISE_SIGMA, rng)


def gen_single_peak(rng):
    K_func = lambda t: 0.05 + 1.5 * np.exp(-((t - 10.0) / 2.5) ** 2)
    r_fine = solve_phase_difference(K_func, 0.8, 2.5, T_SEC, n_fine=2000)
    return add_noise(sample_to_n(r_fine, N_SAMPLES), NOISE_SIGMA, rng)


def gen_delayed_peak(rng):
    K_func = lambda t: 0.05 + 1.5 * np.exp(-((t - 30.0) / 2.5) ** 2)
    r_fine = solve_phase_difference(K_func, 0.8, 2.5, T_SEC, n_fine=2000)
    return add_noise(sample_to_n(r_fine, N_SAMPLES), NOISE_SIGMA, rng)


# =====================================================================
# Exact mean-matching
# =====================================================================

def generate_and_match(gen_a, gen_b, n_gen=N_GENERATE, n_keep=N_KEEP,
                     seed=SEED):
    """Generate n_gen samples per condition, then 1:1 match on mean_sync.

    Returns
    -------
    matched_a : list of ndarray
    matched_b : list of ndarray
    mean_diffs : list of float  (for diagnostics)
    """
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 9999)

    # Generate pool A
    pool_a = []
    for i in range(n_gen):
        rng = np.random.default_rng(seed + i)
        sync = gen_a(rng)
        ms = float(np.mean(sync))
        feats = extract_features_clean(sync)
        pool_a.append({"sync": sync, "mean_sync": ms, "feats": feats})

    # Generate pool B
    pool_b = []
    for i in range(n_gen):
        rng = np.random.default_rng(seed + 9999 + i)
        sync = gen_b(rng)
        ms = float(np.mean(sync))
        feats = extract_features_clean(sync)
        pool_b.append({"sync": sync, "mean_sync": ms, "feats": feats})

    # Match: for each a, find nearest b within caliper
    # Greedy matching WITH caliper check
    used_b = set()
    matched_a = []
    matched_b = []
    mean_diffs = []

    # Sort pool_a by mean_sync for stable matching
    pool_a_sorted = sorted(pool_a, key=lambda x: x["mean_sync"])

    for a in pool_a_sorted:
        best_b = None
        best_diff = CALIPER + 1.0
        for j, b in enumerate(pool_b):
            if j in used_b:
                continue
            diff = abs(a["mean_sync"] - b["mean_sync"])
            if diff < best_diff:
                best_diff = diff
                best_b = j
        # CALIPER CHECK — skip if no match within caliper
        if best_b is not None and best_diff <= CALIPER:
            used_b.add(best_b)
            matched_a.append(a)
            matched_b.append(pool_b[best_b])
            mean_diffs.append(best_diff)
        if len(matched_a) >= n_keep:
            break

    if len(matched_a) < n_keep:
        print(f"  WARNING: only {len(matched_a)} matched pairs found "
              f"(caliper={CALIPER})")

    print(f"  Matched {len(matched_a)} pairs, "
          f"mean |Δmean_sync|={np.mean(mean_diffs):.6f}, "
          f"max |Δ|={np.max(mean_diffs):.6f}")

    # Return as feature dicts (for incremental AUC)
    rows_a = []
    rows_b = []
    for a in matched_a:
        row = a["feats"].copy()
        row["condition"] = "A"
        rows_a.append(row)
    for b in matched_b:
        row = b["feats"].copy()
        row["condition"] = "B"
        rows_b.append(row)

    return rows_a + rows_b, np.array(mean_diffs)


def extract_features_clean(sync):
    """Extract features, return as dict (no 'condition' key)."""
    df = extract_features(sync, hz=HZ, wcc_window_sec=WCC_WINDOW_SEC)
    return df.to_dict()


# =====================================================================
# Incremental AUC (same as v2)
# =====================================================================

FEATURE_STEPS = [
    ("L1: mean_sync",       ["mean_synchrony"]),
    ("L1: +peak_ampl",     ["mean_synchrony", "peak_amplitude"]),
    ("L2: +dwell",         ["mean_synchrony", "peak_amplitude", "dwell_time"]),
    ("L2: +switch",        ["mean_synchrony", "peak_amplitude",
                              "dwell_time", "switching_rate"]),
    ("L2: +entropy",       ["mean_synchrony", "peak_amplitude",
                              "dwell_time", "switching_rate",
                              "synchrony_entropy"]),
    ("L3: +onset",        ["mean_synchrony", "peak_amplitude",
                              "dwell_time", "switching_rate",
                              "synchrony_entropy",
                              "onset_latency"]),
    ("L3: +rise+rec",     ["mean_synchrony", "peak_amplitude",
                              "dwell_time", "switching_rate",
                              "synchrony_entropy",
                              "onset_latency", "rise_time", "recovery_time"]),
]


def _prepare_X(df, feat_names):
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


def compute_auc(X, y, n_splits=5, random_state=42):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0,
                              solver="lbfgs", random_state=random_state),
        )
        n_actual = min(n_splits, max(2, min(np.bincount(y))))
        cv = StratifiedKFold(n_splits=n_actual,
                             shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())
    except Exception:
        return 0.5


# =====================================================================
# Run one contrast
# =====================================================================

def run_contrast(name, gen_a, gen_b, label_a, label_b):
    print(f"\n{'='*60}")
    print(f"{name}: {label_a} vs {label_b}")
    print(f"{'='*60}")

    rows, diffs = generate_and_match(gen_a, gen_b)
    df = pd.DataFrame(rows)

    # Diagnostics
    for cond in ["A", "B"]:
        sub = df[df["condition"] == cond]
        ms = sub["mean_synchrony"]
        print(f"  {cond}: mean_sync={ms.mean():.6f}±{ms.std():.6f}  "
              f"N={len(sub)}")

    print(f"  |Δmean_sync| max={np.max(diffs):.6f}")

    y = (df["condition"] == "B").astype(int).values

    results = []
    for step_name, feat_names in FEATURE_STEPS:
        X, used = _prepare_X(df, feat_names)
        if X is None:
            auc = float("nan")
        else:
            auc = compute_auc(X, y)
        results.append({
            "step": step_name,
            "auc": auc,
            "features_used": ",".join(used),
        })
        print(f"  {step_name:25s} AUC={auc:.4f}  [{','.join(used)}]")

    # Delta AUC
    aucs = [r["auc"] for r in results]
    for i in range(len(results)):
        if i == 0:
            results[i]["delta_auc"] = 0.0
        else:
            d = aucs[i] - aucs[i-1]
            results[i]["delta_auc"] = d if not np.isnan(d) else 0.0

    # Show delta
    print(f"\n  ΔAUC:")
    for r in results:
        if r["delta_auc"] > 0.01:
            print(f"    {r['step']:25s} ΔAUC={r['delta_auc']:+.4f}  ***")
        elif r["delta_auc"] > 0.005:
            print(f"    {r['step']:25s} ΔAUC={r['delta_auc']:+.4f}  **")
        else:
            print(f"    {r['step']:25s} ΔAUC={r['delta_auc']:+.4f}")

    return df, results


# =====================================================================
# Main
# =====================================================================

def main():
    all_results = {}

    # L2 Structure
    df_l2, res_l2 = run_contrast(
        "L2_Structure", gen_sustained, gen_single_peak,
        "sustained", "single_peak")
    all_results["L2_Structure"] = res_l2

    # L3 Temporal
    df_l3, res_l3 = run_contrast(
        "L3_Temporal", gen_single_peak, gen_delayed_peak,
        "early_peak", "delayed_peak")
    all_results["L3_Temporal"] = res_l3

    # Reference (no matching)
    print(f"\n{'='*60}")
    print("REFERENCE: sustained vs delayed_peak (no matching)")
    rows_ref = []
    for i in range(N_KEEP):
        rng = np.random.default_rng(SEED + i)
        sa = extract_features_clean(gen_sustained(rng))
        sa["condition"] = "sustained"
        rows_ref.append(sa)
        rng2 = np.random.default_rng(SEED + i + N_KEEP)
        sb = extract_features_clean(gen_delayed_peak(rng2))
        sb["condition"] = "delayed_peak"
        rows_ref.append(sb)
    df_ref = pd.DataFrame(rows_ref)
    y_ref = (df_ref["condition"] == "delayed_peak").astype(int).values
    res_ref = []
    for step_name, feat_names in FEATURE_STEPS:
        X, used = _prepare_X(df_ref, feat_names)
        auc = compute_auc(X, y_ref) if X is not None else float("nan")
        res_ref.append({"step": step_name, "auc": auc, "features_used": ",".join(used)})
        print(f"  {step_name:25s} AUC={auc:.4f}")
    all_results["Reference"] = res_ref

    # Save
    with open(OUT_DIR / "kuramoto_l23_v3_auc.json", "w") as f:
        json.dump({"results": all_results, "params": {
            "T_sec": T_SEC, "hz": HZ, "n_keep": N_KEEP,
            "n_generate": N_GENERATE, "caliper": CALIPER,
            "noise_sigma": NOISE_SIGMA, "seed": SEED,
        }}, f, indent=2)

    df_all = pd.concat([df_l2, df_l3, df_ref], ignore_index=True)
    df_all.to_csv(OUT_DIR / "kuramoto_l23_v3_features.csv", index=False)

    print(f"\nSaved to {OUT_DIR / 'kuramoto_l23_v3_*'}")
    return all_results


if __name__ == "__main__":
    main()

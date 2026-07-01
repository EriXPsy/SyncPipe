"""DEPRECATED / ARCHIVED — DO NOT USE AS AN EXISTENCE NULL.

This script preserves the *falsified* naive circular time-shift null for the
two exploratory timing descriptors, kept only as reproducible negative
evidence.  The circular shift preserves a trace's amplitude distribution,
autocorrelation, and spectrum — i.e. exactly the statistics that *determine*
``first_peak_time`` and ``inter_peak_cv`` — so the null is near-trivially
satisfied and carries no power (sustained quasi-null rejection ~0.10-0.12,
far above alpha, while true-peak conditions reject at ~0.00).  It was
therefore retired in favour of the cyclic block-bootstrap null in
``scripts/validate_timing_descriptors.py`` (Part 1).  See METHOD_LOG.md
§7c.2 for the full rationale and the side-by-side comparison.

The original docstring follows verbatim:

  Part 1 — L2 circular time-shift null (Kuramoto EGT synchrony traces).
           For each generated synchrony trace we build a null distribution by
           circularly shifting the trace (which preserves its amplitude
           distribution, autocorrelation, and spectrum while destroying the
           temporal anchoring of the peak), recompute the descriptor, and
           obtain a two-tailed Phipson-Smyth p-value.  We report rejection
           rates per condition.  ``sustained`` (no localized peak) serves as
           a quasi-null reference for first_peak_time.

  Part 2 — Incremental AUC (baseline = mean_synchrony, then +inter_peak_cv,
           +first_peak_time) on:
             (a) EGT-1 structure contrast  (sustained vs single_peak, mean-matched)
             (b) EGT-2 temporal contrast   (single_peak vs delayed_peak, mean-matched)
             (c) Lerique 2024 real data    (rest1 vs trials_concat, from WCC traces)
             (d) Gordon 2025 real data     (exp1 vs exp4, from WCC traces; illustrative)

NOTE on substrate: in the Kuramoto EGT setup the synchrony order parameter
r(t) IS the measurement substrate (there is no separate raw signal passed
through WCC), so a trace-level circular shift is the faithful L2 null here.
For the real datasets only WCC traces are available (no raw signals), so we
report incremental AUC only and do NOT run a raw-signal L2 null on them.

Run from multisync-core/:
    python scripts/validate_timing_descriptors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multisync.feature_definitions import (  # noqa: E402
    ONSET_THRESHOLD,
    compute_first_peak_time,
    compute_inter_peak_cv,
    compute_mean_synchrony,
)
from multisync.validation import phipson_smyth_p  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "timing_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Kuramoto EGT constants (matched to run_kuramoto_l23_taxonomy.py) ----
T_SEC = 60.0
N_SAMPLES = 300
HZ = N_SAMPLES / T_SEC  # 5.0 Hz
NOISE_SIGMA = 0.10
N_GENERATE = 300
N_KEEP = 60
CALIPER = 0.005
SEED = 42
N_SHIFT_NULL = 499
ALPHA = 0.05

TIMING = ("inter_peak_cv", "first_peak_time")


def _solve(K_func, domega, theta0):
    def ode(t, y):
        K = K_func(t) if callable(K_func) else K_func
        return [domega - K * np.sin(y[0])]
    t_eval = np.linspace(0, T_SEC, 2000)
    sol = solve_ivp(ode, [0, T_SEC], [theta0], t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-12)
    return np.abs(np.cos(sol.y[0] / 2.0))


def _resample(r):
    idx = np.linspace(0, len(r) - 1, N_SAMPLES).astype(int)
    return r[idx]


def _noise(r, rng):
    return np.clip(r + rng.normal(0, NOISE_SIGMA, size=len(r)), 0.0, 1.0)


def gen_sustained(rng):
    return _noise(_resample(_solve(0.05, 0.7, 0.0)), rng)


def gen_single_peak(rng):
    Kf = lambda t: 0.05 + 1.5 * np.exp(-((t - 10.0) / 2.5) ** 2)
    return _noise(_resample(_solve(Kf, 0.8, 2.5)), rng)


def gen_delayed_peak(rng):
    Kf = lambda t: 0.05 + 1.5 * np.exp(-((t - 30.0) / 2.5) ** 2)
    return _noise(_resample(_solve(Kf, 0.8, 2.5)), rng)


def timing_features(trace):
    return {
        "inter_peak_cv": compute_inter_peak_cv(trace, hz=HZ, threshold=ONSET_THRESHOLD),
        "first_peak_time": compute_first_peak_time(trace, hz=HZ, threshold=ONSET_THRESHOLD),
    }


# =====================================================================
# Part 1 — L2 circular time-shift null
# =====================================================================
def circular_shift_null(trace, feat_name, rng, n=N_SHIFT_NULL):
    """Null distribution from circular shifts of the trace."""
    nsamp = len(trace)
    vals = []
    # use distinct, non-trivial shifts
    shifts = rng.choice(np.arange(1, nsamp), size=min(n, nsamp - 1), replace=False)
    for s in shifts:
        shifted = np.roll(trace, int(s))
        v = timing_features(shifted)[feat_name]
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def run_l2_null(gen, label, n_traces=60, seed=SEED):
    """For each trace, two-tailed Phipson-Smyth p vs its circular-shift null."""
    rows = []
    for i in range(n_traces):
        rng = np.random.default_rng(seed + i)
        trace = gen(rng)
        obs = timing_features(trace)
        null_rng = np.random.default_rng(seed + 10_000 + i)
        for f in TIMING:
            o = obs[f]
            if not np.isfinite(o):
                rows.append(dict(condition=label, trace=i, feature=f,
                                 defined=False, p=np.nan, reject=False))
                continue
            null = circular_shift_null(trace, f, null_rng)
            if null.size < 10:
                rows.append(dict(condition=label, trace=i, feature=f,
                                 defined=True, p=np.nan, reject=False))
                continue
            # two-tailed: how extreme is obs vs the shift null (either direction)
            med = np.median(null)
            p_up = phipson_smyth_p(o, null, tail="upper")
            p_lo = phipson_smyth_p(-o, -null, tail="upper")
            p = min(1.0, 2.0 * min(p_up, p_lo))
            rows.append(dict(condition=label, trace=i, feature=f,
                             defined=True, p=float(p), reject=bool(p < ALPHA)))
    return rows


# =====================================================================
# Part 2 — Incremental AUC
# =====================================================================
INCREMENTAL_ORDER = [
    ("baseline: mean_synchrony", ["mean_synchrony"]),
    ("+inter_peak_cv", ["mean_synchrony", "inter_peak_cv"]),
    ("+first_peak_time", ["mean_synchrony", "inter_peak_cv", "first_peak_time"]),
]


def incremental_auc(df, label_col="label"):
    y = df[label_col].to_numpy().astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return {"error": "insufficient class balance", "n": len(df),
                "n_pos": int(y.sum())}
    results = []
    n_splits = min(5, int(min(np.bincount(y))))
    for name, feats in INCREMENTAL_ORDER:
        X = df[feats].to_numpy(dtype=float)
        # median-impute NaNs (timing descriptors may be undefined)
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.any(~np.isfinite(col)):
                med = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                col[~np.isfinite(col)] = med
                X[:, j] = col
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.transform(X[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], prob))
        results.append(dict(step=name, n_features=len(feats),
                            auc=float(np.mean(aucs)) if aucs else float("nan"),
                            auc_sd=float(np.std(aucs)) if aucs else float("nan")))
    # incremental deltas
    for k in range(1, len(results)):
        results[k]["delta_auc"] = results[k]["auc"] - results[k - 1]["auc"]
    return {"n": len(df), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "n_splits": n_splits, "steps": results}


def egt_contrast_df(gen_a, gen_b, label_a, label_b, mean_match=True):
    """Generate two conditions, optionally caliper-match on mean_synchrony."""
    rng = np.random.default_rng(SEED)
    pool_a, pool_b = [], []
    for i in range(N_GENERATE):
        ta = gen_a(np.random.default_rng(SEED + i))
        pool_a.append((float(compute_mean_synchrony(ta)), ta))
        tb = gen_b(np.random.default_rng(SEED + 9999 + i))
        pool_b.append((float(compute_mean_synchrony(tb)), tb))
    pairs_a, pairs_b = [], []
    if mean_match:
        used = set()
        for ms_a, ta in sorted(pool_a, key=lambda x: x[0]):
            best, bd = None, CALIPER + 1
            for j, (ms_b, _tb) in enumerate(pool_b):
                if j in used:
                    continue
                d = abs(ms_a - ms_b)
                if d < bd:
                    bd, best = d, j
            if best is not None and bd <= CALIPER:
                used.add(best)
                pairs_a.append(ta)
                pairs_b.append(pool_b[best][1])
            if len(pairs_a) >= N_KEEP:
                break
    else:
        pairs_a = [t for _, t in pool_a[:N_KEEP]]
        pairs_b = [t for _, t in pool_b[:N_KEEP]]

    rows = []
    for t in pairs_a:
        f = timing_features(t)
        f["mean_synchrony"] = float(compute_mean_synchrony(t))
        f["label"] = 0
        rows.append(f)
    for t in pairs_b:
        f = timing_features(t)
        f["mean_synchrony"] = float(compute_mean_synchrony(t))
        f["label"] = 1
        rows.append(f)
    df = pd.DataFrame(rows)
    return df, len(pairs_a), len(pairs_b)


def real_data_df(dataset, cond0, cond1):
    """Recompute mean_synchrony + timing descriptors per WCC trace."""
    src = ROOT / "artifacts" / "wcc_traces" / f"{dataset}_wcc_traces.csv"
    raw = pd.read_csv(src)
    raw = raw[raw["condition"].isin([cond0, cond1])]
    rows = []
    for _, r in raw.iterrows():
        wcc = np.asarray(json.loads(r["wcc_json"]), dtype=float)
        f = timing_features(wcc)
        f["mean_synchrony"] = float(compute_mean_synchrony(wcc))
        f["label"] = 0 if r["condition"] == cond0 else 1
        f["modality"] = r["modality"]
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    summary = {}

    # ---- Part 1 ----
    print("=" * 72)
    print("PART 1 — L2 circular time-shift null (Kuramoto EGT traces)")
    print("two-tailed Phipson-Smyth p; reject at alpha=0.05")
    print("=" * 72)
    l2_rows = []
    for gen, lab in ((gen_sustained, "sustained"),
                     (gen_single_peak, "single_peak"),
                     (gen_delayed_peak, "delayed_peak")):
        l2_rows.extend(run_l2_null(gen, lab))
    l2 = pd.DataFrame(l2_rows)
    l2.to_csv(OUT_DIR / "l2_circular_shift_null.csv", index=False)
    part1 = {}
    for (cond, feat), sub in l2.groupby(["condition", "feature"]):
        defined = sub["defined"].mean()
        sub_def = sub[sub["defined"] & sub["p"].notna()]
        rej = sub_def["reject"].mean() if len(sub_def) else float("nan")
        part1.setdefault(cond, {})[feat] = dict(
            definedness=round(float(defined), 3),
            n_testable=int(len(sub_def)),
            rejection_rate=round(float(rej), 3) if len(sub_def) else None,
        )
        print(f"  [{cond:12s}] {feat:16s} defined={defined:.2f} "
              f"n_testable={len(sub_def):3d} reject_rate="
              f"{rej:.3f}" if len(sub_def) else
              f"  [{cond:12s}] {feat:16s} defined={defined:.2f} (no testable)")
    summary["part1_l2_null"] = part1

    # ---- Part 2 ----
    print("\n" + "=" * 72)
    print("PART 2 — Incremental AUC (baseline mean_synchrony -> +timing)")
    print("=" * 72)
    part2 = {}

    df_struct, na, nb = egt_contrast_df(gen_sustained, gen_single_peak,
                                        "sustained", "single_peak")
    part2["EGT1_structure"] = dict(n_a=na, n_b=nb, **incremental_auc(df_struct))

    df_temp, na, nb = egt_contrast_df(gen_single_peak, gen_delayed_peak,
                                      "single_peak", "delayed_peak")
    part2["EGT2_temporal"] = dict(n_a=na, n_b=nb, **incremental_auc(df_temp))

    # Lerique: pool modalities (label rest vs task), and per-modality
    df_ler = real_data_df("lerique", "rest1", "trials_concat")
    part2["lerique_pooled"] = incremental_auc(df_ler)
    for mod, sub in df_ler.groupby("modality"):
        part2[f"lerique_{mod}"] = incremental_auc(sub.reset_index(drop=True))

    # Gordon: exp1 vs exp4 (illustrative; condition semantics not mapped)
    df_gor = real_data_df("gordon", "exp1", "exp4")
    part2["gordon_exp1_vs_exp4_illustrative"] = incremental_auc(df_gor)

    summary["part2_incremental_auc"] = part2

    for key, res in part2.items():
        print(f"\n[{key}]")
        if "error" in res:
            print(f"  skipped: {res['error']} (n={res.get('n')})")
            continue
        print(f"  n={res['n']} pos={res.get('n_pos')} neg={res.get('n_neg')} "
              f"folds={res.get('n_splits')}")
        for s in res["steps"]:
            d = s.get("delta_auc")
            dtxt = f"  Δ={d:+.3f}" if d is not None else ""
            print(f"    {s['step']:28s} AUC={s['auc']:.3f}±{s['auc_sd']:.3f}{dtxt}")

    (OUT_DIR / "timing_validation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nWrote {OUT_DIR}/timing_validation_summary.json")
    print(f"Wrote {OUT_DIR}/l2_circular_shift_null.csv")


if __name__ == "__main__":
    main()

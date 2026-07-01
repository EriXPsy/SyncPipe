"""Validation of the two exploratory timing descriptors
(``inter_peak_cv`` and ``first_peak_time``):

  Part 1 — Cyclic block-bootstrap peak-timing existence null.
           This null cuts each trace into equal-length blocks
           (default 5 s) and randomly permutes the BLOCK ORDER before
           re-concatenating.  This preserves within-block short-range
           autocorrelation / local peak shape AND the global marginal
           distribution, while destroying the long-range TEMPORAL ANCHORING
           of peaks (which block a peak falls in).  We recompute the
           descriptor on each permutation, obtain a two-tailed Phipson-Smyth
           p-value per trace, and report per-condition rejection rates.

           Success criteria (used to decide if this can ever be an existence
           test): true-peak conditions (single_peak, delayed_peak) should
           reject at a rate WELL above alpha; the no-localized-peak
           ``sustained`` quasi-null should reject NEAR alpha (=0.05).  If
           sustained also rejects far above alpha, this null is likewise
           unfit and we report the failure rather than over-claim.

  Part 2 — Incremental AUC (baseline = mean_synchrony, then +inter_peak_cv,
           +first_peak_time) on:
             (a) EGT-1 structure contrast  (sustained vs single_peak, mean-matched)
             (b) EGT-2 temporal contrast   (single_peak vs delayed_peak, mean-matched)
             (c) Lerique 2024 real data    (rest1 vs trials_concat, from WCC traces)
             (d) Gordon 2025 real data     (exp1 vs exp4, from WCC traces; illustrative)

NOTE on substrate: in the Kuramoto EGT setup the synchrony order parameter
r(t) IS the measurement substrate (no separate raw signal passed through
WCC), so a trace-level block permutation is a faithful existence null there.
For the real datasets only WCC traces (not raw signals) are available; we
ALSO run the block-permutation null on those traces but flag it as a
TRACE-LEVEL (not signal-level) null whose interpretation is weaker — it
asks whether the observed peak timing exceeds what block-reordering of the
*synchrony trace itself* would produce, not what raw-signal surrogates would.

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
N_BLOCK_NULL = 499
ALPHA = 0.05
BLOCK_SEC = 5.0  # block length for cyclic block bootstrap (see Part 1 docstring)

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


def timing_features_hz(trace, hz):
    return {
        "inter_peak_cv": compute_inter_peak_cv(trace, hz=hz, threshold=ONSET_THRESHOLD),
        "first_peak_time": compute_first_peak_time(trace, hz=hz, threshold=ONSET_THRESHOLD),
    }


def timing_features(trace):
    return timing_features_hz(trace, HZ)


# =====================================================================
# Part 1 — Cyclic block-bootstrap peak-timing existence null
# =====================================================================
def _block_permute(trace, block_len, rng):
    """Cut trace into equal blocks (last block may be short), permute the
    block ORDER, re-concatenate.  Preserves within-block structure + global
    marginal distribution; destroys long-range temporal anchoring."""
    n = len(trace)
    starts = list(range(0, n, block_len))
    blocks = [trace[s:s + block_len] for s in starts]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[k] for k in order])


def block_permute_null(trace, feat_name, block_len, rng, n=N_BLOCK_NULL):
    """Null distribution of a descriptor under block-order permutation."""
    vals = []
    for _ in range(n):
        perm = _block_permute(trace, block_len, rng)
        v = timing_features(perm)[feat_name]
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def run_block_null(gen, label, hz, block_len, n_traces=60, seed=SEED):
    """For each trace, two-tailed Phipson-Smyth p vs its block-permute null."""
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
            null = block_permute_null(trace, f, block_len, null_rng)
            if null.size < 10:
                rows.append(dict(condition=label, trace=i, feature=f,
                                 defined=True, p=np.nan, reject=False))
                continue
            # two-tailed: how extreme is obs vs the block-permute null
            p_up = phipson_smyth_p(o, null, tail="upper")
            p_lo = phipson_smyth_p(-o, -null, tail="upper")
            p = min(1.0, 2.0 * min(p_up, p_lo))
            rows.append(dict(condition=label, trace=i, feature=f,
                             defined=True, p=float(p), reject=bool(p < ALPHA)))
    return rows


def run_block_null_real(dataset, cond, hz_default, seed=SEED):
    """Block-permute null on real WCC traces (TRACE-LEVEL, weaker null)."""
    src = ROOT / "artifacts" / "wcc_traces" / f"{dataset}_wcc_traces.csv"
    raw = pd.read_csv(src)
    raw = raw[raw["condition"] == cond].reset_index(drop=True)
    rows = []
    for i, r in raw.iterrows():
        trace = np.asarray(json.loads(r["wcc_json"]), dtype=float)
        hz = float(r["hz"]) if "hz" in r and np.isfinite(r["hz"]) else hz_default
        block_len = max(2, int(round(BLOCK_SEC * hz)))
        obs = timing_features_hz(trace, hz)
        null_rng = np.random.default_rng(seed + 20_000 + i)
        for f in TIMING:
            o = obs[f]
            if not np.isfinite(o):
                rows.append(dict(dataset=dataset, condition=cond, trace=i,
                                 modality=r.get("modality"), feature=f,
                                 defined=False, p=np.nan, reject=False))
                continue
            vals = []
            for _ in range(N_BLOCK_NULL):
                perm = _block_permute(trace, block_len, null_rng)
                v = timing_features_hz(perm, hz)[f]
                if np.isfinite(v):
                    vals.append(v)
            null = np.asarray(vals, dtype=float)
            if null.size < 10:
                rows.append(dict(dataset=dataset, condition=cond, trace=i,
                                 modality=r.get("modality"), feature=f,
                                 defined=True, p=np.nan, reject=False))
                continue
            p_up = phipson_smyth_p(o, null, tail="upper")
            p_lo = phipson_smyth_p(-o, -null, tail="upper")
            p = min(1.0, 2.0 * min(p_up, p_lo))
            rows.append(dict(dataset=dataset, condition=cond, trace=i,
                             modality=r.get("modality"), feature=f,
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
    block_len_egt = max(2, int(round(BLOCK_SEC * HZ)))
    print("=" * 72)
    print(f"PART 1 — Cyclic block-bootstrap peak-timing null "
          f"(block={BLOCK_SEC:.0f}s = {block_len_egt} samples @ {HZ:.1f}Hz)")
    print("two-tailed Phipson-Smyth p; reject at alpha=0.05")
    print("=" * 72)

    # 1a — Kuramoto EGT (synchrony trace IS the substrate -> faithful null)
    print("\n-- 1a. Kuramoto EGT (faithful substrate-level null) --")
    egt_rows = []
    for gen, lab in ((gen_sustained, "sustained"),
                     (gen_single_peak, "single_peak"),
                     (gen_delayed_peak, "delayed_peak")):
        egt_rows.extend(run_block_null(gen, lab, HZ, block_len_egt))
    egt = pd.DataFrame(egt_rows)
    egt.to_csv(OUT_DIR / "block_permute_null_egt.csv", index=False)
    part1_egt = {}
    for (cond, feat), sub in egt.groupby(["condition", "feature"]):
        defined = sub["defined"].mean()
        sub_def = sub[sub["defined"] & sub["p"].notna()]
        rej = sub_def["reject"].mean() if len(sub_def) else float("nan")
        part1_egt.setdefault(cond, {})[feat] = dict(
            definedness=round(float(defined), 3),
            n_testable=int(len(sub_def)),
            rejection_rate=round(float(rej), 3) if len(sub_def) else None,
        )
        rate_txt = f"reject_rate={rej:.3f}" if len(sub_def) else "(no testable)"
        print(f"  [{cond:12s}] {feat:16s} defined={defined:.2f} "
              f"n_testable={len(sub_def):3d} {rate_txt}")

    # 1b — Real WCC traces (TRACE-LEVEL null; weaker interpretation)
    print("\n-- 1b. Real WCC traces (trace-level null; weaker interpretation) --")
    real_rows = []
    for ds, conds, hz_def in (("lerique", ("rest1", "trials_concat"), 2.0),
                              ("gordon", ("exp1", "exp4"), 2.0)):
        for c in conds:
            real_rows.extend(run_block_null_real(ds, c, hz_def))
    realdf = pd.DataFrame(real_rows)
    realdf.to_csv(OUT_DIR / "block_permute_null_real.csv", index=False)
    part1_real = {}
    for (ds, cond, feat), sub in realdf.groupby(["dataset", "condition", "feature"]):
        defined = sub["defined"].mean()
        sub_def = sub[sub["defined"] & sub["p"].notna()]
        rej = sub_def["reject"].mean() if len(sub_def) else float("nan")
        part1_real.setdefault(f"{ds}/{cond}", {})[feat] = dict(
            definedness=round(float(defined), 3),
            n_testable=int(len(sub_def)),
            rejection_rate=round(float(rej), 3) if len(sub_def) else None,
        )
        rate_txt = f"reject_rate={rej:.3f}" if len(sub_def) else "(no testable)"
        print(f"  [{ds}/{cond:14s}] {feat:16s} defined={defined:.2f} "
              f"n_testable={len(sub_def):3d} {rate_txt}")

    summary["part1_block_null"] = {
        "block_sec": BLOCK_SEC,
        "egt_substrate_level": part1_egt,
        "real_trace_level": part1_real,
    }

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
    print(f"Wrote {OUT_DIR}/block_permute_null_egt.csv")
    print(f"Wrote {OUT_DIR}/block_permute_null_real.csv")


if __name__ == "__main__":
    main()

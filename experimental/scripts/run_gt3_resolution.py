"""
GT-3: Estimator resolution comparison + SRF deconvolution.
==========================================================
Question (user, 2026-06): the timing family (onset/rise/recovery) shows
near-zero ground-truth recovery on GT-2 because sliding-window WCC acts as
a ~1/W low-pass filter that smears second-scale event edges.

This script tests two fixes:
  (A) Instantaneous / finer estimators — does swapping WCC(W=60) for
      WCC(W=20/10), WCLC, or PLV improve timing-family recovery?
  (B) SRF deconvolution — model synchrony as an SCR/ERP-like event process
      (Benedek & Kaernbach 2010 CDA analogue): observed trace ≈ driver ⊛ kernel.
      Deconvolve to recover a sharper driver, then extract timing features.

Metrics:
  - trace_fidelity: Pearson r between estimator trace and the GROUND-TRUTH
    coupling envelope (resampled to trace timepoints). This is the cleanest
    resolution measure — it does not depend on feature definitions.
  - feature recovery: Spearman ρ of extracted timing features vs GT params
    (switch_freq, recovery_rate), aggregated across the grid.

GT generator is reused from run_gt2_temporal but extended to also RETURN the
coupling(t) envelope (the ground-truth driver).

Output:
  artifacts/gt3_estimator_resolution.csv      # per-run feature values
  artifacts/gt3_trace_fidelity.csv            # per-run trace fidelity
  artifacts/gt3_summary.csv                   # recovery ρ per estimator×feature
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features
from multisync.metrics import wclc_synchrony, plv_synchrony
from multisync.feature_definitions import CONFIRMATORY_FEATURES

HZ = 1.0
DURATION = 300
N_SEEDS = 15
COUPLING = 0.5

SWITCH_FREQS = [0.5, 2.0, 5.0]      # switches per minute
RECOVERY_RATES = [0.02, 0.1, 0.5]   # 1/recovery_time

TIMING_FEATS = ["onset_latency", "rise_time", "recovery_time"]
SLOW_FEATS = ["dwell_time", "switching_rate", "synchrony_entropy"]
ALL_FEATS = list(CONFIRMATORY_FEATURES) + ["mean_synchrony", "synchrony_entropy"]

ART = Path(REPO) / "artifacts"


# ───────────────────────────────────────────────────────────────────────
# GT generator (extended to return coupling envelope = ground-truth driver)
# ───────────────────────────────────────────────────────────────────────
def generate_temporal_gt(switch_freq, recovery_rate, duration, hz, seed):
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    event_rate = switch_freq / 60.0
    is_on = np.zeros(n, dtype=bool)
    state = False
    next_event = int(rng.exponential(1.0 / event_rate)) if event_rate > 0 else n
    for i in range(n):
        if i >= next_event:
            state = not state
            next_event = i + int(max(1, rng.exponential(1.0 / event_rate)))
        is_on[i] = state
    coupling = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if is_on[i]:
            cur = min(cur + 0.3, 1.0)
        else:
            cur *= (1.0 - recovery_rate)
        coupling[i] = cur * COUPLING
    shared = np.sin(np.linspace(0, duration * 2 * np.pi / 60, n))
    shared += 0.5 * np.sin(np.linspace(0, duration * 2 * np.pi / 20, n))
    shared /= np.std(shared) + 1e-10
    a = coupling * shared + 0.3 * rng.normal(0, 1, n)
    b = coupling * shared + 0.3 * rng.normal(0, 1, n)
    return a, b, coupling


# ───────────────────────────────────────────────────────────────────────
# SRF deconvolution (Benedek & Kaernbach 2010 CDA analogue)
# ───────────────────────────────────────────────────────────────────────
def srf_kernel(hz, tau_rise=2.0, tau_decay=8.0, length_sec=40.0):
    """Canonical synchrony-response kernel = difference of two exponentials.
    Mirrors the bi-exponential SCR shape (fast rise, slow decay)."""
    t = np.arange(0, int(length_sec * hz)) / hz
    h = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    h[h < 0] = 0.0
    s = h.sum()
    return h / s if s > 0 else h


def richardson_lucy_1d(obs, kernel, iters=50):
    """Non-negative 1-D deconvolution (Richardson-Lucy)."""
    obs = np.clip(obs, 1e-6, None)
    kernel = kernel / (kernel.sum() + 1e-12)
    k_flip = kernel[::-1]
    est = np.full_like(obs, obs.mean())
    for _ in range(iters):
        conv = np.convolve(est, kernel, mode="same")
        conv = np.clip(conv, 1e-6, None)
        relative = obs / conv
        est *= np.convolve(relative, k_flip, mode="same")
        est = np.clip(est, 0, None)
    return est


def deconvolved_trace(a, b, hz, win_sec=10, step=1,
                      tau_rise=2.0, tau_decay=8.0):
    """Fine-window WCC → deconvolve with SRF kernel → sharpened driver."""
    obs = sliding_window_wcc(a, b, window_size=int(win_sec * hz),
                             hz=hz, step_samples=step)
    if len(obs) < 5:
        return np.array([])
    obs = np.nan_to_num(obs, nan=0.0)
    trace_hz = hz / step
    kern = srf_kernel(trace_hz, tau_rise, tau_decay,
                      length_sec=min(40.0, len(obs) / trace_hz / 2))
    if len(kern) < 2:
        return obs
    driver = richardson_lucy_1d(np.clip(obs, 0, None), kern, iters=40)
    return driver


# ───────────────────────────────────────────────────────────────────────
# Estimator registry: name -> (trace_fn, trace_hz)
# ───────────────────────────────────────────────────────────────────────
def make_estimators(hz):
    return {
        "WCC_W60_S10": (lambda a, b: sliding_window_wcc(a, b, window_size=int(60*hz), hz=hz, step_samples=10), hz/10),
        "WCC_W20_S2":  (lambda a, b: sliding_window_wcc(a, b, window_size=int(20*hz), hz=hz, step_samples=2),  hz/2),
        "WCC_W10_S1":  (lambda a, b: sliding_window_wcc(a, b, window_size=int(10*hz), hz=hz, step_samples=1),  hz/1),
        "WCLC_W60_S10":(lambda a, b: wclc_synchrony(a, b, window_size=int(60*hz), step=10, max_lag_samples=15), hz/10),
        "PLV_W20_S2":  (lambda a, b: plv_synchrony(a, b, window_size=int(20*hz), step=2, fs=hz), hz/2),
        "PLV_W10_S1":  (lambda a, b: plv_synchrony(a, b, window_size=int(10*hz), step=1, fs=hz), hz/1),
        "SRF_deconv_W10":(lambda a, b: deconvolved_trace(a, b, hz, win_sec=10, step=1), hz/1),
    }


def resample_to(env, n_out):
    """Resample ground-truth envelope to n_out points (block-mean)."""
    if n_out <= 0:
        return np.array([])
    idx = np.linspace(0, len(env) - 1, n_out).astype(int)
    return env[idx]


def main():
    rows_feat, rows_fid = [], []
    estimators = make_estimators(HZ)

    total = len(SWITCH_FREQS) * len(RECOVERY_RATES) * N_SEEDS
    i = 0
    for sf in SWITCH_FREQS:
        for rr in RECOVERY_RATES:
            for seed in range(N_SEEDS):
                i += 1
                a, b, env = generate_temporal_gt(sf, rr, DURATION, HZ, 2000 + seed)
                for est_name, (fn, trace_hz) in estimators.items():
                    try:
                        trace = fn(a, b)
                    except Exception:
                        continue
                    if trace is None or len(trace) < 5:
                        continue
                    trace = np.nan_to_num(np.asarray(trace, float), nan=0.0)
                    # trace fidelity vs GT envelope
                    env_rs = resample_to(env, len(trace))
                    if np.std(trace) > 1e-9 and np.std(env_rs) > 1e-9:
                        fid = pearsonr(trace, env_rs)[0]
                    else:
                        fid = np.nan
                    rows_fid.append({"switch_freq": sf, "recovery_rate": rr,
                                     "seed": seed, "estimator": est_name,
                                     "trace_fidelity": fid, "trace_hz": trace_hz})
                    # feature extraction (skip PLV/SRF that aren't on r-scale for
                    # threshold features? keep all for fairness — onset thresh=0.5)
                    feats = extract_dynamic_features(trace, hz=trace_hz,
                                                     wcc_window_sec=60)
                    for feat in ALL_FEATS:
                        v = getattr(feats, feat, None)
                        if v is not None and np.isfinite(v):
                            rows_feat.append({"switch_freq": sf, "recovery_rate": rr,
                                              "seed": seed, "estimator": est_name,
                                              "feature": feat, "value": float(v)})
                if i % 30 == 0:
                    print(f"[{i}/{total}]", flush=True)

    df_feat = pd.DataFrame(rows_feat)
    df_fid = pd.DataFrame(rows_fid)
    df_feat.to_csv(ART / "gt3_estimator_resolution.csv", index=False)
    df_fid.to_csv(ART / "gt3_trace_fidelity.csv", index=False)

    # ── summary 1: trace fidelity per estimator ──
    print("\n=== TRACE FIDELITY (Pearson r vs GT coupling envelope) ===")
    fid_summary = df_fid.groupby("estimator")["trace_fidelity"].agg(["mean", "std", "count"])
    print(fid_summary.round(3).to_string())

    # ── summary 2: feature recovery ρ per estimator × timing feature ──
    print("\n=== TIMING-FAMILY GT RECOVERY (Spearman ρ, |ρ| higher = better) ===")
    summary_rows = []
    for est in estimators:
        for feat in TIMING_FEATS + SLOW_FEATS:
            sub = df_feat[(df_feat.estimator == est) & (df_feat.feature == feat)]
            if len(sub) < 10:
                summary_rows.append({"estimator": est, "feature": feat,
                                     "rho_vs_switch": np.nan, "rho_vs_recovery": np.nan,
                                     "n": len(sub)})
                continue
            rho_s = spearmanr(sub.switch_freq, sub.value)[0]
            rho_r = spearmanr(sub.recovery_rate, sub.value)[0]
            summary_rows.append({"estimator": est, "feature": feat,
                                 "rho_vs_switch": rho_s, "rho_vs_recovery": rho_r,
                                 "n": len(sub)})
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(ART / "gt3_summary.csv", index=False)

    for feat in TIMING_FEATS:
        print(f"\n--- {feat} (best |ρ vs recovery_rate| wins) ---")
        sub = df_sum[df_sum.feature == feat].copy()
        sub["abs_rec"] = sub.rho_vs_recovery.abs()
        sub = sub.sort_values("abs_rec", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {r.estimator:16s} ρ_switch={r.rho_vs_switch:+.3f}  "
                  f"ρ_recovery={r.rho_vs_recovery:+.3f}  n={int(r.n)}")

    print(f"\nSaved: gt3_estimator_resolution.csv / gt3_trace_fidelity.csv / gt3_summary.csv")


if __name__ == "__main__":
    main()

"""
GT-3b: Event-related ground truth with a CONVOLUTIONAL generative model.
========================================================================
GT-2/GT-3 finding (honest): the timing family "failed" synthetic recovery,
but the GT itself could not test onset/rise:
  - onset_latency had NO controllable GT parameter (events were Poisson-random)
  - rise_time had a FIXED slope (+0.3/sample) -> nothing to recover
  - recovery_time was the ONLY timing feature with a real GT knob
    (recovery_rate) and it DID recover (rho=-0.47 on WCC_W60).
Also, SRF deconvolution failed (fidelity 0.08) because GT-2's coupling is a
multiplicative gate, NOT a driver (x) kernel process -- the deconvolution
model was mismatched to the generator.

GT-3b fixes both by generating synchrony as an explicit convolution:
    driver(t)   = sparse event train at KNOWN onset times
    kernel(t)   = bi-exponential SRF with controllable (tau_rise, tau_decay)
    coupling(t) = driver (x) kernel        # ground-truth synchrony envelope
This GT HAS the parameters the fast family claims to measure, and gives the
deconvolution a MATCHED generative model (fair test).

Two questions:
  Q1 (feature validity): do onset_latency / rise_time / recovery_time recover
     onset_delay / tau_rise / tau_decay when the GT actually parameterizes them?
  Q2 (deconvolution): does SRF deconvolution localize the driver event better
     than the raw WCC trace, when the kernel is matched?

Output:
  artifacts/gt3b_feature_recovery.csv   # per-run feature values + GT params
  artifacts/gt3b_trace_fidelity.csv     # per-run trace fidelity + driver loc
  artifacts/gt3b_summary.csv            # recovery rho per estimator x feature
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
from multisync.metrics import plv_synchrony

HZ = 1.0
DURATION = 300
N_SEEDS = 12
PEAK_AMP = 0.85          # peak coupling (well above ONSET_THRESHOLD=0.5)
NOISE = 0.25

# GT knobs -- the parameters the fast family CLAIMS to measure
ONSET_DELAYS = [40.0, 90.0, 150.0]     # seconds to first event
TAU_RISES = [2.0, 6.0, 12.0]           # rise time-constant (sec)
TAU_DECAYS = [6.0, 18.0, 40.0]         # decay time-constant (sec)

TIMING_FEATS = ["onset_latency", "rise_time", "recovery_time"]
ALL_FEATS = ["onset_latency", "rise_time", "peak_amplitude",
             "recovery_time", "dwell_time", "switching_rate",
             "mean_synchrony", "synchrony_entropy"]

ART = Path(REPO) / "artifacts"


# ----------------------------------------------------------------------
# Convolutional GT: coupling(t) = single event (x) bi-exponential kernel
# ----------------------------------------------------------------------
def bi_exp(t, tau_rise, tau_decay):
    h = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    h[t < 0] = 0.0
    h[h < 0] = 0.0
    return h


def generate_event_gt(onset_delay, tau_rise, tau_decay, duration, hz, seed):
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    t = np.arange(n) / hz
    # single clean event: bi-exponential pulse starting at onset_delay
    pulse = bi_exp(t - onset_delay, tau_rise, tau_decay)
    if pulse.max() > 0:
        pulse = pulse / pulse.max() * PEAK_AMP
    coupling = pulse  # ground-truth synchrony envelope
    # shared oscillation modulated by the coupling envelope
    shared = np.sin(2 * np.pi * t / 8.0) + 0.5 * np.sin(2 * np.pi * t / 3.0)
    shared /= np.std(shared) + 1e-10
    a = coupling * shared + NOISE * rng.normal(0, 1, n)
    b = coupling * shared + NOISE * rng.normal(0, 1, n)
    # true driver = delta at onset (for deconvolution localization test)
    driver = np.zeros(n)
    onset_idx = int(onset_delay * hz)
    if 0 <= onset_idx < n:
        driver[onset_idx] = 1.0
    return a, b, coupling, driver


# ----------------------------------------------------------------------
# SRF deconvolution (now with a kernel MATCHED to the generator)
# ----------------------------------------------------------------------
def srf_kernel(hz, tau_rise, tau_decay, length_sec):
    t = np.arange(0, max(2, int(length_sec * hz))) / hz
    h = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    h[h < 0] = 0.0
    s = h.sum()
    return h / s if s > 0 else h


def richardson_lucy_1d(obs, kernel, iters=60):
    obs = np.clip(obs, 1e-6, None)
    kernel = kernel / (kernel.sum() + 1e-12)
    k_flip = kernel[::-1]
    est = np.full_like(obs, obs.mean())
    for _ in range(iters):
        conv = np.convolve(est, kernel, mode="same")
        conv = np.clip(conv, 1e-6, None)
        est *= np.convolve(obs / conv, k_flip, mode="same")
        est = np.clip(est, 0, None)
    return est


def deconvolved_trace(a, b, hz, win_sec, step, tau_rise, tau_decay):
    obs = sliding_window_wcc(a, b, window_size=int(win_sec * hz),
                             hz=hz, step_samples=step)
    if len(obs) < 5:
        return np.array([]), np.array([])
    obs = np.nan_to_num(obs, nan=0.0)
    trace_hz = hz / step
    kern = srf_kernel(trace_hz, tau_rise, tau_decay,
                      length_sec=min(60.0, len(obs) / trace_hz / 2))
    if len(kern) < 2:
        return obs, obs
    driver = richardson_lucy_1d(np.clip(obs, 0, None), kern, iters=50)
    return driver, obs


def resample_to(env, n_out):
    if n_out <= 0:
        return np.array([])
    idx = np.linspace(0, len(env) - 1, n_out).astype(int)
    return env[idx]


def make_estimators(hz):
    return {
        "WCC_W60_S5": (lambda a, b: sliding_window_wcc(a, b, window_size=int(60*hz), hz=hz, step_samples=5), hz/5),
        "WCC_W20_S2": (lambda a, b: sliding_window_wcc(a, b, window_size=int(20*hz), hz=hz, step_samples=2), hz/2),
        "WCC_W10_S1": (lambda a, b: sliding_window_wcc(a, b, window_size=int(10*hz), hz=hz, step_samples=1), hz/1),
        "PLV_W20_S2": (lambda a, b: plv_synchrony(a, b, window_size=int(20*hz), step=2, fs=hz), hz/2),
    }


def main():
    rows_feat, rows_fid = [], []
    estimators = make_estimators(HZ)
    grid = [(od, tr, td) for od in ONSET_DELAYS for tr in TAU_RISES for td in TAU_DECAYS]
    total = len(grid) * N_SEEDS
    i = 0
    for (od, tr, td) in grid:
        for seed in range(N_SEEDS):
            i += 1
            a, b, env, driver = generate_event_gt(od, tr, td, DURATION, HZ, 3000 + seed)
            # --- windowed estimators ---
            for est_name, (fn, trace_hz) in estimators.items():
                try:
                    trace = fn(a, b)
                except Exception:
                    continue
                if trace is None or len(trace) < 5:
                    continue
                trace = np.nan_to_num(np.asarray(trace, float), nan=0.0)
                env_rs = resample_to(env, len(trace))
                fid = (pearsonr(trace, env_rs)[0]
                       if np.std(trace) > 1e-9 and np.std(env_rs) > 1e-9 else np.nan)
                rows_fid.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                 "seed": seed, "estimator": est_name,
                                 "trace_fidelity": fid, "trace_hz": trace_hz,
                                 "driver_peak_err": np.nan})
                feats = extract_dynamic_features(trace, hz=trace_hz, wcc_window_sec=60)
                for feat in ALL_FEATS:
                    v = getattr(feats, feat, None)
                    if v is not None and np.isfinite(v):
                        rows_feat.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                          "seed": seed, "estimator": est_name,
                                          "feature": feat, "value": float(v)})
            # --- SRF deconvolution (matched kernel) ---
            driver_est, obs = deconvolved_trace(a, b, HZ, win_sec=10, step=1,
                                                tau_rise=tr, tau_decay=td)
            if len(driver_est) >= 5:
                trace_hz = HZ / 1
                env_rs = resample_to(env, len(driver_est))
                fid = (pearsonr(driver_est, env_rs)[0]
                       if np.std(driver_est) > 1e-9 and np.std(env_rs) > 1e-9 else np.nan)
                # driver localization: peak position error (sec) vs true onset
                peak_idx = int(np.argmax(driver_est))
                peak_t = peak_idx / trace_hz
                # raw WCC peak for comparison baseline
                obs_peak_t = int(np.argmax(obs)) / trace_hz if len(obs) else np.nan
                rows_fid.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                 "seed": seed, "estimator": "SRF_deconv_W10",
                                 "trace_fidelity": fid, "trace_hz": trace_hz,
                                 "driver_peak_err": abs(peak_t - od)})
                rows_fid.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                 "seed": seed, "estimator": "rawWCC_W10_peak",
                                 "trace_fidelity": np.nan, "trace_hz": trace_hz,
                                 "driver_peak_err": abs(obs_peak_t - od)})
                feats = extract_dynamic_features(driver_est, hz=trace_hz, wcc_window_sec=60)
                for feat in ALL_FEATS:
                    v = getattr(feats, feat, None)
                    if v is not None and np.isfinite(v):
                        rows_feat.append({"onset_delay": od, "tau_rise": tr, "tau_decay": td,
                                          "seed": seed, "estimator": "SRF_deconv_W10",
                                          "feature": feat, "value": float(v)})
            if i % 30 == 0:
                print(f"[{i}/{total}]", flush=True)

    df_feat = pd.DataFrame(rows_feat)
    df_fid = pd.DataFrame(rows_fid)
    df_feat.to_csv(ART / "gt3b_feature_recovery.csv", index=False)
    df_fid.to_csv(ART / "gt3b_trace_fidelity.csv", index=False)

    print("\n=== TRACE FIDELITY (Pearson r vs GT coupling envelope) ===")
    print(df_fid.groupby("estimator")["trace_fidelity"].agg(["mean", "std", "count"]).round(3).to_string())

    print("\n=== DRIVER LOCALIZATION (|peak_t - onset_delay|, sec; lower=better) ===")
    loc = df_fid[df_fid.estimator.isin(["SRF_deconv_W10", "rawWCC_W10_peak"])]
    print(loc.groupby("estimator")["driver_peak_err"].agg(["mean", "std", "count"]).round(2).to_string())

    print("\n=== FEATURE RECOVERY (Spearman rho vs MATCHING GT knob) ===")
    pairs = {"onset_latency": "onset_delay", "rise_time": "tau_rise", "recovery_time": "tau_decay"}
    summary_rows = []
    for est in df_feat.estimator.unique():
        for feat, knob in pairs.items():
            sub = df_feat[(df_feat.estimator == est) & (df_feat.feature == feat)]
            if len(sub) < 10:
                summary_rows.append({"estimator": est, "feature": feat, "knob": knob,
                                     "rho": np.nan, "n": len(sub)})
                continue
            rho = spearmanr(sub[knob], sub.value)[0]
            summary_rows.append({"estimator": est, "feature": feat, "knob": knob,
                                 "rho": rho, "n": len(sub)})
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(ART / "gt3b_summary.csv", index=False)
    for feat, knob in pairs.items():
        print(f"\n--- {feat} vs {knob} (|rho| higher = better recovery) ---")
        sub = df_sum[df_sum.feature == feat].copy()
        sub["abs_rho"] = sub.rho.abs()
        for _, r in sub.sort_values("abs_rho", ascending=False).iterrows():
            print(f"  {r.estimator:16s} rho={r.rho:+.3f}  n={int(r.n)}")

    print("\nSaved: gt3b_feature_recovery.csv / gt3b_trace_fidelity.csv / gt3b_summary.csv")


if __name__ == "__main__":
    main()

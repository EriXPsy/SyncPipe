"""
run_treur_validation.py
========================

Full validation pipeline: Treur dyad simulator → DualStream Epoch analysis.

Evaluates:
  1. Can WCC features recover known switching parameters?
  2. Does WCLC detect lagged Epoch in leader-follower scenarios?
  3. How does adaptive theta_d compare to fixed θ=0.5?
  4. Multi-scale consistency across 5s/30s/120s windows.

Run:  python scripts/run_treur_validation.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure multisync is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multisync.simulation.treur_dyad import (
    TreurDyadSimulator,
    scenario_constant_high_sync,
    scenario_frequent_switching,
    scenario_leader_follower,
    scenario_gradual_emergence,
    scenario_isc_confound,
)
from multisync.epoch_detection import dual_stream_epoch_analysis
from multisync.adaptive_threshold import calibrate_threshold_from_null
from multisync.multiscale_epochs import multiscale_epoch_analysis
from multisync.feature_definitions import ONSET_THRESHOLD


# ── Config ─────────────────────────────────────────────────────────
SEED = 42
DURATION_SEC = 120  # shorter for speed; increase for more reliable estimates
HZ = 10.0
OUTDIR = Path(__file__).resolve().parent.parent / "scripts" / "treur_validation_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "constant_high": {
        "fn": scenario_constant_high_sync,
        "ground_truth": {
            "expect_high_peak": True,
            "expect_low_switching": True,
            "expect_lagged": False,
        },
    },
    "frequent_switching": {
        "fn": lambda: scenario_frequent_switching(DURATION_SEC, HZ, n_switches=2, seed=SEED),
        "ground_truth": {
            "expect_high_peak": True,
            "expect_low_switching": False,
            "expect_lagged": False,
        },
    },
    "leader_follower": {
        "fn": lambda: scenario_leader_follower(DURATION_SEC, HZ, seed=SEED),
        "ground_truth": {
            "expect_high_peak": False,  # WCC misses lagged correlation!
            "expect_low_switching": False,
            "expect_lagged": True,       # ★ WCLC should detect this
        },
    },
    "gradual_emergence": {
        "fn": lambda: scenario_gradual_emergence(DURATION_SEC, HZ, seed=SEED),
        "ground_truth": {
            "expect_high_peak": True,
            "expect_low_switching": False,
            "expect_lagged": False,
        },
    },
    "isc_confound": {
        "fn": lambda: scenario_isc_confound(DURATION_SEC, HZ, seed=SEED),
        "ground_truth": {
            "expect_high_peak": True,
            "expect_low_switching": True,
            "expect_lagged": False,
        },
    },
}

print(f"Running Treur validation on {len(SCENARIOS)} scenarios...")
print(f"  duration = {DURATION_SEC}s, hz = {HZ:.0f}, seed = {SEED}\n")

results = {}
comparisons = []

for name, cfg in SCENARIOS.items():
    t0 = time.time()
    sim_result = cfg["fn"]()
    gt = cfg["ground_truth"]

    a = sim_result.x_A_obs
    b = sim_result.x_B_obs

    # ── 1. Fixed threshold (baseline) ─────────────────────────
    ds_fixed = dual_stream_epoch_analysis(
        a, b, hz=HZ, wcc_window_sec=10.0, wclc_window_sec=20.0,
        max_lag_sec=5.0, onset_threshold=0.5,
    )

    # ── 2. Adaptive threshold ─────────────────────────────────
    from multisync.dynamic_features import sliding_window_wcc
    wcc_test = sliding_window_wcc(a, b, window_size=int(30*HZ), hz=HZ, step_samples=int(5*HZ))
    wcc_clean = wcc_test[~np.isnan(wcc_test)]
    theta_d = calibrate_threshold_from_null(wcc_clean, n_surrogates=100, seed=SEED)
    ds_adaptive = dual_stream_epoch_analysis(
        a, b, hz=HZ, wcc_window_sec=10.0, wclc_window_sec=20.0,
        max_lag_sec=5.0, onset_threshold=theta_d,
    )

    # ── 3. Multi-scale ────────────────────────────────────────
    ms = multiscale_epoch_analysis(a, b, hz=HZ, threshold=theta_d)

    elapsed = time.time() - t0

    # ── Collect results ───────────────────────────────────────
    f0 = ds_fixed.features_0lag
    fl = ds_fixed.features_lagged

    entry = {
        "scenario": name,
        "theta_fixed": 0.5,
        "theta_adaptive": round(theta_d, 4),
        "W_AB_mean": round(float(np.mean(sim_result.W_AB)), 4),
        "W_AB_std": round(float(np.std(sim_result.W_AB)), 4),
        "features_fixed": {
            "onset_latency": round(f0.onset_latency, 2) if not np.isnan(f0.onset_latency) else None,
            "rise_time": round(f0.rise_time, 2) if not np.isnan(f0.rise_time) else None,
            "peak_amplitude": round(f0.peak_amplitude, 3),
            "recovery_time": round(f0.recovery_time, 2) if not np.isnan(f0.recovery_time) else None,
            "dwell_time": round(f0.dwell_time, 2),
            "switching_rate": round(f0.switching_rate, 2),
            "mean_synchrony": round(f0.mean_synchrony, 4),
            "synchrony_entropy": round(f0.synchrony_entropy, 3),
        },
        "features_lagged_fixed": {
            "lagged_dwell": round(fl.dwell_time, 2) if not np.isnan(fl.dwell_time) else None,
            "lagged_switching": round(fl.switching_rate, 2),
            "lagged_peak": round(ds_fixed.wclc_peak, 3) if not np.isnan(ds_fixed.wclc_peak) else None,
            "lag_consistency": round(ds_fixed.lag_consistency, 3) if not np.isnan(ds_fixed.lag_consistency) else None,
            "lag_direction": round(ds_fixed.lag_direction, 3) if not np.isnan(ds_fixed.lag_direction) else None,
            "overlap_ratio": round(ds_fixed.overlap_ratio, 4),
        },
        "features_adaptive": {
            "dwell_time": round(ds_adaptive.features_0lag.dwell_time, 2),
            "switching_rate": round(ds_adaptive.features_0lag.switching_rate, 2),
        },
        "multiscale": {
            "consistency_2scale": round(ms.consistency_2scale_mean, 4),
            "consistency_3scale": round(ms.consistency_3scale_mean, 4),
        },
        "ground_truth_checks": {},
        "elapsed_sec": round(elapsed, 1),
    }

    # Ground-truth verification
    gt_checks = {}
    gt_checks["peak_as_expected"] = (
        (f0.peak_amplitude > 0.6) == gt["expect_high_peak"]
    )
    gt_checks["switching_as_expected"] = (
        (f0.switching_rate < 1.5) == gt["expect_low_switching"]
    )
    gt_checks["lagged_detected"] = (
        (ds_fixed.lagged_dwell_time > 2.0 if not np.isnan(ds_fixed.lagged_dwell_time) else False)
        == gt["expect_lagged"]
    )
    entry["ground_truth_checks"] = gt_checks

    results[name] = entry

    # Summary line
    gt_pass = sum(gt_checks.values())
    print(f"  {name:25s} θ_fixed=0.5 θ_adapt={theta_d:.3f}  "
          f"peak={f0.peak_amplitude:.3f}  dwell={f0.dwell_time:.1f}s  "
          f"switch={f0.switching_rate:.2f}/min  overlap={ds_fixed.overlap_ratio:.3f}  "
          f"GT checks: {gt_pass}/3  ({elapsed:.1f}s)")

    # Comparison rows
    comparisons.append({
        "scenario": name,
        "metric": "dwell_time",
        "fixed": round(f0.dwell_time, 2),
        "adaptive": round(ds_adaptive.features_0lag.dwell_time, 2),
        "delta": round(ds_adaptive.features_0lag.dwell_time - f0.dwell_time, 2),
    })
    comparisons.append({
        "scenario": name,
        "metric": "switching_rate",
        "fixed": round(f0.switching_rate, 2),
        "adaptive": round(ds_adaptive.features_0lag.switching_rate, 2),
        "delta": round(ds_adaptive.features_0lag.switching_rate - f0.switching_rate, 2),
    })

# ── Save results ─────────────────────────────────────────────────
out_json = OUTDIR / "treur_validation_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── Summary table ────────────────────────────────────────────────
print(f"\n{'='*85}")
print("FEATURE RECOVERY SUMMARY (fixed θ=0.5)")
print(f"{'='*85}")
print(f"{'Scenario':<22s} {'peak':>8s} {'dwell':>8s} {'switch':>8s} {'lag_det':>8s} {'overlap':>8s}")
print("-" * 62)
for name, r in results.items():
    f = r["features_fixed"]
    print(f"{name:<22s} {f['peak_amplitude']:>8.3f} {f['dwell_time']:>8.1f} {f['switching_rate']:>8.2f} "
          f"{'YES' if r['features_lagged_fixed']['lagged_dwell'] and r['features_lagged_fixed']['lagged_dwell'] > 1 else 'NO':>8s} "
          f"{r['features_lagged_fixed']['overlap_ratio']:>8.4f}")

print(f"\n{'='*85}")
print("THRESHOLD COMPARISON (fixed θ=0.5 vs adaptive θ_d)")
print(f"{'='*85}")
for c in comparisons:
    print(f"  {c['scenario']:<22s} {c['metric']:<16s} fixed={c['fixed']:<8.2f} adaptive={c['adaptive']:<8.2f} delta={c['delta']:+.2f}")

# Ground truth check summary
print(f"\n{'='*85}")
print("GROUND TRUTH VERIFICATION")
print(f"{'='*85}")
total_checks = 0
passed_checks = 0
for name, r in results.items():
    gt = r["ground_truth_checks"]
    for check, passed in gt.items():
        total_checks += 1
        if passed:
            passed_checks += 1
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark} {name}/{check}")
print(f"\n  Overall: {passed_checks}/{total_checks} checks passed")

print(f"\nResults saved to: {out_json}")

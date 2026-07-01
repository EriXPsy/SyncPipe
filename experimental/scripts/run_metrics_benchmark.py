"""
Metrics Benchmark: WCC vs CRQA vs MI vs PLV on synthetic dyadic data.

Generates coupled sine-wave dyads at known coupling strengths (c=0, 0.3, 0.6)
and measures which metric best discriminates coupling from noise.
"""
import sys
sys.path.insert(0, r'<REPO>')

import numpy as np
import pandas as pd
from multisync.metrics import crqa_synchrony, mi_synchrony, plv_synchrony
from multisync.dynamic_features import extract_dynamic_features, sliding_window_wcc
from multisync.feature_definitions import CONFIRMATORY_FEATURES

N_SEEDS = 20
COUPLINGS = [0.0, 0.3, 0.6]
DURATION = 300  # seconds
HZ = 1.0
WINDOW = 30
STEP = 5

FEATS = list(CONFIRMATORY_FEATURES) + ['mean_synchrony', 'synchrony_entropy']

# Generate dyad signals: P1 = sine, P2 = c*sine + (1-c)*independent_noise
def generate_dyad(coupling: float, noise_ratio: float, duration: int, hz: float, seed: int):
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    t = np.linspace(0, duration, n)

    # Shared signal (the coupling component)
    shared = np.sin(2 * np.pi * t / 60) + 0.5 * np.sin(2 * np.pi * t / 20)
    shared = shared / np.std(shared)

    # Individual noise
    noise = noise_ratio * rng.normal(0, 1, n)

    # Person 1: signal + independent noise component
    a = np.sqrt(coupling) * shared + np.sqrt(1 - coupling) * rng.normal(0, 1, n)
    # Person 2: same shared signal + different independent noise
    b = np.sqrt(coupling) * shared + np.sqrt(1 - coupling) * rng.normal(0, 1, n)

    # Add overall measurement noise
    a = a + noise_ratio * rng.normal(0, 0.2, n)
    b = b + noise_ratio * rng.normal(0, 0.2, n)

    return a, b

results = []
for c in COUPLINGS:
    print(f"c={c}")
    for seed in range(N_SEEDS):
        if seed % 5 == 0:
            print(f"  seed={seed}/{N_SEEDS}")
        a, b = generate_dyad(c, 0.3, DURATION, HZ, 1000 + seed)

        for name, func in [
            ('wcc', lambda x, y: sliding_window_wcc(x, y, window_size=WINDOW, hz=HZ, step_samples=STEP)),
            ('crqa', lambda x, y: crqa_synchrony(x, y, window_size=WINDOW, step=STEP)),
            ('mi', lambda x, y: mi_synchrony(x, y, window_size=WINDOW, step=STEP)),
            ('plv', lambda x, y: plv_synchrony(x, y, window_size=WINDOW, step=STEP, fs=HZ)),
        ]:
            trace = func(a, b)
            if len(trace) < 10:
                continue
            trace[np.isnan(trace)] = 0
            wcc_hz = HZ / STEP
            feats = extract_dynamic_features(trace, hz=wcc_hz, wcc_window_sec=WINDOW)
            for feat in FEATS:
                v = getattr(feats, feat, None)
                if v is not None and np.isfinite(v):
                    results.append({
                        'coupling': c, 'seed': seed, 'metric': name,
                        'feature': feat, 'value': float(v),
                    })

df = pd.DataFrame(results)
out = r'<REPO>\artifacts\metrics_benchmark_gt1.csv'
df.to_csv(out, index=False)
print(f"\n{len(df)} rows -> {out}")

# Summary
print("\n=== METRIC DETECTION POWER (c=0 vs c=0.6, N=20 seeds) ===")
for feat in ['peak_amplitude', 'mean_synchrony', 'dwell_time', 'switching_rate',
              'synchrony_entropy', 'onset_latency', 'rise_time', 'recovery_time']:
    print(f"\n{feat}:")
    best_metric, best_detect = None, -1
    for m in ['wcc', 'crqa', 'mi', 'plv']:
        c0 = df[(df.metric == m) & (df.coupling == 0.0) & (df.feature == feat)]['value']
        c6 = df[(df.metric == m) & (df.coupling == 0.6) & (df.feature == feat)]['value']
        if len(c0) < 2 or len(c6) < 2:
            continue
        m0, s0 = c0.mean(), c0.std()
        m6, s6 = c6.mean(), c6.std()
        psd = np.sqrt((s0**2 + s6**2) / 2 + 1e-10)
        d = (m6 - m0) / psd
        # detection: fraction of c=0.6 seeds > 95th percentile of c=0
        thresh = np.percentile(c0, 95)
        detect = np.mean(c6.values > thresh)
        bar = '#' * int(detect * 20)
        print(f"  {m:5s}: d={d:+.2f} m0={m0:.2f} m6={m6:.2f} detect={detect:.0%} {bar}")
        if detect > best_detect:
            best_detect = detect
            best_metric = m
    if best_metric:
        print(f"  -> BEST: {best_metric} (detect={best_detect:.0%})")

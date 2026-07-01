"""
.. deprecated:: 2026-06-11
    SUPERSEDED by ``run_kuramoto_benchmark.py``.

    This white-box benchmark had two fatal design flaws:
      1. Bug A: LAG=30 equals signal period (sin(2πt/30)) → shifted signal = original
      2. Bug B: sigmoid(P1) has Pearson r=0.967 with P1 → not truly nonlinear
    Both bugs made all three "coupling structures" effectively identical
    (P1's component + noise), causing WCC to score highest on EVERY structure —
    including those it should theoretically fail on.

    The replacement Kuramoto gray-box benchmark uses emergent synchrony
    from coupled oscillators against a known ground truth r(t), making
    the comparison fair and reviewer-proof.

    KEPT FOR ARCHIVAL REFERENCE ONLY. DO NOT RUN FOR PUBLICATION.
"""
import sys
sys.path.insert(0, r'<REPO>')

import numpy as np
import pandas as pd
from multisync.metrics import crqa_synchrony, mi_synchrony, plv_synchrony
from multisync.dynamic_features import extract_dynamic_features, sliding_window_wcc
from multisync.feature_definitions import CONFIRMATORY_FEATURES

N_SEEDS = 15
COUPLINGS = [0.0, 0.3, 0.6]
DURATION = 300
HZ = 1.0
WINDOW = 60  # larger window for better MI stability
STEP = 10
LAG = 30     # samples lag for LAGGED coupling (30s)

FEATS = list(CONFIRMATORY_FEATURES) + ['mean_synchrony', 'synchrony_entropy']


def wclc_synchrony(a, b, window_size=60, lag_samples=30, step=10):
    """Windowed Cross-Lagged Correlation (max |r| across lags [-L, +L])."""
    n = min(len(a), len(b))
    n_windows = max(1, (n - window_size) // step + 1)
    rr = np.full(n_windows, np.nan)
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n: break
        aw = a[start:end]
        best_r = 0.0
        for lag in range(-lag_samples, lag_samples + 1):
            if lag < 0:
                x, y = aw[-lag:], b[start:end + lag]
            elif lag > 0:
                x, y = aw[:-lag] if lag < len(aw) else aw, b[start + lag:end]
            else:
                x, y = aw, b[start:end]
            ml = min(len(x), len(y))
            if ml < 5: continue
            r = np.corrcoef(x[:ml], y[:ml])[0, 1]
            if np.isfinite(r) and abs(r) > abs(best_r):
                best_r = r
        rr[i] = best_r
    return rr


def generate_linear(c, noise_r, duration, hz, seed):
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    t = np.linspace(0, duration, n)
    shared = np.sin(2*np.pi*t/60) + 0.5*np.sin(2*np.pi*t/20)
    shared /= np.std(shared) + 1e-10
    a = shared + 0.3 * rng.normal(0, 1, n)
    b = c * shared + (1-c) * rng.normal(0, 1, n) + 0.3 * rng.normal(0, 0.2, n)
    return a, b


def generate_lagged(c, noise_r, duration, hz, seed, lag=LAG):
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    t = np.linspace(0, duration, n)
    p1 = np.sin(2*np.pi*t/30) + 0.3 * rng.normal(0, 1, n)
    p2_base = np.zeros(n)
    p2_base[lag:] = p1[:-lag]
    p2 = c * p2_base + (1-c) * rng.normal(0, 1, n) + noise_r * rng.normal(0, 0.2, n)
    return p1, p2


def generate_nonlinear(c, noise_r, duration, hz, seed):
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    t = np.linspace(0, duration, n)
    p1 = np.sin(2*np.pi*t/40) + 0.3 * rng.normal(0, 1, n)
    # Nonlinear coupling: P2 = c * sigmoid(P1) + (1-c) * noise
    p1_norm = p1 / (np.std(p1) + 1e-10)
    sigmoid = 1.0 / (1.0 + np.exp(-3 * p1_norm))
    p2 = c * sigmoid + (1-c) * rng.normal(0, 1, n)
    p2 += noise_r * rng.normal(0, 0.2, n)
    return p1, p2


METRICS = {
    'wcc': lambda a, b: sliding_window_wcc(a, b, window_size=WINDOW, hz=HZ, step_samples=STEP),
    'wclc': lambda a, b: wclc_synchrony(a, b, window_size=WINDOW, lag_samples=LAG, step=STEP),
    'crqa': lambda a, b: crqa_synchrony(a, b, window_size=WINDOW, step=STEP),
    'mi': lambda a, b: mi_synchrony(a, b, window_size=WINDOW, step=STEP, n_bins=15),
    'plv': lambda a, b: plv_synchrony(a, b, window_size=WINDOW, step=STEP, fs=HZ),
}

GENERATORS = {
    'linear': generate_linear,
    'lagged': generate_lagged,
    'nonlinear': generate_nonlinear,
}

results = []

for gen_name, gen_func in GENERATORS.items():
    print(f"\n{'='*50}")
    print(f"Structure: {gen_name}")
    print(f"{'='*50}")
    for c in COUPLINGS:
        for seed in range(N_SEEDS):
            a, b = gen_func(c, 0.3, DURATION, HZ, 1000 + seed)
            for m_name, m_func in METRICS.items():
                trace = m_func(a, b)
                if len(trace) < 10: continue
                trace[np.isnan(trace)] = 0
                wcc_hz = HZ / STEP
                feats = extract_dynamic_features(trace, hz=wcc_hz, wcc_window_sec=WINDOW)
                for feat in FEATS:
                    v = getattr(feats, feat, None)
                    if v is not None and np.isfinite(v):
                        results.append({
                            'structure': gen_name, 'coupling': c, 'seed': seed,
                            'metric': m_name, 'feature': feat, 'value': float(v),
                        })

df = pd.DataFrame(results)
out = r'<REPO>\artifacts\metrics_benchmark_v2.csv'
df.to_csv(out, index=False)
print(f"\n{len(df)} rows -> {out}")


# Summary
print("\n" + "="*80)
print("BENCHMARK: detection power (c=0 vs c=0.6, fraction > 95th pct of c=0)")
print("="*80)

KEY_FEATS = ['peak_amplitude', 'mean_synchrony', 'switching_rate', 'synchrony_entropy']

for structure in ['linear', 'lagged', 'nonlinear']:
    print(f"\n### {structure.upper()} COUPLING ###")
    sub = df[df['structure'] == structure]
    header = f"{'feature':>20s}"
    for m in METRICS: header += f"  {m:>6s}"
    print(header)
    print("-" * (20 + 7 * len(METRICS)))
    for feat in KEY_FEATS:
        row = f"{feat:>20s}"
        for m in METRICS:
            c0 = sub[(sub.metric==m)&(sub.coupling==0.0)&(sub.feature==feat)]['value']
            c6 = sub[(sub.metric==m)&(sub.coupling==0.6)&(sub.feature==feat)]['value']
            if len(c0) >= 2 and len(c6) >= 2:
                thresh = np.percentile(c0, 95)
                detect = np.mean(c6.values > thresh)
                row += f"  {detect:5.0%}"
            else:
                row += f"     --"
        print(row)

# Winner per structure
print(f"\n{'='*80}")
print("WINNER PER STRUCTURE (by total detection across features)")
print("="*80)
for structure in ['linear', 'lagged', 'nonlinear']:
    sub = df[df['structure'] == structure]
    scores = {}
    for m in METRICS:
        total = 0
        for feat in KEY_FEATS:
            c0 = sub[(sub.metric==m)&(sub.coupling==0.0)&(sub.feature==feat)]['value']
            c6 = sub[(sub.metric==m)&(sub.coupling==0.6)&(sub.feature==feat)]['value']
            if len(c0) >= 2 and len(c6) >= 2:
                thresh = np.percentile(c0, 95)
                total += np.mean(c6.values > thresh)
        scores[m] = total
    best = max(scores, key=scores.get)
    score_str = '  '.join(f'{m}: {s:.0%}' for m, s in sorted(scores.items(), key=lambda x: -x[1]))
    print(f"  {structure:>10s}: {score_str}  -> BEST: {best}")

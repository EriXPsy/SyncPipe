"""
GT-2: Temporal-dynamics ground truth for onset/rise/recovery.

Extends GT-1 (coupling strength only) by adding:
  - switching_frequency: how often coupling switches on/off (σ_t)
  - recovery_rate: how fast coupling decays after peaks (τ)

This provides ground truth for timing-family features that
have 0-30% power on GT-1's coupling-strength-only axis.

Grid: 3 switching frequencies × 3 recovery rates × 15 seeds
Expected runtime: ~5 min.
"""
import sys
sys.path.insert(0, r'<REPO>')

import numpy as np
import pandas as pd
from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features
from multisync.feature_definitions import CONFIRMATORY_FEATURES

HZ = 1.0
DURATION = 300
WINDOW = 60
STEP = 10
N_SEEDS = 15
COUPLING = 0.5  # Fixed moderate coupling

SWITCH_FREQS = [0.5, 2.0, 5.0]   # switches per minute
RECOVERY_RATES = [0.02, 0.1, 0.5]  # 1/recovery_time (higher = faster recovery)

FEATS = list(CONFIRMATORY_FEATURES) + ['mean_synchrony','synchrony_entropy']

def generate_temporal_gt(switch_freq, recovery_rate, duration, hz, seed):
    """Generate dyad with controlled switching frequency and recovery rate.
    
    Creates a shared coupling signal that:
    - switches on/off at rate = switch_freq (Poisson process)
    - when "on", decays exponentially at rate = recovery_rate
    - P1 and P2 are coupled through this shared signal
    """
    rng = np.random.default_rng(seed)
    n = duration * int(hz)
    
    # Generate switching signal: Poisson on/off events
    event_rate = switch_freq / 60.0  # events per sample
    is_on = np.zeros(n, dtype=bool)
    state = False
    next_event = int(rng.exponential(1.0 / event_rate) if event_rate > 0 else n)
    for i in range(n):
        if i >= next_event:
            state = not state
            next_event = i + int(max(1, rng.exponential(1.0 / event_rate)))
        is_on[i] = state
    
    # Generate coupling amplitude with exponential recovery
    coupling = np.zeros(n)
    current_amp = 0.0
    for i in range(n):
        if is_on[i]:
            current_amp = min(current_amp + 0.3, 1.0)  # gradual rise
        else:
            current_amp *= (1.0 - recovery_rate)  # exponential decay
        coupling[i] = current_amp * COUPLING
    
    # Generate individual signals
    shared = np.sin(np.linspace(0, duration * 2 * np.pi / 60, n))
    shared += 0.5 * np.sin(np.linspace(0, duration * 2 * np.pi / 20, n))
    shared /= np.std(shared) + 1e-10
    
    a = coupling * shared + 0.3 * rng.normal(0, 1, n)
    b = coupling * shared + 0.3 * rng.normal(0, 1, n)
    return a, b

results = []
for sf in SWITCH_FREQS:
    for rr in RECOVERY_RATES:
        print(f'switch={sf:.1f}/min recovery={rr:.3f}')
        for seed in range(N_SEEDS):
            a, b = generate_temporal_gt(sf, rr, DURATION, HZ, 2000 + seed)
            wcc = sliding_window_wcc(a, b, window_size=int(WINDOW*HZ), hz=HZ, step_samples=STEP)
            if len(wcc) < 10: continue
            wcc[np.isnan(wcc)] = 0
            feats = extract_dynamic_features(wcc, hz=HZ/STEP, wcc_window_sec=WINDOW)
            for feat in FEATS:
                v = getattr(feats, feat, None)
                if v is not None and np.isfinite(v):
                    results.append({
                        'switch_freq': sf, 'recovery_rate': rr, 'seed': seed,
                        'feature': feat, 'value': float(v),
                    })

df = pd.DataFrame(results)
out = r'<REPO>\artifacts\gt2_temporal_dynamics.csv'
df.to_csv(out, index=False)

# Summary
print('\n=== GT-2 TEMPORAL DYNAMICS SUMMARY ===')
print(f'Response of each feature to switching_frequency (low→high)')
print(f'and recovery_rate (slow→fast) at fixed c={COUPLING}\n')

for feat in ['onset_latency','rise_time','recovery_time','switching_rate',
             'dwell_time','peak_amplitude','mean_synchrony','synchrony_entropy']:
    print(f'--- {feat} ---')
    for sf in SWITCH_FREQS:
        vals = df[(df.switch_freq==sf)&(df.feature==feat)]['value'].dropna()
        if len(vals)==0: continue
        row_vals = []
        for rr in RECOVERY_RATES:
            v = df[(df.switch_freq==sf)&(df.recovery_rate==rr)&(df.feature==feat)]['value'].dropna()
            row_vals.append(v.mean() if len(v)>0 else np.nan)
        v_str = '  '.join(f'{x:7.2f}' if np.isfinite(x) else '    N/A' for x in row_vals)
        print(f'  sf={sf:.1f}: {v_str}')

    # Effect size: high vs low switch_freq
    low = df[(df.switch_freq==SWITCH_FREQS[0])&(df.feature==feat)]['value'].dropna()
    high = df[(df.switch_freq==SWITCH_FREQS[-1])&(df.feature==feat)]['value'].dropna()
    if len(low)>0 and len(high)>0:
        d = (high.mean() - low.mean()) / (np.sqrt((low.std()**2 + high.std()**2)/2) + 1e-10)
        print(f'  d(switch low→high) = {d:+.2f}')

print(f'\nSaved: {out}')

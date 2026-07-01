"""
Focused surrogate-calibrated threshold optimization for SyncPipe.
Runs Level 3 grid at onset_threshold in [0.3, 0.4, 0.5, 0.6]
at c=0 (FPR) and c=0.6 (power), noise=0.3, 30 seeds, PRTF 499 surrogates.

Expected runtime: ~15 min.
"""
import sys
sys.path.insert(0, r'<REPO>')

import numpy as np
import pandas as pd
from multisync.validation.pgt1_intensity import Level3Config, run_level3_grid, bh_fdr
from multisync.feature_definitions import CONFIRMATORY_FEATURES

FEATS = list(CONFIRMATORY_FEATURES)

thresholds = [0.3, 0.4, 0.5, 0.6]
couplings = [0.0, 0.6]
noises = [0.3]
n_seeds = 30

all_fdr = []

for theta in thresholds:
    print(f"\nThreshold={theta}...")
    config = Level3Config(
        n_surrogates=499,
        seeds=list(range(1000, 1000 + n_seeds)),
        onset_threshold=theta,
        couplings=couplings,
        noise_ratios=noises,
    )
    df = run_level3_grid(config)
    # Compute FDR per cell
    for feat in FEATS:
        p_col = f'p_{feat}'
        if p_col not in df.columns:
            continue
        # Group by noise+coupling, compute rejection rate
        grouped = df.groupby(['noise_ratio','coupling'])
        for (noise, coup), sub in grouped:
            p_vals = sub[p_col].dropna().values
            if len(p_vals) == 0:
                continue
            mask = bh_fdr(p_vals, q=0.05)
            rej_rate = np.mean(mask)
            all_fdr.append({
                'onset_threshold': theta,
                'noise_ratio': noise,
                'coupling': coup,
                'feature': feat,
                'n_seeds': len(p_vals),
                'reject_rate': rej_rate,
            })

df_out = pd.DataFrame(all_fdr)
out = r'<REPO>\artifacts\threshold_calibration_fdr.csv'
df_out.to_csv(out, index=False)

# Summary
print("\n" + "="*60)
print("SURROGATE-CALIBRATED THRESHOLD OPTIMIZATION")
print("="*60)

for theta in thresholds:
    sub = df_out[df_out['onset_threshold'] == theta]
    print(f"\n--- theta = {theta} ---")
    for c in couplings:
        sub_c = sub[sub['coupling'] == c]
        tag = 'FPR(@c=0)' if c == 0 else 'POWER(@c=0.6)'
        print(f"  {tag}:")
        for feat in FEATS:
            row = sub_c[sub_c['feature'] == feat]
            v = row['reject_rate'].values[0] if len(row) > 0 else np.nan
            if np.isfinite(v):
                bar = '#' * int(v * 20)
                print(f"    {feat:20s}: {v:.1%} {bar}")

# Optimal threshold
print("\n--- OPTIMAL ---")
best = None
best_score = -1
for theta in thresholds:
    sub = df_out[df_out['onset_threshold'] == theta]
    fpr = sub[(sub['coupling'] == 0.0) & (sub['feature'] == 'peak_amplitude')]['reject_rate'].values
    pwr = sub[(sub['coupling'] == 0.6) & (sub['feature'] == 'peak_amplitude')]['reject_rate'].values
    if len(fpr) > 0 and len(pwr) > 0:
        score = pwr[0] - fpr[0] * 2  # penalize FPR twice
        print(f"  theta={theta}: FPR={fpr[0]:.1%}, Power={pwr[0]:.1%}, score={score:.3f}")
        if score > best_score and fpr[0] <= 0.05:
            best_score = score
            best = theta

print(f"\nBest threshold: {best}")
print(f"Saved: {out}")

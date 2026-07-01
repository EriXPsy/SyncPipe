"""
Analyze fixed PGT-2 results: test H2.4 (peak_amplitude invariant).
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv('artifacts/pgt2_grid_results.csv')

print('=== PGT-2 (Fixed) — peak_amplitude Drift Test ===')
print()

# H2.4: peak_amplitude should NOT vary with n_epochs
print('--- H2.4: peak_amplitude by (epoch_duration, n_epochs) ---')
pivot = df.pivot_table(values='peak_amplitude', 
                        index='epoch_duration', 
                        columns='n_epochs', 
                        aggfunc=['mean','std'])
print(pivot)
print()

# Also check mean_synchrony
print('--- mean_synchrony by (epoch_duration, n_epochs) ---')
pivot2 = df.pivot_table(values='mean_synchrony', 
                         index='epoch_duration', 
                         columns='n_epochs', 
                         aggfunc=['mean','std'])
print(pivot2)
print()

# Test H2.4: one-way ANOVA on peak_amplitude ~ n_epochs (within each epoch_duration)
print('--- H2.4 test: peak_amplitude ~ n_epochs (ANOVA) ---')
for d in [15.0, 30.0, 60.0]:
    sub = df[df.epoch_duration == d]
    groups = [sub[sub.n_epochs == n]['peak_amplitude'].values for n in [2, 4, 8]]
    f_stat, p_val = stats.f_oneway(*groups)
    status = "PASS" if p_val > 0.05 else "FAIL"
    print(f'  epoch={d}s: F={f_stat:.3f}, p={p_val:.4f} {status} (no drift)')
print()

# Also test other hypotheses
print('--- H2.1: dwell_time vs epoch_duration ---')
for d in [15.0, 30.0, 60.0]:
    vals = df[df.epoch_duration == d]['dwell_time']
    print(f'  epoch={d}s: dwell={vals.mean():.1f} +/- {vals.std():.1f}')
r = df['epoch_duration'].corr(df['dwell_time'], method='spearman')
print(f'  Spearman rho = {r:.3f}')

print()
print('--- H2.2: switching_rate vs n_epochs ---')
for n in [2, 4, 8]:
    vals = df[df.n_epochs == n]['switching_rate']
    print(f'  n_epochs={n}: switching={vals.mean():.3f} +/- {vals.std():.3f}')
r = df['n_epochs'].corr(df['switching_rate'], method='spearman')
print(f'  Spearman rho = {r:.3f}')

print()
print('--- H2.3: synchrony_entropy vs n_epochs ---')
for n in [2, 4, 8]:
    vals = df[df.n_epochs == n]['synchrony_entropy']
    print(f'  n_epochs={n}: entropy={vals.mean():.3f} +/- {vals.std():.3f}')
r = df['n_epochs'].corr(df['synchrony_entropy'], method='spearman')
print(f'  Spearman rho = {r:.3f}')
print()

# WCC window size used
print('--- Diagnostic: n_wcc_samples (should vary with epoch_duration) ---')
print(df[['epoch_duration', 'n_wcc_samples']].drop_duplicates().sort_values('epoch_duration'))

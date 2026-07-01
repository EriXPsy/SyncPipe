"""
Diagnostic: investigate why peak_amplitude drifts with n_epochs in PGT-2.

Hypothesis: wcc_window_sec=30s is too large for epoch_duration=15s,
causing temporal smoothing that attenuates WCC peaks.

Fix: set wcc_window_sec = epoch_duration / 2 (or match epoch_duration).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from multisync.simulation.shared_signal_model import generate_signals, alternating_coupling
from multisync.dynamic_features import sliding_window_wcc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Config ---
epoch_duration = 15.0
n_epochs = 4
c_high = 0.80
c_low = 0.15
wcc_windows = [5, 10, 15, 30]  # test different window sizes
hz = 1.0
seed = 2000

# --- Generate signals ---
c_func = alternating_coupling(c_high=c_high, c_low=c_low,
                               epoch_duration=epoch_duration, n_epochs=n_epochs)
result = generate_signals(
    c_t=c_func,
    duration_sec=2 * n_epochs * epoch_duration + 20,  # extra padding
    hz=hz,
    noise_sigma=0.3,
    seed=seed,
)

# --- Compute WCC with different window sizes ---
fig, axes = plt.subplots(len(wcc_windows) + 1, 1, figsize=(12, 3*(len(wcc_windows)+1)))
fig.suptitle(f"PGT-2 Diagnostic: epoch={epoch_duration}s, n={n_epochs}, c_high={c_high}")

# Plot c(t)
ax = axes[0]
ax.plot(result.t, result.c_t, 'k-', linewidth=2, label='c(t)')
ax.set_ylabel('c(t)')
ax.set_ylim(-0.05, 1.05)
ax.legend()

# Compute and plot WCC for each window size
for i, wcc_win in enumerate(wcc_windows):
    ax = axes[i+1]
    wcc = sliding_window_wcc(result.x_A, result.x_B,
                              window_size=int(wcc_win * hz),
                              hz=hz)
    # Time axis for WCC (center of window)
    wcc_t = np.arange(len(wcc)) / hz + wcc_win / 2
    ax.plot(wcc_t, wcc, label=f'WCC (win={wcc_win}s)')
    ax.axhline(y=c_high, color='r', linestyle='--', alpha=0.5, label=f'c_high={c_high}')
    ax.axhline(y=c_low, color='b', linestyle='--', alpha=0.5, label=f'c_low={c_low}')
    ax.set_ylabel(f'WCC (win={wcc_win}s)')
    ax.legend()
    
    # Compute peak_amplitude and mean_synchrony
    pa = np.nanmax(wcc)
    ms = np.nanmean(wcc)
    print(f"  wcc_window={wcc_win:2d}s: peak_amplitude={pa:.3f}, mean_synchrony={ms:.3f}")

ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('artifacts/pgt2_diagnostic_wcc_window.png', dpi=150)
print(f"\nSaved to artifacts/pgt2_diagnostic_wcc_window.png")

# --- Also check: does peak_amplitude drift with n_epochs for fixed epoch_duration? ---
print("\n=== peak_amplitude vs n_epochs (epoch_duration=15s, wcc_window=15s) ===")
for n_ep in [2, 4, 8]:
    c_func = alternating_coupling(c_high=c_high, c_low=c_low,
                                   epoch_duration=epoch_duration, n_epochs=n_ep)
    result = generate_signals(
        c_t=c_func,
        duration_sec=2 * n_ep * epoch_duration + 20,
        hz=hz,
        noise_sigma=0.3,
        seed=seed,
    )
    wcc = sliding_window_wcc(result.x_A, result.x_B,
                              window_size=int(epoch_duration * hz),  # match epoch
                              hz=hz)
    pa = np.nanmax(wcc)
    ms = np.nanmean(wcc)
    print(f"  n_epochs={n_ep}: peak={pa:.3f}, mean={ms:.3f}")

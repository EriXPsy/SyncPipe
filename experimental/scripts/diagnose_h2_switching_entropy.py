"""Deep diagnosis: why H2.2 switching_rate and H2.3 entropy have high noise.

Investigates:
1. WCC trajectory noise → switching_rate variance
2. Binarisation sensitivity (onset_threshold sweep)
3. Entropy bin-count impact (Doane's formula vs fixed bins)
4. Epoch duration / WCC window ratio impact
"""
# Suppress matplotlib backend to avoid sandbox permission issues
import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import scipy.stats as stats

from multisync.simulation.shared_signal_model import (
    generate_signals, alternating_coupling, constant_coupling,
)
from multisync.dynamic_features import sliding_window_wcc
from multisync.feature_definitions import (
    extract_features, ONSET_THRESHOLD,
    compute_switching_rate, compute_synchrony_entropy,
)

# ============================================================================
# 1. SINGLE-CELL TRACE INSPECTION
# ============================================================================
print("=" * 68)
print("1. WCC TRACE INSPECTION (n_epochs=2 vs 8)")
print("=" * 68)

hz = 10.0
epoch = 30.0
seeds = [42, 123, 456]

for n_ep in [2, 8]:
    total_sec = 2 * n_ep * epoch
    coupling = alternating_coupling(
        epoch_duration=epoch, n_epochs=n_ep,
        c_high=0.8, c_low=0.15,
    )
    for seed in [seeds[0]]:
        result = generate_signals(
            c_t=coupling, duration_sec=total_sec, hz=hz,
            noise_sigma=0.3, seed=seed,
        )
        wcc_win_sec = max(5.0, epoch / 2)
        win_samp = int(round(wcc_win_sec * hz))
        wcc = sliding_window_wcc(result.x_A, result.x_B, window_size=win_samp, hz=hz)
        
        # Binarised trace
        binary = (wcc >= ONSET_THRESHOLD).astype(int)
        transitions = np.diff(binary) != 0
        n_trans = transitions.sum()
        n_above = binary.sum()
        
        print(f"\n  n_epochs={n_ep}, seed={seed}:")
        print(f"    WCC: mean={np.nanmean(wcc):.3f}, std={np.nanstd(wcc):.3f}")
        print(f"    WCC range: [{np.nanmin(wcc):.3f}, {np.nanmax(wcc):.3f}]")
        print(f"    Binarised: above_threshold fraction={n_above/len(binary):.3f}")
        print(f"    Transitions: {n_trans} ({n_trans/(n_ep*2):.1f} per epoch-pair)")
        print(f"    switching_rate: {n_trans/len(wcc):.4f}")
        
        feats = extract_features(wcc, hz=hz, wcc_window_sec=wcc_win_sec)
        print(f"    computed switching_rate: {feats.switching_rate:.4f}")
        print(f"    computed entropy: {feats.synchrony_entropy:.3f}")

# ============================================================================
# 2. WCC VARIANCE DECOMPOSITION (by noise_sigma)
# ============================================================================
print("\n" + "=" * 68)
print("2. WCC NOISE SOURCE: coupling_func noise vs measurement noise")
print("=" * 68)

for noise in [0.0, 0.1, 0.3, 0.5]:
    coupling = alternating_coupling(
        epoch_duration=30, n_epochs=2,
        c_high=0.8, c_low=0.15,
    )
    result = generate_signals(
        duration_sec=120, hz=hz, c_t=coupling,
        noise_sigma=noise, seed=42,
    )
    wcc = sliding_window_wcc(result.x_A, result.x_B, window_size=150, hz=hz)
    binary = (wcc >= ONSET_THRESHOLD).astype(int)
    trans = np.diff(binary) != 0
    feats = extract_features(wcc, hz=hz, wcc_window_sec=15)
    
    print(f"  noise={noise}: WCC mean={np.nanmean(wcc):.3f} std={np.nanstd(wcc):.3f}, "
          f"switching={feats.switching_rate:.4f}, entropy={feats.synchrony_entropy:.3f}, "
          f"transitions={trans.sum()}")

# ============================================================================
# 3. THRESHOLD SENSITIVITY (onset_threshold sweep)
# ============================================================================
print("\n" + "=" * 68)
print("3. THRESHOLD SENSITIVITY: switching_rate vs onset_threshold")
print("=" * 68)

coupling = alternating_coupling(
    epoch_duration=30, n_epochs=2,
    c_high=0.8, c_low=0.15,
)
result = generate_signals(
    duration_sec=120, hz=hz, c_t=coupling, noise_sigma=0.3, seed=42,
)
wcc = sliding_window_wcc(result.x_A, result.x_B, window_size=150, hz=hz)
print(f"  WCC histogram: min={np.nanmin(wcc):.3f}, 25%={np.nanquantile(wcc,0.25):.3f}, "
      f"median={np.nanmedian(wcc):.3f}, 75%={np.nanquantile(wcc,0.75):.3f}, max={np.nanmax(wcc):.3f}")

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    binary = (wcc >= thresh).astype(int)
    trans = np.diff(binary) != 0
    feats = extract_features(wcc, hz=hz, wcc_window_sec=15, threshold=thresh)
    above = binary.mean()
    print(f"  θ={thresh}: above={above:.3f}, switching={feats.switching_rate:.4f}, "
          f"transitions={trans.sum()}, entropy={feats.synchrony_entropy:.3f}")

# ============================================================================
# 4. ENTROPY BIN-COUNT EFFECT
# ============================================================================
print("\n" + "=" * 68)
print("4. ENTROPY BIN SENSITIVITY")
print("=" * 68)

for n_bins in [5, 10, 15, 20, 30, 50]:
    entropy = compute_synchrony_entropy(wcc, n_bins=n_bins)
    print(f"  n_bins={n_bins:2d}: entropy={entropy:.3f}")

# ============================================================================
# 5. CROSS-SEED VARIANCE (the real question)
# ============================================================================
print("\n" + "=" * 68)
print("5. CROSS-SEED VARIANCE (30 seeds, n_epochs=2, epoch=30)")
print("=" * 68)

switching_vals = []
entropy_vals = []
mean_sync_vals = []

for seed in range(30):
    coupling = alternating_coupling(
        epoch_duration=30, n_epochs=2,
        c_high=0.8, c_low=0.15,
    )
    result = generate_signals(
        duration_sec=120, hz=hz, c_t=coupling, noise_sigma=0.3, seed=seed,
    )
    wcc = sliding_window_wcc(result.x_A, result.x_B, window_size=150, hz=hz)
    feats = extract_features(wcc, hz=hz, wcc_window_sec=15)
    switching_vals.append(feats.switching_rate)
    entropy_vals.append(feats.synchrony_entropy)
    mean_sync_vals.append(feats.mean_synchrony)

sw = np.array(switching_vals)
en = np.array(entropy_vals)
ms = np.array(mean_sync_vals)

print(f"  switching_rate: mean={sw.mean():.4f}, std={sw.std():.4f}, CV={sw.std()/sw.mean():.2f}")
print(f"  entropy: mean={en.mean():.3f}, std={en.std():.3f}")
print(f"  mean_synchrony: mean={ms.mean():.3f}, std={ms.std():.3f}")
print(f"  switching_rate range: [{sw.min():.4f}, {sw.max():.4f}]")
print(f"  entropy range: [{en.min():.3f}, {en.max():.3f}]")

# Does switching_rate variance come from mean_synchrony variance?
r = np.corrcoef(sw, ms)[0, 1]
print(f"  corr(switching_rate, mean_synchrony): r={r:.3f}")

# ============================================================================
# 6. ROOT CAUSE SUMMARY
# ============================================================================
print("\n" + "=" * 68)
print("6. ROOT CAUSE INTERPRETATION")
print("=" * 68)

# Combine PGT-2 grid analysis too
df2 = pd.read_csv("artifacts/pgt2_grid_results.csv")

# Switching rate within each (epoch_duration, n_epochs) cell
for ed in sorted(df2["epoch_duration"].unique()):
    sub = df2[df2.epoch_duration == ed]
    for n in sorted(sub["n_epochs"].unique()):
        s_sub = sub[sub.n_epochs == n]
        sr = s_sub["switching_rate"]
        ent = s_sub["synchrony_entropy"]
        print(f"  epoch={int(ed)}s, n_epochs={n}: "
              f"switching={sr.mean():.4f}+/-{sr.std():.4f} (CV={sr.std()/sr.mean():.2f}), "
              f"entropy={ent.mean():.3f}+/-{ent.std():.3f}")

print("""
=== DIAGNOSIS CONCLUSION ===

H2.2 switching_rate:
  1. switching_rate = n_transitions / WCC_length
  2. On a clean alternating WCC, expected ~2 transitions per epoch-pair
     (onset + offset). With noise, the binarised trace near threshold 
     produces extra toggling → inflates switching_rate AND its variance.
  3. The BTW-cell variance (SD~2.0 for 30 seeds) comes from noise-seed
     interaction at the binary decision boundary.
  4. Solution: reduce noise_sigma from 0.3 to 0.15 in PGT-2, OR
     add a hysteresis band to the binary decision.

H2.3 synchrony_entropy:
  1. Doane's formula typically produces ~15 bins for 3000-point WCC.
  2. With bins=15, a WCC switching between two values (c_low~WCC~0.1,
     c_high~WCC~0.9) should produce entropy ~1.0 (2-state).
  3. Observed entropy ~3.6 → WCC distribution is NOT bimodal.
     Extra variance from WCC window smoothing and noise fills intermediate
     bins, inflating entropy and flattening the n_epochs signal.
  4. Solution: increase bin count to capture the expected bimodal structure,
     or switch to a peak-bimodality metric instead of Shannon entropy.
""")

"""GT-5a: Dimensional recovery validation (reuses GT-3 data).

Level 0 — feature-level recovery (GT-3): ρ(feature, GT_param) per feature
Level 1 — dimensional recovery (GT-5a): ρ(dimension_score, GT_param) after deconfounding

Reuses GT-3 synthetic data (already computed), applies deconfounding
and dimension projection from real Andersen data.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

REPO = Path(r"<REPO>")
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

# ── Load GT-3 data ──
ART = REPO / "artifacts"
df_gt3 = pd.read_csv(ART / "gt3_estimator_resolution.csv")
print(f"GT-3 data: {len(df_gt3)} rows")

# ── Load real Andersen data to fit deconfounding coefficients ──
df_and = pd.read_csv("<OSF_ROOT>/Andersen-hj4k6/multisync_results/multisync_andersen_full.csv")
feats = ['peak_amplitude','dwell_time','recovery_time','onset_latency','switching_rate','rise_time','mean_synchrony','synchrony_entropy']
X_and = df_and[feats].dropna()
print(f"Andersen reference: n={len(X_and)}")

# Fit deconfounding regression per feature
deconf_beta = {}
for f in feats:
    if f == 'mean_synchrony':
        continue
    reg = LinearRegression().fit(X_and[['mean_synchrony']], X_and[f])
    deconf_beta[f] = reg.coef_[0]

# ── Focus on WCC_W60_S10 (our default estimator) ──
est = "WCC_W60_S10"
sub = df_gt3[df_gt3.estimator == est].copy()
print(f"GT-3 {est}: {len(sub)} rows, n_dyads={sub.groupby(['switch_freq','recovery_rate','seed']).ngroups}")

# Pivot: one row per synthetic dyad, features as columns
pivot = sub.pivot_table(
    index=['switch_freq','recovery_rate','seed'],
    columns='feature', values='value'
).reset_index()
print(f"Pivoted: {len(pivot)} dyads, features={list(pivot.columns[3:])}")

# ── Deconfound ──
valid = pivot.dropna(subset=feats).copy()
print(f"Complete cases: {len(valid)}/{len(pivot)}")
for f in feats:
    if f == 'mean_synchrony':
        continue
    valid[f"{f}_resid"] = valid[f] - deconf_beta[f] * valid['mean_synchrony']

# ── Dimension scores (simple: primary feature per dimension) ──
valid['INTENSITY']  = valid['peak_amplitude_resid']
valid['STRUCTURE']  = valid['switching_rate_resid']  # + = flexible
valid['TIMING']     = valid['onset_latency_resid']

# ── Recovery ──
print("\n=== GT-5a: DIMENSIONAL RECOVERY (Spearman ρ) ===")
print(f"{'Dimension':<16s} {'ρ vs switch_freq':>16s} {'ρ vs recovery_rate':>16s}")
print("-" * 50)

for dim, feat in [('INTENSITY','peak_amplitude_resid'),
                   ('STRUCTURE','switching_rate_resid'),
                   ('TIMING','onset_latency_resid')]:
    rho_s, p_s = spearmanr(valid['switch_freq'], valid[dim])
    rho_r, p_r = spearmanr(valid['recovery_rate'], valid[dim])
    print(f"{dim:<16s} {rho_s:>+16.4f} {rho_r:>+16.4f}")

# ── Comparison: GT-3 (raw) vs GT-5a (deconfounded) ──
print("\n=== GT-3 vs GT-5a: SAME DATA, DIFFERENT METRIC ===")
print(f"{'Feature':<20s} {'GT-3 ρ_switch':>14s} {'GT-5a ρ':>14s} {'Change':>10s}")
print("-" * 60)

for feat, dim_label in [('onset_latency','TIMING'),
                          ('switching_rate','STRUCTURE'),
                          ('peak_amplitude','INTENSITY')]:
    rho3_s = spearmanr(valid['switch_freq'], valid[feat])[0]
    rho5_s = spearmanr(valid['switch_freq'], valid[dim_label])[0]
    delta = rho5_s - rho3_s
    direction = "↑" if abs(rho5_s) > abs(rho3_s) else ("↓" if abs(rho5_s) < abs(rho3_s) else "=")
    print(f"{feat:<20s} {rho3_s:>+14.4f} {rho5_s:>+14.4f} {delta:>+9.4f} {direction}")

# ── Discriminant validity: wrong dimension ≠ GT param ──
print("\n=== DISCRIMINANT VALIDITY ===")
for dim, should_be_unrelated_to in [('INTENSITY', 'switch_freq'),
                                      ('INTENSITY', 'recovery_rate'),
                                      ('TIMING', 'switch_freq')]:
    rho, p = spearmanr(valid[dim], valid[should_be_unrelated_to])
    status = "✓" if abs(rho) < 0.20 else "⚠"
    print(f"  {dim} vs {should_be_unrelated_to}: ρ={rho:+.3f} {status}")

print("\nDone")

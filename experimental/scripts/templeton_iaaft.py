"""Templeton IAAFT surrogate analysis.

For each dyad, generates IAAFT surrogates of the connection ratings,
recomputes WCC on surrogate pairs, and compares real vs surrogate features.
"""
import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

CORE = Path(r"<REPO>")
sys.path.insert(0, str(CORE))
from multisync.validation.pgt1_intensity import iaaft_surrogate
from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features

warnings.filterwarnings("ignore")

OUT = Path("<OSF_ROOT>/Templeton-Twz3s/multisync_results")
OUT.mkdir(exist_ok=True)
LOG = OUT / "templeton_iaaft.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("templeton_iaaft")

CONFIRMATORY = ["onset_latency","rise_time","peak_amplitude","recovery_time","dwell_time","switching_rate"]
DIAGNOSTICS = ["mean_synchrony","synchrony_entropy"]
ALL_F = CONFIRMATORY + DIAGNOSTICS
DIRECTION = {"onset_latency":"less","rise_time":"less","switching_rate":"less",
             "peak_amplitude":"greater","recovery_time":"greater","dwell_time":"greater",
             "mean_synchrony":"greater","synchrony_entropy":"less"}

N_SURR, SEED, HZ = 499, 42, 10.0
W, S = 300, 100  # W=30s, step=10s at 10Hz

# Load all dyads
df_full = pd.read_csv("<OSF_ROOT>/Templeton-Twz3s/multisync_results/multisync_templeton_full.csv")
dyad_ids = df_full["dyad_id"].unique()
log.info(f"Dyads: {len(dyad_ids)}")

rows = []
for i, did in enumerate(dyad_ids):
    sub = df_full[df_full.dyad_id == did].iloc[0]
    group = sub["group"]
    id_a, id_b = did.split("_")

    # Load raw ratings
    fa = f"<OSF_ROOT>/Templeton-Twz3s/Data/continuous_connection_ratings/{group}/{id_a}_{id_b}.csv"
    fb = f"<OSF_ROOT>/Templeton-Twz3s/Data/continuous_connection_ratings/{group}/{id_b}_{id_a}.csv"
    try:
        s1 = pd.read_csv(fa); s2 = pd.read_csv(fb)
    except Exception:
        continue
    n = min(len(s1), len(s2))
    a = (s1.Rating.values[:n] - s1.Rating.values[:n].mean()) / (s1.Rating.values[:n].std() + 1e-10)
    b = (s2.Rating.values[:n] - s2.Rating.values[:n].mean()) / (s2.Rating.values[:n].std() + 1e-10)

    # Real WCC + features
    wcc_real = sliding_window_wcc(a, b, window_size=W, hz=HZ, step_samples=S)
    feats_real = extract_dynamic_features(wcc_real, hz=HZ/S, wcc_window_sec=30.)
    real_d = {f: getattr(feats_real, f, np.nan) for f in ALL_F}

    # IAAFT surrogates
    rng = np.random.default_rng(SEED + i)
    surr_feats = {f: [] for f in ALL_F}
    for _ in range(N_SURR):
        b_surr = iaaft_surrogate(b, rng)
        wcc_s = sliding_window_wcc(a, b_surr, window_size=W, hz=HZ, step_samples=S)
        fs = extract_dynamic_features(wcc_s, hz=HZ/S, wcc_window_sec=30.)
        for f in ALL_F:
            surr_feats[f].append(getattr(fs, f, np.nan))

    row = {"dyad_id": did, "group": group, "n_surr": N_SURR}
    for f in ALL_F:
        row[f"{f}_real"] = real_d[f]
        sv = np.array(surr_feats[f])
        row[f"{f}_surr_mean"] = np.nanmean(sv)
        row[f"{f}_surr_std"] = np.nanstd(sv)
    rows.append(row)

    if (i+1) % 50 == 0:
        log.info(f"  {i+1}/{len(dyad_ids)}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "templeton_iaaft_per_dyad.csv", index=False)
log.info(f"Saved {len(df)} rows to per_dyad")

# ── Summary: Wilcoxon paired real vs surr_mean ──
log.info("\n=== IAAFT SUMMARY ===")
summary_rows = []
for f in ALL_F:
    r = df[f"{f}_real"].dropna()
    s = df[f"{f}_surr_mean"].dropna()
    valid = r.notna() & s.notna()
    n = valid.sum()
    if n < 10:
        log.info(f"  {f}: n={n} < 10, skip")
        continue
    alt = DIRECTION[f]
    stat, p = stats.wilcoxon(r[valid], s[valid], alternative=alt)
    summary_rows.append({"feature": f, "n_dyads": n, "real_median": r[valid].median(),
                         "surr_median": s[valid].median(), "p_raw": p, "alternative": alt})

df_s = pd.DataFrame(summary_rows)
# BH-FDR within confirmatory
confirm_mask = df_s["feature"].isin(CONFIRMATORY)
p_confirm = df_s.loc[confirm_mask, "p_raw"].values
if len(p_confirm) > 0:
    from multisync.validation.pgt1_intensity import bh_fdr as _bh
    reject = _bh(p_confirm, q=0.05)
    df_s.loc[confirm_mask, "p_fdr"] = p_confirm * len(p_confirm) / np.arange(1, len(p_confirm)+1)  # simple BH
    # Actually use proper BH
    n = len(p_confirm)
    ranked = np.argsort(p_confirm)
    p_fdr = np.minimum(1, p_confirm[ranked] * n / (np.arange(n) + 1))
    for j in range(n-2, -1, -1):
        p_fdr[j] = min(p_fdr[j], p_fdr[j+1])
    p_fdr_out = np.zeros(n)
    p_fdr_out[ranked] = p_fdr
    df_s.loc[confirm_mask, "p_fdr"] = p_fdr_out
    df_s["sig_05"] = False
    df_s.loc[confirm_mask, "sig_05"] = p_fdr_out < 0.05

df_s.to_csv(OUT / "templeton_iaaft_summary.csv", index=False)
log.info("\n" + df_s.to_string())
log.info("Done")

"""
WCLC lag/sign diagnosis on the Gordon (Mayo & Gordon 2025) raw behavioral data.

Question
--------
Gordon's "Contextual Pulls" game places two players on a circle. Under some
contextual pulls the two move with a *consistent rhythm but in opposite
directions* (anti-phase): a high |r| at lag ~= 0 with NEGATIVE sign. The
current epoch classifier masks lagged epochs by |WCLC| >= threshold, so a
zero-lag anti-phase epoch could be misread as "lagged synchrony".

This script does NOT change any production code. It loads the raw signals,
runs the same lag sweep WCLC uses, and reports, per (dyad, condition):
  - peak |r| and the lag at which it occurs
  - the SIGN of r at that peak
  - whether the peak sits at lag ~= 0 (=> NOT leader-follower)

If Gordon is dominated by lag~=0 negative-r windows, the correct fix is the
GENERAL rule "lagged requires |best_lag| > 0", not a Gordon-specific patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(r"<OSF_ROOT>\Gordon-349su\behavioral data")
TARGET_HZ = 10.0
WINDOW = 60      # samples (6 s @ 10 Hz)
STEP = 10
MAX_LAG = 30     # samples (3 s @ 10 Hz)
ZERO_LAG_TOL = 2  # |best_lag| <= 2 samples (0.2 s) counts as "zero-lag"

RAW_COLUMNS = ["time_p1", "R_p1", "theta_p1", "time_p2", "R_p2", "theta_p2"]


def _motion(t, R, theta):
    t = np.asarray(t, float)
    R = np.asarray(R, float)
    theta = np.unwrap(np.asarray(theta, float))
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, np.nan, dt)
    dR = np.gradient(R) / dt
    dth = np.gradient(theta) / dt
    return np.sqrt(dR ** 2 + (R * dth) ** 2)


def _resample(t, x, grid):
    return np.interp(grid, t, x)


def _lagged_r(a, b, lag):
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return np.nan if np.isnan(r) else float(r)


def _window_peaks(a, b):
    """Per-window peak |r|, its lag, and its signed r."""
    n = min(len(a), len(b))
    out = []
    for start in range(0, n - WINDOW + 1, STEP):
        aw = a[start:start + WINDOW]
        bw = b[start:start + WINDOW]
        best_absr, best_lag, best_r = 0.0, 0, 0.0
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            r = _lagged_r(aw, bw, lag)
            if np.isfinite(r) and abs(r) > best_absr:
                best_absr, best_lag, best_r = abs(r), lag, r
        if best_absr > 0:
            out.append((best_absr, best_lag, best_r))
    return out


def main():
    if not DATA_ROOT.exists():
        print(f"[ERR] not found: {DATA_ROOT}")
        sys.exit(1)

    rows = []
    dyad_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    for dyad_dir in dyad_dirs:
        for n in range(1, 5):
            csv = dyad_dir / f"exp{n}.csv"
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv, header=None, names=RAW_COLUMNS)
            except Exception as exc:
                print(f"  skip {csv}: {exc}")
                continue
            if df.shape[1] < 6 or len(df) < WINDOW * 2:
                continue
            ma = _motion(df.time_p1, df.R_p1, df.theta_p1)
            mb = _motion(df.time_p2, df.R_p2, df.theta_p2)
            t1 = df.time_p1.values
            t2 = df.time_p2.values
            t0 = max(t1[0], t2[0])
            t9 = min(t1[-1], t2[-1])
            if t9 <= t0:
                continue
            grid = np.arange(t0, t9, 1.0 / TARGET_HZ)
            a = _resample(t1, ma, grid)
            b = _resample(t2, mb, grid)
            a = np.nan_to_num(a, nan=0.0)
            b = np.nan_to_num(b, nan=0.0)

            peaks = _window_peaks(a, b)
            if not peaks:
                continue
            absr = np.array([p[0] for p in peaks])
            lags = np.array([p[1] for p in peaks])
            sgnr = np.array([p[2] for p in peaks])

            # Elevated windows (synchrony candidates).
            hi = absr >= 0.5
            if hi.sum() == 0:
                hi = absr >= np.percentile(absr, 75)
            zero_lag = np.abs(lags[hi]) <= ZERO_LAG_TOL
            neg = sgnr[hi] < 0

            rows.append({
                "dyad": dyad_dir.name,
                "cond": f"exp{n}",
                "n_win": len(peaks),
                "n_hi": int(hi.sum()),
                "mean_absr": round(float(absr[hi].mean()), 3),
                "pct_zero_lag": round(float(zero_lag.mean()) * 100, 1),
                "pct_neg_r": round(float(neg.mean()) * 100, 1),
                "pct_zero_lag_AND_neg": round(
                    float((zero_lag & neg).mean()) * 100, 1),
                "median_abs_lag": int(np.median(np.abs(lags[hi]))),
            })

    if not rows:
        print("[ERR] no usable dyads parsed.")
        sys.exit(1)

    res = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "gordon_wclc_lag_diagnosis.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)

    print("=" * 78)
    print("GORDON WCLC LAG/SIGN DIAGNOSIS")
    print(f"  window={WINDOW} samp  step={STEP}  max_lag={MAX_LAG} samp  "
          f"zero_lag_tol=+-{ZERO_LAG_TOL} samp @ {TARGET_HZ} Hz")
    print("=" * 78)
    print(f"dyad-conditions parsed : {len(res)}")
    print(f"mean peak |r| (hi win) : {res.mean_absr.mean():.3f}")
    print(f"% windows at zero lag  : {res.pct_zero_lag.mean():.1f}%")
    print(f"% windows negative r   : {res.pct_neg_r.mean():.1f}%")
    print(f"% zero-lag AND neg-r   : {res.pct_zero_lag_AND_neg.mean():.1f}%")
    print(f"median |best lag|      : {res.median_abs_lag.median():.0f} samples "
          f"({res.median_abs_lag.median() / TARGET_HZ:.2f} s)")
    print("-" * 78)
    print("INTERPRETATION:")
    zl = res.pct_zero_lag.mean()
    neg = res.pct_neg_r.mean()
    if zl >= 60 and neg >= 40:
        print("  -> Gordon is dominated by ZERO-LAG, often NEGATIVE-r coupling.")
        print("  -> This is ANTI-PHASE (consistent rhythm, opposite direction),")
        print("     NOT leader-follower lagged synchrony.")
        print("  -> Correct fix: GENERAL rule 'lagged epoch requires |lag|>0';")
        print("     classify zero-lag negative-r windows as anti-phase.")
    else:
        print("  -> Mixed lag structure; inspect per-condition rows in CSV.")
    print(f"\nfull table -> {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

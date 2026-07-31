"""Lerique temporal-prediction gap check — verify the prediction.py
``_compute_effective_gap`` fix on REAL Lerique WCC traces.

Motivation
----------
The pre-fix code (HEAD) used ``horizon_gap_rows = horizon_windows`` (feature
rows) instead of converting the label's future horizon to feature-row units.
The fix (working tree) uses
``horizon_gap_rows = int(np.ceil(horizon_windows * window_size / step))``.
On synthetic data the max |ΔΔAUC| was 0.200 (see _repro_02_delta_auc_gap.py).
This script measures the SAME effect on real Lerique WCC traces: for every
trace we run ``rolling_origin_cv`` twice — once with the FIXED
``_compute_effective_gap`` (current code) and once with the OLD (pre-fix)
implementation monkeypatched in — and compare ``mean_delta_auc`` and the
leakage flag (``mean_delta_auc > LEAKAGE_DELTA_AUC_THRESHOLD`` = 0.14).

Usage
------
    python deliverables/run_lerique_prediction_gapcheck.py

Outputs
-------
    artifacts/prediction/lerique_gapcheck.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Keep output readable: suppress sklearn FutureWarnings / ConvergenceWarnings /
# Bootstrap-CI warnings emitted inside rolling_origin_cv.
warnings.filterwarnings("ignore")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import multisync.prediction as pred  # noqa: E402

WCC_CSV = REPO_ROOT / "artifacts" / "wcc_traces" / "lerique_wcc_traces.csv"
OUT_DIR = REPO_ROOT / "artifacts" / "prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_THRESHOLD = 0.14  # multisync.prediction.LEAKAGE_DELTA_AUC_THRESHOLD

# --- OLD (pre-fix, from git show HEAD:multisync/prediction.py) _compute_effective_gap ---
def _old_compute_effective_gap(gap, window_size, horizon_windows):
    step = max(1, window_size // 2)
    min_physical_gap_rows = int(np.ceil(window_size / step))
    horizon_gap_rows = horizon_windows  # <-- THE BUG: raw int, no conversion
    return max(gap, min_physical_gap_rows + horizon_gap_rows)


# Speed: disable nonlinear baselines (they do not affect mean_delta_auc).
pred._nonlinear_model_factories = lambda seed=42: {}


def _run_one(trace, hw, ns, thr):
    return pred.rolling_origin_cv(
        trace,
        window_size=1,  # <= min_window forces auto-adjust to min(60, n//4)
        hz=1.0,
        horizon_windows=hw,
        n_splits=ns,
        gap=0,  # gap=0 so the auto physical-time gap is what differs old vs new
        threshold=thr,  # median split -> balanced labels, real signal
        seed=42,
        pair_name="lerique",
        mode="intra",
    )


def main():
    import pandas as pd
    df = pd.read_csv(WCC_CSV)
    traces = []
    for _, row in df.iterrows():
        wcc = np.asarray(json.loads(row["wcc_json"]), dtype=float)
        if wcc.size < 30:
            continue
        traces.append((row["modality"], row["condition"], wcc, float(np.nanmedian(wcc))))

    fixed_fn = pred._compute_effective_gap  # current (fixed) implementation

    summary = {}
    for hw in (1, 2):
        ns = 5
        per_mod = {}
        for mod, cond, wcc, thr in traces:
            # NEW (fixed) — wrapped: short traces can raise in TimeSeriesSplit
            # when the (now correctly larger) buffer exceeds the sample count.
            # That is a genuine data-limitation, recorded as data_limited_cv.
            try:
                pred._compute_effective_gap = fixed_fn
                r_new = _run_one(wcc, hw, ns, thr)
                pred._compute_effective_gap = _old_compute_effective_gap
                r_old = _run_one(wcc, hw, ns, thr)
            except Exception as e:
                rec = {
                    "modality": mod,
                    "condition": cond,
                    "n_samples": int(len(wcc)),
                    "delta_new": None,
                    "delta_old": None,
                    "delta_diff": None,
                    "warning_new": f"data_limited_cv:{type(e).__name__}",
                    "warning_old": f"data_limited_cv:{type(e).__name__}",
                    "leak_new": None,
                    "leak_old": None,
                }
                per_mod.setdefault(mod, []).append(rec)
                continue
            d_new = float(r_new.mean_delta_auc)
            d_old = float(r_old.mean_delta_auc)
            warn_new = r_new.warning
            warn_old = r_old.warning
            leak_new = (d_new > LEAKAGE_THRESHOLD) if warn_new is None else None
            leak_old = (d_old > LEAKAGE_THRESHOLD) if warn_old is None else None
            rec = {
                "modality": mod,
                "condition": cond,
                "n_samples": int(len(wcc)),
                "delta_new": d_new,
                "delta_old": d_old,
                "delta_diff": d_new - d_old,
                "warning_new": warn_new,
                "warning_old": warn_old,
                "leak_new": leak_new,
                "leak_old": leak_old,
            }
            per_mod.setdefault(mod, []).append(rec)

        # aggregate per modality
        agg = {}
        for mod, recs in per_mod.items():
            valid = [r for r in recs if r["warning_new"] is None]
            limited = [r for r in recs if r["warning_new"] is not None]
            if valid:
                diffs = [r["delta_diff"] for r in valid]
                flips = sum(
                    1 for r in valid
                    if r["leak_old"] is not None and r["leak_new"] is not None
                    and r["leak_old"] != r["leak_new"]
                )
                agg[mod] = {
                    "n_traces": len(recs),
                    "n_valid_cv": len(valid),
                    "n_data_limited": len(limited),
                    "mean_delta_new": float(np.mean([r["delta_new"] for r in valid])),
                    "mean_delta_old": float(np.mean([r["delta_old"] for r in valid])),
                    "mean_abs_delta_diff": float(np.mean(np.abs(diffs))),
                    "max_abs_delta_diff": float(np.max(np.abs(diffs))),
                    "leak_flag_flips": int(flips),
                    "limited_warnings": sorted({r["warning_new"] for r in limited}),
                }
            else:
                agg[mod] = {
                    "n_traces": len(recs),
                    "n_valid_cv": 0,
                    "n_data_limited": len(limited),
                    "limited_warnings": sorted({r["warning_new"] for r in limited}),
                }
        summary[f"horizon_windows={hw}"] = agg
        # incremental write so partial progress survives interruption
        out_path = OUT_DIR / "lerique_gapcheck.json"
        out_path.write_text(json.dumps({
            "wcc_csv": str(WCC_CSV),
            "leakage_threshold": LEAKAGE_THRESHOLD,
            "note": "gap=0 forces auto physical-time gap; old vs new differ only by "
                    "the horizon_windows->feature-row conversion.",
            "results": summary,
            "partial": True,
        }, indent=2, ensure_ascii=False))
        # console
        print(f"\n=== horizon_windows={hw}, n_splits={ns} ===")
        for mod, a in agg.items():
            if a["n_valid_cv"] > 0:
                print(f"  {mod:5s} valid={a['n_valid_cv']:3d}/{a['n_traces']:3d} "
                      f"ΔAUC new={a['mean_delta_new']:+.4f} old={a['mean_delta_old']:+.4f} "
                      f"|Δdiff|max={a['max_abs_delta_diff']:.4f} flips={a['leak_flag_flips']}")
            else:
                print(f"  {mod:5s} ALL {a['n_traces']} traces data-limited "
                      f"({a.get('limited_warnings')}) — gap fix not observable here")

    out = {
        "wcc_csv": str(WCC_CSV),
        "leakage_threshold": LEAKAGE_THRESHOLD,
        "note": "gap=0 forces auto physical-time gap; old vs new differ only by "
                "the horizon_windows->feature-row conversion. Lerique rest1 traces "
                "(~150 samples) are too short for meaningful temporal CV "
                "(data-limited); trials_concat traces (601/1051) are long enough.",
        "results": summary,
    }
    out_path = OUT_DIR / "lerique_gapcheck.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

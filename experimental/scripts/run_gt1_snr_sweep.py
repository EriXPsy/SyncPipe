"""
Ground-Truth experiment GT-1 + GT-2: SNR x Coupling power curve.

Purpose
-------
GT-1 fixes a sim-level ground-truth coordinate system for the 6
confirmatory epoch features (DECISION-09 family) + 2 diagnostics:

    detectability(feature) = f(noise_ratio, coupling)

The output power curves serve as the reference frame against which
any future "this feature is now more sensitive" claim must be
checked under the Reversal Protocol (docs/DECISION_LOG.md).

GT-2 is performed by post-hoc analysis of the ``coupling = 0.0``
slice of the same grid: at the null, the empirical rejection rate
after BH-FDR within (noise_ratio, seed) groups should be <= q for
every (noise_ratio, feature) cell.  This validates the family-wise
error implementation without spending an extra ~210k surrogate runs.

Design (locked 2026-05-24, see DECISION_LOG.md)
-----------------------------------------------
- SNR axis:   noise_ratio in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}   (7 levels)
- Coupling:   c in {0.0, 0.3, 0.6}                                  (3 levels)
- Seeds:      30 per cell (range(1000, 1030)), matches existing sweeps
- Surrogates: 999 PRTF (smoke: 99)
- Window:     30 s WCC; onset_threshold = 0.5; duration = 300 s; hz = 1 Hz
- FDR:        BH q=0.05 within (noise_ratio, seed), family = 6 confirmatory
              (DECISION-09).  Diagnostics report raw detectability only.

Outputs
-------
- artifacts/gt1/gt1_grid.csv            raw per-cell grid with surrogate p-values
- artifacts/gt1/gt1_grid_fdr.csv        grid with BH-FDR reject_* booleans
- artifacts/gt1/gt1_power_curve.csv     long-format summary
                                        (noise_ratio, coupling, feature,
                                         family, reject_rate, n_seeds)
- artifacts/gt1/gt2_null_fwer.csv       coupling=0 slice with per-feature
                                        FPR vs q audit

Usage
-----
    python scripts/run_gt1_snr_sweep.py --smoke
    python scripts/run_gt1_snr_sweep.py --full
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multisync.feature_definitions import (
    CONFIRMATORY_FEATURES,
    DIAGNOSTIC_FEATURES,
    FEATURE_FAMILY,
)
from multisync.validation.pgt1_intensity import (
    Level3Config,
    run_level3_grid,
    apply_bh_fdr_within_noise,
    summarise_level3,
)


# --- DECISION-locked sweep axes -------------------------------------------
SNR_NOISE_RATIOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
COUPLINGS = (0.0, 0.3, 0.6)
SEEDS = tuple(range(1000, 1030))     # 30 seeds, matches existing sweeps
FDR_Q = 0.05

# Smoke vs full surrogate count.
SMOKE_N_SURROGATES = 99
FULL_N_SURROGATES = 999


def _build_config(n_surrogates: int) -> Level3Config:
    return Level3Config(
        duration_sec=300.0,
        hz=1.0,
        wcc_window_sec=30.0,
        onset_threshold=0.5,
        noise_ratios=SNR_NOISE_RATIOS,
        couplings=COUPLINGS,
        seeds=SEEDS,
        n_surrogates=n_surrogates,
        fdr_q=FDR_Q,
        surrogate_method="prtf",
    )


def _to_long_format(summary: pd.DataFrame) -> pd.DataFrame:
    """Pivot the wide rejection-rate summary to long format with family column.

    summarise_level3 returns one row per (noise_ratio, coupling) with
    columns named ``reject_<feature>_rate`` (one per confirmatory
    feature in the FDR family).  Diagnostics do not get a rejection-rate
    column because BH-FDR is intentionally applied only to the
    confirmatory family (DECISION-09); we attach their raw detectability
    separately by aggregating ``p_<feature> <= alpha`` from the raw grid.
    """
    rate_cols = [
        c for c in summary.columns
        if c.startswith("reject_") and c.endswith("_rate")
    ]
    id_cols = [c for c in summary.columns if c not in rate_cols]
    if not rate_cols:
        raise RuntimeError(
            "summarise_level3 produced no reject_*_rate columns; "
            "expected one per confirmatory feature."
        )
    long_df = summary.melt(
        id_vars=id_cols,
        value_vars=rate_cols,
        var_name="reject_col",
        value_name="reject_rate",
    )
    # Strip both prefix and suffix to recover the feature name.
    long_df["feature"] = (
        long_df["reject_col"]
        .str.replace("reject_", "", regex=False)
        .str.replace("_rate", "", regex=False)
    )
    long_df["family"] = long_df["feature"].map(FEATURE_FAMILY)
    long_df = long_df.drop(columns=["reject_col"])
    return long_df


def _diagnostics_detectability(
    grid_fdr: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """Raw (uncorrected) detectability for the 2 diagnostic features.

    DECISION-09: diagnostics do NOT enter BH-FDR, so we report raw
    rejection rate at alpha = 0.05 for transparency.  This keeps the
    power curve schema symmetric across all 8 features without
    pretending diagnostics share the family-corrected p-value.
    """
    rows = []
    for diag in DIAGNOSTIC_FEATURES:
        pcol = f"p_{diag}"
        if pcol not in grid_fdr.columns:
            continue
        grouped = (
            grid_fdr.assign(_rej=(grid_fdr[pcol] <= alpha).astype(float))
            .groupby(["noise_ratio", "coupling"], as_index=False)["_rej"]
            .mean()
            .rename(columns={"_rej": "reject_rate"})
        )
        grouped["feature"] = diag
        grouped["family"] = "diagnostic"
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_long_summary(
    summary: pd.DataFrame, grid_fdr: pd.DataFrame
) -> pd.DataFrame:
    """Combine confirmatory (FDR-corrected) + diagnostic (raw) into one long table."""
    confirm = _to_long_format(summary)
    diag = _diagnostics_detectability(grid_fdr, alpha=FDR_Q)
    common = ["noise_ratio", "coupling", "feature", "family", "reject_rate"]
    long_df = pd.concat(
        [confirm[common], diag[common] if not diag.empty else diag],
        ignore_index=True,
    )
    long_df["family"] = pd.Categorical(
        long_df["family"],
        categories=["confirmatory", "diagnostic"],
        ordered=True,
    )
    return long_df.sort_values(
        ["coupling", "noise_ratio", "family", "feature"]
    ).reset_index(drop=True)


def _gt2_null_audit(grid_fdr: pd.DataFrame) -> pd.DataFrame:
    """Per (noise_ratio, feature) empirical FPR at coupling = 0 vs FDR_Q.

    Confirmatory features use the BH-FDR-corrected ``reject_<feature>``
    column (DECISION-09 family-wise audit).

    Diagnostics use raw ``p_<feature> <= FDR_Q`` because they do not
    enter BH-FDR by design.  Their FPR under the null is an uncorrected
    per-feature Type-I rate and is reported as a sanity check, NOT as a
    family-wise error claim.
    """
    null_slice = grid_fdr[grid_fdr["coupling"] == 0.0]
    rows = []
    for noise in sorted(null_slice["noise_ratio"].unique()):
        sub = null_slice[null_slice["noise_ratio"] == noise]
        for feat in list(CONFIRMATORY_FEATURES) + list(DIAGNOSTIC_FEATURES):
            family = FEATURE_FAMILY[feat]
            if family == "confirmatory":
                col = f"reject_{feat}"
                if col not in sub.columns:
                    continue
                fpr = float(sub[col].mean())
                source = "bh_fdr_corrected"
            else:
                pcol = f"p_{feat}"
                if pcol not in sub.columns:
                    continue
                fpr = float((sub[pcol] <= FDR_Q).mean())
                source = "raw_uncorrected"
            n = int(sub.shape[0])
            # Wilson 95% CI for binomial proportion
            if n > 0:
                p = fpr
                z = 1.96
                denom = 1 + z * z / n
                centre = (p + z * z / (2 * n)) / denom
                half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
                lo, hi = max(0.0, centre - half), min(1.0, centre + half)
            else:
                lo = hi = float("nan")
            rows.append(
                {
                    "noise_ratio": float(noise),
                    "feature": feat,
                    "family": family,
                    "fpr_source": source,
                    "fpr_empirical": fpr,
                    "fpr_ci_lo": lo,
                    "fpr_ci_hi": hi,
                    "n_seeds": n,
                    "fdr_q": FDR_Q,
                    # Use the 95% Wilson CI LOWER bound vs q as the
                    # acceptance criterion: a cell is flagged only if the
                    # CI strictly excludes q, i.e. we have positive
                    # evidence that the true FPR exceeds q.  With n=30
                    # seeds, a point estimate of 0.10 has CI [0.034, 0.256]
                    # and does NOT exclude q=0.05, so it should not be
                    # flagged.  Without this guard the audit produces
                    # frequent false alarms on small n.
                    "fpr_within_target": bool(lo <= FDR_Q)
                    if n > 0 else False,
                }
            )
    return pd.DataFrame(rows)


def run_gt1(out_dir: Path, n_surrogates: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(n_surrogates=n_surrogates)
    n_cells = len(cfg.noise_ratios) * len(cfg.couplings) * len(cfg.seeds)
    print(
        f"[gt1] grid size: {len(cfg.noise_ratios)} noise x "
        f"{len(cfg.couplings)} coupling x {len(cfg.seeds)} seeds = "
        f"{n_cells} cells x {n_surrogates} surrogates "
        f"= {n_cells * n_surrogates:,} surrogate runs",
        flush=True,
    )
    t0 = time.time()
    grid = run_level3_grid(cfg)
    grid_path = out_dir / "gt1_grid.csv"
    grid.to_csv(grid_path, index=False)
    print(f"[gt1] raw grid -> {grid_path}  ({time.time() - t0:.1f}s)", flush=True)

    grid_fdr = apply_bh_fdr_within_noise(grid, q=FDR_Q)
    fdr_path = out_dir / "gt1_grid_fdr.csv"
    grid_fdr.to_csv(fdr_path, index=False)
    print(f"[gt1] BH-FDR grid -> {fdr_path}", flush=True)

    summary = summarise_level3(grid_fdr)
    long_summary = _build_long_summary(summary, grid_fdr)
    curve_path = out_dir / "gt1_power_curve.csv"
    long_summary.to_csv(curve_path, index=False)
    print(f"[gt1] long-format power curve -> {curve_path}", flush=True)

    # --- GT-2 piggyback: null FWER audit at coupling = 0 ----------------
    null_audit = _gt2_null_audit(grid_fdr)
    audit_path = out_dir / "gt2_null_fwer.csv"
    null_audit.to_csv(audit_path, index=False)
    print(f"[gt2] null FWER audit -> {audit_path}", flush=True)

    # --- Console preview ------------------------------------------------
    print()
    print("=== GT-1 power curve (confirmatory family) ===")
    conf = long_summary[long_summary["family"] == "confirmatory"]
    pivot = conf.pivot_table(
        index=["coupling", "noise_ratio"],
        columns="feature",
        values="reject_rate",
    )
    print(pivot.to_string())
    print()
    print("=== GT-2 null FWER audit (coupling = 0) ===")
    out_of_target = null_audit[~null_audit["fpr_within_target"]]
    if out_of_target.empty:
        print(f"All (noise_ratio, feature) cells have empirical FPR <= q = {FDR_Q}.")
    else:
        print(f"WARNING: {len(out_of_target)} cell(s) with empirical FPR > q:")
        print(
            out_of_target[
                ["noise_ratio", "feature", "family", "fpr_empirical",
                 "fpr_ci_lo", "fpr_ci_hi", "n_seeds"]
            ].to_string(index=False)
        )

    elapsed = time.time() - t0
    print(f"\n[gt1] total elapsed: {elapsed / 60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description="SyncPipe Ground-Truth GT-1 + GT-2")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run with n_surrogates={SMOKE_N_SURROGATES} for path validation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Run with n_surrogates={FULL_N_SURROGATES} (production).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/gt1"),
        help="Output directory.",
    )
    args = parser.parse_args()

    if args.smoke and args.full:
        parser.error("--smoke and --full are mutually exclusive")
    if not args.smoke and not args.full:
        parser.error("must specify --smoke or --full")

    n_surr = SMOKE_N_SURROGATES if args.smoke else FULL_N_SURROGATES
    run_gt1(args.out, n_surrogates=n_surr)


if __name__ == "__main__":
    main()

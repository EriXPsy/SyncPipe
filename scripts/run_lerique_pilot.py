"""Lerique (Lerique et al., 2024) pilot runner — SyncPipe P2 case study.

Pre-registration: docs/PRE_REGISTRATION_PILOTS.md
Decision records: docs/DECISION_LOG.md (Lerique-meta-correction,
                  Lerique-rest1-length-heterogeneity,
                  Lerique-preproc-smoke-pass,
                  Lerique-batch-pipeline-ready)

Pipeline
--------
1. Load every (dyad, modality, condition_unit) record from Lerique-47n3p
   with preprocessing enabled (pre-reg §1.4 protocol).
2. Per record, build a SyncPipe Dyad and run DynamicAnalyzer to obtain
   the v3 confirmatory + diagnostic features.
3. Aggregate into DyadResult objects and pivot to a per-record feature
   table (one row per (dyad, modality, condition_unit)).
4. Run paired group-level contrasts (within-dyad, across condition_units):
        Main:        rest1            vs trials_concat  (confirmatory)
        Sensitivity: rest_postblock   vs trials_concat  (robustness)
        Reference:   rest1            vs rest_postblock (sanity)
   using Wilcoxon signed-rank (DECISION-09 / pre-reg §1.4 locked).
5. BH-FDR within (modality × confirmatory family of 6), **limited to
   the MAIN contrast**. Sensitivity, reference, and diagnostic features
   carry only raw p-values — folding them into FDR would inflate the
   family size and weaken main inference (pre-reg §1.5 FDR family scope).
6. Write all CSVs to ``artifacts/realtest/lerique_2024/``.

Why paired (vs Gordon's unpaired)
---------------------------------
Each Lerique dyad contributes ALL three condition_units, so within-dyad
variance is removed by pairing. This is the higher-power design
relative to Gordon's between-condition contrasts.

Usage (PowerShell)
------------------
    python scripts/run_lerique_pilot.py `
        --data-root "<PATH_TO>/Lerique-47n3p" `
        --out-dir   "artifacts/realtest/lerique_2024"

Smoke-only run (first N records):

    python scripts/run_lerique_pilot.py --data-root ... --limit 9
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Make multisync importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("lerique_pilot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Lerique (2024) pilot with SyncPipe v3 "
                    "paired group-level analysis."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the Lerique-47n3p dataset root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/realtest/lerique_2024"),
        help="Output directory for CSVs.",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=1.0,
        help="Resample rate for IBI / EDA / RESP (Hz). Pre-reg §1.4 locks 1 Hz.",
    )
    parser.add_argument(
        "--wcc-window-sec",
        type=float,
        default=30.0,
        help="WCC sliding window in seconds (SSoT default 30s).",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Onset threshold r (DECISION-01 locked 0.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If given, only process first N records (smoke).",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="If output CSVs already exist, rename them with a "
             "'.partial_<timestamp>' suffix before writing the new run. "
             "Prevents accidental overwrite of interim results "
             "(see DECISION_LOG § Lerique-interim-N10-observation).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1-3: load + per-record analysis + per-record feature table
# ---------------------------------------------------------------------------

def _analyze_all_records(
    records,
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """Run DynamicAnalyzer on each record. Return:

    Returns
    -------
    per_record_df : pd.DataFrame
        One row per successfully analysed (dyad, modality, condition_unit).
    excluded : list[dict]
        Records that were excluded *by design* (e.g. inventory-missing
        P2, sub-floor duration). Not an error — pre-registered exclusion.
    failures : list[dict]
        Records that raised an unexpected exception during analysis.
    """
    from multisync.core import DynamicAnalyzer
    from multisync.batch import DyadResult
    from multisync.realtest.lerique_2024 import lerique_record_to_multisync_dyad
    from multisync.feature_definitions import (
        FDR_FEATURES,
        REFERENCE_FEATURE,
    )

    window_size = max(2, int(round(wcc_window_sec * target_hz)))
    analyzer = DynamicAnalyzer(
        window_size=window_size,
        onset_threshold=onset_threshold,
    )

    rows: List[Dict] = []
    excluded: List[Dict] = []
    failures: List[Dict] = []

    # SSoT family bare names → DyadResult v2-style scalar attribute names
    # The bare names are the FDR-family identifiers; the attribute names
    # are how DyadResult exposes the cross-modality-pair mean.
    SSOT_TO_DYADRESULT = {
        "onset_latency":     "mean_onset_latency",
        "rise_time":         "mean_rise_time",
        "peak_amplitude":    "mean_peak_amplitude",
        "recovery_time":     "mean_recovery_time",
        "dwell_time":        "mean_dwell_time",
        "switching_rate":    "mean_switching_rate",
        "mean_synchrony":    "mean_synchrony",
        "synchrony_entropy": "mean_synchrony_entropy",
    }

    feature_names = list(FDR_FEATURES) + list(REFERENCE_FEATURE)
    n_total = len(records)

    for i, rec in enumerate(records, 1):
        if rec.incomplete:
            failures.append({
                "dyad_label": rec.dyad_label,
                "modality": rec.modality,
                "condition_unit": rec.condition,
                "reason": "incomplete (inventory or duration floor)",
                "meta": str(rec.meta),
            })
            continue
        try:
            dyad = lerique_record_to_multisync_dyad(rec)
            dyad.align(target_hz=target_hz, require_co_start=False)
            dyad.zscore()
            results = analyzer.fit_transform(dyad)
            dr = DyadResult.from_analysis_results(results)

            row = {
                "dyad_label": rec.dyad_label,
                "dyad_id": dr.dyad_id,
                "modality": rec.modality,
                "condition_unit": rec.condition,
                "n_samples": rec.n_samples,
                "duration_sec": rec.duration_sec,
            }
            # Use SSoT bare names as the column key (FDR family identifier);
            # pull the corresponding DyadResult attribute for the value.
            for name in feature_names:
                attr = SSOT_TO_DYADRESULT.get(name)
                row[name] = getattr(dr, attr, np.nan) if attr else np.nan
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%d/%d] FAILED dyad=%s modality=%s unit=%s: %s",
                i, n_total, rec.dyad_label, rec.modality, rec.condition, exc,
            )
            failures.append({
                "dyad_label": rec.dyad_label,
                "modality": rec.modality,
                "condition_unit": rec.condition,
                "reason": f"{type(exc).__name__}: {exc}",
                "meta": str(rec.meta),
            })
        if i % 25 == 0 or i == n_total:
            logger.info("[per-record] %d/%d processed (rows=%d failures=%d)",
                        i, n_total, len(rows), len(failures))

    df = pd.DataFrame(rows)
    return df, failures


# ---------------------------------------------------------------------------
# Step 4: paired Wilcoxon signed-rank contrast helper
# ---------------------------------------------------------------------------

# Three pre-registered contrasts (pre-reg §1.3).
# Each entry: (label, condition_a, condition_b, role)
CONTRASTS: Sequence[Tuple[str, str, str, str]] = (
    ("main",        "rest1",          "trials_concat", "main"),
    ("sensitivity", "rest_postblock", "trials_concat", "sensitivity"),
    ("reference",   "rest1",          "rest_postblock", "reference"),
)


def _paired_wilcoxon(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, int, int, int]:
    """Wilcoxon signed-rank test on paired vectors after dropping NaN pairs.

    Returns
    -------
    statistic, p_value, n_pairs, n_undefined_a, n_undefined_b

    NaN pairs (either side) are excluded **pairwise** — they cannot
    contribute to a within-dyad delta. ``n_undefined_a/b`` count NaNs
    that were present **before** pairwise drop, so the caller can
    distinguish "feature is missing" from "feature is
    phenomenologically undefined" (e.g. onset_latency on a Rest1
    segment with no transitions — pre-reg §1.5).

    p_value is NaN if n_pairs < 3.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return (np.nan, np.nan, 0, 0, 0)

    n_undef_a = int(np.isnan(a).sum())
    n_undef_b = int(np.isnan(b).sum())

    valid = np.isfinite(a) & np.isfinite(b)
    n_pairs = int(valid.sum())
    if n_pairs < 3:
        return (np.nan, np.nan, n_pairs, n_undef_a, n_undef_b)
    # Drop tied-zero diffs (default Wilcoxon behaviour 'wilcox')
    diff = a[valid] - b[valid]
    if np.allclose(diff, 0):
        return (0.0, 1.0, n_pairs, n_undef_a, n_undef_b)
    try:
        # Use scipy's default method handling — older versions used
        # `mode=`, newer use `method=`. Default behaviour (auto/asymp
        # crossover at n ≥ 25) is what we want; don't pin the kwarg.
        result = wilcoxon(a[valid], b[valid], alternative="two-sided",
                          zero_method="wilcox")
        stat = float(result.statistic)
        p = float(result.pvalue)
    except (ValueError, TypeError) as exc:
        logger.warning("Wilcoxon failed (n=%d): %s", n_pairs, exc)
        return (np.nan, np.nan, n_pairs, n_undef_a, n_undef_b)
    return (stat, p, n_pairs, n_undef_a, n_undef_b)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaNs are preserved.

    Returns adjusted p-values of the same shape; non-NaN entries are
    corrected within themselves (one family per call).
    """
    p = np.asarray(pvals, dtype=np.float64)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return p
    sub = p[mask]
    n = len(sub)
    order = np.argsort(sub)
    ranked = sub[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    adj = np.empty(n, dtype=np.float64)
    adj[order] = np.clip(bh, 0.0, 1.0)
    out = p.copy()
    out[mask] = adj
    return out


def _run_paired_contrasts(per_record: pd.DataFrame) -> pd.DataFrame:
    """For each (modality, contrast, feature) compute paired Wilcoxon.

    BH-FDR is applied within (modality × confirmatory family of 6) per
    contrast — diagnostic features get raw p-values only.
    """
    from multisync.feature_definitions import (
        FDR_FEATURES,
        REFERENCE_FEATURE,
    )

    rows: List[Dict] = []

    for modality, mod_df in per_record.groupby("modality"):
        for contrast_label, cond_a, cond_b, role in CONTRASTS:
            wide = mod_df[mod_df["condition_unit"].isin([cond_a, cond_b])]
            if wide.empty:
                continue
            for feature in list(FDR_FEATURES) + list(REFERENCE_FEATURE):
                pivot = wide.pivot_table(
                    index="dyad_label",
                    columns="condition_unit",
                    values=feature,
                    aggfunc="first",
                )
                if cond_a not in pivot.columns or cond_b not in pivot.columns:
                    continue
                a = pivot[cond_a].to_numpy()
                b = pivot[cond_b].to_numpy()
                stat, p, n_pairs, n_undef_a, n_undef_b = _paired_wilcoxon(a, b)
                n_valid_a = int(np.isfinite(a).sum())
                n_valid_b = int(np.isfinite(b).sum())
                n_dyads_in_pivot = int(len(pivot))
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() >= 1:
                    mean_a = float(np.nanmean(a))
                    mean_b = float(np.nanmean(b))
                    median_a = float(np.nanmedian(a))
                    median_b = float(np.nanmedian(b))
                    median_diff = float(np.nanmedian(a[valid] - b[valid]))
                else:
                    mean_a = mean_b = median_a = median_b = median_diff = np.nan
                family = (
                    "confirmatory" if feature in CONFIRMATORY_FEATURES
                    else "diagnostic"
                )
                rows.append({
                    "modality": modality,
                    "contrast": contrast_label,
                    "contrast_role": role,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "feature": feature,
                    "family": family,
                    "n_dyads_in_pivot": n_dyads_in_pivot,
                    "n_valid_a": n_valid_a,
                    "n_valid_b": n_valid_b,
                    "n_undef_a": n_undef_a,
                    "n_undef_b": n_undef_b,
                    "n_pairs": n_pairs,
                    "median_a": median_a,
                    "median_b": median_b,
                    "median_a_minus_b": median_diff,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR within (modality × confirmatory family) — pre-reg §1.5 locks
    # the confirmatory family to the MAIN contrast (rest1 → trials_concat).
    # Sensitivity and reference contrasts are robustness checks and report
    # raw p only; folding them into FDR would inflate the family size and
    # weaken the main inference (DECISION-09).
    df["p_fdr"] = np.nan
    main_conf_mask = (
        (df["family"] == "confirmatory")
        & (df["contrast_role"] == "main")
    )
    for modality, sub in df[main_conf_mask].groupby("modality"):
        idx = sub.index
        df.loc[idx, "p_fdr"] = _bh_fdr(sub["p_raw"].to_numpy())
    df["significant_fdr"] = (df["p_fdr"] < 0.05).where(df["p_fdr"].notna(), other=pd.NA)
    return df


# ---------------------------------------------------------------------------
# Step 5: orchestration + IO
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if not args.data_root.exists():
        logger.error("data_root not found: %s", args.data_root)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir.resolve())

    # Archive existing CSVs if requested (default off — caller must opt-in
    # to avoid silent file moves)
    if args.archive_existing:
        import time
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive_targets = [
            "per_record_features.csv",
            "per_record_failures.csv",
            "group_contrasts_paired.csv",
        ]
        for fname in archive_targets:
            src = out_dir / fname
            if src.exists():
                dst = out_dir / f"{src.stem}.partial_{stamp}{src.suffix}"
                src.rename(dst)
                logger.info("Archived existing %s -> %s", fname, dst.name)

    # ---- Step 1: load all records ----
    from multisync.realtest.lerique_2024 import (
        load_lerique_dataset, MIN_DURATION_SEC,
    )
    logger.info("Loading Lerique dataset (preprocess=True, MIN_DURATION_SEC=%.1fs)…",
                MIN_DURATION_SEC)
    records = load_lerique_dataset(
        data_root=args.data_root,
        preprocess=True,
        drop_incomplete=False,
        drop_misaligned=True,
        drop_short_duration=True,
    )
    logger.info("Loaded %d records.", len(records))
    if args.limit is not None:
        records = records[: args.limit]
        logger.info("--limit %d applied; analyzing %d records.",
                    args.limit, len(records))

    # ---- Step 2-3: per-record analysis ----
    per_record, failures = _analyze_all_records(
        records=records,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )
    per_record_csv = out_dir / "per_record_features.csv"
    per_record.to_csv(per_record_csv, index=False)
    logger.info("Per-record features written: %s (%d rows)",
                per_record_csv, len(per_record))

    if failures:
        failures_csv = out_dir / "per_record_failures.csv"
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
        logger.warning("Failures written: %s (%d rows)",
                       failures_csv, len(failures))

    if per_record.empty:
        logger.error("Per-record table empty — aborting group test.")
        return 2

    # ---- Step 4: paired contrasts + BH-FDR ----
    contrast_df = _run_paired_contrasts(per_record)
    contrast_csv = out_dir / "group_contrasts_paired.csv"
    contrast_df.to_csv(contrast_csv, index=False)
    logger.info("Group contrasts written: %s (%d rows)",
                contrast_csv, len(contrast_df))

    # ---- Step 5: summary ----
    if not contrast_df.empty:
        conf = contrast_df[contrast_df["family"] == "confirmatory"]
        sig = conf[conf["significant_fdr"] == True]  # noqa: E712
        logger.info(
            "Confirmatory family: %d tests total, %d significant (p_FDR<0.05)",
            len(conf), len(sig),
        )
        for _, r in sig.iterrows():
            logger.info(
                "  [%s] %s :: %s × %s | n=%d median_diff=%+.4g p_raw=%.4g p_fdr=%.4g",
                r["contrast"], r["modality"], r["feature"],
                f"{r['condition_a']}->{r['condition_b']}",
                int(r["n_pairs"]) if pd.notna(r["n_pairs"]) else -1,
                r["median_a_minus_b"], r["p_raw"], r["p_fdr"],
            )

    logger.info("Lerique pilot complete. Outputs in: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gordon (Mayo & Gordon, 2025) case study runner.

Section 6.x of the SyncPipe paper.

Pipeline
--------
1. Load every (dyad, condition) record from the Gordon dataset.
2. For each record, build a SyncPipe ``Dyad`` and run
   ``DynamicAnalyzer.fit_transform`` to obtain the six v3 features.
3. Aggregate into ``DyadResult`` objects via
   ``DyadResult.from_analysis_results``.
4. Run condition-level group comparisons (exp1 vs exp2, ..., exp3 vs exp4)
   using the v3 fields with Mann-Whitney U + BH-FDR.
5. Optionally re-do the dyad-level analysis with the Level 3 PRTF
   surrogate test for the strongest condition contrast (``--level3``).
6. Write all CSVs to ``artifacts/realtest/gordon_2025/``.

Usage
-----
PowerShell (default paths):

    python scripts/run_gordon_case_study.py `
        --data-root "<PATH_TO>/gordon_2025" `
        --out-dir   "artifacts/realtest/gordon_2025"

With Level 3 PRTF surrogate (heavier, ~5-10 min for full dataset):

    python scripts/run_gordon_case_study.py --data-root ... --level3

Why this design
---------------
* Every step is independent and writes its own CSV.  If step 5 (the
  expensive surrogate sweep) crashes, you still keep step 1-4 outputs.
* Conditions are the natural comparison unit because Mayo & Gordon
  (2025) report contextual-pull effects within dyad across the four
  experimental cells.
* The script uses ONLY the public SyncPipe API (no internal imports
  except the converter).  This is the same surface a third party would
  use.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("gordon_case_study")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gordon (Mayo & Gordon, 2025) case study with "
                    "SyncPipe v3 group-level analysis."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the Gordon dataset root (containing 'behavior data/').",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/realtest/gordon_2025"),
        help="Output directory for CSVs (default: artifacts/realtest/gordon_2025).",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=10.0,
        help="Resampling rate for the motion-intensity channel (Hz).",
    )
    parser.add_argument(
        "--wcc-window-sec",
        type=float,
        default=30.0,
        help="WCC sliding window (seconds, default 30 = Section 4 default).",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Onset threshold r (default 0.5 = Cohen medium effect).",
    )
    parser.add_argument(
        "--level3",
        action="store_true",
        help="Also run the PRTF surrogate test (slower).",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=999,
        help="Number of PRTF surrogates per dyad (only with --level3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If given, only process the first N records (smoke test).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1-3: load + per-dyad analysis + extract DyadResults
# ---------------------------------------------------------------------------

def _analyze_all_records(
    records,
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
):
    """Run DynamicAnalyzer on every record and return DyadResults.

    Returns
    -------
    list of DyadResult, list of failure dicts
    """
    from multisync.core import DynamicAnalyzer
    from multisync.batch import DyadResult
    from multisync.realtest.gordon_2025 import gordon_record_to_multisync_dyad

    # SyncPipe's DynamicAnalyzer takes ``window_size`` in SAMPLES, not
    # seconds.  Convert here so the user can still think in seconds.
    window_size = max(2, int(round(wcc_window_sec * target_hz)))

    analyzer = DynamicAnalyzer(
        window_size=window_size,
        onset_threshold=onset_threshold,
    )

    dyad_results: List = []
    failures: List[Dict] = []

    for i, rec in enumerate(records, 1):
        try:
            dyad = gordon_record_to_multisync_dyad(rec)
            # Align + normalize before fit_transform.  The Gordon record
            # is already on a uniform grid at target_hz, but Dyad.align
            # also runs the timestamp-type sanity check.
            dyad.align(target_hz=target_hz, require_co_start=False)
            dyad.zscore()
            results = analyzer.fit_transform(dyad)
            dr = DyadResult.from_analysis_results(results)
            dyad_results.append(dr)
            if i % 10 == 0 or i == len(records):
                logger.info("[per-dyad] %d/%d done", i, len(records))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[per-dyad] %d/%d FAILED for %s: %s",
                i, len(records), rec.dyad_id, exc,
            )
            failures.append({
                "dyad_id": rec.dyad_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    return dyad_results, failures


# ---------------------------------------------------------------------------
# Step 4: condition-level group comparisons
# ---------------------------------------------------------------------------

def _split_by_condition(dyad_results) -> Dict[str, list]:
    """Group DyadResults by condition (exp1..exp4) using dyad_id suffix."""
    buckets: Dict[str, list] = {}
    for dr in dyad_results:
        # dyad_id format: "<p1>_<p2>__expN"
        if "__" not in dr.dyad_id:
            continue
        cond = dr.dyad_id.split("__")[-1]
        buckets.setdefault(cond, []).append(dr)
    return buckets


def _run_all_condition_comparisons(buckets: Dict[str, list]) -> pd.DataFrame:
    """Run pairwise Mann-Whitney U on every condition pair.

    Returns a long-format DataFrame with columns:
        condition_a, condition_b, metric, mean_a, mean_b,
        median_a, median_b, p_raw, p_fdr, effect_size, n_a, n_b,
        significant_fdr.
    """
    from multisync.batch import group_comparison

    rows: List[Dict] = []
    conditions = sorted(buckets.keys())
    for ca, cb in combinations(conditions, 2):
        report = group_comparison(
            group_a=buckets[ca],
            group_b=buckets[cb],
            label_a=ca,
            label_b=cb,
        )
        for r in report.test_results:
            rows.append({
                "condition_a": ca,
                "condition_b": cb,
                "metric": r.metric,
                "mean_a": r.mean_a,
                "mean_b": r.mean_b,
                "median_a": r.median_a,
                "median_b": r.median_b,
                "p_raw": r.p_raw,
                "p_fdr": r.p_fdr,
                "effect_size": r.effect_size,
                "effect_size_type": r.effect_size_type,
                "n_a": r.n_a,
                "n_b": r.n_b,
                "test_name": r.test_name,
                "significant_fdr": r.significant_fdr,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5 (optional): Level-3 PRTF surrogate per record
# ---------------------------------------------------------------------------

def _level3_per_record(
    records,
    wcc_window_sec: float,
    onset_threshold: float,
    n_surrogates: int,
) -> pd.DataFrame:
    """Run the PRTF surrogate test on every record's motion channel.

    Uses the same machinery as Level 3 (pgt1_intensity.py).  This is the
    consistency-with-Section-4 evidence: real-data inference uses the
    same null as synthetic-data inference.

    Statistical convention (matching Section 4.4):
    * ``peak_amplitude`` and ``mean_synchrony`` use a one-sided upper-tail
      test (H1: observed > null).
    * ``onset_latency``, ``rise_time``, ``recovery_time`` and
      ``synchrony_entropy`` use a two-sided test (no a priori direction).
    """
    from multisync.validation.pgt1_intensity import (
        prtf_surrogate,
        phipson_smyth_p,
    )
    from multisync.validation.recovery import _extract_six_features
    from multisync.dynamic_features import sliding_window_wcc

    # Tail convention copied from pgt1_intensity.FEATURE_TAILS so the realtest
    # pipeline reports the SAME directional choices as Section 4.4.
    upper_features = {"peak_amplitude", "mean_synchrony"}

    rng = np.random.default_rng(2026_05_23)
    rows: List[Dict] = []

    for i, rec in enumerate(records, 1):
        a = rec.person_a["motion_intensity"].values.astype(float)
        b = rec.person_b["motion_intensity"].values.astype(float)
        hz = float(rec.target_hz)
        window_size = max(2, int(round(wcc_window_sec * hz)))

        wcc_obs = sliding_window_wcc(a, b, window_size=window_size, hz=hz)
        obs = _extract_six_features(
            wcc_obs, hz=hz, onset_threshold=onset_threshold,
        )

        # Build null distribution over the 6 scalar features.
        feature_keys = [
            "peak_amplitude", "mean_synchrony", "onset_latency",
            "rise_time", "recovery_time", "synchrony_entropy",
        ]
        null: Dict[str, list] = {k: [] for k in feature_keys}
        for _ in range(n_surrogates):
            a_s = prtf_surrogate(a, rng)
            b_s = prtf_surrogate(b, rng)
            wcc_s = sliding_window_wcc(
                a_s, b_s, window_size=window_size, hz=hz,
            )
            f_s = _extract_six_features(
                wcc_s, hz=hz, onset_threshold=onset_threshold,
            )
            for k in feature_keys:
                null[k].append(f_s.get(k, np.nan))

        for k, null_vals in null.items():
            obs_val = obs.get(k, np.nan)
            arr = np.asarray(null_vals, dtype=float)
            arr_valid = arr[~np.isnan(arr)]

            if np.isnan(obs_val):
                rows.append({
                    "dyad_id": rec.dyad_id, "feature": k,
                    "obs": np.nan, "p_phipson_smyth": np.nan,
                    "null_mean": (float(np.mean(arr_valid))
                                  if arr_valid.size else np.nan),
                    "n_null": int(arr_valid.size),
                    "note": "obs_undefined",
                })
                continue
            if arr_valid.size < 50:
                rows.append({
                    "dyad_id": rec.dyad_id, "feature": k,
                    "obs": float(obs_val), "p_phipson_smyth": np.nan,
                    "null_mean": (float(np.mean(arr_valid))
                                  if arr_valid.size else np.nan),
                    "n_null": int(arr_valid.size),
                    "note": f"insufficient_null_n={arr_valid.size}",
                })
                continue

            tail = "upper" if k in upper_features else "two"
            p_val = phipson_smyth_p(
                observed=float(obs_val),
                null_values=arr_valid,
                tail=tail,
            )
            rows.append({
                "dyad_id": rec.dyad_id, "feature": k,
                "obs": float(obs_val),
                "null_mean": float(np.mean(arr_valid)),
                "null_sd": float(np.std(arr_valid, ddof=1)),
                "p_phipson_smyth": float(p_val),
                "tail": tail,
                "n_null": int(arr_valid.size),
                "note": "",
            })

        if i % 5 == 0 or i == len(records):
            logger.info("[level3] %d/%d done", i, len(records))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.out_dir)

    # --- Step 1: load -----------------------------------------------------
    from multisync.realtest.gordon_2025 import load_gordon_dataset

    logger.info("Loading Gordon dataset from %s ...", args.data_root)
    records = load_gordon_dataset(
        data_root=args.data_root,
        target_hz=args.target_hz,
    )
    if args.limit is not None:
        records = records[: args.limit]
        logger.info("Truncated to first %d records (--limit).", len(records))

    if not records:
        logger.error("No records loaded from %s.  Aborting.", args.data_root)
        return 2

    logger.info("Loaded %d records.", len(records))

    # --- Step 2-3: per-dyad analysis -------------------------------------
    dyad_results, failures = _analyze_all_records(
        records,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            args.out_dir / "per_dyad_failures.csv", index=False,
        )
    if not dyad_results:
        logger.error("All per-dyad analyses failed.  Aborting.")
        return 3

    # Persist per-dyad table.
    per_dyad_df = pd.DataFrame([dr.to_dict() for dr in dyad_results])
    per_dyad_df.to_csv(args.out_dir / "per_dyad_features.csv", index=False)
    logger.info("Wrote per_dyad_features.csv (%d rows).", len(per_dyad_df))

    # --- Step 4: condition-level group comparisons -----------------------
    buckets = _split_by_condition(dyad_results)
    if len(buckets) >= 2:
        cond_df = _run_all_condition_comparisons(buckets)
        cond_df.to_csv(args.out_dir / "condition_comparisons.csv", index=False)
        logger.info(
            "Wrote condition_comparisons.csv (%d rows across %d condition pairs).",
            len(cond_df),
            len(list(combinations(buckets.keys(), 2))),
        )

        # Pretty summary printout.
        v3_metrics = (
            "mean_peak_amplitude", "mean_rise_time", "mean_recovery_time",
            "mean_synchrony", "mean_synchrony_entropy",
            "onset_defined_rate", "recovery_defined_rate",
        )
        summary = cond_df[cond_df["metric"].isin(v3_metrics)][
            ["condition_a", "condition_b", "metric", "median_a", "median_b",
             "p_fdr", "effect_size", "significant_fdr"]
        ]
        logger.info("\n=== v3 metric summary (condition pairs) ===\n%s",
                    summary.to_string(index=False))
    else:
        logger.warning(
            "Only one condition found; skipping group comparisons.",
        )

    # --- Step 5 (optional): Level-3 PRTF surrogate -----------------------
    if args.level3:
        logger.info(
            "Running Level-3 PRTF surrogate (N=%d) on all %d records ...",
            args.n_surrogates, len(records),
        )
        l3_df = _level3_per_record(
            records,
            wcc_window_sec=args.wcc_window_sec,
            onset_threshold=args.onset_threshold,
            n_surrogates=args.n_surrogates,
        )
        l3_df.to_csv(args.out_dir / "per_dyad_level3_prtf.csv", index=False)
        logger.info("Wrote per_dyad_level3_prtf.csv (%d rows).", len(l3_df))

    logger.info("Gordon case study complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

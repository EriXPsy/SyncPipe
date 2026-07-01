"""Dose-response check: Rest1 < Rest2 < Rest3 < Rest4 (time-on-task drift)?

Logic
-----
- Re-extracts per-record features for *each individual rest segment*
  (rest1_only, rest2_only, rest3_only, rest4_only) instead of the
  pooled `rest_postblock`.
- Then for each (modality, feature), fits a per-dyad slope of feature
  ~ rest_index (1..4) using OLS within dyad, and tests whether the
  across-dyad mean slope differs from zero (one-sample Wilcoxon).
- Also reports per-rest medians + pairwise Wilcoxon to localise
  *where* the monotonic increase (if any) emerges.

Output
------
artifacts/realtest/lerique_2024/dose_response_rests.csv

Pre-reg posture
---------------
This is a *post-hoc sensitivity analysis* on top of the locked
pre-registration. The locked main/reference/sensitivity contrasts are
NOT re-run or re-judged.

Usage (PowerShell)
------------------
    python scripts/dose_response_rest_check.py `
        --data-root "<OSF_ROOT>/Lerique-47n3p" `
        --out-csv "artifacts/realtest/lerique_2024/dose_response_rests.csv"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("dose_response")


def _build_single_rest_records(
    data_root: Path,
    modalities: Sequence[str],
    *,
    target_fs: float,
) -> List:
    """Emit one record per (dyad, modality, single_rest_segment).

    Uses lower-level loader helpers directly, bypassing
    load_lerique_dataset's CONDITION_UNITS registry.
    """
    from multisync.realtest.lerique_2024 import (
        MODALITIES, RAW_FS_HZ, MIN_DURATION_SEC,
        LeriqueDyadCondition,
        _collect_segments_for_person,
        _verify_p1_p2_length_alignment,
        _verify_min_duration,
        _preprocess_ecg, _preprocess_eda, _preprocess_resp,
    )

    root = Path(data_root).expanduser().resolve()
    preprocess_fn = {
        "ECG": _preprocess_ecg,
        "EDA": _preprocess_eda,
        "RESP": _preprocess_resp,
    }

    records: List = []
    n_missing = 0
    for modality in modalities:
        if modality not in MODALITIES:
            continue
        mod_root = root / modality
        if not mod_root.exists():
            logger.warning("Missing modality dir: %s", mod_root)
            continue

        for pce_dir in sorted(d for d in mod_root.iterdir() if d.is_dir()):
            dyad_label = pce_dir.name[:5]
            for rest_idx in (1, 2, 3, 4):
                unit = f"rest{rest_idx}_only"
                a_raw, _, _ = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="1",
                    cond_class="Rest", seg_indices=[rest_idx],
                )
                b_raw, _, _ = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="2",
                    cond_class="Rest", seg_indices=[rest_idx],
                )
                if not _verify_p1_p2_length_alignment(
                    a_raw, b_raw, dyad_label, modality, unit,
                ):
                    n_missing += 1
                    continue
                if not _verify_min_duration(
                    a_raw, b_raw, RAW_FS_HZ, dyad_label, modality, unit,
                    min_duration_sec=MIN_DURATION_SEC,
                ):
                    n_missing += 1
                    continue
                if a_raw is None or b_raw is None:
                    n_missing += 1
                    continue

                # Preprocess: returns (signal, mask) — two numpy arrays
                try:
                    a_sig, a_mask = preprocess_fn[modality](
                        a_raw, RAW_FS_HZ, target_fs,
                        boundary_mask=np.ones(len(a_raw), dtype=bool),
                    )
                    b_sig, b_mask = preprocess_fn[modality](
                        b_raw, RAW_FS_HZ, target_fs,
                        boundary_mask=np.ones(len(b_raw), dtype=bool),
                    )
                except Exception as exc:
                    logger.warning("preprocess failed %s/%s/%s: %s",
                                   dyad_label, modality, unit, exc)
                    continue

                # Truncate to min length so P1/P2 share a common grid
                n_common = min(len(a_sig), len(b_sig))
                mask_out = a_mask[:n_common] & b_mask[:n_common]
                a_sig = a_sig[:n_common]
                b_sig = b_sig[:n_common]

                t = np.arange(n_common, dtype=np.float64) / target_fs
                rec = LeriqueDyadCondition(
                    dyad_id=f"{dyad_label}__{modality}__{unit}",
                    dyad_label=dyad_label,
                    modality=modality,
                    condition=unit,
                    person_a=pd.DataFrame({"time": t, "value": a_sig}),
                    person_b=pd.DataFrame({"time": t, "value": b_sig}),
                    target_hz=float(target_fs),
                    n_samples=int(n_common),
                    duration_sec=float(n_common / target_fs),
                    incomplete=False,
                    discontinuity_mask=mask_out,
                    meta={"rest_index": rest_idx},
                )
                records.append(rec)

    logger.info("Built %d single-rest records (%d skipped)",
                len(records), n_missing)
    return records


def _analyze_records(records, *, target_hz, wcc_window_sec, onset_threshold):
    """Reuse the same DynamicAnalyzer pipeline as run_lerique_pilot.py."""
    from multisync.core import DynamicAnalyzer
    from multisync.batch import DyadResult
    from multisync.realtest.lerique_2024 import lerique_record_to_multisync_dyad
    from multisync.feature_definitions import (
        CONFIRMATORY_FEATURES, DIAGNOSTIC_FEATURES,
    )

    window_size = max(2, int(round(wcc_window_sec * target_hz)))
    analyzer = DynamicAnalyzer(
        window_size=window_size,
        onset_threshold=onset_threshold,
        enable_prediction=False,  # dose-response check only uses
                                  # descriptive features; skip prediction CV.
    )
    SSOT_TO_DR = {
        "onset_latency":  "mean_onset_latency",
        "rise_time":      "mean_rise_time",
        "peak_amplitude": "mean_peak_amplitude",
        "recovery_time":  "mean_recovery_time",
        "dwell_time":     "mean_dwell_time",
        "switching_rate": "mean_switching_rate",
        "mean_synchrony": "mean_synchrony",
        "synchrony_entropy": "mean_synchrony_entropy",
    }
    feats = list(CONFIRMATORY_FEATURES) + list(DIAGNOSTIC_FEATURES)

    rows = []
    for i, rec in enumerate(records, 1):
        try:
            dyad = lerique_record_to_multisync_dyad(rec)
            dyad.align(target_hz=target_hz, require_co_start=False)
            dyad.zscore()
            res = analyzer.fit_transform(dyad)
            dr = DyadResult.from_analysis_results(res)
            row = {
                "dyad_label": rec.dyad_label,
                "modality": rec.modality,
                "condition_unit": rec.condition,
                "rest_index": rec.meta["rest_index"],
            }
            for n in feats:
                attr = SSOT_TO_DR.get(n)
                row[n] = getattr(dr, attr, np.nan) if attr else np.nan
            rows.append(row)
        except Exception as exc:
            logger.debug("FAIL %s/%s/%s: %s",
                         rec.dyad_label, rec.modality, rec.condition, exc)
        if i % 50 == 0:
            logger.info("processed %d/%d", i, len(records))
    return pd.DataFrame(rows)


def _per_dyad_slope_and_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (modality, feature): per-dyad slope + pairwise Wilcoxon."""
    from scipy.stats import wilcoxon
    from multisync.feature_definitions import CONFIRMATORY_FEATURES

    rows = []
    for modality, mod_df in df.groupby("modality"):
        for feature in CONFIRMATORY_FEATURES:
            wide = mod_df.pivot_table(
                index="dyad_label", columns="rest_index",
                values=feature, aggfunc="first",
            )
            needed = [1, 2, 3, 4]
            missing = [c for c in needed if c not in wide.columns]
            if missing:
                continue
            wide = wide[needed]
            # Drop dyads with any NaN in the 4 rests
            wide_clean = wide.dropna(how="any")
            n = len(wide_clean)
            if n < 4:
                rows.append({
                    "modality": modality, "feature": feature,
                    "n_dyads": n,
                    "slope_median": np.nan, "slope_p_raw": np.nan,
                    **{f"rest{i}_median": np.nan for i in needed},
                    "pair_12_p": np.nan, "pair_23_p": np.nan,
                    "pair_34_p": np.nan, "pair_14_p": np.nan,
                    "pair_12_dmed": np.nan, "pair_23_dmed": np.nan,
                    "pair_34_dmed": np.nan, "pair_14_dmed": np.nan,
                })
                continue

            # (a) per-dyad OLS slope: feature ~ rest_index (1..4)
            xs = np.array(needed, dtype=float)
            xbar = xs.mean()
            denom = ((xs - xbar) ** 2).sum()
            slopes = []
            for _, ys_row in wide_clean.iterrows():
                yarr = ys_row.to_numpy(dtype=float)
                ybar = yarr.mean()
                num = ((xs - xbar) * (yarr - ybar)).sum()
                slopes.append(num / denom)
            slopes = np.array(slopes)
            try:
                w = wilcoxon(slopes, alternative="two-sided",
                             zero_method="wilcox")
                slope_p = float(w.pvalue)
            except ValueError:
                slope_p = np.nan
            slope_med = float(np.median(slopes))

            # (b) pairwise median diffs + paired Wilcoxon
            def _pair(a_idx, b_idx):
                a = wide_clean[a_idx].to_numpy()
                b = wide_clean[b_idx].to_numpy()
                d_med = float(np.median(a - b))
                if np.allclose(a, b):
                    return 1.0, d_med
                try:
                    p = float(wilcoxon(a, b, alternative="two-sided",
                                       zero_method="wilcox").pvalue)
                except ValueError:
                    p = np.nan
                return p, d_med

            p12, d12 = _pair(1, 2)
            p23, d23 = _pair(2, 3)
            p34, d34 = _pair(3, 4)
            p14, d14 = _pair(1, 4)

            rows.append({
                "modality": modality, "feature": feature, "n_dyads": n,
                "slope_median": slope_med, "slope_p_raw": slope_p,
                "rest1_median": float(wide_clean[1].median()),
                "rest2_median": float(wide_clean[2].median()),
                "rest3_median": float(wide_clean[3].median()),
                "rest4_median": float(wide_clean[4].median()),
                "pair_12_p": p12, "pair_12_dmed": d12,
                "pair_23_p": p23, "pair_23_dmed": d23,
                "pair_34_p": p34, "pair_34_dmed": d34,
                "pair_14_p": p14, "pair_14_dmed": d14,
            })

    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-csv", type=Path,
                   default=Path("artifacts/realtest/lerique_2024/"
                                "dose_response_rests.csv"))
    p.add_argument("--target-hz", type=float, default=1.0)
    p.add_argument("--wcc-window-sec", type=float, default=30.0)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    args = p.parse_args()

    records = _build_single_rest_records(
        args.data_root, modalities=("ECG", "EDA", "RESP"),
        target_fs=args.target_hz,
    )

    logger.info("Analyzing %d records (%d min estimated)",
                len(records), max(1, len(records) * 2 // 60))
    df = _analyze_records(
        records,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )
    summary = _per_dyad_slope_and_pairwise(df)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s (%d rows)", args.out_csv, len(summary))

    # Print key findings for eyeball inspection
    print("\n=== DOSE-RESPONSE SUMMARY ===")
    robust = [
        ("ECG", "peak_amplitude"),
        ("ECG", "switching_rate"),
        ("RESP", "peak_amplitude"),
    ]
    for mod, feat in robust:
        r = summary[(summary["modality"] == mod) & (summary["feature"] == feat)]
        if r.empty:
            print(f"  {mod} {feat}: NO DATA")
            continue
        row = r.iloc[0]
        mono = "MONOTONIC" if (row["slope_median"] > 0 and row["slope_p_raw"] < 0.05) else "FLAT"
        sat = "SATURATED" if (row["pair_12_p"] < 0.05 and row["pair_23_p"] > 0.05 and row["pair_34_p"] > 0.05) else "not saturated"
        print(f"  {mod:4s} {feat:20s} slope={row['slope_median']:+.2e} p={row['slope_p_raw']:.4f} "
              f"n={int(row['n_dyads'])} [{mono}] R1={row['rest1_median']:.3f} "
              f"R2={row['rest2_median']:.3f} R3={row['rest3_median']:.3f} "
              f"R4={row['rest4_median']:.3f} p12={row['pair_12_p']:.3f} p23={row['pair_23_p']:.3f} "
              f"p34={row['pair_34_p']:.3f} [{sat}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

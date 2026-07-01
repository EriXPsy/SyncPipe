"""Trial-level Part A: per-trial synchrony feature extraction + slope test.

Key contract (verified against multisync.realtest.lerique_2024):
    _preprocess_*(raw, raw_fs, target_fs, boundary_mask=None)
        -> (sig_out: np.ndarray, mask_out: np.ndarray of bool)
    LeriqueDyadCondition fields = (dyad_id, dyad_label, modality,
        condition, person_a, person_b, target_hz, n_samples,
        duration_sec, incomplete, discontinuity_mask, meta)

Usage (PowerShell)
------------------
    python scripts/trial_level_part_a.py `
        --data-root "<OSF_ROOT>/Lerique-47n3p" `
        --out-dir   "artifacts/realtest/lerique_2024"

Optional smoke (first 3 dyads only):
    --dyad-whitelist pce01 pce02 pce03
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("trial_level_a")


def _build_per_trial_records(
    data_root: Path,
    *,
    target_fs: float,
    dyad_whitelist: Sequence[str] | None,
):
    """One LeriqueDyadCondition per (dyad, modality, trial_k=1..18)."""
    from multisync.realtest.lerique_2024 import (
        MODALITIES, RAW_FS_HZ, TRIAL_SEGMENT_COUNT, MIN_DURATION_SEC,
        LeriqueDyadCondition,
        _PREPROC_DISPATCH,
        _collect_segments_for_person,
        _verify_p1_p2_length_alignment,
        _verify_min_duration,
    )

    root = Path(data_root).expanduser().resolve()
    records: List[LeriqueDyadCondition] = []

    for modality in MODALITIES:
        mod_root = root / modality
        if not mod_root.exists():
            logger.warning("Missing modality dir: %s", mod_root)
            continue
        preproc = _PREPROC_DISPATCH[modality]

        for pce_dir in sorted(d for d in mod_root.iterdir() if d.is_dir()):
            dyad_label = pce_dir.name[:5]
            if dyad_whitelist and dyad_label not in dyad_whitelist:
                continue

            for k in range(1, TRIAL_SEGMENT_COUNT + 1):
                unit = f"trial{k:02d}"
                a_raw, _, a_bound = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="1",
                    cond_class="Trial", seg_indices=[k],
                )
                b_raw, _, b_bound = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="2",
                    cond_class="Trial", seg_indices=[k],
                )
                if not _verify_p1_p2_length_alignment(
                    a_raw, b_raw, dyad_label, modality, unit,
                ):
                    continue
                if not _verify_min_duration(
                    a_raw, b_raw, RAW_FS_HZ, dyad_label, modality, unit,
                ):
                    continue
                if a_raw is None or b_raw is None:
                    continue

                try:
                    a_sig, a_mask = preproc(
                        a_raw, RAW_FS_HZ, target_fs,
                        boundary_mask=a_bound,
                    )
                    b_sig, b_mask = preproc(
                        b_raw, RAW_FS_HZ, target_fs,
                        boundary_mask=b_bound,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("preproc failed %s/%s/%s: %s",
                                   dyad_label, modality, unit, exc)
                    continue

                # Build a common time axis at target_fs. Single-trial
                # segments have no internal boundaries, so the mask is
                # all-True — but use the actually returned mask for safety.
                n = int(min(len(a_sig), len(b_sig)))
                if n < 4:
                    continue
                t = np.arange(n, dtype=np.float64) / target_fs
                a_df = pd.DataFrame({"time": t, "value": a_sig[:n]})
                b_df = pd.DataFrame({"time": t, "value": b_sig[:n]})
                disc = (a_mask[:n].astype(bool) & b_mask[:n].astype(bool))

                rec = LeriqueDyadCondition(
                    dyad_id=f"{dyad_label}__{modality}__{unit}",
                    dyad_label=dyad_label,
                    modality=modality,
                    condition=unit,
                    person_a=a_df,
                    person_b=b_df,
                    target_hz=target_fs,
                    n_samples=n,
                    duration_sec=float(n / target_fs),
                    incomplete=False,
                    discontinuity_mask=disc,
                    meta={"trial_index": k},
                )
                records.append(rec)
    return records


def _analyze_per_trial(records, *, target_hz, wcc_window_sec, onset_threshold):
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
        enable_prediction=False,  # trial-level slope test only uses
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
                "trial_index": int(rec.meta["trial_index"]),
                "duration_sec": rec.duration_sec,
            }
            for n in feats:
                attr = SSOT_TO_DR.get(n)
                row[n] = getattr(dr, attr, np.nan) if attr else np.nan
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAIL %s/%s/trial%02d: %s",
                             rec.dyad_label, rec.modality,
                             rec.meta["trial_index"], exc)
        if i % 50 == 0 or i == len(records):
            logger.info("processed %d/%d", i, len(records))
    return pd.DataFrame(rows)


def _trial_slope_test(df: pd.DataFrame) -> pd.DataFrame:
    """Per (modality, feature): per-dyad OLS slope of feature ~ trial_index,
       then one-sample Wilcoxon on slopes against zero."""
    from scipy.stats import wilcoxon
    from multisync.feature_definitions import CONFIRMATORY_FEATURES

    rows = []
    for modality, mod_df in df.groupby("modality"):
        for feature in CONFIRMATORY_FEATURES:
            wide = mod_df.pivot_table(
                index="dyad_label", columns="trial_index",
                values=feature, aggfunc="first",
            )
            slopes = []
            for _, ys in wide.iterrows():
                y = ys.to_numpy(dtype=float)
                x = np.array(ys.index, dtype=float)
                mask = np.isfinite(y)
                if mask.sum() < 5:
                    continue
                xf = x[mask]; yf = y[mask]
                xbar = xf.mean(); ybar = yf.mean()
                denom = ((xf - xbar) ** 2).sum()
                if denom == 0:
                    continue
                slopes.append(((xf - xbar) * (yf - ybar)).sum() / denom)
            slopes = np.array(slopes, dtype=float)
            n = int(len(slopes))
            if n < 5:
                rows.append({
                    "modality": modality, "feature": feature,
                    "n_dyads_with_slope": n,
                    "slope_median": np.nan, "slope_p_raw": np.nan,
                    "frac_positive": np.nan,
                })
                continue
            try:
                p = float(wilcoxon(slopes, alternative="two-sided",
                                   zero_method="wilcox").pvalue)
            except ValueError:
                p = np.nan
            rows.append({
                "modality": modality, "feature": feature,
                "n_dyads_with_slope": n,
                "slope_median": float(np.median(slopes)),
                "slope_p_raw": p,
                "frac_positive": float((slopes > 0).mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/realtest/lerique_2024"))
    p.add_argument("--target-hz", type=float, default=1.0)
    p.add_argument("--wcc-window-sec", type=float, default=30.0)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--dyad-whitelist", nargs="+", default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = _build_per_trial_records(
        args.data_root, target_fs=args.target_hz,
        dyad_whitelist=args.dyad_whitelist,
    )
    logger.info("Built %d per-trial records", len(records))

    feats = _analyze_per_trial(
        records,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )
    feats_path = args.out_dir / "trial_level_features.csv"
    feats.to_csv(feats_path, index=False)
    logger.info("Wrote %s (rows=%d)", feats_path, len(feats))

    slopes = _trial_slope_test(feats)
    slopes_path = args.out_dir / "trial_level_slopes.csv"
    slopes.to_csv(slopes_path, index=False)
    logger.info("Wrote %s (rows=%d)", slopes_path, len(slopes))
    return 0

if __name__ == "__main__":
    sys.exit(main())

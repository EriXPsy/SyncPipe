"""Surrogate controls for Lerique pilot — P0 specificity tests.

Pre-registration posture
------------------------
This is a *post-hoc specificity check* on top of the locked
pre-registration. Main / sensitivity / reference contrasts are NOT
re-judged here. Goal: rule out the two leading nuisance hypotheses
for the rest1 -> trials_concat elevation found in the main analysis:

    H1 (shared noise):   real-pair signal driven by shared environment /
                          time-of-day / instructor voice / room temp ->
                          ruled out IF cross-dyad pseudo-pair < real-pair.
    H2 (autocorrelation): real-pair signal driven by within-person
                          autocorrelation that survives any pairing ->
                          ruled out IF within-dyad time-shifted surrogate
                          < real-pair.

We only run the 4 robust features that survived the main analysis:
    ECG  peak_amplitude
    ECG  switching_rate
    RESP peak_amplitude
    EDA  peak_amplitude

Output
------
artifacts/realtest/lerique_2024/surrogate_pseudo_pair.csv
artifacts/realtest/lerique_2024/surrogate_time_shift.csv

Usage (PowerShell)
------------------
    python scripts/surrogate_controls.py `
        --data-root "<PATH_TO>/Lerique-47n3p" `
        --out-dir   "artifacts/realtest/lerique_2024" `
        --n-pseudo-per-dyad 5 `
        --shift-lags-sec -90 -60 -30 30 60 90 `
        --condition trials_concat
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
logger = logging.getLogger("surrogate_controls")
# Force unbuffered stderr on all logging handlers (Python 3.12+).
# Without this, nohup + -u leaves logging handlers buffered indefinitely.
import sys as _sys
for _h in logging.getLogger().handlers:
    try:
        _h.stream.reconfigure(write_through=True)
    except Exception:
        pass

ROBUST_FEATURES = [
    ("ECG",  "peak_amplitude"),
    ("ECG",  "switching_rate"),
    ("RESP", "peak_amplitude"),
    ("EDA",  "peak_amplitude"),
]
SSOT_TO_DR = {
    "peak_amplitude": "mean_peak_amplitude",
    "switching_rate": "mean_switching_rate",
}


# ---------------------------------------------------------------------------
# Step A: Build per-dyad preprocessed signals once (shared across both controls)
# ---------------------------------------------------------------------------

def _build_preprocessed_signals(
    data_root: Path,
    modalities: Sequence[str],
    condition_unit: str,
    *,
    target_fs: float,
) -> Dict[Tuple[str, str], Dict]:
    """Return {(dyad_label, modality) -> {a_sig, b_sig, n_samples}}.

    Preprocessed once, reused for real / pseudo-pair / time-shift analyses.
    Returns only dyads with BOTH persons present after preprocessing.
    """
    from multisync.realtest.lerique_2024 import (
        MODALITIES, RAW_FS_HZ, MIN_DURATION_SEC,
        _segments_for_condition_unit,
        _collect_segments_for_person,
        _verify_p1_p2_length_alignment,
        _verify_min_duration,
        _PREPROC_DISPATCH,
    )

    cond_class, seg_indices = _segments_for_condition_unit(condition_unit)
    root = Path(data_root).expanduser().resolve()

    cache: Dict[Tuple[str, str], Dict] = {}
    n_skipped = 0

    for modality in modalities:
        if modality not in MODALITIES:
            continue
        mod_root = root / modality
        if not mod_root.exists():
            logger.warning("Missing modality dir: %s", mod_root)
            continue
        preproc_fn = _PREPROC_DISPATCH[modality]

        for pce_dir in sorted(d for d in mod_root.iterdir() if d.is_dir()):
            dyad_label = pce_dir.name[:5]

            # Load raw segments
            a_raw, _, _ = _collect_segments_for_person(
                pce_dir, dyad_label, modality, person="1",
                cond_class=cond_class, seg_indices=list(seg_indices),
            )
            b_raw, _, _ = _collect_segments_for_person(
                pce_dir, dyad_label, modality, person="2",
                cond_class=cond_class, seg_indices=list(seg_indices),
            )

            # Gate: both persons present, aligned, sufficient duration
            if a_raw is None or b_raw is None:
                n_skipped += 1
                continue
            if not _verify_p1_p2_length_alignment(
                a_raw, b_raw, dyad_label, modality, condition_unit,
            ):
                n_skipped += 1
                continue
            if not _verify_min_duration(
                a_raw, b_raw, RAW_FS_HZ, dyad_label, modality,
                condition_unit, min_duration_sec=MIN_DURATION_SEC,
            ):
                n_skipped += 1
                continue

            # Preprocess
            try:
                a_sig, _a_mask = preproc_fn(a_raw, RAW_FS_HZ, target_fs)
                b_sig, _b_mask = preproc_fn(b_raw, RAW_FS_HZ, target_fs)
            except Exception as exc:
                logger.warning("preproc failed %s/%s: %s",
                               dyad_label, modality, exc)
                n_skipped += 1
                continue

            n = min(len(a_sig), len(b_sig))
            if n < 4:
                n_skipped += 1
                continue

            cache[(dyad_label, modality)] = {
                "a_sig": a_sig[:n].copy(),
                "b_sig": b_sig[:n].copy(),
                "n_samples": n,
            }

    logger.info("Built signal cache: %d dyad×modality pairs (%d skipped)",
                len(cache), n_skipped)
    return cache


# ---------------------------------------------------------------------------
# Feature extraction helper (shared by real / pseudo / shifted)
# ---------------------------------------------------------------------------

def _extract_features(
    a_sig: np.ndarray,
    b_sig: np.ndarray,
    modality: str,
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
) -> Dict[str, float]:
    """Run DynamicAnalyzer on a (person_a, person_b) pair and return
    a dict mapping feature bare-name -> scalar value.

    Warning suppression
    -------------------
    Lerique P1/P2 are co-acquired on a single BIOPAC MP160; we build
    both persons' time axes from the same `np.arange(n) / target_fs`,
    so they are co-started by construction. The Dyad.align "relative
    timestamps starting near 0" warning is therefore a false positive
    here and is silenced for clarity (any other UserWarning still
    propagates).
    """
    import warnings
    from multisync.core import DynamicAnalyzer, Dyad
    from multisync.batch import DyadResult

    ch = modality.lower()
    dyad = Dyad(
        hz=target_hz,
        dyad_id="surrogate",
        **{f"{ch}_a": pd.DataFrame({
            "time": np.arange(len(a_sig)) / target_hz,
            "value": a_sig.astype(np.float64),
        }),
           f"{ch}_b": pd.DataFrame({
            "time": np.arange(len(b_sig)) / target_hz,
            "value": b_sig.astype(np.float64),
        })},
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*relative timestamps.*",
            category=UserWarning,
        )
        dyad.align(target_hz=target_hz, require_co_start=False)
    dyad.zscore()

    window_size = max(2, int(round(wcc_window_sec * target_hz)))
    analyzer = DynamicAnalyzer(
        window_size=window_size,
        onset_threshold=onset_threshold,
        enable_prediction=False,  # surrogate test only needs descriptive
                                  # features; skip rolling-origin CV +
                                  # LogisticRegression (dominates runtime).
    )
    results = analyzer.fit_transform(dyad)
    dr = DyadResult.from_analysis_results(results)

    out = {}
    for feat_name, dr_attr in SSOT_TO_DR.items():
        out[feat_name] = getattr(dr, dr_attr, np.nan)
    return out


# ---------------------------------------------------------------------------
# Real-pair analysis
# ---------------------------------------------------------------------------

def _real_pair_features(
    cache: Dict[Tuple[str, str], Dict],
    *,
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
) -> pd.DataFrame:
    """Extract features for each real dyad×modality pair."""
    rows = []
    n_total = len(cache)
    for i, ((dyad, modality), sigs) in enumerate(cache.items(), 1):
        feats = _extract_features(
            sigs["a_sig"], sigs["b_sig"], modality,
            target_hz, wcc_window_sec, onset_threshold,
        )
        row = {"dyad_label": dyad, "modality": modality, "pair_type": "real"}
        row.update(feats)
        rows.append(row)
        if i % 20 == 0 or i == n_total:
            logger.info("real-pair %d/%d", i, n_total)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pseudo-pair (cross-dyad scrambling)
# ---------------------------------------------------------------------------

def _pseudo_pair_features(
    cache: Dict[Tuple[str, str], Dict],
    *,
    n_per_dyad: int,
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
    seed: int = 42,
) -> pd.DataFrame:
    """For each (dyad, modality), create `n_per_dyad` pseudo-pairs by
    pairing this dyad's person_a with a random OTHER dyad's person_b.

    Only dyads sharing the same modality and with sufficient signal
    length are eligible as pseudo-partners.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for (dyad_a, modality), sigs_a in cache.items():
        # Eligible partners: any OTHER dyad with same modality.
        # NOTE: we do NOT filter by length — n_common = min(...) below
        # already truncates to the shorter of the two.
        # The pseudo-pair semantic is "dyad_a's person_a × dyad_b's
        # person_b" — testing dyad-specificity of partner identity
        # against the H1 (shared environment / experimenter / time-of-day)
        # nuisance hypothesis. Same-role (a×a) pseudo would test a
        # different hypothesis (role-specific autocorrelation) and is
        # not run here.
        partners = [
            (dyad_b, sigs_b) for (dyad_b, mod_b), sigs_b in cache.items()
            if mod_b == modality and dyad_b != dyad_a
        ]
        if len(partners) < 1:
            logger.debug("No pseudo-pair partners for %s/%s",
                         dyad_a, modality)
            continue

        # Sample n_per_dyad partners; with replacement only if pool < n.
        replace = n_per_dyad > len(partners)
        chosen = rng.choice(
            len(partners), size=n_per_dyad, replace=replace,
        )
        for idx in chosen:
            dyad_b, sigs_b = partners[idx]
            # Truncate to common length so P1/P2 share a grid.
            n_common = min(sigs_a["n_samples"], sigs_b["n_samples"])
            feats = _extract_features(
                sigs_a["a_sig"][:n_common],
                sigs_b["b_sig"][:n_common],
                modality,
                target_hz, wcc_window_sec, onset_threshold,
            )
            row = {
                "dyad_label": f"{dyad_a}_x_{dyad_b}",
                "modality": modality,
                "pair_type": "pseudo",
                "source_dyad_a": dyad_a,
                "source_dyad_b": dyad_b,
            }
            row.update(feats)
            rows.append(row)

        if (len(rows) + 1) % 100 == 0:
            logger.info("pseudo-pair: %d pairs built", len(rows))

    logger.info("Built %d pseudo-pairs", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time-shift surrogate
# ---------------------------------------------------------------------------

def _time_shift_features(
    cache: Dict[Tuple[str, str], Dict],
    *,
    shift_lags_sec: Sequence[float],
    target_hz: float,
    wcc_window_sec: float,
    onset_threshold: float,
) -> pd.DataFrame:
    """For each (dyad, modality), create time-shifted surrogates by
    LINEARLY shifting person_b's time series by `lag` seconds and
    truncating both signals to the overlapping window. This preserves
    each person's autocorrelation structure while breaking the
    time-alignment between the two persons.

    We use truncation rather than `np.roll` (circular shift) because
    circular shift introduces seam-discontinuities at the wrap point;
    for short conditions (e.g. rest1, 180s) those seams contaminate
    a non-trivial fraction of samples relative to the WCC window.
    """
    rows = []
    n_total = len(cache) * len(shift_lags_sec)
    count = 0
    for (dyad, modality), sigs in cache.items():
        a = sigs["a_sig"]
        b_orig = sigs["b_sig"]
        n = len(a)
        wcc_window_samples = max(2, int(round(wcc_window_sec * target_hz)))

        for lag_sec in shift_lags_sec:
            k = int(round(lag_sec * target_hz))
            if k == 0:
                # lag=0 is the real pair, skip
                continue
            # Need overlap to be at least 2 WCC windows for a stable
            # synchrony estimate.
            overlap = n - abs(k)
            if overlap < 2 * wcc_window_samples:
                logger.debug(
                    "skip lag=%+ds for %s/%s: overlap=%d < 2*window=%d",
                    int(lag_sec), dyad, modality, overlap,
                    2 * wcc_window_samples,
                )
                continue
            # Linear shift with truncation:
            #   k > 0: b leads a by k samples
            #   k < 0: b lags  a by |k| samples
            if k > 0:
                a_use = a[k:]
                b_use = b_orig[:n - k]
            else:
                a_use = a[:n + k]      # k < 0, equivalent to a[:n - |k|]
                b_use = b_orig[-k:]

            feats = _extract_features(
                a_use, b_use, modality,
                target_hz, wcc_window_sec, onset_threshold,
            )
            row = {
                "dyad_label": dyad,
                "modality": modality,
                "pair_type": "time_shift",
                "lag_sec": lag_sec,
                "n_overlap": int(overlap),
            }
            row.update(feats)
            rows.append(row)
            count += 1
            if count % 50 == 0 or count == n_total:
                logger.info("time-shift %d/%d", count, n_total)

    logger.info("Built %d time-shifted surrogates", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical comparison: real vs surrogate
# ---------------------------------------------------------------------------

def _compare_real_vs_surrogate(
    real_df: pd.DataFrame,
    surrogate_df: pd.DataFrame,
    modality: str,
    feature: str,
    surrogate_label: str,
    *,
    surrogate_agg: str = "median",
) -> dict:
    """For a given (modality, feature), compare real-pair feature values
    against surrogate feature values **per dyad** with paired Wilcoxon
    signed-rank.

    Statistical design rationale
    ----------------------------
    Each dyad contributes ONE real value and (typically) several
    surrogate instances (n_per_dyad pseudo partners, or n_lags
    time-shift surrogates). We aggregate the surrogate instances
    within-dyad (default: median) to obtain ONE surrogate value per
    dyad, then run a within-dyad paired test (real vs surrogate).

    This is the correct design because:
      * The dyad is the unit of randomization / observation.
      * It removes between-dyad variance from the noise term,
        gaining power proportional to the within-dyad correlation.
      * It preserves the 1-real-vs-1-surrogate accounting that
        evidence-based surrogate frameworks (e.g. PRTF, IAAFT) use.

    The previous unpaired Mann-Whitney design was wrong because
    n_real (~27) and n_surrogate (~135) were treated as independent
    samples, double-counting within-dyad observations.

    Returns
    -------
    dict with: medians, mean within-dyad delta, paired p, n_dyads,
    and the directional verdict (real > surrogate AND p < 0.05).
    """
    from scipy.stats import wilcoxon

    # Aggregate surrogate instances per dyad.
    if "source_dyad_a" in surrogate_df.columns:
        # Pseudo-pair: aggregate by source_dyad_a (the "ego" dyad)
        agg_key = "source_dyad_a"
    else:
        # Time-shift: aggregate by dyad_label across lags
        agg_key = "dyad_label"

    surr_sub = surrogate_df[surrogate_df["modality"] == modality]
    if surrogate_agg == "median":
        surr_per_dyad = (
            surr_sub.groupby(agg_key)[feature].median().reset_index()
        )
    else:
        surr_per_dyad = (
            surr_sub.groupby(agg_key)[feature].mean().reset_index()
        )
    surr_per_dyad = surr_per_dyad.rename(
        columns={agg_key: "dyad_label", feature: "surrogate_value"}
    )

    real_sub = real_df[real_df["modality"] == modality][
        ["dyad_label", feature]
    ].rename(columns={feature: "real_value"})

    merged = real_sub.merge(surr_per_dyad, on="dyad_label", how="inner")
    merged = merged.dropna(subset=["real_value", "surrogate_value"])
    n = len(merged)

    if n < 3:
        return {
            "modality": modality, "feature": feature,
            "surrogate": surrogate_label,
            "n_dyads": n,
            "real_median": np.nan, "surrogate_median": np.nan,
            "median_delta": np.nan, "frac_real_gtr": np.nan,
            "p_raw": np.nan, "real_gtr_surrogate": False,
            "note": "insufficient_paired_dyads",
        }

    real_vals = merged["real_value"].to_numpy(dtype=float)
    surr_vals = merged["surrogate_value"].to_numpy(dtype=float)
    delta = real_vals - surr_vals

    try:
        # One-sided: real > surrogate (we expect dyad-specific coupling
        # to elevate the real-pair value above the surrogate baseline).
        w = wilcoxon(real_vals, surr_vals, alternative="greater",
                     zero_method="wilcox")
        p = float(w.pvalue)
    except ValueError:
        p = np.nan

    return {
        "modality": modality,
        "feature": feature,
        "surrogate": surrogate_label,
        "n_dyads": int(n),
        "real_median": float(np.median(real_vals)),
        "surrogate_median": float(np.median(surr_vals)),
        "median_delta": float(np.median(delta)),
        "frac_real_gtr": float(np.mean(delta > 0)),
        "p_raw": float(p) if np.isfinite(p) else np.nan,
        "real_gtr_surrogate": bool(
            np.median(delta) > 0
            and np.isfinite(p) and p < 0.05
        ),
    }


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaNs preserved."""
    p = np.asarray(pvals, dtype=np.float64)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return p
    sub = p[mask]
    n_p = len(sub)
    order = np.argsort(sub)
    ranked = sub[order]
    bh = ranked * n_p / (np.arange(n_p) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    adj = np.empty(n_p, dtype=np.float64)
    adj[order] = np.clip(bh, 0.0, 1.0)
    out = p.copy()
    out[mask] = adj
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Surrogate controls for Lerique pilot "
                    "(pseudo-pair + time-shift)."
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/realtest/lerique_2024"))
    p.add_argument("--target-hz", type=float, default=1.0)
    p.add_argument("--wcc-window-sec", type=float, default=30.0)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--condition", type=str, default="trials_concat",
                   choices=["trials_concat", "rest1", "rest_postblock"])
    p.add_argument("--n-pseudo-per-dyad", type=int, default=5,
                   help="Number of pseudo-pairs per (dyad, modality).")
    p.add_argument("--shift-lags-sec", type=float, nargs="+",
                   default=[-90, -60, -30, 30, 60, 90],
                   help="Time-shift lags in seconds.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modalities = [m for m, _ in ROBUST_FEATURES]
    # Deduplicate while preserving order
    seen = set()
    modalities_unique = []
    for m in modalities:
        if m not in seen:
            modalities_unique.append(m)
            seen.add(m)

    # ---- Step A: Preprocess once ----
    logger.info("Building preprocessed signal cache for condition=%s ...",
                args.condition)
    cache = _build_preprocessed_signals(
        args.data_root,
        modalities=modalities_unique,
        condition_unit=args.condition,
        target_fs=args.target_hz,
    )
    if not cache:
        logger.error("Empty signal cache — aborting.")
        return 1

    # ---- Step B: Real-pair features ----
    logger.info("Computing real-pair features ...")
    real_df = _real_pair_features(
        cache,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )
    logger.info("Real pairs: %d rows", len(real_df))

    # ---- Step C: Pseudo-pair surrogates ----
    logger.info("Computing pseudo-pair surrogates "
                "(n_per_dyad=%d) ...", args.n_pseudo_per_dyad)
    pseudo_df = _pseudo_pair_features(
        cache,
        n_per_dyad=args.n_pseudo_per_dyad,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
        seed=args.seed,
    )

    # ---- Step D: Time-shift surrogates ----
    logger.info("Computing time-shift surrogates "
                "(lags=%s) ...", list(args.shift_lags_sec))
    shift_df = _time_shift_features(
        cache,
        shift_lags_sec=args.shift_lags_sec,
        target_hz=args.target_hz,
        wcc_window_sec=args.wcc_window_sec,
        onset_threshold=args.onset_threshold,
    )

    # ---- Step E: Statistical comparison (paired, per-dyad) ----
    rows_pseudo = []
    for modality, feature in ROBUST_FEATURES:
        result = _compare_real_vs_surrogate(
            real_df, pseudo_df, modality, feature, "pseudo_pair",
        )
        rows_pseudo.append(result)
    pseudo_cmp = pd.DataFrame(rows_pseudo)
    # BH-FDR within the 4-feature pseudo-pair family
    pseudo_cmp["p_fdr"] = _bh_fdr(pseudo_cmp["p_raw"].to_numpy())
    pseudo_cmp["sig_fdr"] = (
        (pseudo_cmp["p_fdr"] < 0.05) & (pseudo_cmp["median_delta"] > 0)
    )
    pseudo_path = args.out_dir / "surrogate_pseudo_pair.csv"
    pseudo_cmp.to_csv(pseudo_path, index=False)
    logger.info("Wrote %s (%d rows)", pseudo_path, len(pseudo_cmp))

    # Time-shift: aggregate ALL lags within-dyad (median across lags)
    # then paired test against real. This treats time-shift as a
    # single composite null hypothesis ("autocorrelation alone") rather
    # than over-counting per-lag tests.
    rows_shift = []
    for modality, feature in ROBUST_FEATURES:
        result = _compare_real_vs_surrogate(
            real_df, shift_df, modality, feature, "time_shift_all_lags",
        )
        rows_shift.append(result)
    shift_cmp = pd.DataFrame(rows_shift)
    shift_cmp["p_fdr"] = _bh_fdr(shift_cmp["p_raw"].to_numpy())
    shift_cmp["sig_fdr"] = (
        (shift_cmp["p_fdr"] < 0.05) & (shift_cmp["median_delta"] > 0)
    )
    shift_path = args.out_dir / "surrogate_time_shift.csv"
    shift_cmp.to_csv(shift_path, index=False)
    logger.info("Wrote %s (%d rows)", shift_path, len(shift_cmp))

    # ---- Per-lag breakdown (descriptive only, NOT entered into FDR) ----
    rows_per_lag = []
    for modality, feature in ROBUST_FEATURES:
        for lag in sorted(set(args.shift_lags_sec)):
            lag_df = shift_df[shift_df["lag_sec"] == lag]
            r = _compare_real_vs_surrogate(
                real_df, lag_df, modality, feature, f"shift_{lag:+.0f}s",
            )
            r["lag_sec"] = lag
            rows_per_lag.append(r)
    per_lag_cmp = pd.DataFrame(rows_per_lag)
    per_lag_path = args.out_dir / "surrogate_time_shift_per_lag.csv"
    per_lag_cmp.to_csv(per_lag_path, index=False)
    logger.info("Wrote %s (%d rows, descriptive)",
                per_lag_path, len(per_lag_cmp))

    # ---- Console summary ----
    print("\n=== PSEUDO-PAIR SUMMARY (paired Wilcoxon, BH-FDR within 4-feature family) ===")
    for _, r in pseudo_cmp.iterrows():
        verdict = (
            "REAL > PSEUDO ✓" if bool(r["sig_fdr"])
            else ("trend (raw p<.05)" if r["p_raw"] < 0.05
                  else "real ≈ pseudo ✗")
        )
        print(f"  {r['modality']:4s} {r['feature']:20s} "
              f"n_dyads={int(r['n_dyads']):2d}  "
              f"real={r['real_median']:.3f}  pseudo={r['surrogate_median']:.3f}  "
              f"Δ={r['median_delta']:+.3f}  frac+={r['frac_real_gtr']:.2f}  "
              f"p_raw={r['p_raw']:.4f}  p_fdr={r['p_fdr']:.4f}  [{verdict}]")

    print("\n=== TIME-SHIFT SUMMARY (paired Wilcoxon on per-dyad median across lags, BH-FDR) ===")
    for _, r in shift_cmp.iterrows():
        verdict = (
            "REAL > SHIFTED ✓" if bool(r["sig_fdr"])
            else ("trend (raw p<.05)" if r["p_raw"] < 0.05
                  else "real ≈ shifted ✗")
        )
        print(f"  {r['modality']:4s} {r['feature']:20s} "
              f"n_dyads={int(r['n_dyads']):2d}  "
              f"real={r['real_median']:.3f}  shifted={r['surrogate_median']:.3f}  "
              f"Δ={r['median_delta']:+.3f}  frac+={r['frac_real_gtr']:.2f}  "
              f"p_raw={r['p_raw']:.4f}  p_fdr={r['p_fdr']:.4f}  [{verdict}]")

    print("\n=== TIME-SHIFT PER-LAG (descriptive, no FDR) ===")
    for modality, feature in ROBUST_FEATURES:
        sub = per_lag_cmp[
            (per_lag_cmp["modality"] == modality)
            & (per_lag_cmp["feature"] == feature)
        ].sort_values("lag_sec")
        deltas = "  ".join(
            f"{int(row['lag_sec']):+3d}s:Δ={row['median_delta']:+.3f}(p={row['p_raw']:.2f})"
            for _, row in sub.iterrows()
        )
        print(f"  {modality:4s} {feature:20s}  {deltas}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

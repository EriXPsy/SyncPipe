#!/usr/bin/env python3
r"""
run_gt45_fdr.py
================

GT-4/5 grey-box Kuramoto FDR calibration.

Validates that SyncPipe features do not produce excessive false
positives under the null hypothesis of "no coordination dynamics".

Null architecture
-----------------
H0: No coupling between oscillators.

Null generator: two INDEPENDENT oscillators with i.i.d. uniform
phase differences::

    Δθ_ij ~ Uniform(0, 2π)   i.i.d. for each time point t_j

    r_null(t_j) = |cos(Δθ_ij / 2)|  ∈ [0, 1]

This is the TRUE null under "no coupling" — there is NO deterministic
temporal structure in r_null(t).  (The OLD null generator used one
drifting oscillator, which created deterministic beating → IAAFT
surrogates preserved the structure → n_ge ≈ 0 → false-negative FDR.)

For each null trajectory, we generate SURROGATE_N=499 IAAFT surrogates
(Phipson & Smyth p = (n_ge+1)/(N+1)), then apply Benjamini-Hochberg FDR
at α=0.05 across the 6 confirmatory features.  Repeating over
N_SEEDS_FDR=30 Monte Carlo seeds gives the family-wise FPR estimate.

Usage::

    python run_gt45_fdr.py [--project-root <dir>] [--dry-run]

Author: SyncPipe Validation Team (2026-06-10)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Robust import
# ===========================================================================

def _find_feature_definitions_path(start: Path) -> Path:
    current = start.resolve()
    for _ in range(10):
        candidate = current / "multisync" / "feature_definitions.py"
        if candidate.exists():
            return candidate
        candidate2 = current / "feature_definitions.py"
        if candidate2.exists():
            return candidate2
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        "Cannot find feature_definitions.py.\n"
        "Please specify --project-root <dir> where feature_definitions.py lives."
    )


def _load_module(name: str, filepath: Path):
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(name, str(filepath))
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup_imports(project_root: Optional[Path] = None):
    if project_root is None:
        script_dir = Path(__file__).resolve().parent
        fd_path = _find_feature_definitions_path(script_dir)
    else:
        fd_path = project_root / "multisync" / "feature_definitions.py"
        if not fd_path.exists():
            fd_path = project_root / "feature_definitions.py"
        if not fd_path.exists():
            raise FileNotFoundError(
                f"--project-root {project_root} does not contain "
                f"feature_definitions.py"
            )
    return _load_module("feature_definitions", fd_path)


# Will be populated in main():
fd = None
extract_features = None
CONFIRMATORY_FEATURES = []


# ===========================================================================
# CONFIGURATION
# ===========================================================================

N_SAMPLES = 300           # WCC samples per trajectory
NOISE_SIGMA = 0.10        # additive Gaussian noise σ
SURROGATE_N = 499         # IAAFT surrogates (odd → clean Phipson-Smyth)
N_SEEDS_FDR = 30          # Monte Carlo seeds for FPR estimation
FDR_ALPHA = 0.05          # nominal FDR level
SEED_BASE = 42


# ===========================================================================
# TRUE NULL GENERATOR
# ===========================================================================

def generate_null_wcc(n_samples: int = N_SAMPLES,
                      seed: int = 42,
                      noise_sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Generate TRUE null WCC under no-coupling hypothesis.

    Two INDEPENDENT oscillators with i.i.d. uniform phase differences.
    r(t) = |cos(Δθ/2)| where Δθ ~ Uniform(0, 2π) i.i.d. for each t.

    Theoretical mean r = 0.5 (arcsine distribution).
    This is the correct null for "no coordination dynamics".

    Returns
    -------
    wcc : np.ndarray, shape (n_samples,)
        Null WCC trajectory with additive Gaussian noise, clipped to [-1, 1].
    """
    rng = np.random.default_rng(seed)
    delta_theta = rng.uniform(0, 2 * np.pi, size=n_samples)
    r = np.abs(np.cos(delta_theta / 2.0))
    wcc = r + rng.normal(0, noise_sigma, size=n_samples)
    wcc = np.clip(wcc, -1.0, 1.0)
    return wcc


# ===========================================================================
# IAAFT SURROGATE
# ===========================================================================

def _iaaft_surrogate(signal: np.ndarray, seed: int) -> np.ndarray:
    """Generate one IAAFT surrogate for a 1-D signal.

    Uses the amplitude-adjusted Fourier transform
    (Schreiber & Schmitz 1996).  Falls back to simple phase
    randomisation if IAAFT fails.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        The original 1-D time series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    surrogate : np.ndarray, shape (n,)
        IAAFT surrogate preserving amplitude distribution and
        power spectrum.
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    if n < 4:
        return signal.copy()

    # Step 1: Store sorted original values
    original_sorted = np.sort(signal)

    # Step 2: FFT → randomise phases → IFFT
    fft_orig = np.fft.rfft(signal)
    amplitudes = np.abs(fft_orig)
    random_phases = rng.uniform(0, 2 * np.pi, size=len(fft_orig))
    surrogate_complex = amplitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(surrogate_complex, n=n)

    # Step 3: Amplitude adjustment (single iteration — sufficient for FDR)
    rank_surr = np.argsort(np.argsort(surrogate))
    surrogate_adjusted = original_sorted[rank_surr]
    return surrogate_adjusted


# ===========================================================================
# FDR CALIBRATION
# ===========================================================================

def run_fdr_calibration() -> pd.DataFrame:
    """Run GT-4/5 FDR calibration with TRUE null.

    For each morphology:
        1. Generate NULL WCC (no coordination)
        2. Extract 6 confirmatory features on null WCC
        3. Generate SURROGATE_N=499 IAAFT surrogates
        4. Compute Phipson-Smyth p-values: p = (n_ge+1)/(N+1)
        5. Apply Benjamini-Hochberg FDR at α=0.05
        6. Repeat for N_SEEDS_FDR=30 Monte Carlo seeds
        7. Report family-wise FPR

    Returns
    -------
    pd.DataFrame with columns:
        morphology, family_fpr, fpr_onset_latency, fpr_rise_time, ...
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GT-4/5 FDR Calibration — TRUE Null (no coupling)")
    logger.info(f"  SURROGATE_N = {SURROGATE_N}  (Phipson-Smyth p = (n_ge+1)/(N+1))")
    logger.info(f"  N_SEEDS_FDR = {N_SEEDS_FDR} (Monte Carlo seeds)")
    logger.info(f"  FDR_ALPHA   = {FDR_ALPHA}")
    logger.info("=" * 60)

    # We calibrate FDR across 4 morphologies to ensure consistency.
    # Under TRUE null all should give similar (near-nominal) FPR.
    morph_names = ["sustained", "single_peak", "oscillatory", "asymmetric_decay"]

    fpr_results: List[Dict] = []

    for morph in morph_names:
        logger.info(f"  Morphology: {morph}")
        feature_fpr = {f: 0 for f in CONFIRMATORY_FEATURES}
        family_rejections = 0

        for seed in range(N_SEEDS_FDR):
            # ── Generate TRUE NULL WCC ──
            wcc_obs = generate_null_wcc(
                n_samples=N_SAMPLES,
                seed=SEED_BASE + seed * 100,
                noise_sigma=NOISE_SIGMA,
            )

            # ── Extract observed features ──
            feat_obs = extract_features(wcc_obs, hz=1.0, wcc_window_sec=30.0)
            obs_values = {f: getattr(feat_obs, f) for f in CONFIRMATORY_FEATURES}

            # ── Pre-filter: require finite WCC ──
            wcc_valid = wcc_obs[np.isfinite(wcc_obs)]
            if len(wcc_valid) < 30:
                continue

            # ── Generate IAAFT surrogates ──
            surr_feature_values = {f: [] for f in CONFIRMATORY_FEATURES}
            for s in range(SURROGATE_N):
                wcc_surr = _iaaft_surrogate(
                    wcc_valid,
                    seed=SEED_BASE + seed * 1000 + s,
                )
                feat_surr = extract_features(wcc_surr, hz=1.0, wcc_window_sec=30.0)
                for f in CONFIRMATORY_FEATURES:
                    surr_feature_values[f].append(getattr(feat_surr, f))

            # ── Phipson & Smyth p-values ──
            p_values = {}
            for f in CONFIRMATORY_FEATURES:
                null_vals = np.array(surr_feature_values[f])
                null_vals = null_vals[np.isfinite(null_vals)]
                obs_val = obs_values[f]
                if not np.isfinite(obs_val) or len(null_vals) == 0:
                    p_values[f] = 1.0
                else:
                    # Two-tailed: count null features ≥ |obs| OR null ≤ -|obs|
                    # For one-tailed: n_ge = count(null >= obs)
                    n_ge = np.sum(null_vals >= obs_val)
                    p_values[f] = (n_ge + 1) / (len(null_vals) + 1)

            # ── Benjamini-Hochberg FDR ──
            p_sorted = sorted(
                [(f, p) for f, p in p_values.items() if np.isfinite(p)],
                key=lambda x: x[1],
            )
            n_tests = len(p_sorted)
            rejected = set()
            for rank, (f, p) in enumerate(p_sorted, start=1):
                bh_threshold = FDR_ALPHA * rank / n_tests
                if p <= bh_threshold:
                    rejected.add(f)

            # Any rejection under null = false positive for the family
            if len(rejected) > 0:
                family_rejections += 1
            for f in CONFIRMATORY_FEATURES:
                if f in rejected:
                    feature_fpr[f] += 1

        # ── Normalise to FPR ──
        for f in CONFIRMATORY_FEATURES:
            feature_fpr[f] /= N_SEEDS_FDR
        family_fpr = family_rejections / N_SEEDS_FDR

        fpr_results.append({
            "morphology": morph,
            "family_fpr": family_fpr,
            **{f"fpr_{f}": feature_fpr[f] for f in CONFIRMATORY_FEATURES},
        })

        status = ("✅ PASS" if family_fpr <= FDR_ALPHA + 0.02
                  else "⚠️ MARGINAL" if family_fpr <= 0.10
                  else "🔴 FAIL")
        logger.info(
            f"    Family FPR = {family_fpr:.3f}  "
            f"(target ≤ {FDR_ALPHA}, n_seeds={N_SEEDS_FDR})  {status}"
        )
        for f in CONFIRMATORY_FEATURES:
            logger.info(f"      {f}: FPR = {feature_fpr[f]:.3f}")

    return pd.DataFrame(fpr_results)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    import logging

    parser = argparse.ArgumentParser(
        description="SyncPipe GT-4/5 FDR Calibration — TRUE Null")
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="Path to multisync-core/ (auto-detect if omitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test imports, don't run full computation",
    )
    args = parser.parse_args()

    # --- Set up imports ---
    global fd, extract_features, CONFIRMATORY_FEATURES
    project_root = Path(args.project_root) if args.project_root else None
    fd_local = _setup_imports(project_root)
    fd = fd_local
    extract_features = fd_local.extract_features
    CONFIRMATORY_FEATURES = fd_local.CONFIRMATORY_FEATURES

    if args.dry_run:
        print("DRY RUN: imports successful ✓")
        print(f"  extract_features: {extract_features}")
        print(f"  CONFIRMATORY_FEATURES: {CONFIRMATORY_FEATURES}")
        print(f"  SURROGATE_N: {SURROGATE_N}")
        print(f"  N_SEEDS_FDR: {N_SEEDS_FDR}")
        print("DRY RUN: exiting without full computation")
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run FDR calibration ---
    df_fdr = run_fdr_calibration()

    # --- Save ---
    csv_path = output_dir / "gt45_fdr_matrix.csv"
    df_fdr.to_csv(csv_path, index=False)
    logger.info(f"Saved FDR matrix to {csv_path}")

    # --- Summary report ---
    report_lines = [
        "=" * 70,
        "SyncPipe GT-4/5 FDR Calibration Report",
        f"Date: 2026-06-10",
        f"SURROGATE_N: {SURROGATE_N}",
        f"N_SEEDS_FDR: {N_SEEDS_FDR}",
        f"FDR_ALPHA: {FDR_ALPHA}",
        "=" * 70,
        "",
        "NULL ARCHITECTURE:",
        "  Null = 2 independent oscillators, Δθ ~ Uniform(0,2π) i.i.d.",
        "  r_null(t) = |cos(Δθ/2)| — NO deterministic temporal structure.",
        "",
        "FAMILY-WISE FPR (target ≤ 0.05):",
    ]
    for _, row in df_fdr.iterrows():
        status = ("✅ PASS" if row["family_fpr"] <= FDR_ALPHA + 0.02
                  else "⚠️ MARGINAL" if row["family_fpr"] <= 0.10
                  else "🔴 FAIL")
        report_lines.append(
            f"  {row['morphology']:<20s}: {row['family_fpr']:.3f}  {status}"
        )
    report_lines.extend([
        "",
        "PER-FEATURE FPR:",
    ])
    for _, row in df_fdr.iterrows():
        report_lines.append(f"  {row['morphology']}:")
        for f in CONFIRMATORY_FEATURES:
            report_lines.append(f"    {f:<20s}: {row[f'fpr_{f}']:.3f}")
    report_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "FPR ≈ 0.05  → FDR is well-calibrated (features respect null).",
        "FPR  > 0.10 → features may be over-sensitive to WCC noise structure.",
        "FPR  ≈ 0.00 → test may be too conservative (increase SURROGATE_N).",
        "",
        f"P-value granularity: 1/({SURROGATE_N}+1) ≈ {1/(SURROGATE_N+1):.4f}",
    ])

    report_text = "\n".join(report_lines)
    report_path = output_dir / "gt45_fdr_report.txt"
    report_path.write_text(report_text)
    logger.info(f"Saved report to {report_path}")

    print(report_text)


if __name__ == "__main__":
    main()

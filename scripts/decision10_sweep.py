"""DECISION-10 threshold recalibration: 30-seed sine-vs-noise sweep.

Calibrates ``LEAKAGE_DELTA_AUC_THRESHOLD`` for the CURRENT joint feature set
(DECISION-10)::

    FEATURE_NAMES = [onset_latency, rise_time, peak_amplitude,
                     recovery_time, dwell_time, switching_rate]

with ``mean_synchrony`` riding its own AR-baseline channel (not in the joint
model).  The 0.30 threshold was calibrated under the OLD feature set
(sine delta_AUC ~0.366, noise ~0).  The new AR baseline is stronger, so the
joint-vs-baseline gap shrank (sine delta_AUC ~0.273).  We re-sweep to pick a
threshold that (a) still flags perfectly-predictable structured signals
(sine) as ``leakage_suspected`` and (b) never flags structureless noise.

Method (mirrors tests/test_core.py leakage-audit harness):
  * Per seed: generate a sine-wave series (controlled amplitude / frequency /
    phase, simulating real synchrony structure) and an independent Gaussian
    noise series (no structure).
  * Run ``rolling_origin_cv`` on each (single 1-D synchrony series), recording
    ``mean_delta_auc`` (joint - max(naive, AR)) and the leakage ``warning``.
  * Nonlinear baselines (RandomForest / SVM) are disabled for the sweep — they
    do not affect ``mean_delta_auc`` and only add compute.

Why single-series ``rolling_origin_cv`` (intra mode): that is exactly the
calibration surface the SSoT docstring describes ("sine wave ... vs random
noise") and the surface the leakage-audit tests exercise.  The same
``mean_delta_auc`` + ``LEAKAGE_DELTA_AUC_THRESHOLD`` comparison applies in both
intra and cross_modal modes.

Usage:
    python scripts/decision10_sweep.py [N_SEEDS] [--log PATH]

Outputs a JSON summary to stdout (and the log) with the sine / noise
delta_AUC distributions and a recommended threshold.
"""

from __future__ import annotations

import json
import sys
import time
import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from syncpipe.prediction import rolling_origin_cv

# --- Sweep hyperparameters (consistent across sine & noise = a clean pair) ---
# Calibration surface chosen to be directly comparable to the leakage-audit
# tests in tests/test_core.py and to the cited DECISION-10 numbers
# (sine delta_AUC ~0.27 under the NEW joint feature set).  A 30-seed sweep
# over a FIXED canonical sine (controlled amplitude/frequency = "real
# synchrony structure" proxy) plus 30 independent Gaussian-noise nulls.
#
# Why a fixed canonical sine instead of random frequency: a pure sine's
# delta_AUC under the current pipeline is dominated by window/period/phase
# alignment artifacts (a uniform-frequency Monte-Carlo spans -0.6..+0.4 and
# overlaps the noise null).  A controlled period-80 sine is the SSoT's
# definition of "perfectly autocorrelated / leakage-like" (the audit test
# uses exactly this signal); we sweep its phase to characterize the positive
# control's spread.
N_SAMPLES = 2000          # series length
WINDOW_SIZE = 60          # > ONSET window requirement; stable feature extraction
N_SPLITS = 3
GAP = 5
HZ = 1.0
THRESHOLD = 0.0           # label = mean of future windows (balanced)

SINE_PERIOD = 80.0        # canonical, controlled (matches audit test)
SINE_AMP = 1.0            # canonical, controlled
RNG_BASE = 5000           # noise seed offset (independent of test RNG)


@dataclass
class SeedResult:
    seed: int
    kind: str                      # "sine" | "noise"
    mean_delta_auc: float
    warning: Optional[str]
    n_samples: int
    n_features_used: int
    n_folds: int


def _make_sine(seed: int) -> np.ndarray:
    # Controlled amplitude/frequency; phase swept deterministically across
    # the full [0, 2pi) range to characterize the positive-control spread.
    phase = 2.0 * np.pi * (seed / 30.0)
    t = np.arange(N_SAMPLES, dtype=float)
    return SINE_AMP * np.sin(2.0 * np.pi * t / SINE_PERIOD + phase)


def _make_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_BASE + seed)
    return rng.standard_normal(N_SAMPLES)


def _run_one(series: np.ndarray, seed: int, kind: str, log) -> SeedResult:
    pred = rolling_origin_cv(
        series,
        window_size=WINDOW_SIZE,
        hz=HZ,
        n_splits=N_SPLITS,
        gap=GAP,
        threshold=THRESHOLD,
        seed=seed,
        pair_name=f"{kind}_seed{seed}",
    )
    res = SeedResult(
        seed=seed,
        kind=kind,
        mean_delta_auc=float(pred.mean_delta_auc),
        warning=pred.warning,
        n_samples=int(pred.n_samples),
        n_features_used=int(pred.n_features_used),
        n_folds=len(pred.folds),
    )
    log(f"  {kind:5s} seed={seed:2d} delta={res.mean_delta_auc:+.4f} "
        f"warn={res.warning} folds={res.n_folds} nfeat={res.n_features_used}")
    return res


def _pct(vals: List[float], p: float) -> float:
    return float(np.percentile(vals, p))


def _summarize(vals: List[float]) -> dict:
    vals = sorted(vals)
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "p05": _pct(vals, 5),
        "p25": _pct(vals, 25),
        "p75": _pct(vals, 75),
        "p95": _pct(vals, 95),
    }


def recommend_threshold(sine_deltas: List[float], noise_deltas: List[float]) -> dict:
    """Pick a threshold maximizing separation with conservative margins.

    Design goal (from SSoT docstring): the threshold must flag perfectly
    predictable structured signals (sine) as ``leakage_suspected`` yet never
    flag structureless noise.  We therefore want TPR(sine) high and
    FPR(noise) ~0.

    The valid threshold region is the GAP between the noise upper tail and the
    sine lower tail: T in (noise_max, sine_min) yields TPR = 100%, FPR = 0%.
    We pick the point in that gap that MAXIMIZES the margin to both sides
    (midpoint of the realized gap), which is the most robust operating point.
    We also report the original-calibration analog (~0.07 below the sine
    median) for comparison.
    """
    sine = sorted(sine_deltas)
    noise = sorted(noise_deltas)
    sine_min = float(min(sine))
    sine_max = float(max(sine))
    sine_median = float(np.median(sine))
    noise_min = float(min(noise))
    noise_max = float(max(noise))

    def tpr(tp_t: float) -> float:
        return float(np.mean([d > tp_t for d in sine]))

    def fpr(fp_t: float) -> float:
        return float(np.mean([d > fp_t for d in noise]))

    # Gap between the two empirical distributions.
    gap_lo = noise_max
    gap_hi = sine_min
    midpoint = (gap_lo + gap_hi) / 2.0

    # Valid region: T > noise_max (FPR=0) and T < sine_min (TPR=100%).
    # Choose the midpoint for max robustness; round to 2 dp for the SSoT.
    chosen = round(midpoint, 2)

    # Sanity: if rounding pushed T out of the valid gap, snap back inside.
    if chosen <= gap_lo or chosen >= gap_hi:
        chosen = midpoint

    # Original-calibration analog: sine_median - 0.07 (mirrors the 0.37->0.30
    # margin that the old 0.30 threshold used).
    analog = round(max(0.05, sine_median - 0.07), 2)

    cands = {
        "gap_midpoint": {
            "threshold": float(chosen), "tpr_sine": tpr(chosen),
            "fpr_noise": fpr(chosen),
        },
        "median_minus_0.07": {
            "threshold": float(analog), "tpr_sine": tpr(analog),
            "fpr_noise": fpr(analog),
        },
    }

    return {
        "sine_min": sine_min,
        "sine_max": sine_max,
        "sine_median": sine_median,
        "noise_min": noise_min,
        "noise_max": noise_max,
        "gap": [gap_lo, gap_hi],
        "candidates": cands,
        "recommended_threshold": float(chosen),
        "recommended_tpr_sine": tpr(chosen),
        "recommended_fpr_noise": fpr(chosen),
        "analog_threshold": float(analog),
        "analog_tpr_sine": tpr(analog),
        "analog_fpr_noise": fpr(analog),
    }


def main() -> dict:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    log_path = "/tmp/syncpipe_decision10.log"

    # Disable nonlinear baselines for speed (they do not affect mean_delta_auc).
    import syncpipe.prediction as _pred
    _pred._nonlinear_model_factories = lambda seed=42: {}  # type: ignore[assignment]

    # Logging to file + stdout
    logger = logging.getLogger("decision10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    def log(msg: str) -> None:
        logger.info(msg)

    log("=" * 78)
    log(f"DECISION-10 sweep start: n_seeds={n_seeds} "
        f"N={N_SAMPLES} win={WINDOW_SIZE} splits={N_SPLITS} gap={GAP}")
    log(f"FEATURE_NAMES (NEW joint set) used by rolling_origin_cv:")
    from syncpipe.prediction import FEATURE_NAMES
    log(f"  {FEATURE_NAMES}")
    log("Nonlinear baselines disabled for sweep speed.")
    t0 = time.time()

    sine_results: List[SeedResult] = []
    noise_results: List[SeedResult] = []
    for seed in range(n_seeds):
        log(f"-- seed {seed} --")
        sine_results.append(_run_one(_make_sine(seed), seed, "sine", log))
        noise_results.append(_run_one(_make_noise(seed), seed, "noise", log))

    elapsed = time.time() - t0
    sine_deltas = [r.mean_delta_auc for r in sine_results]
    noise_deltas = [r.mean_delta_auc for r in noise_results]

    sine_sum = _summarize(sine_deltas)
    noise_sum = _summarize(noise_deltas)
    rec = recommend_threshold(sine_deltas, noise_deltas)

    log(f"Sweep done in {elapsed:.1f}s")
    log(f"SINE  delta_AUC: {json.dumps(sine_sum)}")
    log(f"NOISE delta_AUC: {json.dumps(noise_sum)}")
    log(f"RECOMMENDATION: {json.dumps(rec)}")

    # Any invalid runs?
    bad = [r for r in sine_results + noise_results
           if r.warning in ("data_too_short_for_cv", "insufficient_samples",
                            "class_imbalance", "no_valid_folds")]
    if bad:
        log(f"WARNING: {len(bad)} runs returned a hard-stop warning:")
        for r in bad:
            log(f"  {r.kind} seed={r.seed} warn={r.warning} folds={r.n_folds}")

    summary = {
        "n_seeds": n_seeds,
        "params": {
            "N_SAMPLES": N_SAMPLES, "WINDOW_SIZE": WINDOW_SIZE,
            "N_SPLITS": N_SPLITS, "GAP": GAP, "HZ": HZ, "THRESHOLD": THRESHOLD,
        },
        "sine": sine_sum,
        "noise": noise_sum,
        "recommendation": rec,
        "elapsed_sec": elapsed,
        "invalid_runs": len(bad),
    }
    log("SUMMARY_JSON " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    out = main()
    print("\nFINAL_SUMMARY " + json.dumps(out, indent=2))

#!/usr/bin/env python
"""Autocorrelation-robustness calibration for the signal-level existence audit.

Measures the false-positive rate (FPR) and power of the IAAFT synchrony-
existence audit under two noise regimes:

  - ``white`` — unit-variance white noise (the earlier GT regime).
  - ``ar1``   — stationary AR(1) noise, lag-1 autocorrelation ``phi``,
                matching real low-frequency physiological envelopes.

The key question is whether the existence audit holds its ~5% false-positive
rate on *autocorrelated* (realistic) data, since that is the regime where IAAFT
must actually preserve autocorrelation to be a valid null.

Usage:
    python scripts/run_autocorr_robustness.py            # default (fast) scale
    python scripts/run_autocorr_robustness.py --n-dyads 100 --surrogate-n 199 \
        --seed 0 -o artifacts/autocorr_robustness.json   # larger scale

Output: a JSON summary under `artifacts/autocorr_robustness.json` (default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from syncpipe.validation.autocorr_robustness import run_autocorr_robustness  # noqa: E402


def _sanitize(obj):
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-dyads", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--surrogate-n", type=int, default=99)
    p.add_argument("--phi", type=float, default=0.9)
    p.add_argument("--coupling", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", default="artifacts/autocorr_robustness.json")
    args = p.parse_args(argv)

    res = run_autocorr_robustness(
        noise_models=("white", "ar1"),
        phi=args.phi,
        coupling=args.coupling,
        n_dyads=args.n_dyads,
        n_samples=args.n_samples,
        window=args.window,
        surrogate_n=args.surrogate_n,
        seed=args.seed,
    )

    print("Autocorrelation-robustness calibration (IAAFT existence audit)\n")
    print(f"  {'regime':<7} {'FPR':>7} {'power':>7}   (alpha=0.05)")
    print("  " + "-" * 30)
    for nm in ("white", "ar1"):
        fpr = res["fpr"][nm]
        pwr = res["power"][nm]
        print(f"  {nm:<7} {fpr['fpr']:>7.3f} {pwr['power']:>7.3f}")

    ar1_fpr = res["fpr"]["ar1"]["fpr"]
    print("\nReading:")
    print(f"  - FPR on autocorrelated (ar1) data = {ar1_fpr:.3f}: "
          f"{'≈ nominal (good)' if abs(ar1_fpr - 0.05) < 0.05 else 'check calibration'}")
    print("  - A valid existence null keeps FPR ≈ alpha on realistic data; the")
    print("    ar1 regime is the one that exercises IAAFT's autocorrelation")
    print("    preservation, which white noise does not.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(res), indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

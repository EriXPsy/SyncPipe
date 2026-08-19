#!/usr/bin/env python
"""Validate peak_amplitude duration bias and finite-record dependability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from syncpipe.dynamic_features import sliding_window_wcc
from syncpipe.validation.discriminant import generate_discriminant_pair
from syncpipe.validation.peak_duration import (
    duration_dependability_curve,
    moving_block_peak_stability,
    simulate_peak_duration_bias,
)


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-dyads", type=int, default=30)
    p.add_argument("--n-replicates", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--durations", type=int, nargs="+", default=[120, 300, 600])
    p.add_argument("--block-length", type=int, default=60)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--out-dir", default="artifacts/peak_duration_validation")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bias_values, bias_summary = simulate_peak_duration_bias(
        durations=args.durations,
        n_replicates=args.n_replicates,
        window_size=args.window,
        seed=args.seed,
    )
    bias_values.to_csv(out / "null_duration_values.csv", index=False)
    bias_summary.to_csv(out / "null_duration_summary.csv", index=False)

    traces = {}
    strengths = np.linspace(0.08, 0.32, args.n_dyads)
    for d, strength in enumerate(strengths):
        a, b = generate_discriminant_pair(
            "reciprocal_var",
            n_samples=args.n_samples,
            seed=args.seed + d,
            interaction_strength=float(strength),
        )
        traces[f"dyad_{d:03d}"] = sliding_window_wcc(a, b, args.window)
    blocks, dependability = duration_dependability_curve(
        traces, durations=args.durations
    )
    blocks.to_csv(out / "dependability_blocks.csv", index=False)
    dependability.to_csv(out / "dependability_summary.csv", index=False)

    first_trace = next(iter(traces.values()))
    stability = moving_block_peak_stability(
        first_trace,
        block_length=args.block_length,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    (out / "stability_example.json").write_text(
        json.dumps(_json_safe(stability), indent=2), encoding="utf-8"
    )

    manifest = {
        "descriptor": "peak_amplitude",
        "claim": "validation_only_no_new_descriptor",
        "parameters": vars(args),
        "dependability_ground_truth": {
            "model": "reciprocal_var",
            "interaction_strength_range": [float(strengths[0]), float(strengths[-1])],
            "purpose": "known between-dyad heterogeneity",
        },
        "warning": (
            "The block-resampling stability interval is not a confidence "
            "interval for a population maximum."
        ),
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
    )
    print(bias_summary.to_string(index=False))
    print("\nDependability\n", dependability.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

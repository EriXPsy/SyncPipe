#!/usr/bin/env python
"""Run adversarial discriminant-validity controls for peak_amplitude."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from syncpipe.validation.discriminant import (
    SCENARIO_METADATA,
    run_discriminant_benchmark,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-replicates", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--surrogate-n", type=int, default=99)
    p.add_argument("--phi", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIO_METADATA),
        choices=list(SCENARIO_METADATA),
    )
    p.add_argument("-o", "--out-dir", default="artifacts/discriminant_validity")
    args = p.parse_args(argv)

    values, summary, controls = run_discriminant_benchmark(
        scenarios=args.scenarios,
        n_replicates=args.n_replicates,
        n_samples=args.n_samples,
        window_size=args.window,
        surrogate_n=args.surrogate_n,
        phi=args.phi,
        seed=args.seed,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    values.to_csv(out / "replicate_results.csv", index=False)
    summary.to_csv(out / "scenario_summary.csv", index=False)
    controls.to_csv(out / "design_control_summary.csv", index=False)
    (out / "MANIFEST.json").write_text(
        json.dumps({
            "descriptor": "peak_amplitude",
            "claim": "adversarial_validation_not_construct_proof",
            "parameters": vars(args),
            "scenario_metadata": SCENARIO_METADATA,
        }, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nDesign controls\n", controls.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

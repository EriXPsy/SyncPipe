#!/usr/bin/env python3
"""
Level 3 surrogate-based significance testing runner.

Three sub-experiments:
    3a (Type I error):  coupling=0.0, 5 noise levels, 30 seeds
    3b (Power):         coupling=0.3, 5 noise levels, 30 seeds
    3c (Sanity):        coupling=0.7, 5 noise levels, 10 seeds

Usage::

    python -m scripts.run_level3_validation --out level3_outputs
    python -m scripts.run_level3_validation --out level3_outputs --only 3a
    python -m scripts.run_level3_validation --out level3_outputs --surrogate prtf
    python -m scripts.run_level3_validation --out level3_outputs --surrogate iaaft
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# HACK: allow running from multisync-core/ without installing
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multisync.validation import (
    Level3Config,
    run_level3_grid,
    apply_bh_fdr_within_noise,
    summarise_level3,
)


def run_sub_experiment(
    name: str,
    cfg: Level3Config,
    out_dir: Path,
) -> None:
    """
    Run one Level 3 sub-experiment and write CSV outputs.

    Parameters
    ----------
    name : str
        Sub-experiment label (e.g. ``"3a_typeI"``).
    cfg : Level3Config
        Experiment configuration.
    out_dir : Path
        Output directory (created if missing).
    """
    t0 = time.perf_counter()
    method = cfg.surrogate_method
    print(f"\n=== Sub-experiment {name} ({method}): "
          f"{cfg.n_cells} cells, "
          f"{cfg.n_surrogates} surrogates each ===")

    df = run_level3_grid(cfg)
    df_fdr = apply_bh_fdr_within_noise(df, q=cfg.fdr_q)
    summary = summarise_level3(df_fdr)

    stem = f"level3_{name}_{method}"
    df.to_csv(out_dir / f"{stem}_raw.csv", index=False)
    df_fdr.to_csv(out_dir / f"{stem}_fdr.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"  -> {out_dir}/{stem}_raw.csv    ({len(df)} rows)")
    print(f"  -> {out_dir}/{stem}_summary.csv")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print()
    # Print summary with truncation for readability
    with pd.option_context("display.max_columns", 12,
                           "display.width", 120,
                           "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Level 3 surrogate significance tests.",
    )
    parser.add_argument(
        "--out", type=str, default="level3_outputs",
        help="Output directory (default: level3_outputs/)",
    )
    parser.add_argument(
        "--only", type=str, default="all",
        choices=["all", "3a", "3b", "3c"],
        help="Run only one sub-experiment (default: all)",
    )
    parser.add_argument(
        "--surrogate", type=str, default="prtf",
        choices=["prtf", "iaaft"],
        help="Surrogate method: prtf (primary) or iaaft (robustness, Appendix C)",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=999,
        help="Number of surrogates per cell (default: 999)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Level 3a: Type I error (coupling = 0.0) ----
    if args.only in ("all", "3a"):
        cfg_3a = Level3Config(
            couplings=(0.0,),
            seeds=tuple(range(1000, 1030)),
            n_surrogates=args.n_surrogates,
            surrogate_method=args.surrogate,
        )
        run_sub_experiment("3a_typeI", cfg_3a, out_dir)

    # ---- Level 3b: Power (coupling = 0.3) ----
    if args.only in ("all", "3b"):
        cfg_3b = Level3Config(
            couplings=(0.3,),
            seeds=tuple(range(1000, 1030)),
            n_surrogates=args.n_surrogates,
            surrogate_method=args.surrogate,
        )
        run_sub_experiment("3b_power", cfg_3b, out_dir)

    # ---- Level 3c: Sanity check (coupling = 0.7, 10 seeds) ----
    if args.only in ("all", "3c"):
        cfg_3c = Level3Config(
            couplings=(0.7,),
            seeds=tuple(range(1000, 1010)),   # only 10 seeds
            n_surrogates=args.n_surrogates,
            surrogate_method=args.surrogate,
        )
        run_sub_experiment("3c_sanity", cfg_3c, out_dir)


if __name__ == "__main__":
    main()

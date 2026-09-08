#!/usr/bin/env python3
"""Convergent-validity channel: SyncPipe WCC descriptors vs PyRQA measures.

CONSTRUCT_VALIDITY.md §4.3 requires comparing SyncPipe descriptors against
"independently implemented, theoretically related measures".  This script
computes, over a battery of synthetic dyads with graded coupling:

- SyncPipe side: mean_synchrony, dwell_time, switching_rate extracted from
  the zero-lag WCC trace (canonical smoothed-trace pipeline);
- PyRQA side (Apache-2.0, optional dependency ``pip install syncpipe[rqa]``):
  standard RQA measures (determinism, laminarity, average diagonal length,
  trapping time) on the SAME WCC trace — both sides quantify temporal
  organization of the synchrony trace, so moderate (not perfect) convergence
  is the expected signature.

and writes a Spearman correlation matrix plus the per-dyad table to
``artifacts/crqa_convergence/``.

PyRQA is OPTIONAL by design (GPL-3 传染性 rules keep pyspi/pyphysio out of
the dependency tree entirely; PyRQA is Apache-2.0 so it *may* be a
dependency, but it pulls in PyOpenCL, so it stays optional).  Without
PyRQA the script prints installation guidance and exits cleanly — the
SyncPipe-side descriptor table is still produced.

Usage:
    python scripts/run_crqa_convergence.py --n-dyads 24 -o artifacts/crqa_convergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from syncpipe.dynamic_features import sliding_window_wcc, extract_dynamic_features  # noqa: E402

WINDOW_SIZE = 60
HZ = 10.0


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _synthetic_dyads(n_dyads: int, n: int = 900, seed: int = 20260908):
    """Yield (label, x, y) with graded coupling: shared AR component whose
    weight grows from 0 (independent) to 0.9 (strong coupling)."""
    rng = np.random.default_rng(seed)
    for k in range(n_dyads):
        coupling = 0.9 * k / max(n_dyads - 1, 1)
        base = _ar1(rng, n, phi=0.5)
        x = base * coupling + 0.7 * _ar1(rng, n, phi=0.3)
        y = base * coupling + 0.7 * _ar1(rng, n, phi=0.3)
        yield f"dyad_{k:03d}_c{coupling:.2f}", x, y


def syncpipe_descriptors(x: np.ndarray, y: np.ndarray) -> dict:
    wcc = sliding_window_wcc(x, y, window_size=WINDOW_SIZE, hz=HZ)
    feats = extract_dynamic_features(wcc, hz=HZ)
    return {
        "mean_synchrony": float(np.nanmean(wcc)),
        "dwell_time": float(getattr(feats, "dwell_time", np.nan)),
        "switching_rate": float(getattr(feats, "switching_rate", np.nan)),
    }


def pyrqa_measures(wcc: np.ndarray) -> dict:
    """RQA on the WCC trace (auto-recurrence of the synchrony trace).

    Settings are fixed a priori: embedding dimension 1 (the trace is already
    a scalar index series), delay 1, fixed-radius neighbourhood at 10% of
    the trace's value range, Theiler window 1.
    """
    from pyrqa.computation import RQAComputation
    from pyrqa.metric import EuclideanMetric
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.settings import Settings

    trace = np.asarray(wcc, dtype=float)
    trace = trace[np.isfinite(trace)]
    radius = 0.1 * float(trace.max() - trace.min())
    settings = Settings(
        trace,
        embedding_dimension=1,
        time_delay=1,
        neighbourhood=FixedRadius(radius),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
        min_diagonal_line_length=2,
        min_vertical_line_length=2,
        min_white_vertical_line_length=2,
    )
    result = RQAComputation.create(settings, verbose=False).run()
    return {
        "rqa_determinism": float(result.determinism),
        "rqa_laminarity": float(result.laminarity),
        "rqa_avg_diag": float(result.average_diagonal_line),
        "rqa_trapping_time": float(result.trapping_time),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-dyads", type=int, default=24)
    ap.add_argument("-o", "--output", default="artifacts/crqa_convergence")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyrqa  # noqa: F401

        have_pyrqa = True
    except ImportError:
        have_pyrqa = False
        print(
            "[skip] PyRQA is not installed — SyncPipe-side descriptor table "
            "will still be written.  To enable the RQA convergence channel:  "
            "pip install syncpipe[rqa]  (requires a working OpenCL runtime)"
        )

    rows = []
    for label, x, y in _synthetic_dyads(args.n_dyads):
        row = {"dyad": label, **syncpipe_descriptors(x, y)}
        if have_pyrqa:
            wcc = sliding_window_wcc(x, y, window_size=WINDOW_SIZE, hz=HZ)
            row.update(pyrqa_measures(wcc))
        rows.append(row)

    df = pd.DataFrame(rows)
    table_path = out_dir / "per_dyad_table.csv"
    df.to_csv(table_path, index=False)

    report: dict = {
        "n_dyads": int(len(df)),
        "window_size": WINDOW_SIZE,
        "pyrqa_used": bool(have_pyrqa),
        "table": str(table_path),
    }

    if have_pyrqa:
        corr = df.drop(columns=["dyad"]).corr(method="spearman")
        corr_path = out_dir / "spearman_matrix.csv"
        corr.to_csv(corr_path)
        report["spearman_matrix"] = str(corr_path)
        print(corr.round(3).to_string())
        print(
            "\nInterpretation: moderate positive convergence between "
            "dwell_time/switching_rate and rqa_determinism/laminarity is the "
            "expected signature (§4.3); near-perfect convergence would "
            "suggest estimator redundancy instead."
        )
    else:
        print(f"SyncPipe descriptor table written to {table_path}")

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

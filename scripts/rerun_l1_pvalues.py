#!/usr/bin/env python3
"""BUG-4 remediation: regenerate L1 p-values with the protocol null.

Every historical L1 result (p_dwell_time / p_switching_rate) computed under
the pre-2026-09-08 default null (``state_shuffle``) is void: state_shuffle
re-orders whole elevated/baseline segments, making both statistics invariant
by construction (p == 1.0 on every input).  The protocol null for L1 is
WCC-level IAAFT (V1_PROTOCOL §10).

This script reads stored WCC traces, re-runs the L1 surrogate test with
``null_model='iaaft'`` for every trace, and writes a fresh p-value table.
It is deterministic given the master seed.

Usage:
    python scripts/rerun_l1_pvalues.py \
        --traces-csv artifacts/wcc_traces/lerique_wcc_traces.csv \
        -o artifacts/realdata_audit --surrogate-n 499
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from syncpipe.dynamic_features import wcc_surrogate_test  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-csv", required=True)
    ap.add_argument("-o", "--output", default="artifacts/realdata_audit")
    ap.add_argument("--surrogate-n", type=int, default=499)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N traces")
    args = ap.parse_args()

    src = REPO_ROOT / args.traces_csv
    df = pd.read_csv(src)
    if args.limit:
        df = df.head(args.limit)

    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for i, rec in df.iterrows():
        wcc = np.asarray(json.loads(rec["wcc_json"]), dtype=float)
        res = wcc_surrogate_test(
            wcc,
            hz=float(rec["hz"]),
            surrogate_n=args.surrogate_n,
            seed=args.seed + i,  # per-trace seeds: independent draws, reproducible
            raw_signals=None,
            null_model="iaaft",
        )
        rows.append({
            "id": rec["id"],
            "dyad": rec["dyad"],
            "modality": rec["modality"],
            "condition": rec["condition"],
            "hz": rec["hz"],
            "n_wcc_points": int(np.isfinite(wcc).sum()),
            "obs_dwell_time": res.get("obs_dwell_time"),
            "obs_switching_rate": res.get("obs_switching_rate"),
            "p_dwell_time": res.get("p_dwell_time"),
            "p_switching_rate": res.get("p_switching_rate"),
            "n_surrogates": args.surrogate_n,
            "null_model": "iaaft",
        })
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(df)}] {time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    table = out_dir / "realdata_l1_pvalues_iaaft_post_bug4.csv"
    out.to_csv(table, index=False)

    meta = {
        "source_traces": str(src),
        "n_traces": int(len(out)),
        "surrogate_n": args.surrogate_n,
        "master_seed": args.seed,
        "null_model": "iaaft",
        "reason": "BUG-4 remediation: pre-fix L1 p-values used state_shuffle, "
                  "under which dwell_time/switching_rate are invariant and "
                  "p == 1.0 by construction.  Regenerated with the protocol "
                  "null (WCC-level IAAFT, V1_PROTOCOL §10) on 2026-09-08.",
        "table": str(table),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out_dir / "realdata_l1_pvalues_iaaft_post_bug4.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Done: {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

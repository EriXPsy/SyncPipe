"""rerun_l0_existence.py — post-BUG-3 L0 existence recompute on real data.

Remediates the deferred item in METHOD_LOG §10: the prior real-data
existence-audit pass rates (e.g. ``<OSF>/Lerique-47n3p/multisync_results/
lerique_surrogate_summary.csv``) were produced by the pre-BUG-3 package,
whose shared master seed correlated surrogate draws across dyads
(r ≈ +0.33 on null peaks) and inflated the second-order group-null spread
~1.46x — a systematic conservative bias.

This script re-runs ONLY the L0 synchrony-existence audit through the
canonical governance path (``InferencePipeline.run_synchrony_existence_audit``,
per-pair derived seeds) with data prep mirrored exactly from
``realdata_full_new_pipeline.py`` (same loaders, hz, window, seed), so the
result is an apples-to-apples post-BUG-3 replacement of the prior pass-rate
tables.  Raw signals are read from ``E:/OSF`` (local copy of the OSF
datasets) — no network download.

Outputs (artifacts/realdata_audit/):
    realdata_l0_existence_post_bug3_<dataset>.csv   per-pair rows
    realdata_l0_existence_post_bug3.json            summary + gate results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from realdata_full_new_pipeline import (  # noqa: E402
    load_lerique,
    load_gordon,
    load_bizzego,
    load_han,
    load_andersen,
    _stratified_cap,
    SEED as PRIOR_SEED,
)
from syncpipe.pipeline_bridge import records_to_inference_inputs  # noqa: E402
from syncpipe.inference_pipeline import (  # noqa: E402
    InferencePipeline,
    _existence_gate_by_modality,
    PRIMARY_EXISTENCE_MODALITIES,
)
from syncpipe.feature_definitions import (  # noqa: E402
    EXISTENCE_GATE_ALPHA,
    PRIMARY_EXISTENCE_ENDPOINT,
)

OUT = ROOT / "artifacts" / "realdata_audit"

LOADERS = {
    "Lerique": load_lerique,
    "Gordon": load_gordon,
    "Bizzego": load_bizzego,
    "Han": load_han,
    "Andersen": load_andersen,
}

# Prior (pre-BUG-3) baseline tables, if present, for side-by-side deltas.
PRIOR_BASELINES = {
    "Lerique": Path("E:/OSF/Lerique-47n3p/multisync_results/lerique_surrogate_summary.csv"),
}


def _rate_table(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-pair rows into a modality x condition pass-rate table."""
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    pcols = [c for c in df.columns if c.startswith("sig_")]
    out = []
    for (mod, cond), g in ok.groupby(["modality", "condition"]):
        row = {"modality": mod, "condition": cond, "n_records": len(g)}
        for c in sorted(pcols):
            feat = c[len("sig_"):]
            n_sig = int(pd.Series(g[c]).fillna(False).astype(bool).sum())
            row[f"{feat}_sig"] = n_sig
            row[f"{feat}_pct"] = round(100.0 * n_sig / len(g), 1)
        out.append(row)
    return pd.DataFrame(out)


def _prior_lookup(prior_path: Path) -> dict[tuple[str, str], dict]:
    """Load the prior pass-rate table keyed by (modality, condition)."""
    if not prior_path.exists():
        return {}
    prior = pd.read_csv(prior_path)
    lut = {}
    for _, r in prior.iterrows():
        lut[(str(r["modality"]), str(r["condition"]))] = {
            c: r[c] for c in prior.columns
            if c not in ("modality", "condition")
        }
    return lut


def _compare(new_table: pd.DataFrame, prior_lut: dict) -> list[dict]:
    """Build the prior-vs-post comparison rows for shared features."""
    rows = []
    for _, r in new_table.iterrows():
        key = (str(r["modality"]), str(r["condition"]))
        prior = prior_lut.get(key)
        if not prior:
            continue
        for feat in ("mean_synchrony", "peak_amplitude", "dwell_time",
                     "switching_rate", "bimodality_coefficient"):
            new_pct = r.get(f"{feat}_pct")
            old_pct = prior.get(f"{feat}_pct")
            if new_pct is None or old_pct is None or pd.isna(old_pct):
                continue
            rows.append({
                "modality": key[0], "condition": key[1], "feature": feat,
                "prior_pct": float(old_pct), "post_bug3_pct": float(new_pct),
                "delta_pct": round(float(new_pct) - float(old_pct), 1),
            })
    return rows


def run_one(name: str, args) -> dict:
    t0 = time.time()
    recs, cfg = LOADERS[name]()
    recs = [r for r in recs if not getattr(r, "incomplete", False)]
    if args.max_dyads and args.max_dyads > 0:
        chosen = _stratified_cap(recs, args.max_dyads, seed=args.seed)
        recs = [r for r in recs if r.dyad_label in chosen]
    hz, window = cfg["hz"], cfg["window"]
    inputs = records_to_inference_inputs(
        recs, hz=hz, window_size=window,
        onset_threshold="session_pooled",
        design_condition=cfg.get("design_condition"),
    )
    pipe = InferencePipeline(
        inputs.features_df, hz=hz, wcc_window_sec=float(window) / hz,
        surrogate_n=args.surrogate_n, seed=args.seed, n_workers=args.n_workers,
    )
    audit = pipe.run_synchrony_existence_audit(
        inputs.raw_signals, wcc_window_size=window,
    )
    results = audit.get("results", {})

    rows = []
    for label, res in results.items():
        parts = str(label).split("__")
        dyad = parts[0] if parts else label
        mod = parts[1] if len(parts) > 1 else ""
        cond = parts[2] if len(parts) > 2 else ""
        row = {
            "label": label, "dyad": dyad, "modality": mod, "condition": cond,
            "status": res.get("status", "?"),
            "n_samples": res.get("n_samples"),
            "n_wcc": res.get("n_wcc"),
        }
        row.update({f"p_{k}": v for k, v in res.get("p_values", {}).items()})
        row.update({
            f"sig_{k}": v
            for k, v in res.get("per_feature_significant", {}).items()
        })
        rows.append(row)

    gate = _existence_gate_by_modality(
        results,
        primary_modalities=list(PRIMARY_EXISTENCE_MODALITIES),
        alpha=EXISTENCE_GATE_ALPHA,
        endpoint=PRIMARY_EXISTENCE_ENDPOINT,
    )

    table = _rate_table(rows)
    prior_lut = _prior_lookup(PRIOR_BASELINES.get(name, Path("?")))
    comparison = _compare(table, prior_lut)

    csv_path = OUT / f"realdata_l0_existence_post_bug3_{name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    return _clean({
        "dataset": name,
        "n_pairs_audited": len(results),
        "n_pairs_ok": sum(1 for r in rows if r["status"] == "ok"),
        "hz": hz, "window": window,
        "surrogate_n": args.surrogate_n, "seed": args.seed,
        "note": cfg.get("note"),
        "runtime_sec": round(time.time() - t0, 1),
        "gate": gate,
        "pass_rate_table": table.to_dict(orient="records"),
        "prior_vs_post_bug3": comparison,
        "per_pair_csv": str(csv_path),
    })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-BUG-3 L0 existence recompute from E:/OSF raw data.")
    ap.add_argument("--datasets", nargs="+", default=["Lerique"],
                    choices=sorted(LOADERS))
    ap.add_argument("--max-dyads", type=int, default=0,
                    help="0 = all dyads (stratified cap otherwise)")
    ap.add_argument("--surrogate-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=PRIOR_SEED)
    ap.add_argument("--n-workers", type=int, default=0,
                    help="0 = min(12, os.cpu_count())")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for the summary JSON filename, so parallel "
                         "single-dataset runs cannot clobber each other.")
    args = ap.parse_args()
    if args.n_workers <= 0:
        args.n_workers = min(12, os.cpu_count() or 1)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.datasets:
        print(f"\n===== L0 existence recompute: {name} =====", flush=True)
        try:
            summary[name] = run_one(name, args)
        except Exception as e:  # noqa: BLE001 — keep other datasets running
            import traceback
            traceback.print_exc()
            summary[name] = {"dataset": name, "error": f"{type(e).__name__}: {e}"}
        print(f"  done: {json.dumps(summary[name], default=str)[:400]}",
              flush=True)

    out_json = OUT / (
        "realdata_l0_existence_post_bug3"
        + (f"_{args.out_suffix}" if args.out_suffix else "") + ".json"
    )
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                   default=str))
    print(f"\nWrote {out_json}", flush=True)


if __name__ == "__main__":
    main()

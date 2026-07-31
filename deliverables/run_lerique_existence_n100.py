"""Lerique existence (L0 signal-level IAAFT) re-audit at surrogate_n=100.

The FAST-CONFIRMATION pipeline (realdata_full_new_pipeline.py) sets
SURROGATE_N=30, which makes the two-tailed Phipson-Smyth existence test
statistically incapable of reaching p<0.05 (min achievable two-tailed
p = 2/31 ~= 0.0645). That is why the canonical rerun reported
existence pass_rate=0.0 (0/176) -- a config artifact, NOT a finding.

The project's own docstring states the intended existence audit uses
surrogate_n=100 (design_controls.synchrony_existence_audit -> surrogate_n=100),
which gives min two-tailed p = 2/101 ~= 0.0198 < 0.05.  This script re-runs
ONLY the existence audit on the SAME raw Lerique signals at n=100 so the
real pass rate can be reported and reconciled with the historical
~56.7% EDA figure.

Output: artifacts/realdata_full/lerique_existence_n100.json
"""
from __future__ import annotations

import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import realdata_full_new_pipeline as rp  # noqa: E402
from multisync.inference_pipeline import InferencePipeline  # noqa: E402
from multisync.pipeline_bridge import records_to_inference_inputs  # noqa: E402

OUT = REPO / "artifacts" / "realdata_full"
OUT.mkdir(parents=True, exist_ok=True)

N = 100  # documented existence standard (not FAST-CONFIRMATION's 30)


def main():
    recs, cfg = rp.load_lerique()
    hz, window = cfg["hz"], cfg["window"]
    inputs = records_to_inference_inputs(
        recs, hz=hz, window_size=window,
        onset_threshold="session_pooled",
        design_condition=cfg.get("design_condition"),
    )
    pipe = InferencePipeline(
        inputs.features_df, hz=hz,
        wcc_window_sec=float(window) / hz,
        surrogate_n=N, seed=rp.SEED,
    )
    # ONLY the existence audit (fast) -- not the full chain.
    exist = pipe.run_synchrony_existence_audit(
        inputs.raw_signals, wcc_window_size=window,
        discontinuity_mask=inputs.discontinuity_mask,
    )
    results = exist.get("results", {})

    by_mod = defaultdict(lambda: {"n": 0, "sig": 0})
    by_mod_cond = defaultdict(lambda: {"n": 0, "sig": 0})
    by_mod_feat = defaultdict(lambda: defaultdict(lambda: {"n": 0, "sig": 0}))
    per_pair = []
    for label, pv in results.items():
        if not isinstance(pv, dict):
            continue
        parts = label.split("__")
        mod = parts[1] if len(parts) >= 3 else label
        cond = parts[2] if len(parts) >= 3 else "na"
        pfs = pv.get("per_feature_significant", {})
        any_sig = any(pfs.values())
        by_mod[mod]["n"] += 1
        by_mod[mod]["sig"] += int(any_sig)
        by_mod_cond[f"{mod}|{cond}"]["n"] += 1
        by_mod_cond[f"{mod}|{cond}"]["sig"] += int(any_sig)
        for f, s in pfs.items():
            by_mod_feat[mod][f]["n"] += 1
            by_mod_feat[mod][f]["sig"] += int(bool(s))
        per_pair.append({"label": label, "modality": mod, "condition": cond,
                         "per_feature_significant": pfs})

    summary = {
        "surrogate_n": N,
        "n_pairs_audited": len(results),
        "note": "n=100 matches the documented existence standard; "
                "FAST-CONFIRMATION's surrogate_n=30 is too small for the "
                "two-tailed test (min p=0.0645>0.05), which produced the "
                "artifactual 0/176.",
        "by_modality": {
            m: {"n_pairs": v["n"], "n_significant": v["sig"],
                "pass_rate": (v["sig"] / v["n"]) if v["n"] else None}
            for m, v in by_mod.items()
        },
        "by_modality_condition": {
            k: {"n_pairs": v["n"], "n_significant": v["sig"],
                "pass_rate": (v["sig"] / v["n"]) if v["n"] else None}
            for k, v in by_mod_cond.items()
        },
        "by_modality_feature": {
            m: {f: {"n": d["n"], "n_significant": d["sig"],
                    "pass_rate": (d["sig"] / d["n"]) if d["n"] else None}
                for f, d in feats.items()}
            for m, feats in by_mod_feat.items()
        },
    }
    out = {"dataset": "Lerique", "summary": summary, "per_pair": per_pair}
    (OUT / "lerique_existence_n100.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[existence n={N}] audited={len(results)} pairs", flush=True)
    print("by_modality pass_rate:", flush=True)
    for m, v in summary["by_modality"].items():
        print(f"  {m}: {v['n_significant']}/{v['n_pairs']} = "
              f"{v['pass_rate']:.3f}", flush=True)
    print("by_modality x feature:", flush=True)
    for m, feats in summary["by_modality_feature"].items():
        for f, d in feats.items():
            print(f"  {m}.{f}: {d['n_significant']}/{d['n']} = "
                  f"{d['pass_rate']:.3f}", flush=True)
    print(f"wrote {OUT / 'lerique_existence_n100.json'}", flush=True)


if __name__ == "__main__":
    main()

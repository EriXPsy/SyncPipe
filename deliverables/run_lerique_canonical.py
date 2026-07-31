"""Lerique-only canonical 3-pipeline rerun (post-fix confirmation).

Imports the FAST-CONFIRMATION analysis functions from
``realdata_full_new_pipeline.py`` WITHOUT modifying that project script, and
runs ONLY the Lerique dataset end-to-end through the current
ComputationPipeline + InferencePipeline + MorphologyAnalyzer.

This confirms that the STANDARD audited-evidence chain (L0 existence ->
L1 WCC -> L2 between-condition FDR -> morphology) reproduces on raw Lerique
signals under the post-fix code.  The gap fix (prediction.py) does NOT touch
this path, so this run is the "complete analysis" baseline that the
prediction-path gap check (run_lerique_prediction_gapcheck.py) complements.

Params match the FAST-CONFIRMATION defaults in realdata_full_new_pipeline.py
(surrogate_n=30, n_perm=2000, n_pseudo=8) so the run fits in a session, but we
run ALL available clean Lerique dyads (no 12-dyad cap) for a genuine rerun.

Output: artifacts/realdata_full/realdata_full_Lerique.json
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import realdata_full_new_pipeline as rp  # noqa: E402

OUT = REPO / "artifacts" / "realdata_full"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    recs, cfg = rp.load_lerique()
    print(f"[canonical] loaded {len(recs)} raw Lerique records "
          f"(status={cfg['status']})", flush=True)
    res = rp.run_dataset("Lerique", recs, cfg)
    out = rp._jsonify(res)
    (OUT / "realdata_full_Lerique.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    # compact console summary
    ex = res.get("existence", {})
    print(f"[canonical] error={res.get('error')}", flush=True)
    print(f"[canonical] existence pass_rate={ex.get('pass_rate')} "
          f"({ex.get('n_pairs_significant')}/{ex.get('n_pairs_audited')})",
          flush=True)
    l2 = res.get("l2_pooled", {})
    if isinstance(l2, dict) and "n_significant" in l2:
        print(f"[canonical] L2 pooled n_significant={l2['n_significant']} "
              f"({l2.get('condition_a')} vs {l2.get('condition_b')})", flush=True)
    pm = res.get("l2_per_modality", {})
    if isinstance(pm, dict):
        for mod, m in pm.items():
            if isinstance(m, dict) and "n_significant" in m:
                print(f"[canonical]   L2 {mod}: n_significant="
                      f"{m['n_significant']}", flush=True)
    ff = res.get("l2_full_family", {})
    if isinstance(ff, dict) and "n_significant" in ff:
        print(f"[canonical] L2 full-family(12feat) n_significant="
              f"{ff['n_significant']}", flush=True)
    print(f"[canonical] wrote {OUT / 'realdata_full_Lerique.json'}", flush=True)


if __name__ == "__main__":
    main()

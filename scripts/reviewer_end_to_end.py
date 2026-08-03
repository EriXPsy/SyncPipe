"""Reviewer end-to-end walkthrough of the THREE SyncPipe pipeline files.

Pipeline 1  feature_pipeline.py     -> consult / select features
Pipeline 2  computation_pipeline.py -> load -> WCC -> features -> DataFrame
Pipeline 3  inference_pipeline.py   -> audited evidence chain

This script is written *as a fresh reviewer would operate SyncPipe* on a
real dataset (Lerique-47n3p).  Because the published .mat files are not
shipped with the repo, it generates a FAITHFUL synthetic proxy using the
package's own ``multisync.synthetic.generate_ground_truth_dyad`` (true
inter-personal coupling in the TASK condition, ~zero coupling in REST),
which reproduces the exact data contract of
``multisync.realtest.lerique_2024.LeriqueDyadCondition``.

Pass ``--data-root <path/to/Lerique-47n3p>`` to use the REAL loader instead.

Run:
    python scripts/reviewer_end_to_end.py --out-dir artifacts/reviewer_audit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("reviewer")

TARGET_HZ = 1.0
WCC_WINDOW_SEC = 30.0
WCC_WINDOW_SIZE = int(round(WCC_WINDOW_SEC * TARGET_HZ))  # 30 samples
ONSET_THRESHOLD = 0.5
TASK_CONDITION = "task"
REST_CONDITION = "rest"


# ---------------------------------------------------------------------------
# Records (faithful to LeriqueDyadCondition contract)
# ---------------------------------------------------------------------------
@dataclass
class Record:
    dyad_label: str
    modality: str
    condition: str
    person_a: pd.DataFrame
    person_b: pd.DataFrame
    target_hz: float
    duration_sec: float
    incomplete: bool = False


def _make_synthetic_records(n_dyads: int = 10, seed: int = 0) -> List[Record]:
    """Build (dyad, modality, condition) records with genuine coupling in TASK.

    Uses the package's own coupled-signal generator so the WCC features and
    the synchrony-existence IAAFT audit behave exactly as on real data.
    """
    from multisync.synthetic import generate_ground_truth_dyad

    recs: List[Record] = []
    modalities = {"EDA": "behavior", "RESP": "neural"}
    for d in range(1, n_dyads + 1):
        for cond, coupling in ((REST_CONDITION, 0.05), (TASK_CONDITION, 0.70)):
            ds = generate_ground_truth_dyad(
                lead_modality="behavior", lag_modality="neural",
                coupling=coupling, duration_sec=180.0, hz=TARGET_HZ,
                noise_ratio=0.25, n_bursts=5, burst_sigma=4.0,
                gap_prob=0.0, morphology="identical",
                seed=1000 * d + (0 if cond == REST_CONDITION else 777),
            )
            for mod_out, mod_in in modalities.items():
                df = ds.modalities[mod_in]
                n = len(df)
                recs.append(Record(
                    dyad_label=f"pce{d:02d}", modality=mod_out, condition=cond,
                    person_a=df[["time", "person_a"]].rename(columns={"person_a": "value"}),
                    person_b=df[["time", "person_b"]].rename(columns={"person_b": "value"}),
                    target_hz=TARGET_HZ, duration_sec=float(n) / TARGET_HZ,
                ))
    return recs


def _load_real_records(data_root: Path, limit: int | None = None) -> List[Record]:
    from multisync.realtest.lerique_2024 import load_lerique_dataset

    raw = load_lerique_dataset(
        data_root=data_root, preprocess=True,
        drop_incomplete=False, drop_misaligned=True, drop_short_duration=True,
    )
    out: List[Record] = []
    for r in raw:
        if r.incomplete:
            continue
        if r.modality not in ("EDA", "RESP"):
            continue
        if r.condition not in ("rest1", "trials_concat"):
            continue
        out.append(Record(
            dyad_label=r.dyad_label, modality=r.modality, condition=r.condition,
            person_a=r.person_a, person_b=r.person_b,
            target_hz=r.target_hz, duration_sec=r.duration_sec,
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Stage 1 — Feature pipeline (consult / select)
# ---------------------------------------------------------------------------
def stage_feature_consult():
    from multisync.feature_pipeline import print_feature_table, recommend_features
    log.info("=== STAGE 1 (feature_pipeline): consult & select ===")
    print(print_feature_table())
    rec = recommend_features("general")
    log.info("FDR primary family (per SSoT): %s", rec["primary"])
    log.info("Reference comparator: %s", rec["reference"])
    return rec


# ---------------------------------------------------------------------------
# Stages 2+3 — bridge -> computation -> inference
# ---------------------------------------------------------------------------
def run_pipeline(recs, surrogate_n: int):
    from multisync.pipeline_bridge import records_to_inference_inputs
    from multisync.inference_pipeline import InferencePipeline
    from multisync.feature_definitions import FDR_FEATURES

    design_condition = TASK_CONDITION if any(
        r.condition == TASK_CONDITION for r in recs) else None

    log.info("=== STAGE 2+3 (computation + inference pipelines via bridge) ===")
    inputs = records_to_inference_inputs(
        recs, hz=TARGET_HZ, window_size=WCC_WINDOW_SIZE,
        onset_threshold="session_pooled", design_condition=design_condition,
    )
    log.info("features_df: %d rows x %d cols", *inputs.features_df.shape)
    log.info("raw_signals (existence audit): %d pairs", len(inputs.raw_signals))
    log.info("design_pairs (design control): %d units", len(inputs.design_pairs))

    pipe = InferencePipeline(
        features_df=inputs.features_df, hz=TARGET_HZ,
        wcc_window_sec=WCC_WINDOW_SEC, surrogate_n=surrogate_n, seed=42,
    )
    chain = pipe.run_audited_evidence_chain(
        raw_signals=inputs.raw_signals,
        wcc_window_size=WCC_WINDOW_SIZE,
        design_signal_pairs=inputs.design_pairs,
        condition_col=inputs.condition_col,
        dyad_col=inputs.dyad_col,
        feature_cols=list(FDR_FEATURES),
        fdr_alpha=0.05,
        n_permutations=2000,
    )
    print(pipe.summarize())
    print("\n--- v1 evidence-chain summary ---")
    print(chain["summary"])

    # Per-modality L2 (rigor on multimodal data)
    log.info("=== STAGE 3b: per-modality L2 (test_l2_by_modality) ===")
    by_mod = pipe.test_l2_by_modality(
        modality_col="modality", condition_col=inputs.condition_col,
        dyad_col=inputs.dyad_col, feature_cols=list(FDR_FEATURES),
        n_permutations=2000,
    )
    for mod, res in by_mod.items():
        log.info("  modality=%s -> %d significant FDR feature(s)",
                 mod, res.get("n_significant", 0))
    return pipe, chain, by_mod


def _markdown_report(recs, chain, by_mod, surrogate_n) -> str:
    se = chain.get("synchrony_existence", {})
    n_pairs = se.get("n_pairs", 0)
    design = chain.get("design_controls") or {}
    group = chain.get("group_condition_inference") or {}
    lines = [
        "# SyncPipe Reviewer Walkthrough — Audited Evidence Chain",
        "",
        f"- Records analysed: **{len(recs)}** (dyad × modality × condition units)",
        f"- WCC window: **{WCC_WINDOW_SEC:.0f} s** | onset threshold: **session_pooled** "
        f"| existence-surrogate n: **{surrogate_n}**",
        "",
        "## Step 1 — Synchrony-existence audit (signal-level IAAFT)",
        f"- Pairs audited: **{n_pairs}**.",
        "  A significant pair means aligned WCC features exceed what independent "
        "autocorrelated signals can produce. Necessary-but-not-sufficient for "
        "interpersonal coupling; shared-stimulus / co-presence need Step 2.",
        "",
        "## Step 2 — Design-control audit (pseudo-pair + time-shift)",
        f"- Units audited: **{design.get('n_dyads', 0)}**.",
        "  Pseudo-pair asks whether real partners exceed mismatched partners; "
        "time-shift asks whether the effect depends on the original alignment.",
        "",
        "## Step 3 — Group condition inference (dyad-paired permutation + BH-FDR)",
    ]
    # 1c: `group` is always {modality: l2_dict}. The previous version read
    # group["n_significant"] directly, which does not exist on the modality-keyed
    # shape, so this report silently claimed "0 significant" for every
    # multimodal dataset. Walk the modalities instead.
    lines.append(
        "L2 is evaluated per modality; there is no cross-modality pooled number, "
        "because pooling across modalities can cancel opposing real effects."
    )
    for mod in sorted(group.keys(), key=str):
        sub = group[mod]
        if not isinstance(sub, dict) or "error" in sub:
            err = sub.get("error", "invalid") if isinstance(sub, dict) else "invalid"
            lines.append(f"- **{mod}**: not testable ({err})")
            continue
        lines.append(
            f"- **{mod}**: {int(sub.get('n_significant', 0))}"
            f"/{int(sub.get('n_tested', 0))} FDR feature(s) condition-differentiated"
        )
    lines += [
        "",
        "### Per-modality L2 re-run (explicit `test_l2_by_modality` call)",
    ]
    for mod, res in by_mod.items():
        lines.append(
            f"- **{mod}**: {res.get('n_significant', 0)} significant FDR feature(s)")
    lines += [
        "",
        "## Interpretation guard-rail",
        "All positive findings are *audited evidence*, not causal proof. Shared-"
        "stimulus and co-presence alternatives require the Step-2 design controls "
        "and, for segmented shared-stimulus designs, the across-stimulus shuffle.",
        "",
        f"_Generated by `scripts/reviewer_end_to_end.py` (SyncPipe v1)._",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Path to Lerique-47n3p root (uses REAL loader).")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/reviewer_audit"))
    ap.add_argument("--n-dyads", type=int, default=14,
                    help="Synthetic dyads (ignored if --data-root given).")
    ap.add_argument("--surrogate-n", type=int, default=25,
                    help="IAAFT surrogate iterations per pair (existence audit).")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.data_root is not None:
        log.info("REAL mode: loading Lerique records from %s", args.data_root)
        recs = _load_real_records(args.data_root)
    else:
        log.info("SYNTHETIC mode: faithful coupled proxy (%d dyads)", args.n_dyads)
        recs = _make_synthetic_records(n_dyads=args.n_dyads)
    log.info("Records: %d", len(recs))

    stage_feature_consult()
    pipe, chain, by_mod = run_pipeline(recs, surrogate_n=args.surrogate_n)

    json_path = out / "reviewer_inference_results.json"
    # Combine the v1 chain JSON with the per-modality L2 results.
    base = json.loads(pipe.to_json() or "{}")
    base["per_modality_l2"] = {
        mod: {
            "n_significant": res.get("n_significant", 0),
            "condition_a": res.get("condition_a"),
            "condition_b": res.get("condition_b"),
            "per_feature": [
                {
                    "feature": r.feature, "observed_diff": r.observed_diff,
                    "p_raw": r.p_raw, "p_fdr": r.p_fdr,
                    "significant_05": r.significant_05, "n_dyads": r.n_dyads,
                }
                for r in res.get("per_feature", [])
            ],
        }
        for mod, res in by_mod.items()
    }
    json_path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", json_path)

    md = _markdown_report(recs, chain, by_mod, args.surrogate_n)
    md_path = out / "REVIEWER_RESULTS.md"
    md_path.write_text(md, encoding="utf-8")
    log.info("Wrote %s", md_path)

    log.info("DONE. Outputs in %s", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

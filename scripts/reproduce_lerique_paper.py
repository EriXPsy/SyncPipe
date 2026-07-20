"""
reproduce_lerique_paper.py
===========================

M2 reproducibility scaffold for Lerique et al. (2024).

Two modes:

  --fast : SYNTHETIC PROXY. Builds a handful of toy dyad signals
           (no real data, NO OSF access) and runs the canonical
           three-pipeline end-to-end:
               records  -> pipeline_bridge.records_to_inference_inputs
                        -> InferencePipeline.run_audited_evidence_chain
           so CI / reviewers can verify the pipeline wiring and the
           admissible-claims ceilings WITHOUT the protected dataset.
           This is a wiring/SANITY check, not a scientific claim.

  --pub  : REAL OSF DATA. Requires --data-root pointing at a local
           mirror of the Lerique-47n3p OSF component. Lazily imports
           the existing loader (multisync.realtest.lerique_2024
           .load_lerique_dataset) and runs the SAME canonical chain.
           Writes derived tables to artifacts/paper_lerique/.
           Errors clearly (no fabricated data) if --data-root is absent
           or the loader cannot read it.

Outputs (artifacts/paper_lerique/):
  reproduce_<mode>_features.csv   per-(dyad, modality, condition) table
  reproduce_<mode>_summary.json   chain version + admissible-claims summary

NOTE: This script fabricates NO OSF data value and NO license text.
Data-source placeholders live in artifacts/paper_lerique/MANIFEST.json.
The canonical (defensible) path is pipeline_bridge +
InferencePipeline.run_audited_evidence_chain (synchrony-existence audit
-> design controls -> group inference), using a single fixed/pooled onset
threshold — NOT the legacy per-dyad DynamicAnalyzer path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multisync.pipeline_bridge import records_to_inference_inputs
from multisync.inference_pipeline import InferencePipeline

logger = logging.getLogger("reproduce_lerique_paper")

# --- canonical, fixed/pooled parameters (NOT per-dyad) -------------------
OSET_THRESHOLD = 0.5
WCC_WINDOW_SEC = 30.0
TARGET_HZ = 1.0
WINDOW_SIZE = int(round(WCC_WINDOW_SEC * TARGET_HZ))
OUT_DIR = ROOT / "artifacts" / "paper_lerique"


# ---------------------------------------------------------------------------
# Toy synthetic record (--fast mode, no real data, no OSF)
# ---------------------------------------------------------------------------
@dataclass
class _ToyRecord:
    """Minimal bridge-shaped record for the synthetic proxy."""
    dyad_label: str
    modality: str
    condition: str
    person_a: Any
    person_b: Any
    target_hz: float = TARGET_HZ
    incomplete: bool = False


def _make_toy_records(
    seed: int = 42,
    n_dyads: int = 6,
    duration: int = 300,
    lag: int = 4,
    cond_effect: float = 0.15,
) -> List[_ToyRecord]:
    """Toy dyad signals.

    Condition 'trials' has slightly stronger coupling (smaller lag) than
    'rest'. Just enough structure to exercise the full pipeline; this is a
    WIRING/SANITY proxy and is NOT meant to reproduce any real effect.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(duration, dtype=float)
    base = np.sin(2 * np.pi * t / 50.0) + 0.3 * np.sin(2 * np.pi * t / 20.0)
    records: List[_ToyRecord] = []
    for d in range(n_dyads):
        for cond, eff in (("rest", 0.0), ("trials", cond_effect)):
            lag_c = max(1, lag - int(round(eff * lag)))
            a = base + rng.normal(0.0, 0.3, size=t.size)
            b = np.zeros_like(a)
            b[lag_c:] = a[:-lag_c]
            b[:lag_c] = a[:lag_c]  # no NaN; keep length == duration
            records.append(_ToyRecord(
                dyad_label=f"toy{d:02d}",
                modality="neural__behavior",
                condition=cond,
                person_a=pd.DataFrame({"value": a}),
                person_b=pd.DataFrame({"value": b}),
            ))
    return records


# ---------------------------------------------------------------------------
# Canonical chain (shared by both modes)
# ---------------------------------------------------------------------------
def _run_chain(
    records: List[Any],
    *,
    surrogate_n: int,
    n_permutations: int,
    design_condition: str,
) -> Tuple[Any, Dict[str, Any]]:
    inputs = records_to_inference_inputs(
        records,
        hz=TARGET_HZ,
        window_size=WINDOW_SIZE,
        onset_threshold=OSET_THRESHOLD,
        design_condition=design_condition,
    )
    pipe = InferencePipeline(
        inputs.features_df,
        hz=TARGET_HZ,
        wcc_window_sec=WCC_WINDOW_SEC,
        surrogate_n=surrogate_n,
    )
    chain = pipe.run_audited_evidence_chain(
        raw_signals=inputs.raw_signals,
        wcc_window_size=WINDOW_SIZE,
        design_signal_pairs=inputs.design_pairs,
        condition_col=inputs.condition_col,
        dyad_col=inputs.dyad_col,
        n_permutations=n_permutations,
        # Canonical pipeline uses ONE fixed/pooled onset threshold, so the
        # A11 per-dyad-threshold WARNING must NOT fire here.
        threshold_scope="fixed",
    )
    return inputs, chain


def _write_outputs(mode: str, inputs: Any, chain: Dict[str, Any]) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_csv = OUT_DIR / f"reproduce_{mode}_features.csv"
    inputs.features_df.to_csv(feat_csv, index=False)
    summary = {
        "mode": mode,
        "canonical_path": "three_pipeline_v1",
        "evidence_chain_version": chain.get("evidence_chain_version"),
        "n_rows_features": int(len(inputs.features_df)),
        "summary": chain.get("summary"),
        "note": (
            "Synthetic proxy (--fast) or OSF-derived (--pub). NOT a "
            "fabricated OSF result; no OSF data value or license text is "
            "invented here."
        ),
    }
    sum_json = OUT_DIR / f"reproduce_{mode}_summary.json"
    sum_json.write_text(json.dumps(summary, indent=2))
    return feat_csv, sum_json


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_fast() -> int:
    logger.info("FAST mode: synthetic toy-dyad proxy (NO OSF access).")
    records = _make_toy_records()
    inputs, chain = _run_chain(
        records,
        surrogate_n=10,
        n_permutations=200,
        design_condition="trials",
    )
    feat_csv, sum_json = _write_outputs("fast", inputs, chain)
    logger.info("Wrote %s and %s", feat_csv.name, sum_json.name)
    return 0


def run_pub(data_root: Optional[str]) -> int:
    if not data_root:
        logger.error("--pub requires --data-root <local OSF mirror path>")
        return 2
    root = Path(data_root)
    if not root.exists():
        logger.error("--data-root %s does not exist", root)
        return 2
    logger.info("PUB mode: loading Lerique-47n3p from %s", root)
    # Lazy import: --fast must never pull the dataset / neurokit2 stack.
    try:
        from multisync.realtest.lerique_2024 import load_lerique_dataset
    except Exception as e:  # pragma: no cover - environment dependent
        logger.error("Could not import the Lerique loader: %s", e)
        return 2
    try:
        records = load_lerique_dataset(str(root), preprocess=False, target_fs=TARGET_HZ)
    except Exception as e:  # pragma: no cover - depends on real data
        logger.error("Lerique loader failed on %s: %s", root, e)
        return 2
    if not records:
        logger.error("Lerique loader returned 0 records from %s", root)
        return 2
    inputs, chain = _run_chain(
        records,
        surrogate_n=30,
        n_permutations=2000,
        design_condition="trials_concat",
    )
    feat_csv, sum_json = _write_outputs("pub", inputs, chain)
    logger.info("Wrote %s and %s", feat_csv.name, sum_json.name)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce Lerique et al. (2024) via the canonical three-pipeline."
    )
    p.add_argument("--fast", action="store_true",
                   help="Synthetic proxy; no OSF needed (default if neither flag given).")
    p.add_argument("--pub", action="store_true",
                   help="Real OSF data (requires --data-root).")
    p.add_argument("--data-root", default=None,
                   help="Local OSF mirror path (--pub only).")
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help="Output directory (default: artifacts/paper_lerique).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    if args.pub:
        return run_pub(args.data_root)
    # Default to --fast (safe, no data dependency, CI-friendly).
    return run_fast()


if __name__ == "__main__":
    sys.exit(main())

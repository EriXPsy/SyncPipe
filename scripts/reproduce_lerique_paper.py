"""
reproduce_lerique_paper.py
============================

M2 reproducibility scaffold for Lerique / ECSU-PCE (OSF 47n3p).

Two modes:

  --fast : SYNTHETIC PROXY. Toy dyads, NO OSF. Wiring/sanity only.
  --pub  : REAL OSF DATA via --data-root. Publication-floor defaults:
             surrogate_n >= 100, n_permutations >= 10000
           (override with --surrogate-n / --n-permutations).

Canonical path:
  pipeline_bridge.records_to_inference_inputs
    -> InferencePipeline.run_audited_evidence_chain
       (existence -> design controls -> group inference)

Writes under artifacts/paper_lerique/:
  reproduce_<mode>_features.csv
  reproduce_<mode>_summary.json
  MANIFEST.json   (parameters actually used; no fabricated license text)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from syncpipe.pipeline_bridge import records_to_inference_inputs
from syncpipe.inference_pipeline import InferencePipeline

logger = logging.getLogger("reproduce_lerique_paper")

# --- locked analysis geometry (NOT per-dyad thresholds) -------------------
ONSET_THRESHOLD = 0.5
WCC_WINDOW_SEC = 30.0
TARGET_HZ = 1.0
WINDOW_SIZE = int(round(WCC_WINDOW_SEC * TARGET_HZ))
OUT_DIR = ROOT / "artifacts" / "paper_lerique"

# Publication floor (P1-R4) vs fast smoke
PUB_SURROGATE_N = 100
PUB_N_PERMUTATIONS = 10000
FAST_SURROGATE_N = 10
FAST_N_PERMUTATIONS = 200

# Preferred contrast labels when present in features_df
PREFERRED_CONTRAST = ("rest1", "trials_concat")
FALLBACK_CONTRAST = ("rest", "trials")


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
            b[:lag_c] = a[:lag_c]
            records.append(_ToyRecord(
                dyad_label=f"toy{d:02d}",
                modality="neural__behavior",
                condition=cond,
                person_a=pd.DataFrame({"value": a}),
                person_b=pd.DataFrame({"value": b}),
            ))
    return records


def _resolve_contrast(features_df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Pick an explicit contrast when labels are present (P1-R1)."""
    if "condition" not in features_df.columns:
        return None
    labels = set(features_df["condition"].dropna().astype(str).unique())
    for pair in (PREFERRED_CONTRAST, FALLBACK_CONTRAST):
        if pair[0] in labels and pair[1] in labels:
            return pair
    return None


def _git_meta() -> Dict[str, Any]:
    meta: Dict[str, Any] = {"commit": None, "repo_dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
        meta["commit"] = commit
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
        meta["repo_dirty"] = bool(dirty != 0)
    except Exception:
        pass
    return meta


def _write_manifest(
    *,
    mode: str,
    surrogate_n: int,
    n_permutations: int,
    design_condition: str,
    contrast: Optional[Tuple[str, str]],
    data_root: Optional[str],
    n_feature_rows: int,
) -> Path:
    from syncpipe import __version__

    manifest = {
        "schema": "syncpipe.paper_manifest/v1",
        "paper": "ECSU-PCE / Lerique OSF 47n3p reproduction scaffold",
        "canonical_path": "three_pipeline_v1",
        "mode": mode,
        "git": _git_meta(),
        "package_version": __version__,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": {
            "osf_project_url": "https://osf.io/47n3p/",
            "osf_project_title": "Perceptual Crossing Dataset Paper",
            "access": "public_download",
            "license_metadata_on_osf": "No License",
            "preprint": {
                "url": "https://osf.io/preprints/osf/6hjfy",
                "doi": "10.31219/osf.io/6hjfy",
                "license": "CC-BY 4.0 (preprint manuscript; does not automatically license raw .mat)",
            },
            "local_path": data_root,
            "note": (
                "SyncPipe does not redistribute raw physiological files. "
                "See docs/DATA_ACCESS.md."
            ),
        },
        "canonical_pipeline": {
            "bridge": "syncpipe.pipeline_bridge.records_to_inference_inputs",
            "inference": "syncpipe.inference_pipeline.InferencePipeline.run_audited_evidence_chain",
            "evidence_chain": [
                "synchrony_existence_signal_level_IAAFT",
                "design_controls_pseudo_pair_time_shift",
                "group_condition_BH_FDR",
            ],
        },
        "parameters": {
            "target_hz": TARGET_HZ,
            "wcc_window_sec": WCC_WINDOW_SEC,
            "window_size_samples": WINDOW_SIZE,
            "onset_threshold": "session_pooled",
            "threshold_scope": "session_pooled",
            "surrogate_n": surrogate_n,
            "n_permutations": n_permutations,
            "fdr_alpha": 0.05,
            "design_condition": design_condition,
            "contrast": list(contrast) if contrast else None,
            "publication_floor": {
                "surrogate_n_min": PUB_SURROGATE_N,
                "n_permutations_min": PUB_N_PERMUTATIONS,
            },
        },
        "outputs": {
            "n_feature_rows": n_feature_rows,
        },
        "status": (
            "POPULATED_BY_RUN" if mode == "pub" else "FAST_SMOKE_ONLY"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _run_chain(
    records: List[Any],
    *,
    surrogate_n: int,
    n_permutations: int,
    design_condition: str,
    contrast: Optional[Tuple[str, str]] = None,
) -> Tuple[Any, Dict[str, Any], Optional[Tuple[str, str]]]:
    inputs = records_to_inference_inputs(
        records,
        hz=TARGET_HZ,
        window_size=WINDOW_SIZE,
        onset_threshold="session_pooled",
        design_condition=design_condition,
    )
    resolved = contrast or _resolve_contrast(inputs.features_df)
    pipe = InferencePipeline(
        inputs.features_df,
        hz=TARGET_HZ,
        wcc_window_sec=WCC_WINDOW_SEC,
        surrogate_n=surrogate_n,
        seed=42,
    )
    chain = pipe.run_audited_evidence_chain(
        raw_signals=inputs.raw_signals,
        wcc_window_size=WINDOW_SIZE,
        design_signal_pairs=inputs.design_pairs,
        condition_col=inputs.condition_col,
        dyad_col=inputs.dyad_col,
        n_permutations=n_permutations,
        threshold_scope="session_pooled",
        contrast=resolved,
        discontinuity_mask=getattr(inputs, "discontinuity_mask", None),
    )
    return inputs, chain, resolved


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
            "fabricated scientific claim beyond wiring + audited chain output."
        ),
    }
    sum_json = OUT_DIR / f"reproduce_{mode}_summary.json"
    sum_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return feat_csv, sum_json


def run_fast(surrogate_n: int, n_permutations: int) -> int:
    logger.info(
        "FAST mode: synthetic toy-dyad proxy (NO OSF). surrogate_n=%s n_perm=%s",
        surrogate_n, n_permutations,
    )
    records = _make_toy_records()
    inputs, chain, contrast = _run_chain(
        records,
        surrogate_n=surrogate_n,
        n_permutations=n_permutations,
        design_condition="trials",
    )
    feat_csv, sum_json = _write_outputs("fast", inputs, chain)
    man = _write_manifest(
        mode="fast",
        surrogate_n=surrogate_n,
        n_permutations=n_permutations,
        design_condition="trials",
        contrast=contrast,
        data_root=None,
        n_feature_rows=int(len(inputs.features_df)),
    )
    logger.info("Wrote %s, %s, %s", feat_csv.name, sum_json.name, man.name)
    return 0


def run_pub(
    data_root: Optional[str],
    surrogate_n: int,
    n_permutations: int,
) -> int:
    if not data_root:
        logger.error("--pub requires --data-root <local OSF mirror path>")
        return 2
    if surrogate_n < PUB_SURROGATE_N or n_permutations < PUB_N_PERMUTATIONS:
        logger.warning(
            "PUB parameters below publication floor "
            "(surrogate_n=%s < %s or n_permutations=%s < %s). "
            "Results must not be reported as publication-grade.",
            surrogate_n, PUB_SURROGATE_N, n_permutations, PUB_N_PERMUTATIONS,
        )
    root = Path(data_root)
    if not root.exists():
        logger.error("--data-root %s does not exist", root)
        return 2
    logger.info("PUB mode: loading ECSU-PCE / Lerique from %s", root)
    try:
        from syncpipe.realtest.lerique_2024 import load_lerique_dataset
    except Exception as e:  # pragma: no cover
        logger.error("Could not import the Lerique loader: %s", e)
        return 2
    try:
        records = load_lerique_dataset(str(root), preprocess=False, target_fs=TARGET_HZ)
    except Exception as e:  # pragma: no cover
        logger.error("Lerique loader failed on %s: %s", root, e)
        return 2
    if not records:
        logger.error("Lerique loader returned 0 records from %s", root)
        return 2
    inputs, chain, contrast = _run_chain(
        records,
        surrogate_n=surrogate_n,
        n_permutations=n_permutations,
        design_condition="trials_concat",
    )
    feat_csv, sum_json = _write_outputs("pub", inputs, chain)
    man = _write_manifest(
        mode="pub",
        surrogate_n=surrogate_n,
        n_permutations=n_permutations,
        design_condition="trials_concat",
        contrast=contrast,
        data_root=str(root),
        n_feature_rows=int(len(inputs.features_df)),
    )
    logger.info("Wrote %s, %s, %s", feat_csv.name, sum_json.name, man.name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce ECSU-PCE / Lerique via the canonical three-pipeline."
    )
    p.add_argument("--fast", action="store_true",
                   help="Synthetic proxy; no OSF (default if neither flag given).")
    p.add_argument("--pub", action="store_true",
                   help="Real OSF data (requires --data-root).")
    p.add_argument("--data-root", default=None,
                   help="Local OSF mirror path (--pub only).")
    p.add_argument("--out-dir", default=str(OUT_DIR),
                   help="Output directory (default: artifacts/paper_lerique).")
    p.add_argument("--surrogate-n", type=int, default=None,
                   help=f"Override surrogate_n (fast default {FAST_SURROGATE_N}, "
                        f"pub default {PUB_SURROGATE_N}).")
    p.add_argument("--n-permutations", type=int, default=None,
                   help=f"Override L2 n_permutations (fast default {FAST_N_PERMUTATIONS}, "
                        f"pub default {PUB_N_PERMUTATIONS}).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    if args.pub:
        sn = PUB_SURROGATE_N if args.surrogate_n is None else args.surrogate_n
        np_ = PUB_N_PERMUTATIONS if args.n_permutations is None else args.n_permutations
        return run_pub(args.data_root, sn, np_)
    sn = FAST_SURROGATE_N if args.surrogate_n is None else args.surrogate_n
    np_ = FAST_N_PERMUTATIONS if args.n_permutations is None else args.n_permutations
    return run_fast(sn, np_)


if __name__ == "__main__":
    sys.exit(main())

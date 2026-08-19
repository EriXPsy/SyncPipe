"""Canonical report-bundle serialization."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from ..contracts import SCHEMA_NAMES, ManifestRecord, SyncPipeConfig, load_preprocessing_provenance, load_schema
from ..pipeline_bridge import InferenceInputs
from ..preparation import PreparationExclusion
from .report import build_report_markdown
from .runtime import json_safe

def _sha256(path: Optional[str]) -> Optional[str]:
    """SHA-256 of a file's bytes, or None when unreadable."""
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _write_report_bundle(
    *,
    records: List[ManifestRecord],
    cfg: SyncPipeConfig,
    chain: Dict[str, Any],
    qc: Dict[str, Any],
    exclusions: List[PreparationExclusion],
    inputs: InferenceInputs,
    claimability: Dict[str, Any],
    environment: Dict[str, Any],
    output_dir: Path,
    base_dir: Path,
) -> Dict[str, str]:
    """Write the 12-file unified report bundle. Returns a path map."""
    out = str(output_dir)
    paths: Dict[str, str] = {}

    def _dump_json(name: str, payload: Any) -> None:
        p = output_dir / name
        p.write_text(
            json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        paths[name] = str(p)

    # manifest_resolved.json: resolved paths + content hashes, not only the
    # original relative strings. This makes an analysis auditable after files
    # move or are replaced (Gate 1 residual, vulnerability A).
    resolved_rows = []
    for r in records:
        rr = r.resolve(base_dir)
        item = vars(rr).copy()
        item["person_a_sha256"] = _sha256(rr.person_a_path)
        item["person_b_sha256"] = _sha256(rr.person_b_path)
        item["mask_sha256"] = _sha256(rr.mask_path)
        item["preprocessing_sha256"] = _sha256(rr.preprocessing_path)
        item["preprocessing"] = load_preprocessing_provenance(
            rr.preprocessing_path,
            expected_signal_type=rr.signal_type,
            expected_unit=rr.unit,
        )
        resolved_rows.append(item)
    _dump_json("manifest_resolved.json", {
        "base_dir": str(base_dir.resolve()),
        "schema_ids": {
            name: load_schema(name).get("$id") for name in SCHEMA_NAMES
        },
        "rows": resolved_rows,
        "hz_uniform": qc["hz"],
    })

    # config_resolved.toml
    try:
        import tomllib as _t  # noqa: F401  (presence check)
    except ModuleNotFoundError:
        pass
    cfg_path = output_dir / "config_resolved.toml"

    def _toml_str(value: object) -> str:
        # Escape backslash and double-quote so an identifier/condition label
        # containing them still produces valid TOML that parse_config can read
        # back (preserving CLI/API parity). TOML basic strings require these two.
        s = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    lines = [
        "[analysis]",
        f'window_size = {cfg.window_size}',
        f'window_type = {_toml_str(cfg.window_type)}',
        (f'contrast = [{_toml_str(cfg.contrast[0])}, {_toml_str(cfg.contrast[1])}]'
         if cfg.contrast else 'contrast = []'),
        f'fdr_scope = {_toml_str(cfg.fdr_scope)}',
        f'undefined_policy = {_toml_str(cfg.undefined_policy)}',
        f'observation_policy = {_toml_str(cfg.observation_policy)}',
        f'eligibility_policy = {_toml_str(cfg.eligibility_policy)}',
        f'n_min_dyads = {cfg.n_min_dyads}',
        (f'onset_threshold = {_toml_str(cfg.onset_threshold)}'
         if isinstance(cfg.onset_threshold, str)
         else f'onset_threshold = {cfg.onset_threshold}'),
        f'n_permutations = {cfg.n_permutations}',
        f'seed = {cfg.seed}',
        f'surrogate_n = {cfg.surrogate_n}',
        f'design_threshold = {cfg.design_threshold}',
        f'design_condition = {_toml_str(cfg.resolved_design_condition())}',
        f'main_measure = {_toml_str(cfg.resolved_primary_endpoint())}',
        "main_modalities = [" + ", ".join(
            _toml_str(x) for x in cfg.resolved_primary_modalities()
        ) + "]",
        f'existence_alpha = {cfg.existence_alpha}',
        # Recorded for provenance only: n_workers changes wall-clock, never
        # numbers (each pair seeds its own Generator). Replaying this config
        # with a different n_workers reproduces the same results.
        f'n_workers = {cfg.n_workers}',
    ]
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["config_resolved.toml"] = str(cfg_path)

    # environment.json
    _dump_json("environment.json", environment)

    # qc_report.json
    _dump_json("qc_report.json", qc)

    # exclusion_report.csv
    exc_path = output_dir / "exclusion_report.csv"
    exclusion_columns = [
        "key", "dyad_id", "modality", "condition", "code", "stage",
        "detail", "claim_effect", "reason",
    ]
    if exclusions:
        exc_df = pd.DataFrame([item.to_dict() for item in exclusions])[exclusion_columns]
    else:
        exc_df = pd.DataFrame(columns=exclusion_columns)
    exc_df.to_csv(exc_path, index=False)
    paths["exclusion_report.csv"] = str(exc_path)

    # features.csv
    feat_path = output_dir / "features.csv"
    inputs.features_df.to_csv(feat_path, index=False)
    paths["features.csv"] = str(feat_path)

    # wcc_traces/
    wcc_dir = output_dir / "wcc_traces"
    wcc_dir.mkdir(exist_ok=True)
    if inputs.wcc_traces:
        for key, trace in inputs.wcc_traces.items():
            safe = key.replace("/", "_").replace("\\", "_")
            pd.DataFrame({"wcc": np.asarray(trace, dtype=float)}).to_csv(
                wcc_dir / f"wcc_{safe}.csv", index=False
            )
    paths["wcc_traces/"] = str(wcc_dir)

    # existence / design / group inference / claimability
    _dump_json("existence_audit.json", chain.get("synchrony_existence"))
    _dump_json("existence_gate.json", chain.get("existence_gate"))
    _dump_json("design_control_audit.json", chain.get("design_controls"))
    _dump_json("group_inference.json", chain.get("group_condition_inference"))
    _dump_json("evidence_graph.json", chain.get("evidence_graph"))
    _dump_json("claimability.json", claimability)

    # REPORT.md. `paths` is passed so the report's "Output files" section is
    # derived from what was actually written rather than a hand-kept literal
    # list — the previous hard-coded list had silently gone stale and omitted
    # existence_gate.json, i.e. the existence-gate verdict itself.
    report_md = build_report_markdown(
        records, cfg, chain, qc, exclusions, environment, paths
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    paths["REPORT.md"] = str(report_path)

    return paths



write_report_bundle = _write_report_bundle
__all__ = ["write_report_bundle"]

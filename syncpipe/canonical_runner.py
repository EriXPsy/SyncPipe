"""
Canonical scientific runner for SyncPipe v1.

This module is the *single* entry point for paper-level analyses. Both the
Python API (:func:`run_canonical`) and the CLI (``syncpipe analyze``) route
through it, which is exactly what guarantees CLI/API parity (Gate 1 pass
criterion: same manifest + config through either entry point yields
byte-equivalent results).

Design note (reuse, do not rewrite):
    The scientific core already exists and is tested:
      * :func:`syncpipe.pipeline_bridge.records_to_inference_inputs` turns
        loader records into WCC + descriptor ``InferenceInputs`` (ComputationPipeline).
      * :meth:`syncpipe.inference_pipeline.InferencePipeline.run_audited_evidence_chain`
        runs the v1 audited evidence chain (L0 existence -> design controls ->
        L2 paired inference -> FDR / definedness / eligibility governance).
    This module only adds the *outer shell*: manifest/config parsing,
    orchestration, and the unified 12-file report bundle. No inference math
    is touched.

Pipeline:
    manifest + config
        -> parse_manifest / parse_config
        -> records_to_inference_inputs   (ComputationPipeline: WCC + features)
        -> InferencePipeline.run_audited_evidence_chain
        -> unified report bundle (12 files)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as tomllib  # type: ignore

from .pipeline_bridge import records_to_inference_inputs
from .preparation import LoaderRecord, PreparationExclusion, load_manifest_record
from .evidence import derive_claimability
from .export import capture_environment, safe_float, write_report_bundle
from .inference_pipeline import InferencePipeline
from .feature_definitions import FDR_FEATURES, REFERENCE_FEATURE
from .contracts import (
    AnalysisSpec,
    EndpointSpec,
    ModalitySpec,
    NullSpec,
    SyncPipeConfig,
    ManifestRecord,
    analysis_spec_from_mapping,
    load_preprocessing_provenance,
    load_schema,
    parse_manifest,
)

__all__ = [
    "ManifestRecord",
    "AnalysisSpec",
    "SyncPipeConfig",
    "EndpointSpec",
    "NullSpec",
    "ModalitySpec",
    "LoaderRecord",
    "CanonicalResult",
    "DEFAULT_CONFIG",
    "parse_manifest",
    "parse_config",
    "load_schema",
    "run_canonical",
]

# Compatibility alias: new code should use AnalysisSpec.
# SyncPipeConfig is imported from syncpipe.contracts.


DEFAULT_CONFIG = SyncPipeConfig()


# ──────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────
def parse_config(path: Union[str, Path]) -> AnalysisSpec:
    """Parse TOML through the immutable :class:`AnalysisSpec` contract."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("analysis", data) if isinstance(data, dict) else {}
    if not isinstance(section, dict):
        raise ValueError("config [analysis] must be a TOML table")
    return analysis_spec_from_mapping(section, require_declarations=True)


# ──────────────────────────────────────────────────────────────────────────
# Manifest record -> loader record adapter
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# JSON sanitizer (mirrors InferencePipeline.to_json)
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CanonicalResult:
    """Structured result returned by :func:`run_canonical`."""

    output_dir: str
    manifest: List[ManifestRecord]
    config: SyncPipeConfig
    chain: Dict[str, Any]
    qc: Dict[str, Any]
    exclusions: List[PreparationExclusion]
    features_df: pd.DataFrame
    wcc_traces: Dict[str, np.ndarray]
    environment: Dict[str, Any]
    claimability: Dict[str, Any]
    report_paths: Dict[str, str]


def run_canonical(
    manifest: Union[str, Path, List[ManifestRecord]],
    config: Union[str, Path, SyncPipeConfig],
    output_dir: Union[str, Path],
) -> CanonicalResult:
    """Run the full v1 canonical scientific path and write the report bundle.

    Both the Python API and the CLI call this function, which is what makes
    their outputs byte-equivalent.

    Parameters
    ----------
    manifest : path or list of ManifestRecord
        Strict manifest CSV path, or pre-built records.
    config : path or SyncPipeConfig
        TOML config path, or a pre-built config.
    output_dir : path
        Directory for the 12-file report bundle (created if missing).

    Returns
    -------
    CanonicalResult
    """
    records = parse_manifest(manifest) if not isinstance(manifest, list) else list(manifest)
    cfg = parse_config(config) if not isinstance(config, AnalysisSpec) else config
    contrast = cfg.resolved_contrast()
    endpoint_spec = cfg.resolved_endpoint_spec()
    primary_endpoint = endpoint_spec.name
    primary_modalities = cfg.resolved_primary_modalities()
    design_condition = cfg.resolved_design_condition()
    available_modalities = {r.modality for r in records}
    missing_primary = sorted(set(primary_modalities) - available_modalities)
    if missing_primary:
        raise ValueError(
            "config.primary_modalities contains label(s) absent from the manifest: "
            f"{missing_primary}; available modalities are {sorted(available_modalities)}"
        )
    modality_specs = cfg.modality_specs(tuple(sorted(available_modalities)))
    available_conditions = {r.condition for r in records}
    if design_condition not in available_conditions:
        raise ValueError(
            f"config.design_condition={design_condition!r} is not present "
            f"in the manifest conditions {sorted(available_conditions)}"
        )

    hz = records[0].hz  # uniform (parse_manifest guarantees this)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(manifest).parent if isinstance(manifest, (str, Path)) else Path.cwd()

    # Provenance is an analysis contract, not participant-level bad data. Any
    # missing/malformed document fails before load errors can be downgraded to
    # row exclusions.
    for rec in records:
        resolved = rec.resolve(base_dir)
        load_preprocessing_provenance(
            resolved.preprocessing_path,
            expected_signal_type=resolved.signal_type,
            expected_unit=resolved.unit,
        )

    # --- preflight QC: resolve loader records; capture load failures ---
    loader_records: List[LoaderRecord] = []
    load_exclusions: List[PreparationExclusion] = []
    for rec in records:
        try:
            loader_records.append(load_manifest_record(rec, base_dir))
        except (OSError, pd.errors.ParserError) as e:
            # Expected data/input failures become exclusions. Unexpected
            # programming/runtime errors must propagate instead of being
            # mislabeled as bad participant data.
            load_exclusions.append(PreparationExclusion(
                key=f"{rec.dyad_id}__{rec.modality}__{rec.condition}",
                dyad_id=rec.dyad_id, modality=rec.modality,
                condition=rec.condition, code="load_error", stage="loading",
                detail=str(e),
            ))

    # records_to_inference_inputs fails loud on duplicate key / hz mismatch /
    # ambiguous column. It silently skips incomplete / too-short rows, so we
    # diff the included keys afterward to populate exclusion_report.
    inputs: InferenceInputs = records_to_inference_inputs(
        loader_records,
        hz=hz,
        window_size=cfg.window_size,
        onset_threshold=cfg.onset_threshold,
        design_condition=design_condition,
        window_type=cfg.window_type,
        initial_exclusions=load_exclusions,
    )
    exclusions = list(inputs.prepared_cohort.exclusions)

    # Pairing summary is separate from row-level inclusion. A loadable orphan
    # condition row is not a paired dyad-level observation and must not look
    # like a usable confirmatory unit in qc_report.json.
    pair_summary: Dict[str, Any] = {}
    for modality, mod_df in inputs.features_df.groupby("modality"):
        by_condition = {
            str(cond): set(mod_df.loc[mod_df["condition"] == cond, "dyad_id"].astype(str))
            for cond in contrast
        }
        paired = set.intersection(*by_condition.values()) if by_condition else set()
        union = set.union(*by_condition.values()) if by_condition else set()
        pair_summary[str(modality)] = {
            "n_rows": int(len(mod_df)),
            "n_by_condition": {k: len(v) for k, v in by_condition.items()},
            "n_paired_dyads": len(paired),
            "n_orphan_dyads": len(union - paired),
        }

    # --- inference pipeline (v1 audited evidence chain) ---
    pipe = InferencePipeline(
        features_df=inputs.features_df,
        hz=hz,
        wcc_window_sec=cfg.window_size / hz,
        surrogate_n=cfg.surrogate_n,
        seed=cfg.seed,
        n_workers=cfg.n_workers,
    )
    threshold_scope = (
        "per_modality" if isinstance(cfg.onset_threshold, str)
        and cfg.onset_threshold == "session_pooled" else "fixed"
    )
    # Design-control masks and thresholds must be keyed to the SELECTED design
    # condition, not to whichever condition happened to appear first in the
    # manifest (Gate 1 residual P0-2).
    design_keys = set(inputs.design_pairs)
    design_masks: Optional[Dict[str, np.ndarray]] = None
    if inputs.prepared_cohort is not None:
        design_masks = {
            f"{obs.dyad_id}__{obs.modality}": obs.geometry.analysis_mask
            for obs in inputs.prepared_cohort.observations
            if obs.condition == design_condition
        }
        if set(design_masks) != design_keys:
            raise ValueError(
                "prepared design masks must cover exactly the design_signal_pairs keys"
            )

    # P0-1: when onset thresholds are session-pooled, the design-control audit
    # must use the SAME per-modality effective threshold as the feature table,
    # not the fixed config.design_threshold. Pass a per-pair mapping keyed by
    # dyad__modality so dwell/switching/fraction_above_threshold share one
    # measurement definition across both stages.
    if cfg.onset_threshold == "session_pooled":
        mod_of_key = {
            f"{lr.dyad_label}__{lr.modality}": lr.modality
            for lr in loader_records
            if lr.condition == design_condition
        }
        design_threshold: Union[float, Dict[str, float]] = {
            key: float(inputs.thresholds_by_modality[mod_of_key[key]])
            for key in design_keys
        }
    else:
        design_threshold = cfg.design_threshold

    chain = pipe.run_audited_evidence_chain(
        raw_signals=inputs.raw_signals,
        wcc_window_size=cfg.window_size,
        design_signal_pairs=inputs.design_pairs,
        contrast=contrast,
        fdr_scope=cfg.fdr_scope,
        undefined_policy=cfg.undefined_policy,
        observation_policy=cfg.observation_policy,
        eligibility_policy=cfg.eligibility_policy,
        n_min_dyads=cfg.n_min_dyads,
        threshold_scope=threshold_scope,
        discontinuity_mask=inputs.discontinuity_mask,
        design_discontinuity_mask=design_masks,
        window_type=cfg.window_type,
        n_permutations=cfg.n_permutations,
        design_threshold=design_threshold,
        feature_cols=list(FDR_FEATURES) + list(REFERENCE_FEATURE),
        primary_endpoint=primary_endpoint,
        primary_modalities=primary_modalities,
        existence_alpha=cfg.existence_alpha,
    )

    claimability = derive_claimability(chain)
    environment = capture_environment(cfg.seed)

    qc = {
        "total_rows": len(records),
        # Every row excluded for ANY reason (load error OR too-short/incomplete)
        # is removed from the analyzable set. included = total - exclusions.
        "included": len(records) - len(exclusions),
        "excluded": len(exclusions),
        "hz": hz,
        "window_size": cfg.window_size,
        "thresholds_by_modality": inputs.thresholds_by_modality,
        "modality_roles": {spec.label: spec.role for spec in modality_specs},
        "endpoint_contract": {
            "name": endpoint_spec.name,
            "estimand": endpoint_spec.estimand,
            "null": endpoint_spec.null.name,
            "tail": endpoint_spec.null.tail,
            "duration_policy": endpoint_spec.duration_policy,
        },
        "pair_summary": pair_summary,
        "preparation": inputs.preparation_diagnostics,
        "design_threshold_scope": (
            "per_modality_pooled" if cfg.onset_threshold == "session_pooled" else "fixed"
        ),
        "n_wcc_points_per_cell": {
            f"{r['dyad_id']}__{r['modality']}__{r['condition']}": safe_float(
                r.get("n_wcc_points")
            )
            for _, r in inputs.features_df.iterrows()
        },
    }

    paths = write_report_bundle(
        records=records, cfg=cfg, chain=chain, qc=qc, exclusions=exclusions,
        inputs=inputs, claimability=claimability, environment=environment,
        output_dir=output_dir, base_dir=base_dir,
    )

    return CanonicalResult(
        output_dir=str(output_dir),
        manifest=records,
        config=cfg,
        chain=chain,
        qc=qc,
        exclusions=exclusions,
        features_df=inputs.features_df,
        wcc_traces=inputs.wcc_traces or {},
        environment=environment,
        claimability=claimability,
        report_paths=paths,
    )

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

import hashlib
import json
import os
from importlib import resources
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as tomllib  # type: ignore

from .__about__ import __version__, CONFIG_SCHEMA_VERSION, ANALYSIS_SCHEMA_VERSION
from .io import load_csv
from .pipeline_bridge import InferenceInputs, records_to_inference_inputs
from .inference_pipeline import InferencePipeline
from .feature_definitions import FDR_FEATURES, REFERENCE_FEATURE
from .contracts import (
    AnalysisSpec,
    EndpointSpec,
    ModalitySpec,
    NullSpec,
    SyncPipeConfig,
    analysis_spec_from_mapping,
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

# Repo root (syncpipe/) for git hash resolution at runtime.
_REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_COLUMNS = [
    "dyad_id", "modality", "condition",
    "person_a_path", "person_b_path", "hz",
    "signal_type", "unit", "preprocessing_path", "mask_path",
]

_SCHEMA_NAMES = {
    "config": "config.schema.json",
    "manifest_record": "manifest-record.schema.json",
    "preprocessing": "preprocessing.schema.json",
}


def load_schema(name: str) -> Dict[str, Any]:
    """Load a packaged JSON Schema by stable public name."""
    if name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown schema {name!r}; choose from {sorted(_SCHEMA_NAMES)}")
    resource = resources.files("syncpipe").joinpath("schemas", _SCHEMA_NAMES[name])
    return json.loads(resource.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# Input contracts
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ManifestRecord:
    """One row of the strict manifest CSV.

    Columns include signal identity/unit and a required structured preprocessing
    provenance file; ``mask_path`` remains optional.
    """

    dyad_id: str
    modality: str
    condition: str
    person_a_path: str
    person_b_path: str
    hz: float
    signal_type: str
    unit: str
    preprocessing_path: str
    mask_path: Optional[str] = None

    def resolve(self, base_dir: Optional[Union[str, Path]] = None) -> "ManifestRecord":
        """Return a copy with all paths made absolute against ``base_dir``."""
        base = Path(base_dir) if base_dir else Path.cwd()

        def _abs(p: str) -> str:
            p = Path(p)
            return str(p if p.is_absolute() else base / p)

        return ManifestRecord(
            dyad_id=self.dyad_id,
            modality=self.modality,
            condition=self.condition,
            person_a_path=_abs(self.person_a_path),
            person_b_path=_abs(self.person_b_path),
            hz=self.hz,
            signal_type=self.signal_type,
            unit=self.unit,
            preprocessing_path=_abs(self.preprocessing_path),
            mask_path=_abs(self.mask_path) if self.mask_path else None,
        )


# Compatibility alias: new code should use AnalysisSpec.
# SyncPipeConfig is imported from syncpipe.contracts.


@dataclass
class LoaderRecord:
    """Adapter record fed to ``records_to_inference_inputs``.

    Mirrors the attribute contract that loaders (e.g. ``LeriqueDyadCondition``)
    satisfy: ``dyad_label``, ``modality``, ``condition``, ``person_a``,
    ``person_b``, ``target_hz``, ``discontinuity_mask``, ``incomplete``.
    """

    dyad_label: str
    modality: str
    condition: str
    person_a: Any
    person_b: Any
    target_hz: float
    discontinuity_mask: Optional[np.ndarray] = None
    incomplete: bool = False


DEFAULT_CONFIG = SyncPipeConfig()


# ──────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────
def parse_manifest(path: Union[str, Path]) -> List[ManifestRecord]:
    """Parse the strict manifest CSV into :class:`ManifestRecord` objects.

    All rows must share one ``hz`` (fail-loud otherwise); ``mask_path`` is
    optional per row.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("manifest must contain at least one data row")
    required = [
        "dyad_id", "modality", "condition", "person_a_path", "person_b_path",
        "hz", "signal_type", "unit", "preprocessing_path",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    records: List[ManifestRecord] = []
    hz_set = set()
    for _, row in df.iterrows():
        hz = float(row["hz"])
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError(
                f"manifest row dyad={row.get('dyad_id')}: hz must be finite "
                f"positive, got {hz!r}"
            )
        hz_set.add(round(hz, 9))
        mp_raw = row["mask_path"] if "mask_path" in df.columns else None
        mask_path: Optional[str] = None
        if mp_raw is not None and not (isinstance(mp_raw, float) and np.isnan(mp_raw)):
            s = str(mp_raw).strip()
            if s:
                mask_path = s
        records.append(
            ManifestRecord(
                dyad_id=str(row["dyad_id"]),
                modality=str(row["modality"]),
                condition=str(row["condition"]),
                person_a_path=str(row["person_a_path"]),
                person_b_path=str(row["person_b_path"]),
                hz=hz,
                signal_type=str(row["signal_type"]).strip(),
                unit=str(row["unit"]).strip(),
                preprocessing_path=str(row["preprocessing_path"]).strip(),
                mask_path=mask_path,
            )
        )

    keys = [(r.dyad_id, r.modality, r.condition) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("manifest contains duplicate (dyad_id, modality, condition) rows")
    for r in records:
        if any(
            not str(getattr(r, field)).strip()
            for field in (
                "dyad_id", "modality", "condition", "person_a_path",
                "person_b_path", "signal_type", "unit", "preprocessing_path",
            )
        ):
            raise ValueError(
                "manifest contains an empty identifier, signal identity/unit, "
                "or required path"
            )

    if len(hz_set) > 1:
        raise ValueError(
            f"manifest hz must be uniform across rows; found {sorted(hz_set)}. "
            "Resample explicitly before pooling."
        )
    return records


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
def _load_mask(path: str) -> np.ndarray:
    """Load a discontinuity mask CSV (single 0/1 or boolean column)."""
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"mask file {path} has no numeric column.")
    values = num.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"mask file {path} must contain only finite 0/1 values")
    return values.astype(bool)


def _load_preprocessing_provenance(
    path: str, *, expected_signal_type: str, expected_unit: str
) -> Dict[str, Any]:
    """Load and validate the mandatory preprocessing provenance document."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid preprocessing provenance {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("preprocessing provenance must be a JSON object")
    required = ("schema_version", "signal_type", "output_unit", "software", "steps")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"preprocessing provenance missing required fields: {missing}")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"preprocessing schema_version must be {CONFIG_SCHEMA_VERSION!r}"
        )
    if str(payload["signal_type"]).strip() != expected_signal_type:
        raise ValueError(
            "preprocessing signal_type does not match the manifest declaration"
        )
    if str(payload["output_unit"]).strip() != expected_unit:
        raise ValueError(
            "preprocessing output_unit does not match the manifest unit"
        )
    software = payload["software"]
    if not isinstance(software, dict) or not all(
        str(software.get(k, "")).strip() for k in ("name", "version")
    ):
        raise ValueError("preprocessing software requires non-empty name and version")
    steps = payload["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("preprocessing steps must be a non-empty list")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not str(step.get("name", "")).strip():
            raise ValueError(f"preprocessing step {index} requires a non-empty name")
        if not isinstance(step.get("parameters"), dict):
            raise ValueError(f"preprocessing step {index} parameters must be an object")
    return payload


def _validate_aligned_signal_frames(a_df: pd.DataFrame, b_df: pd.DataFrame, *, hz: float, key: str) -> None:
    """Validate the time axes before the bridge discards them.

    Guards against manufacturing spurious synchrony when person A/B are not
    sampled on the same time grid (Gate 1 residual, vulnerability B).
    """
    if "time" not in a_df.columns or "time" not in b_df.columns:
        raise ValueError(f"{key}: both signal files must contain a 'time' column")
    ta = a_df["time"].to_numpy(dtype=float)
    tb = b_df["time"].to_numpy(dtype=float)
    if ta.ndim != 1 or tb.ndim != 1 or ta.size != tb.size:
        raise ValueError(f"{key}: person A/B time axes must be one-dimensional and equal length")
    if ta.size < 2 or not np.isfinite(ta).all() or not np.isfinite(tb).all():
        raise ValueError(f"{key}: time axes must contain at least two finite samples")
    if not (np.all(np.diff(ta) > 0) and np.all(np.diff(tb) > 0)):
        raise ValueError(f"{key}: time axes must be strictly increasing")
    if not np.allclose(ta, tb, rtol=0.0, atol=max(1e-6, 0.1 / hz)):
        raise ValueError(f"{key}: person A/B time axes are not aligned")
    dt = float(np.median(np.diff(ta)))
    if not np.isclose(dt, 1.0 / hz, rtol=0.01, atol=1e-9):
        raise ValueError(f"{key}: time step {dt:g} does not match manifest hz={hz:g}")


def _manifest_record_to_loader_record(
    rec: ManifestRecord, base_dir: Optional[Union[str, Path]] = None
) -> LoaderRecord:
    """Resolve paths and load signals/mask for one manifest row."""
    r = rec.resolve(base_dir)
    a_df = load_csv(r.person_a_path)
    b_df = load_csv(r.person_b_path)
    key = f"{r.dyad_id}__{r.modality}__{r.condition}"
    _validate_aligned_signal_frames(a_df, b_df, hz=r.hz, key=key)
    mask = _load_mask(r.mask_path) if r.mask_path else None
    if mask is not None and mask.size != len(a_df):
        raise ValueError(f"{key}: mask length {mask.size} does not match signal length {len(a_df)}")
    return LoaderRecord(
        dyad_label=r.dyad_id,
        modality=r.modality,
        condition=r.condition,
        person_a=a_df,
        person_b=b_df,
        target_hz=r.hz,
        discontinuity_mask=mask,
        incomplete=False,
    )


# ──────────────────────────────────────────────────────────────────────────
# JSON sanitizer (mirrors InferencePipeline.to_json)
# ──────────────────────────────────────────────────────────────────────────
def _json_safe(obj: Any) -> Any:
    """Recursively convert results to JSON-safe structures.

    Non-finite floats -> null; dataclasses -> dict; ndarray -> list. Mirrors the
    sanitizer in ``InferencePipeline.to_json`` so report files are strict-JSON.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    try:
        from dataclasses import is_dataclass, asdict

        if is_dataclass(obj) and not isinstance(obj, type):
            return _json_safe(asdict(obj))
    except Exception:
        pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    return obj


def _safe_float(x: Any) -> Optional[float]:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if xf != xf or xf in (float("inf"), float("-inf")):
        return None
    return xf


# ──────────────────────────────────────────────────────────────────────────
# Environment + claimability helpers
# ──────────────────────────────────────────────────────────────────────────
def _environment(seed: int) -> Dict[str, Any]:
    git_hash = "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=10,
        )
        if out.returncode == 0:
            git_hash = out.stdout.strip()
    except Exception:
        pass
    try:
        import pandas as _pandas
        import scipy as _scipy
        import sklearn as _sklearn
        dependency_versions = {
            "pandas": _pandas.__version__,
            "scipy": _scipy.__version__,
            "scikit_learn": _sklearn.__version__,
        }
    except Exception:
        dependency_versions = {}
    return {
        "python_version": platform.python_version(),
        "syncpipe_version": __version__,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "dependency_versions": dependency_versions,
        "platform": platform.platform(),
        "git_hash": git_hash,
        "seed": seed,
    }


def _derive_claimability(chain: Dict[str, Any]) -> Dict[str, Any]:
    """Extract per-feature claimability from the L2 result of the chain.

    ``group_condition_inference`` is always the modality-keyed shape
    ``{modality: l2_dict}`` (1c), so each entry is walked and the modality is
    recorded on every row — a claim about ECG and a claim about EDA are distinct
    hypotheses and must not be collapsed into one anonymous feature list.
    """
    group = chain.get("group_condition_inference") or {}
    per_feature: List[Dict[str, Any]] = []

    def _collect(modality: str, l2: Any) -> None:
        if not isinstance(l2, dict):
            return
        pf = l2.get("per_feature")
        if not pf:
            return
        elig = l2.get("eligibility_status")
        for r in pf:
            feat = getattr(r, "feature", None)
            if feat is None:
                continue
            per_feature.append({
                "modality": modality,
                "feature": feat,
                "p_fdr": _safe_float(getattr(r, "p_fdr", None)),
                "significant_05": bool(getattr(r, "significant_05", False)),
                "claimable": getattr(r, "claimable", None),
                "definedness_status": getattr(r, "definedness_status", None),
                "eligibility_status": (elig.get(feat) if isinstance(elig, dict) else elig),
                "n_dyads": _safe_float(getattr(r, "n_dyads", None)),
                "observed_diff": _safe_float(getattr(r, "observed_diff", None)),
                "difference_q25": _safe_float(getattr(r, "difference_q25", None)),
                "difference_q75": _safe_float(getattr(r, "difference_q75", None)),
                "median_ci_low": _safe_float(getattr(r, "median_ci_low", None)),
                "median_ci_high": _safe_float(getattr(r, "median_ci_high", None)),
                "median_ci_bounded": bool(getattr(r, "median_ci_bounded", False)),
                "permutation_method": getattr(r, "permutation_method", None),
                "n_null_draws": int(getattr(r, "n_null_draws", 0)),
                "min_attainable_p": _safe_float(getattr(r, "min_attainable_p", None)),
                "approx_monte_carlo_se": _safe_float(
                    getattr(r, "approx_monte_carlo_se", None)
                ),
            })

    for mod in sorted(group.keys(), key=str):
        _collect(str(mod), group[mod])

    return {
        "stage_status": chain.get("stage_status", {}),
        "claim_ceiling": chain.get("claim_ceiling"),
        "per_feature": per_feature,
    }


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
    exclusions: List[Dict[str, Any]]
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
        _load_preprocessing_provenance(
            resolved.preprocessing_path,
            expected_signal_type=resolved.signal_type,
            expected_unit=resolved.unit,
        )

    # --- preflight QC: resolve loader records; capture load failures ---
    loader_records: List[LoaderRecord] = []
    exclusions: List[Dict[str, Any]] = []
    for rec in records:
        try:
            loader_records.append(_manifest_record_to_loader_record(rec, base_dir))
        except (OSError, pd.errors.ParserError) as e:
            # Expected data/input failures become exclusions. Unexpected
            # programming/runtime errors must propagate instead of being
            # mislabeled as bad participant data.
            exclusions.append({
                "dyad_id": rec.dyad_id, "modality": rec.modality,
                "condition": rec.condition, "reason": f"load_error: {e}",
            })

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
    )

    included_keys = set(
        f"{r['dyad_id']}__{r['modality']}__{r['condition']}"
        for _, r in inputs.features_df.iterrows()
    )
    for lr in loader_records:
        key = f"{lr.dyad_label}__{lr.modality}__{lr.condition}"
        if key not in included_keys:
            exclusions.append({
                "dyad_id": lr.dyad_label, "modality": lr.modality,
                "condition": lr.condition, "reason": "too_short_or_incomplete",
            })

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

    claimability = _derive_claimability(chain)
    environment = _environment(cfg.seed)

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
            f"{r['dyad_id']}__{r['modality']}__{r['condition']}": _safe_float(
                r.get("n_wcc_points")
            )
            for _, r in inputs.features_df.iterrows()
        },
    }

    paths = _write_report_bundle(
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


# ──────────────────────────────────────────────────────────────────────────
# Report bundle writer
# ──────────────────────────────────────────────────────────────────────────
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
    exclusions: List[Dict[str, Any]],
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
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
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
        item["preprocessing"] = _load_preprocessing_provenance(
            rr.preprocessing_path,
            expected_signal_type=rr.signal_type,
            expected_unit=rr.unit,
        )
        resolved_rows.append(item)
    _dump_json("manifest_resolved.json", {
        "base_dir": str(base_dir.resolve()),
        "schema_ids": {
            name: load_schema(name).get("$id") for name in _SCHEMA_NAMES
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
        f'primary_endpoint = {_toml_str(cfg.resolved_primary_endpoint())}',
        "primary_modalities = [" + ", ".join(
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
    if exclusions:
        exc_df = pd.DataFrame(exclusions)[["dyad_id", "modality", "condition", "reason"]]
    else:
        exc_df = pd.DataFrame(columns=["dyad_id", "modality", "condition", "reason"])
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
    _dump_json("claimability.json", claimability)

    # REPORT.md. `paths` is passed so the report's "Output files" section is
    # derived from what was actually written rather than a hand-kept literal
    # list — the previous hard-coded list had silently gone stale and omitted
    # existence_gate.json, i.e. the existence-gate verdict itself.
    report_md = _build_report_md(
        records, cfg, chain, qc, exclusions, environment, paths
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    paths["REPORT.md"] = str(report_path)

    return paths


def _build_report_md(
    records, cfg, chain, qc, exclusions, environment, paths=None
) -> str:
    """Human-readable markdown summary of a canonical run."""
    endpoint = cfg.resolved_endpoint_spec()
    lines = [
        "# SyncPipe v1 — Canonical Analysis Report",
        "",
        f"- SyncPipe version: **{environment.get('syncpipe_version')}**",
        f"- Git hash: `{environment.get('git_hash')}`",
        f"- Seed: {environment.get('seed')}",
        f"- hz: {qc.get('hz')} | window_size: {cfg.window_size} | "
        f"window_type: {cfg.window_type}",
        f"- Contrast: {cfg.contrast}",
        f"- Declared primary endpoint: {endpoint.name} | "
        f"primary modalities: {cfg.primary_modalities}",
        f"- Endpoint estimand: {endpoint.estimand}",
        f"- Endpoint null: {endpoint.null.name} ({endpoint.null.tail})",
        f"- FDR scope: {cfg.fdr_scope} | undefined_policy: {cfg.undefined_policy} | "
        f"observation_policy: {cfg.observation_policy} | eligibility_policy: {cfg.eligibility_policy}",
        f"- Rows in manifest: {qc.get('total_rows')} | included: {qc.get('included')} | "
        f"excluded: {qc.get('excluded')}",
        "",
        "## Pipeline summary",
        "",
        chain.get("summary", ""),
        "",
        "## Claim ceiling",
        "",
        chain.get("claim_ceiling", ""),
        "",
    ]
    if exclusions:
        lines += ["## Exclusions", ""]
        for e in exclusions:
            lines.append(
                f"- dyad={e['dyad_id']} modality={e['modality']} "
                f"condition={e['condition']}: {e['reason']}"
            )
        lines.append("")
    lines += ["## Output files", ""]
    if paths:
        # Derived from the actual write log, so a newly dumped artifact can never
        # go unreported. REPORT.md itself is not in `paths` yet at this point,
        # which is correct: this is that file.
        for name in sorted(paths, key=str):
            lines.append(f"- `{name}`")
    else:
        lines.append(
            "(Output inventory unavailable: report rendered without a write log.)"
        )
    return "\n".join(lines)

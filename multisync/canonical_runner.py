"""
Canonical scientific runner for SyncPipe v1.

This module is the *single* entry point for paper-level analyses. Both the
Python API (:func:`run_canonical`) and the CLI (``syncpipe analyze``) route
through it, which is exactly what guarantees CLI/API parity (Gate 1 pass
criterion: same manifest + config through either entry point yields
byte-equivalent results).

Design note (reuse, do not rewrite):
    The scientific core already exists and is tested:
      * :func:`multisync.pipeline_bridge.records_to_inference_inputs` turns
        loader records into WCC + descriptor ``InferenceInputs`` (ComputationPipeline).
      * :meth:`multisync.inference_pipeline.InferencePipeline.run_audited_evidence_chain`
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

import json
import os
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

from . import __version__
from .io import load_csv
from .pipeline_bridge import InferenceInputs, records_to_inference_inputs
from .inference_pipeline import InferencePipeline
from .feature_definitions import FDR_FEATURES, REFERENCE_FEATURE

__all__ = [
    "ManifestRecord",
    "SyncPipeConfig",
    "LoaderRecord",
    "CanonicalResult",
    "DEFAULT_CONFIG",
    "parse_manifest",
    "parse_config",
    "run_canonical",
]

# Repo root (syncpipe/) for git hash resolution at runtime.
_REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_COLUMNS = [
    "dyad_id", "modality", "condition",
    "person_a_path", "person_b_path", "hz", "mask_path",
]


# ──────────────────────────────────────────────────────────────────────────
# Input contracts
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ManifestRecord:
    """One row of the strict manifest CSV.

    Columns: dyad_id, modality, condition, person_a_path, person_b_path, hz,
    mask_path (optional).
    """

    dyad_id: str
    modality: str
    condition: str
    person_a_path: str
    person_b_path: str
    hz: float
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
            mask_path=_abs(self.mask_path) if self.mask_path else None,
        )


@dataclass
class SyncPipeConfig:
    """Resolved v1 analysis configuration (filled with protocol defaults)."""

    window_size: int = 30
    window_type: str = "rect"
    contrast: Optional[Tuple[str, str]] = None
    fdr_scope: str = "global"
    undefined_policy: str = "gate"
    observation_policy: str = "raise"
    eligibility_policy: str = "raise"
    n_min_dyads: int = 10
    onset_threshold: Union[float, str] = "session_pooled"
    n_permutations: int = 10000
    seed: int = 42
    surrogate_n: int = 100
    design_threshold: float = 0.5
    design_condition: Optional[str] = None

    def resolved_contrast(self) -> Tuple[str, str]:
        if not self.contrast or len(self.contrast) != 2:
            raise ValueError(
                "config.contrast is required and must list exactly two "
                "pre-specified conditions, e.g. ['rest', 'task']."
            )
        return (str(self.contrast[0]), str(self.contrast[1]))


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
    required = ["dyad_id", "modality", "condition", "person_a_path", "person_b_path", "hz"]
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
                mask_path=mask_path,
            )
        )

    if len(hz_set) > 1:
        raise ValueError(
            f"manifest hz must be uniform across rows; found {sorted(hz_set)}. "
            "Resample explicitly before pooling."
        )
    return records


def parse_config(path: Union[str, Path]) -> SyncPipeConfig:
    """Parse a TOML config file into a :class:`SyncPipeConfig`.

    Accepts either a flat table or an ``[analysis]`` section. Missing keys are
    filled from :data:`DEFAULT_CONFIG` (protocol defaults). ``contrast`` is
    required.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("analysis", data) if isinstance(data, dict) else {}

    contrast = section.get("contrast", None)
    if contrast is None:
        raise ValueError(
            "config.contrast is required: specify [analysis] "
            "contrast = ['cond_a', 'cond_b']."
        )
    if not (isinstance(contrast, (list, tuple)) and len(contrast) == 2):
        raise ValueError("config.contrast must be a list/tuple of exactly two condition labels.")
    contrast = (str(contrast[0]), str(contrast[1]))

    onset = section.get("onset_threshold", DEFAULT_CONFIG.onset_threshold)
    if isinstance(onset, str):
        if onset != "session_pooled":
            raise ValueError("config.onset_threshold string must be 'session_pooled' or a numeric value.")
    elif isinstance(onset, (int, float)):
        onset = float(onset)
        if not -1.0 <= onset <= 1.0:
            raise ValueError("numeric config.onset_threshold must lie in [-1, 1].")
    else:
        raise ValueError("config.onset_threshold must be 'session_pooled' or a numeric value.")

    design_condition = section.get("design_condition", None)
    if design_condition is not None:
        design_condition = str(design_condition)

    return SyncPipeConfig(
        window_size=int(section.get("window_size", DEFAULT_CONFIG.window_size)),
        window_type=str(section.get("window_type", DEFAULT_CONFIG.window_type)),
        contrast=contrast,
        fdr_scope=str(section.get("fdr_scope", DEFAULT_CONFIG.fdr_scope)),
        undefined_policy=str(section.get("undefined_policy", DEFAULT_CONFIG.undefined_policy)),
        observation_policy=str(section.get("observation_policy", DEFAULT_CONFIG.observation_policy)),
        eligibility_policy=str(section.get("eligibility_policy", DEFAULT_CONFIG.eligibility_policy)),
        n_min_dyads=int(section.get("n_min_dyads", DEFAULT_CONFIG.n_min_dyads)),
        onset_threshold=onset,
        n_permutations=int(section.get("n_permutations", DEFAULT_CONFIG.n_permutations)),
        seed=int(section.get("seed", DEFAULT_CONFIG.seed)),
        surrogate_n=int(section.get("surrogate_n", DEFAULT_CONFIG.surrogate_n)),
        design_threshold=float(section.get("design_threshold", DEFAULT_CONFIG.design_threshold)),
        design_condition=design_condition,
    )


# ──────────────────────────────────────────────────────────────────────────
# Manifest record -> loader record adapter
# ──────────────────────────────────────────────────────────────────────────
def _load_mask(path: str) -> np.ndarray:
    """Load a discontinuity mask CSV (single 0/1 or boolean column)."""
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"mask file {path} has no numeric column.")
    return num.iloc[:, 0].to_numpy(dtype=float).astype(bool)


def _manifest_record_to_loader_record(
    rec: ManifestRecord, base_dir: Optional[Union[str, Path]] = None
) -> LoaderRecord:
    """Resolve paths and load signals/mask for one manifest row."""
    r = rec.resolve(base_dir)
    a_df = load_csv(r.person_a_path)
    b_df = load_csv(r.person_b_path)
    mask = _load_mask(r.mask_path) if r.mask_path else None
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
    return {
        "python_version": platform.python_version(),
        "syncpipe_version": __version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_hash": git_hash,
        "seed": seed,
    }


def _derive_claimability(chain: Dict[str, Any]) -> Dict[str, Any]:
    """Extract per-feature claimability from the L2 result of the chain."""
    group = chain.get("group_condition_inference") or {}
    per_feature: List[Dict[str, Any]] = []

    def _collect(l2: Any) -> None:
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
                "feature": feat,
                "p_fdr": _safe_float(getattr(r, "p_fdr", None)),
                "significant_05": bool(getattr(r, "significant_05", False)),
                "claimable": getattr(r, "claimable", None),
                "definedness_status": getattr(r, "definedness_status", None),
                "eligibility_status": (elig.get(feat) if isinstance(elig, dict) else elig),
                "n_dyads": _safe_float(getattr(r, "n_dyads", None)),
            })

    if "per_feature" in group:
        _collect(group)
    else:
        for mod, sub in group.items():
            if isinstance(sub, dict):
                _collect(sub)

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
    cfg = parse_config(config) if not isinstance(config, SyncPipeConfig) else config
    contrast = cfg.resolved_contrast()
    if cfg.design_condition is None:
        cfg.design_condition = contrast[1]

    hz = records[0].hz  # uniform (parse_manifest guarantees this)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(manifest).parent if isinstance(manifest, (str, Path)) else Path.cwd()

    # --- preflight QC: resolve loader records; capture load failures ---
    loader_records: List[LoaderRecord] = []
    exclusions: List[Dict[str, Any]] = []
    for rec in records:
        try:
            loader_records.append(_manifest_record_to_loader_record(rec, base_dir))
        except Exception as e:  # load error -> excluded, not a hard crash
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
        design_condition=cfg.design_condition,
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

    # --- inference pipeline (v1 audited evidence chain) ---
    pipe = InferencePipeline(
        features_df=inputs.features_df,
        hz=hz,
        wcc_window_sec=cfg.window_size / hz,
        surrogate_n=cfg.surrogate_n,
        seed=cfg.seed,
    )
    threshold_scope = (
        "per_modality" if isinstance(cfg.onset_threshold, str)
        and cfg.onset_threshold == "session_pooled" else "fixed"
    )
    # design-control audit masks must be keyed by dyad__mod (matching design_pairs),
    # not by dyad__mod__cond. Remap from the cond-keyed discontinuity mask.
    design_masks: Optional[Dict[str, np.ndarray]] = None
    if inputs.discontinuity_mask:
        design_masks = {}
        for key, mask in inputs.discontinuity_mask.items():
            dm = key.rsplit("__", 1)[0]
            design_masks.setdefault(dm, mask)

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
        design_threshold=cfg.design_threshold,
        feature_cols=list(FDR_FEATURES) + list(REFERENCE_FEATURE),
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
        output_dir=output_dir,
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

    # manifest_resolved.json
    _dump_json("manifest_resolved.json", {
        "rows": [vars(r) for r in records],
        "hz_uniform": qc["hz"],
    })

    # config_resolved.toml
    try:
        import tomllib as _t  # noqa: F401  (presence check)
    except ModuleNotFoundError:
        pass
    cfg_path = output_dir / "config_resolved.toml"
    lines = [
        "[analysis]",
        f'window_size = {cfg.window_size}',
        f'window_type = "{cfg.window_type}"',
        f'contrast = ["{cfg.contrast[0]}", "{cfg.contrast[1]}"]' if cfg.contrast else 'contrast = []',
        f'fdr_scope = "{cfg.fdr_scope}"',
        f'undefined_policy = "{cfg.undefined_policy}"',
        f'observation_policy = "{cfg.observation_policy}"',
        f'eligibility_policy = "{cfg.eligibility_policy}"',
        f'n_min_dyads = {cfg.n_min_dyads}',
        f'onset_threshold = "{cfg.onset_threshold}"' if isinstance(cfg.onset_threshold, str)
        else f'onset_threshold = {cfg.onset_threshold}',
        f'n_permutations = {cfg.n_permutations}',
        f'seed = {cfg.seed}',
        f'surrogate_n = {cfg.surrogate_n}',
        f'design_threshold = {cfg.design_threshold}',
        f'design_condition = "{cfg.design_condition}"' if cfg.design_condition else 'design_condition = ""',
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
    _dump_json("design_control_audit.json", chain.get("design_controls"))
    _dump_json("group_inference.json", chain.get("group_condition_inference"))
    _dump_json("claimability.json", claimability)

    # REPORT.md
    report_md = _build_report_md(records, cfg, chain, qc, exclusions, environment)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    paths["REPORT.md"] = str(report_path)

    return paths


def _build_report_md(
    records, cfg, chain, qc, exclusions, environment
) -> str:
    """Human-readable markdown summary of a canonical run."""
    lines = [
        "# SyncPipe v1 — Canonical Analysis Report",
        "",
        f"- SyncPipe version: **{environment.get('syncpipe_version')}**",
        f"- Git hash: `{environment.get('git_hash')}`",
        f"- Seed: {environment.get('seed')}",
        f"- hz: {qc.get('hz')} | window_size: {cfg.window_size} | "
        f"window_type: {cfg.window_type}",
        f"- Contrast: {cfg.contrast}",
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
    lines.append("See `manifest_resolved.json`, `config_resolved.toml`, "
                 "`features.csv`, `wcc_traces/`, `existence_audit.json`, "
                 "`design_control_audit.json`, `group_inference.json`, "
                 "`claimability.json`, `qc_report.json`, `exclusion_report.csv`, "
                 "`environment.json`.")
    return "\n".join(lines)

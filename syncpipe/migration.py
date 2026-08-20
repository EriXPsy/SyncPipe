"""Explicit migration from legacy SyncPipe v1 canonical inputs to v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .__about__ import CONFIG_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
from .contracts import MANIFEST_COLUMNS, analysis_spec_from_mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_manifest_contract(path: str | Path) -> str:
    columns = set(pd.read_csv(path, nrows=0).columns)
    required_v2 = {"signal_type", "unit", "preprocessing_path"}
    return MANIFEST_SCHEMA_VERSION if required_v2 <= columns else "1.x-legacy"


def detect_config_contract(path: str | Path) -> str:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    section = data.get("analysis", data) if isinstance(data, dict) else {}
    has_measure = "main_measure" in section or "primary_endpoint" in section
    has_modalities = "main_modalities" in section or "primary_modalities" in section
    return CONFIG_SCHEMA_VERSION if isinstance(section, dict) and "contrast" in section and has_measure and has_modalities else "1.x-legacy"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def migrate_manifest_v1_to_v2(
    source: str | Path,
    destination: str | Path,
    *,
    signal_type: str,
    unit: str,
    preprocessing_path: str,
) -> Dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    detected = detect_manifest_contract(source)
    frame = pd.read_csv(source)
    added = []
    supplied = {
        "signal_type": str(signal_type),
        "unit": str(unit),
        "preprocessing_path": str(preprocessing_path),
    }
    for column, value in supplied.items():
        if column not in frame:
            frame[column] = value
            added.append(column)
    if "mask_path" not in frame:
        frame["mask_path"] = ""
        added.append("mask_path")
    missing_core = [
        column for column in MANIFEST_COLUMNS
        if column not in frame.columns and column != "mask_path"
    ]
    if missing_core:
        raise ValueError(f"legacy manifest cannot be migrated; missing core columns {missing_core}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[MANIFEST_COLUMNS].to_csv(destination, index=False)
    return {
        "kind": "manifest",
        "source": str(source),
        "destination": str(destination),
        "detected_contract": detected,
        "target_contract": MANIFEST_SCHEMA_VERSION,
        "added_fields": added,
        "user_supplied_assumptions": supplied,
        "source_sha256": _sha256(source),
        "destination_sha256": _sha256(destination),
    }


def migrate_config_v1_to_v2(
    source: str | Path,
    destination: str | Path,
    *,
    primary_endpoint: str,
    primary_modalities: Sequence[str],
) -> Dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    detected = detect_config_contract(source)
    with open(source, "rb") as handle:
        data = tomllib.load(handle)
    section = dict(data.get("analysis", data))
    supplied = {
        "primary_endpoint": str(primary_endpoint),
        "primary_modalities": [str(item) for item in primary_modalities],
    }
    added = []
    for key, value in supplied.items():
        if key not in section:
            section[key] = value
            added.append(key)
    # Validate through the actual v2 contract before writing.
    spec = analysis_spec_from_mapping(section, require_declarations=True)
    resolved = spec.to_dict()
    resolved["main_measure"] = resolved.pop("primary_endpoint")
    resolved["main_modalities"] = resolved.pop("primary_modalities")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[analysis]"]
    for key, value in resolved.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "kind": "config",
        "source": str(source),
        "destination": str(destination),
        "detected_contract": detected,
        "target_contract": CONFIG_SCHEMA_VERSION,
        "added_fields": added,
        "user_supplied_assumptions": supplied,
        "source_sha256": _sha256(source),
        "destination_sha256": _sha256(destination),
    }


def migrate_v1_project(
    *,
    manifest: str | Path,
    config: str | Path,
    output_dir: str | Path,
    signal_type: str,
    unit: str,
    preprocessing_path: str,
    primary_modalities: Sequence[str],
    primary_endpoint: str = "peak_amplitude",
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_report = migrate_manifest_v1_to_v2(
        manifest, output / "manifest.v2.csv", signal_type=signal_type,
        unit=unit, preprocessing_path=preprocessing_path,
    )
    config_report = migrate_config_v1_to_v2(
        config, output / "config.v2.toml", primary_endpoint=primary_endpoint,
        primary_modalities=primary_modalities,
    )
    report = {
        "migration": "SyncPipe v1 canonical inputs -> v2",
        "manifest": manifest_report,
        "config": config_report,
        "warnings": [
            "signal_type, unit, preprocessing provenance, endpoint and modality roles are scientific assumptions supplied by the migrator; review them before analysis",
            "migration changes input contracts only and does not certify scientific comparability with historical outputs",
        ],
    }
    report_path = output / "MIGRATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


__all__ = [
    "detect_manifest_contract", "detect_config_contract",
    "migrate_manifest_v1_to_v2", "migrate_config_v1_to_v2",
    "migrate_v1_project",
]

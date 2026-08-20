"""Canonical manifest and preprocessing-provenance contracts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..__about__ import PREPROCESSING_SCHEMA_VERSION

MANIFEST_COLUMNS = [
    "dyad_id", "modality", "condition",
    "person_a_path", "person_b_path", "hz",
    "signal_type", "unit", "preprocessing_path", "mask_path",
]

SCHEMA_NAMES = {
    "config": "config.schema.json",
    "manifest_record": "manifest-record.schema.json",
    "preprocessing": "preprocessing.schema.json",
}


def load_schema(name: str) -> Dict[str, Any]:
    """Load a packaged JSON Schema by stable public name."""
    if name not in SCHEMA_NAMES:
        raise ValueError(f"unknown schema {name!r}; choose from {sorted(SCHEMA_NAMES)}")
    resource = resources.files("syncpipe").joinpath("schemas", SCHEMA_NAMES[name])
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


def load_preprocessing_provenance(
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
    if payload["schema_version"] != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError(
            f"preprocessing schema_version must be {PREPROCESSING_SCHEMA_VERSION!r}"
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



__all__ = [
    "ManifestRecord", "MANIFEST_COLUMNS", "SCHEMA_NAMES", "load_schema",
    "parse_manifest", "load_preprocessing_provenance",
]

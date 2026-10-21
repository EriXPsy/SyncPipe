"""Canonical manifest and preprocessing-provenance contracts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..__about__ import PREPROCESSING_SCHEMA_VERSION

MANIFEST_COLUMNS = [
    "dyad_id", "modality", "condition",
    "person_a_path", "person_b_path", "hz",
    "signal_type", "unit", "preprocessing_path", "mask_path",
]

# Optional columns (2026-10-16, owner-ratified R7 item 2). Absent (or
# blank) they change nothing: one row is one series, as always.
# MANIFEST_COLUMNS is deliberately NOT extended -- migration.py writes
# manifests with it, and adding the trial columns there would force them
# onto every migrated manifest.
OPTIONAL_TRIAL_COLUMNS = ("trial_id", "stimulus_key")

SCHEMA_NAMES = {
    "config": "config.schema.json",
    "manifest_record": "manifest-record.schema.json",
    "preprocessing": "preprocessing.schema.json",
}


def load_schema(name: str) -> dict[str, Any]:
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
    mask_path: str | None = None
    # Optional trial columns (OPTIONAL_TRIAL_COLUMNS): when trial_id is
    # given, several rows may share one (dyad_id, modality, condition) --
    # each row is one TRIAL of that series, loaded as separate segments no
    # window can straddle. stimulus_key names the trial's stimulus (used
    # to check that trials are comparable across dyads).
    trial_id: str | None = None
    stimulus_key: str | None = None

    def resolve(self, base_dir: str | Path | None = None) -> ManifestRecord:
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


def _optional_text(row: Any, df: pd.DataFrame, column: str) -> str | None:
    """A stripped string from an optional column, or ``None`` if the column or the cell is empty."""
    if column not in df.columns:
        return None
    raw = row[column]
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    return text or None


def _validate_one_series(key: tuple, rows: list[ManifestRecord]) -> None:
    """Trial rules for ONE (dyad_id, modality, condition) series."""
    dyad, modality, condition = key
    label = f"(dyad_id={dyad}, modality={modality}, condition={condition})"
    has_trial = [r.trial_id is not None for r in rows]
    if not any(has_trial):
        if len(rows) > 1:
            raise ValueError("manifest contains duplicate (dyad_id, modality, condition) rows")
        return
    if not all(has_trial):
        raise ValueError(f"manifest series {label} mixes rows with and without trial_id")
    ids = [r.trial_id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"manifest series {label} has duplicate trial_id values")
    keys = [r.stimulus_key for r in rows]
    if any(k is not None for k in keys):
        if any(k is None for k in keys):
            raise ValueError(f"manifest series {label}: stimulus_key must be given for every trial or for none")
        if len(set(keys)) != len(keys):
            raise ValueError(f"manifest series {label} repeats a stimulus_key; give each trial a distinct key")


def _validate_cross_dyad_stimuli(series: dict[tuple, list[ManifestRecord]]) -> None:
    """Every dyad of a (modality, condition) cell must declare the same
    stimuli in the same order (the stranger control pairs trials by
    position)."""
    by_cell: dict[tuple, dict[str, list[str]]] = {}
    for (dyad, modality, condition), rows in series.items():
        if not rows or rows[0].stimulus_key is None:
            continue
        cell = (modality, condition)
        by_cell.setdefault(cell, {})[dyad] = [r.stimulus_key for r in rows]
    for cell, per_dyad in by_cell.items():
        reference = next(iter(per_dyad.values()))
        for dyad, keys in per_dyad.items():
            if keys != reference:
                raise ValueError(
                    f"manifest cell modality={cell[0]}, condition={cell[1]}: dyad {dyad!r} "
                    f"has stimulus order {keys}, expected {reference}; the stranger control "
                    "pairs trials by position, so the stimulus order differs between dyads"
                )


def _validate_series_keys(records: list[ManifestRecord]) -> None:
    """Uniqueness and trial rules for ``(dyad_id, modality, condition)`` series.

    Rows without ``trial_id`` keep the original rule (one row per series).
    Rows with ``trial_id`` may share a series, but a series is either all
    trial rows or a single plain row, trial ids are unique inside it, and
    ``stimulus_key`` (all or none, unique) is only allowed with
    ``trial_id``. Where stimulus keys are given, every dyad of a
    (modality, condition) must show the same stimuli in the same order,
    because the stranger control pairs trials by position.
    """
    series: dict[tuple, list[ManifestRecord]] = {}
    for r in records:
        if r.stimulus_key is not None and r.trial_id is None:
            raise ValueError(f"manifest row dyad={r.dyad_id}: stimulus_key requires trial_id")
        series.setdefault((r.dyad_id, r.modality, r.condition), []).append(r)

    for key, rows in series.items():
        _validate_one_series(key, rows)
    _validate_cross_dyad_stimuli(series)


def parse_manifest(path: str | Path) -> list[ManifestRecord]:
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

    records: list[ManifestRecord] = []
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
        mask_path: str | None = None
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
                trial_id=_optional_text(row, df, "trial_id"),
                stimulus_key=_optional_text(row, df, "stimulus_key"),
            )
        )

    _validate_series_keys(records)
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
) -> dict[str, Any]:
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
    "ManifestRecord", "MANIFEST_COLUMNS", "OPTIONAL_TRIAL_COLUMNS", "SCHEMA_NAMES", "load_schema",
    "parse_manifest", "load_preprocessing_provenance",
]

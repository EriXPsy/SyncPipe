"""Explicit v1-to-v2 input migration contracts."""
from __future__ import annotations

import json

import pandas as pd

from syncpipe.canonical_runner import parse_config, parse_manifest
from syncpipe.cli import build_parser
from syncpipe.migration import (
    detect_config_contract,
    detect_manifest_contract,
    migrate_v1_project,
)


def _legacy_inputs(tmp_path):
    manifest = tmp_path / "manifest.v1.csv"
    pd.DataFrame([{
        "dyad_id": "d1", "modality": "EDA", "condition": "rest",
        "person_a_path": "a.csv", "person_b_path": "b.csv", "hz": 1.0,
    }]).to_csv(manifest, index=False)
    config = tmp_path / "config.v1.toml"
    config.write_text(
        "[analysis]\ncontrast = ['rest', 'task']\nwindow_size = 20\n",
        encoding="utf-8",
    )
    provenance = tmp_path / "preprocessing.json"
    provenance.write_text(json.dumps({
        "schema_version": "1.0.0", "signal_type": "EDA_envelope",
        "output_unit": "z_score",
        "software": {"name": "legacy", "version": "1"},
        "steps": [{"name": "unknown_legacy_pipeline", "parameters": {}}],
    }), encoding="utf-8")
    return manifest, config, provenance


def test_contract_detection_distinguishes_legacy_inputs(tmp_path):
    manifest, config, _ = _legacy_inputs(tmp_path)
    assert detect_manifest_contract(manifest) == "1.x-legacy"
    assert detect_config_contract(config) == "1.x-legacy"


def test_project_migration_writes_valid_v2_inputs_and_report(tmp_path):
    manifest, config, provenance = _legacy_inputs(tmp_path)
    report = migrate_v1_project(
        manifest=manifest, config=config, output_dir=tmp_path / "migrated",
        signal_type="EDA_envelope", unit="z_score",
        preprocessing_path=str(provenance), primary_modalities=["EDA"],
    )
    migrated_manifest = report["manifest"]["destination"]
    migrated_config = report["config"]["destination"]
    records = parse_manifest(migrated_manifest)
    spec = parse_config(migrated_config)
    assert records[0].signal_type == "EDA_envelope"
    assert spec.resolved_primary_modalities() == ("EDA",)
    assert report["manifest"]["source_sha256"] != ""
    assert (tmp_path / "migrated" / "MIGRATION_REPORT.json").exists()
    assert detect_manifest_contract(migrated_manifest) == "2.0.0"
    assert detect_config_contract(migrated_config) == "2.0.0"


def test_cli_exposes_explicit_migrate_command():
    args = build_parser().parse_args([
        "migrate", "-m", "old.csv", "-c", "old.toml", "-o", "new",
        "--signal-type", "EDA_envelope", "--unit", "z_score",
        "--preprocessing-path", "prep.json", "--primary-modalities", "EDA",
    ])
    assert args.command == "migrate"
    assert args.primary_endpoint == "peak_amplitude"

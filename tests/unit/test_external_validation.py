"""External validation kit and structural audit contracts."""
from __future__ import annotations

import json
from pathlib import Path

from syncpipe.canonical_runner import parse_config, parse_manifest, run_canonical
from syncpipe.cli import build_parser
from syncpipe.external_validation import (
    audit_external_bundle,
    create_external_validation_kit,
)


def test_external_kit_contains_blind_protocol_and_valid_inputs(tmp_path):
    paths = create_external_validation_kit(tmp_path / "kit")
    records = parse_manifest(paths["manifest"])
    spec = parse_config(paths["config"])
    assert len(records) == 8
    assert spec.resolved_primary_modalities() == ("SYNTH",)
    protocol = json.loads(Path(paths["protocol"]).read_text(encoding="utf-8"))
    assert protocol["pre_registered_questions"]
    assert (tmp_path / "kit" / "FEEDBACK.md").exists()
    args = build_parser().parse_args(["external-kit", "-o", "kit"])
    assert args.command == "external-kit"


def test_external_check_fails_incomplete_bundle(tmp_path):
    result = audit_external_bundle(tmp_path)
    assert result["structural_pass"] is False
    assert "evidence_graph.json" in result["missing_files"]


def test_external_kit_canonical_run_passes_structural_audit(tmp_path):
    paths = create_external_validation_kit(tmp_path / "kit")
    result_dir = tmp_path / "kit" / "results"
    run_canonical(paths["manifest"], paths["config"], result_dir)
    audit = audit_external_bundle(result_dir)
    assert audit["structural_pass"] is True
    assert audit["claim_decision"]["forbidden_claims"]
    assert "not external construct validation" in audit["scope"]

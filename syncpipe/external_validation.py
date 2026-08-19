"""External-validation kit generation and structural result audit.

The kit tests independent usability and reporting contracts. It is not evidence
of construct validity until an unaffiliated researcher executes and critiques it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .__about__ import PACKAGE_VERSION


REQUIRED_RESULT_FILES = (
    "manifest_resolved.json", "config_resolved.toml", "environment.json",
    "qc_report.json", "exclusion_report.csv", "features.csv",
    "existence_audit.json", "existence_gate.json", "design_control_audit.json",
    "group_inference.json", "evidence_graph.json", "claimability.json",
    "REPORT.md",
)


def create_external_validation_kit(
    output_dir: str | Path, *, seed: int = 20260819, n_dyads: int = 4
) -> Dict[str, str]:
    """Create a deterministic, publication-ineligible external usability kit."""
    if n_dyads < 4:
        raise ValueError("external kit requires at least 4 dyads")
    root = Path(output_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 240
    time = np.arange(n, dtype=float)

    provenance = root / "preprocessing.json"
    provenance.write_text(json.dumps({
        "schema_version": "1.0.0",
        "signal_type": "synthetic_continuous_envelope",
        "output_unit": "z_score",
        "software": {"name": "syncpipe-external-kit", "version": PACKAGE_VERSION},
        "steps": [
            {"name": "deterministic_synthetic_generation", "parameters": {"seed": seed}},
            {"name": "within_trace_zscore", "parameters": {}},
        ],
        "notes": "Usability fixture only; not physiological validation data."
    }, indent=2), encoding="utf-8")

    rows = []
    for dyad in range(n_dyads):
        for condition, shared_weight in (("condition_x", 0.25), ("condition_y", 0.65)):
            shared = np.sin(np.linspace(0, 10 * np.pi, n)) + 0.2 * rng.normal(size=n)
            a = shared_weight * shared + rng.normal(scale=0.7, size=n)
            b = shared_weight * shared + rng.normal(scale=0.7, size=n)
            a = (a - a.mean()) / a.std()
            b = (b - b.mean()) / b.std()
            pa = data_dir / f"d{dyad:02d}_{condition}_a.csv"
            pb = data_dir / f"d{dyad:02d}_{condition}_b.csv"
            pd.DataFrame({"time": time, "value": a}).to_csv(pa, index=False)
            pd.DataFrame({"time": time, "value": b}).to_csv(pb, index=False)
            rows.append({
                "dyad_id": f"d{dyad:02d}", "modality": "SYNTH",
                "condition": condition,
                "person_a_path": str(pa.relative_to(root)),
                "person_b_path": str(pb.relative_to(root)), "hz": 1.0,
                "signal_type": "synthetic_continuous_envelope",
                "unit": "z_score",
                "preprocessing_path": str(provenance.relative_to(root)),
                "mask_path": "",
            })
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    config = root / "config.toml"
    config.write_text(
        "[analysis]\n"
        "window_size = 20\n"
        "contrast = ['condition_x', 'condition_y']\n"
        "primary_endpoint = 'peak_amplitude'\n"
        "primary_modalities = ['SYNTH']\n"
        "undefined_policy = 'gate'\n"
        "observation_policy = 'raise'\n"
        "eligibility_policy = 'warn'\n"
        "n_min_dyads = 10\n"
        "onset_threshold = 'session_pooled'\n"
        "surrogate_n = 20\n"
        "n_permutations = 100\n"
        f"seed = {seed}\n",
        encoding="utf-8",
    )

    (root / "BLIND_PROTOCOL.json").write_text(json.dumps({
        "protocol": "external usability and adversarial interpretation check",
        "package_version": PACKAGE_VERSION,
        "instructions": [
            "Install SyncPipe in a fresh environment without maintainer assistance.",
            "Record installation problems before consulting source code.",
            "Run the command in RUNBOOK.md without changing config.",
            "Interpret evidence_graph.json before inspecting signal generation.",
            "Do not treat this synthetic fixture as construct validation.",
        ],
        "pre_registered_questions": [
            "Can the user identify the strongest permitted claim?",
            "Can the user identify unresolved rival explanations?",
            "Do exclusions and preparation opportunity agree with input data?",
            "Are any report labels misleading without maintainer explanation?",
        ],
    }, indent=2), encoding="utf-8")

    (root / "RUNBOOK.md").write_text(
        "# External Validation Runbook\n\n"
        "This fixture tests installation, contracts, execution, and interpretation—not construct validity.\n\n"
        "```bash\n"
        "python -m venv .venv\n"
        "# activate the environment, then install the candidate wheel/release\n"
        "syncpipe --version\n"
        "syncpipe analyze -m manifest.csv -c config.toml -o results\n"
        "syncpipe external-check -i results -o EXTERNAL_CHECK.json\n"
        "```\n\n"
        "Complete FEEDBACK.md and INDEPENDENT_REPORT.md before contacting the maintainer.\n",
        encoding="utf-8",
    )
    (root / "FEEDBACK.md").write_text(
        "# External User Feedback\n\n"
        "- Research background:\n- Operating system/Python:\n- Installation friction:\n"
        "- Input-contract confusion:\n- Runtime/errors:\n- Misleading labels:\n"
        "- Strongest claim you believe the output permits:\n"
        "- Rival explanations still unresolved:\n- Suggested changes:\n",
        encoding="utf-8",
    )
    (root / "INDEPENDENT_REPORT.md").write_text(
        "# Independent Reproduction Report\n\n"
        "## Identity and independence\nState relationship to the maintainer/project.\n\n"
        "## Environment\n\n## Procedure and deviations\n\n## Structural audit\n\n"
        "## Scientific interpretation\n\n## Disagreements with SyncPipe decisions\n\n"
        "## Reproduction outcome\nPass / partial / fail, with reasons.\n",
        encoding="utf-8",
    )
    expected = root / "EXPECTED_STRUCTURE.json"
    expected.write_text(json.dumps({
        "required_result_files": list(REQUIRED_RESULT_FILES),
        "required_evidence_stages": ["E0", "E1", "E2", "E3", "E4", "E5", "L2"],
        "numerical_results_predeclared": False,
        "publication_eligible": False,
    }, indent=2), encoding="utf-8")
    return {
        "root": str(root), "manifest": str(manifest), "config": str(config),
        "runbook": str(root / "RUNBOOK.md"), "protocol": str(root / "BLIND_PROTOCOL.json"),
    }


def audit_external_bundle(result_dir: str | Path) -> Dict[str, Any]:
    """Audit result structure without asserting expected scientific findings."""
    root = Path(result_dir)
    missing = [name for name in REQUIRED_RESULT_FILES if not (root / name).exists()]
    errors = []
    graph = None
    if not missing:
        try:
            graph = json.loads((root / "evidence_graph.json").read_text(encoding="utf-8"))
            stage_ids = [stage.get("stage_id") for stage in graph.get("stages", [])]
            if stage_ids != ["E0", "E1", "E2", "E3", "E4", "E5", "L2"]:
                errors.append(f"unexpected evidence stages: {stage_ids}")
            decision = graph.get("decision", {})
            if not decision.get("permitted_claim") or not decision.get("forbidden_claims"):
                errors.append("claim decision lacks permitted/forbidden claims")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid evidence_graph.json: {exc}")
        try:
            features = pd.read_csv(root / "features.csv")
            if features.empty:
                errors.append("features.csv is empty")
        except Exception as exc:
            errors.append(f"invalid features.csv: {exc}")
    return {
        "structural_pass": not missing and not errors,
        "missing_files": missing,
        "errors": errors,
        "claim_decision": graph.get("decision") if isinstance(graph, dict) else None,
        "scope": "structural usability audit only; not external construct validation",
    }


__all__ = ["REQUIRED_RESULT_FILES", "create_external_validation_kit", "audit_external_bundle"]

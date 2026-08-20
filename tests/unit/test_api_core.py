from __future__ import annotations

import pytest

# === source: test_syncpipe_namespace.py ===
def test_syncpipe_namespace_exposes_v1_public_api():
    import syncpipe as sp

    for name in [
        "analyze",
        "make_example",
        "Dyad",
        "DynamicAnalyzer",
        "InferencePipeline",
        "AnalysisSpec",
        "EndpointSpec",
        "PreparedObservation",
        "PreparedCohort",
        "EvidenceChain",
        "EvidenceStageResult",
        "EvidenceProfile",
        "ClaimDecision",
        "migrate_v1_project",
        "create_external_validation_kit",
        "audit_external_bundle",
        "MANIFEST_SCHEMA_VERSION",
        "EVIDENCE_SCHEMA_VERSION",
        "feature_status_table",
        "feature_status_latex",
        "explain_feature",
        "run_quality_check",
        "format_qc_report",
        "DataQualityError",
        "compute_session_pooled_threshold",
    ]:
        assert hasattr(sp, name), name

def test_syncpipe_version_available():
    import syncpipe as sp

    assert isinstance(sp.__version__, str)


def test_syncpipe_submodules_import_natively():
    """Real package: `syncpipe.<submodule>` imports without any alias shim."""
    import importlib

    for submodule in (
        "feature_definitions",
        "feature_pipeline",
        "inference_pipeline",
        "computation_pipeline",
        "pipeline_bridge",
        "canonical_runner",
        "validation.l2_between_condition",
        "realtest.lerique_2024",
    ):
        assert importlib.import_module(f"syncpipe.{submodule}") is not None


def test_absent_submodule_still_raises_module_not_found():
    """A typo must fail loudly instead of resolving to something unexpected."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("syncpipe.definitely_not_a_module")


def test_cli_prog_name_uses_canonical_brand():
    """`--help` must say `syncpipe`, matching the distribution and the docs."""
    from syncpipe.cli import build_parser

    assert build_parser().prog == "syncpipe"

# === source: test_reproduce_smoke.py ===
"""
Smoke test for the M2 reproducibility scaffold (A12).

Runs ``scripts/reproduce_lerique_paper.py --fast`` end-to-end on a
SYNTHETIC toy-dyad proxy (NO OSF data, no network). Asserts:
  * the subprocess exits 0, and
  * the derived feature table is written to
    artifacts/paper_lerique/reproduce_fast_features.csv.

This verifies the canonical three-pipeline wiring
(pipeline_bridge -> InferencePipeline.run_audited_evidence_chain)
without requiring the protected dataset. It is a wiring/sanity check,
NOT a scientific claim.
"""
import subprocess
import sys
from pathlib import Path

import pytest

# This file lives at <repo>/tests/unit/, so the repo root is parents[2].
# `parent.parent` resolved to <repo>/tests, where scripts/ does not exist, so
# the skipif below silently skipped this reproducibility smoke test on every
# run it was ever collected in.
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reproduce_lerique_paper.py"


@pytest.mark.slow
@pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="reproduce_lerique_paper.py not present",
)
def test_reproduce_fast_smoke(tmp_path):
    """--fast must run the canonical chain and emit a derived CSV."""
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    output_dir = tmp_path / "paper_lerique"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fast", "--out-dir", str(output_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, (
        f"reproduce_lerique_paper.py --fast exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out_csv = output_dir / "reproduce_fast_features.csv"
    assert out_csv.exists(), (
        f"expected derived table not written: {out_csv}\n"
        f"STDERR:\n{proc.stderr}"
    )

# === source: test_review_fixes.py ===
"""
Regression tests for the reviewer-identified fixes (2026-07-07).

Covers:
- cli.py demo existence-audit summary now leads with BH-FDR-corrected count
  (no more uncorrected p<0.05 headline contradiction).
- feature_pipeline.recommend_features raises on unknown input (case-insensitive
  match); list_features is case-insensitive and warns on unknown tier/axis.
- report.single_dyad_report renders UNDEFINED features as "—" (never drops the
  row), and announces when feature rows are truncated.
"""

import warnings

import numpy as np
import pytest

from syncpipe.batch import _bh_fdr_correction
from syncpipe.core import AnalysisResults
from syncpipe.feature_pipeline import list_features, recommend_features
from syncpipe.report import ReportGenerator


def test_analysis_results_roundtrip_preserves_threshold_metadata():
    original = AnalysisResults(
        dyad_id="d1",
        threshold_meta={"eda__pair": {"mode": "session_pooled", "threshold": 0.61}},
    )
    restored = AnalysisResults.from_dict(original.to_dict())
    assert restored.threshold_meta == original.threshold_meta


# ---------------------------------------------------------------------------
# feature_pipeline: no silent fallback
# ---------------------------------------------------------------------------

def test_recommend_features_case_insensitive():
    # "Structure" (capitalised) must map to the "structure" preset, not fall
    # back silently to "general".
    rec = recommend_features("Structure")
    assert rec["primary"] == ["dwell_time", "switching_rate"]


def test_recommend_features_unknown_raises():
    with pytest.raises(ValueError) as exc:
        recommend_features("temporal")  # not a preset key
    assert "Valid options" in str(exc.value)


def test_list_features_case_insensitive():
    assert len(list_features("Core")) == len(list_features("core"))


def test_list_features_unknown_warns_and_empty():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = list_features(tier="Bogus")
    assert res == []
    assert len(caught) == 1
    assert "Valid tiers" in str(caught[0].message)


# ---------------------------------------------------------------------------
# report: undefined features shown as "—", not hidden
# ---------------------------------------------------------------------------

def _make_results(n_features: int = 3, undefined: bool = True) -> AnalysisResults:
    feats = {"pair_A__B": {}}
    for i in range(n_features):
        name = f"feat_{i}"
        if undefined and i == 0:
            feats["pair_A__B"][name] = float("nan")
        else:
            feats["pair_A__B"][name] = 0.42
    return AnalysisResults(dyad_id="test_dyad", dynamic_features=feats)


def test_report_shows_undefined_as_dash(tmp_path):
    res = _make_results(n_features=3, undefined=True)
    path = tmp_path / "ignore.html"
    ReportGenerator().single_dyad_report(res, filepath=str(path))
    html = open(path, encoding="utf-8").read()
    # The undefined feature's row must still be present (not dropped)...
    assert "Feat 0" in html
    # ...and rendered as an em-dash, exactly like other tables in this module.
    assert "—" in html


def test_report_does_not_drop_undefined_rows(tmp_path):
    res = _make_results(n_features=3, undefined=True)
    path = tmp_path / "ignore2.html"
    ReportGenerator().single_dyad_report(res, filepath=str(path))
    html = open(path, encoding="utf-8").read()
    # All three feature rows present (1 undefined + 2 defined).
    assert html.count("<tr") >= 4  # header + >=3 data rows


def test_report_truncation_notice(tmp_path):
    # 45 features -> only first 40 shown, with an explicit notice.
    res = _make_results(n_features=45, undefined=True)
    path = tmp_path / "ignore3.html"
    ReportGenerator().single_dyad_report(res, filepath=str(path))
    html = open(path, encoding="utf-8").read()
    assert "Showing first 40 of 45" in html


# ---------------------------------------------------------------------------
# cli demo FDR summary logic (mirrors cli.cmd_demo)
# ---------------------------------------------------------------------------

def test_demo_fdr_summary_logic():
    # Replicates the cli.cmd_demo summary computation: headline must be the
    # FDR-corrected count, not the raw p<0.05 count.
    raw_p = [0.04, 0.04, 0.05, 0.5]  # 2 raw < 0.05, but BH-FDR rejects both
    n_raw = int(sum(1 for p in raw_p if p < 0.05))
    _, fdr_sig = _bh_fdr_correction(raw_p, alpha=0.05)
    n_fdr = int(sum(fdr_sig))
    assert n_raw == 2
    assert n_fdr == 0  # corrected headline would NOT claim significance


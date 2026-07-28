from __future__ import annotations

# === source: test_syncpipe_namespace.py ===
def test_syncpipe_namespace_exposes_v1_public_api():
    import syncpipe as sp

    for name in [
        "Dyad",
        "DynamicAnalyzer",
        "InferencePipeline",
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


def test_legacy_multisync_namespace_still_available():
    import multisync as ms

    assert hasattr(ms, "Dyad")
    assert hasattr(ms, "DynamicAnalyzer")

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

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "reproduce_lerique_paper.py"
OUT_CSV = ROOT / "artifacts" / "paper_lerique" / "reproduce_fast_features.csv"


@pytest.mark.slow
@pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="reproduce_lerique_paper.py not present",
)
def test_reproduce_fast_smoke():
    """--fast must run the canonical chain and emit a derived CSV."""
    assert SCRIPT.exists(), f"missing {SCRIPT}"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fast"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, (
        f"reproduce_lerique_paper.py --fast exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert OUT_CSV.exists(), (
        f"expected derived table not written: {OUT_CSV}\n"
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

from multisync.batch import _bh_fdr_correction
from multisync.core import AnalysisResults
from multisync.feature_pipeline import list_features, recommend_features
from multisync.report import ReportGenerator


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


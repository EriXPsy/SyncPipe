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

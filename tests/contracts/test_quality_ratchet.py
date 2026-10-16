"""Quality ratchet (R6, 2026-10-15).

The committed baseline (quality/baseline.json, written by
``scripts/quality_report.py --update-baseline``) may only improve: any
core metric that is "lower is better" must not exceed the locked value.
Tools (ruff/radon) missing locally or in a plain CI job degrade the
check to the metrics that need no external tooling; the quality job in
CI installs them and enforces the full set.

Metric scoping note: ``strict_history_tag_mentions`` counts audit/BUG-/
DECISION- tags in the STRICT packages only — the legacy core carries
them on purpose for governance traceability, and the ratchet must not
penalise that practice (review deviation from the upstream changeset).
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "syncpipe"
BASELINE = ROOT / "quality" / "baseline.json"
NON_CORE = ("validation", "simulation", "realtest", "experimental")
LOWER_IS_BETTER = (
    "ruff_core_E_F", "ruff_core_extended", "undocumented_public_callables",
    "strict_history_tag_mentions", "files_over_800_lines", "functions_cc_over_20",
)


def _core_files() -> list[pathlib.Path]:
    return [p for p in PKG.rglob("*.py") if not any(part in NON_CORE for part in p.relative_to(PKG).parts)]


def _measure_no_tools() -> dict[str, int]:
    total = documented = 0
    for path in _core_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.FunctionDef | ast.ClassDef) and not node.name.startswith("_"):
                total += 1
                documented += bool(ast.get_docstring(node))
    return {
        "undocumented_public_callables": total - documented,
        "files_over_800_lines": sum(len(p.read_text(encoding="utf-8-sig").splitlines()) > 800 for p in _core_files()),
    }


def _ruff_available() -> bool:
    result = subprocess.run(
        [__import__("sys").executable, "-m", "ruff", "--version"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def test_baseline_exists_and_is_well_formed():
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["measured_at_commit"], "baseline must record the commit it was measured at"
    metrics = data["metrics"]
    assert set(LOWER_IS_BETTER) <= set(metrics), "baseline is missing a ratchet metric"
    for key in LOWER_IS_BETTER:
        assert metrics[key] is None or isinstance(metrics[key], int), key


def test_core_metrics_do_not_regress():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    current = _measure_no_tools()
    if _ruff_available():
        out = subprocess.run(
            [__import__("sys").executable, "-m", "ruff", "check",
             *[str(p.relative_to(ROOT)) for p in _core_files()],
             "--select", "E,F", "--ignore", "E501", "--output-format", "concise"],
            capture_output=True, text=True, cwd=ROOT,
        )
        current["ruff_core_E_F"] = sum(
            1 for line in out.stdout.splitlines() if re.match(r"^\S+:\d+:\d+:", line)
        )
    regressions = {
        k: (baseline.get(k), v) for k, v in current.items()
        if k in LOWER_IS_BETTER and baseline.get(k) is not None and v > baseline[k]
    }
    assert not regressions, (
        f"quality ratchet: core regressed on {regressions}; fix the regression "
        f"or run scripts/quality_report.py --update-baseline after IMPROVING"
    )


def test_strict_packages_have_no_history_tags():
    """New code describes behaviour, not history (history -> DECISION_LOG)."""
    tags = re.compile(r"audit|BUG-|DECISION-|gstack|Karpathy|R\d+-")
    findings = []
    for path in (ROOT / "syncpipe" / "surface").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if tags.search(line):
                findings.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not findings, (
        f"strict packages must not carry history tags: {findings}"
    )

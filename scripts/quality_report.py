#!/usr/bin/env python
"""Measure SyncPipe's code-quality indicators and (optionally) tighten the committed baseline.

    python scripts/quality_report.py                 # print the current numbers
    python scripts/quality_report.py --update-baseline   # lock in improvements (refuses regressions)

Quality bar (docs/QUALITY_BAR.md, R6 2026-10-15): a small, thoroughly
tested and documented public surface — bsync's maturity profile, not its
size. The baseline lives in quality/baseline.json and is enforced by
tests/contracts/test_quality_ratchet.py: legacy core code may only get
better, never worse; packages in STRICT_PACKAGES must meet the strict
gates from the start.

Metric design note (review deviation from the upstream changeset):
``history_tag_mentions`` counts audit/BUG-/DECISION- tags, which this
repository writes INTO CODE ON PURPOSE for traceability. Counting them
across the legacy core would penalise a deliberate governance practice,
so the metric is SCOPED to the strict packages only (new code carries
no history tags; history belongs in docs/DECISION_LOG.md).
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "syncpipe"
BASELINE = ROOT / "quality" / "baseline.json"
NON_CORE = ("validation", "simulation", "realtest", "experimental")
STRICT_PACKAGES = ("syncpipe/surface",)
LOWER_IS_BETTER = (
    "ruff_core_E_F", "ruff_core_extended", "undocumented_public_callables",
    "strict_history_tag_mentions", "files_over_800_lines", "functions_cc_over_20",
)
_HISTORY_TAGS = re.compile(r"audit|BUG-|DECISION-|gstack|Karpathy|R\d+-")


def core_files() -> list[pathlib.Path]:
    return [p for p in PKG.rglob("*.py") if not any(part in NON_CORE for part in p.relative_to(PKG).parts)]


def strict_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for rel in STRICT_PACKAGES:
        out.extend((ROOT / rel).rglob("*.py"))
    return out


def _ruff_count(select: str, paths: list[str]) -> int | None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths, "--select", select, "--ignore", "E501",
         "--output-format", "concise"], capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode not in (0, 1):  # ruff missing or crashed
        return None
    return sum(1 for line in result.stdout.splitlines() if re.match(r"^\S+:\d+:\d+:", line))


def public_callables(files: list[pathlib.Path]) -> tuple[int, int]:
    """(public functions/classes, how many have a docstring)."""
    total = documented = 0
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.FunctionDef | ast.ClassDef) and not node.name.startswith("_"):
                total += 1
                documented += bool(ast.get_docstring(node))
    return total, documented


def complexity(files: list[pathlib.Path]) -> tuple[int | None, int | None]:
    """(functions with cyclomatic complexity > 20, maximum), or (None, None) without radon."""
    try:
        from radon.complexity import cc_visit
    except ImportError:
        return None, None
    values = [block.complexity for p in files for block in cc_visit(p.read_text(encoding="utf-8-sig"))]
    return sum(v > 20 for v in values), max(values, default=0)


def measure() -> dict[str, Any]:
    files = core_files()
    sfiles = strict_files()
    total, documented = public_callables(files)
    over20, worst = complexity(files)
    core_paths = [str(p.relative_to(ROOT)) for p in files]
    strict_paths = [str(p.relative_to(ROOT)) for p in sfiles]
    # History tags: scoped to STRICT packages (see module docstring).
    strict_tags = sum(
        len(_HISTORY_TAGS.findall(p.read_text(encoding="utf-8-sig"))) for p in sfiles
    )
    return {
        "ruff_core_E_F": _ruff_count("E,F", core_paths),
        "ruff_core_extended": _ruff_count("B,UP,SIM,C4", core_paths),
        "undocumented_public_callables": total - documented,
        "public_callables": total,
        "strict_history_tag_mentions": strict_tags,
        "files_over_800_lines": sum(len(p.read_text(encoding="utf-8-sig").splitlines()) > 800 for p in files),
        "functions_cc_over_20": over20,
        "max_cyclomatic_complexity": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    current = measure()
    print(json.dumps(current, indent=2))
    if not args.update_baseline:
        return 0
    old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    if old:
        worse = [k for k in LOWER_IS_BETTER if current.get(k) is not None and old["metrics"].get(k) is not None
                 and current[k] > old["metrics"][k]]
        if worse:
            print(f"refusing to raise the baseline for {worse}: fix the regression instead", file=sys.stderr)
            return 1
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps({"measured_at_commit": commit, "metrics": current}, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written to {BASELINE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

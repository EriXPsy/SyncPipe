#!/usr/bin/env python
"""Audit external SyncPipe branding during the multisync -> syncpipe transition.

The v1 release policy is intentionally staged:

- External/user-facing command and import examples should prefer ``syncpipe``.
- The older ``multisync`` namespace remains as a documented compatibility alias.
- Internal implementation files may still live under / import from ``multisync``.

This script scans strict external-facing files for *disallowed* multisync usage
(e.g., ``python -m multisync demo`` or ``import multisync as ms`` in user docs),
while classifying allowed mentions such as compatibility notes and real file
paths like ``multisync/feature_definitions.py``.

Run:
    python scripts/audit_syncpipe_branding.py
    python scripts/audit_syncpipe_branding.py --outdir updates/branding_audit
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


STRICT_EXTERNAL_FILES = [
    "README.md",
    "docs/USER_MANUAL.md",
    "docs/SKILL.md",
    ".github/workflows/ci.yml",
    ".github/workflows/ci.yaml",
    "Dockerfile",
    "pyproject.toml",
    "syncpipe/__init__.py",
    "syncpipe/__main__.py",
]

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".egg-info",
    "build",
    "dist",
    "updates",
}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv",
    ".tex", ".html", ".gitignore", "",
}


@dataclass
class Finding:
    path: str
    line_no: int
    status: str  # allowed | violation
    category: str
    text: str


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _contains_multisync(text: str) -> bool:
    lowered = text.lower()
    return "multisync" in lowered


def _classify_line(path: Path, line: str) -> Tuple[str, str] | None:
    """Return (status, category) for a line mentioning multisync."""
    if not _contains_multisync(line):
        return None

    stripped = line.strip()
    lower = stripped.lower()

    # Third-party tool mention is allowed and scientifically important.
    if "multisyncpy" in lower:
        return "allowed", "third_party_multiSyncPy"

    # Explicitly allowed compatibility wording.
    if "legacy" in lower or "compatibility alias" in lower or "older `multisync`" in lower:
        return "allowed", "legacy_alias_note"

    # Real file/module/package paths are allowed until physical package rename.
    if "multisync/" in stripped or "multisync." in stripped:
        return "allowed", "internal_path_or_module_reference"

    # Packaging and compatibility alias in pyproject.
    if path.name == "pyproject.toml" and ("multisync =" in stripped or "multisync*" in stripped):
        return "allowed", "packaging_compatibility_alias"

    # Docker must copy the implementation package while it physically exists.
    if path.name == "Dockerfile" and "multisync" in stripped:
        return "allowed", "docker_internal_package_copy"

    # syncpipe wrapper necessarily imports multisync during transition.
    if path.parts and path.parts[0] == "syncpipe":
        return "allowed", "syncpipe_wrapper_internal"

    # Disallowed user-facing command/import examples.
    disallowed_snippets = [
        "python -m multisync",
        "multisync demo",
        "multisync analyze",
        "multisync --version",
        "import multisync as",
        "from multisync import",
    ]
    if any(snippet in lower for snippet in disallowed_snippets):
        return "violation", "user_facing_multisync_command_or_import"

    # Anything else in strict external files deserves review.
    return "violation", "unclassified_external_multisync_mention"


def strict_external_findings(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    for rel in STRICT_EXTERNAL_FILES:
        path = root / rel
        if not path.exists():
            continue
        text = _read_text(path)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            cls = _classify_line(path.relative_to(root), line)
            if cls is None:
                continue
            status, category = cls
            findings.append(Finding(rel, i, status, category, line.strip()))
    return findings


def all_text_occurrences(root: Path) -> Tuple[int, int, dict[str, int]]:
    files = 0
    occurrences = 0
    by_top: dict[str, int] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel.as_posix() == "scripts/audit_syncpipe_branding.py":
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = _read_text(path)
        if text is None:
            continue
        count = text.lower().count("multisync")
        if count:
            files += 1
            occurrences += count
            rel = path.relative_to(root)
            top = rel.parts[0]
            by_top[top] = by_top.get(top, 0) + count
    return files, occurrences, dict(sorted(by_top.items(), key=lambda kv: (-kv[1], kv[0])))


def _write_findings_csv(findings: Iterable[Finding], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "line_no", "status", "category", "text"])
        writer.writeheader()
        for item in findings:
            writer.writerow({
                "path": item.path,
                "line_no": item.line_no,
                "status": item.status,
                "category": item.category,
                "text": item.text,
            })


def _write_report(findings: List[Finding], totals: Tuple[int, int, dict[str, int]], path: Path) -> None:
    files, occurrences, by_top = totals
    violations = [f for f in findings if f.status == "violation"]
    allowed = [f for f in findings if f.status == "allowed"]

    lines = [
        "# SyncPipe external branding audit",
        "",
        "## Policy",
        "",
        "- User-facing command/import examples should prefer `syncpipe`.",
        "- `multisync` is allowed only as a documented compatibility alias or as an internal file/module path while the physical implementation package remains named `multisync/`.",
        "- Mentions of the third-party tool `multiSyncPy` are allowed in relationship/lineage sections.",
        "",
        "## Strict external-facing scan",
        "",
        f"- Violations: **{len(violations)}**",
        f"- Allowed `multisync` mentions in strict files: **{len(allowed)}**",
        "",
    ]
    if violations:
        lines.append("### Violations")
        lines.append("")
        for f in violations:
            lines.append(f"- `{f.path}:{f.line_no}` [{f.category}] — {f.text}")
        lines.append("")
    else:
        lines.append("No strict external-facing violations found.")
        lines.append("")

    lines.extend([
        "### Allowed mentions in strict files",
        "",
    ])
    if allowed:
        for f in allowed:
            lines.append(f"- `{f.path}:{f.line_no}` [{f.category}] — {f.text}")
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Whole-repository residual count",
        "",
        f"- Files containing `multisync`/`multiSyncPy`: **{files}**",
        f"- Total case-insensitive `multisync` occurrences: **{occurrences}**",
        "",
        "Residual counts include internal implementation, tests, scripts, experimental files, and allowed compatibility aliases. They are not failures by themselves.",
        "",
        "### Occurrences by top-level path",
        "",
    ])
    for top, count in by_top.items():
        lines.append(f"- `{top}`: {count}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit external SyncPipe branding.")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    parser.add_argument("--outdir", default=None, help="Optional directory for CSV/Markdown report outputs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = strict_external_findings(root)
    totals = all_text_occurrences(root)
    violations = [f for f in findings if f.status == "violation"]

    print("SyncPipe external branding audit")
    print(f"  strict external violations: {len(violations)}")
    print(f"  allowed strict mentions:    {sum(1 for f in findings if f.status == 'allowed')}")
    print(f"  repo residual files/occ:    {totals[0]} files / {totals[1]} occurrences")
    if violations:
        print("\nViolations:")
        for f in violations:
            print(f"  {f.path}:{f.line_no} [{f.category}] {f.text}")

    if args.outdir:
        outdir = Path(args.outdir)
        _write_findings_csv(findings, outdir / "branding_findings.csv")
        _write_report(findings, totals, outdir / "branding_audit_report.md")
        print(f"\nWrote report to {outdir}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

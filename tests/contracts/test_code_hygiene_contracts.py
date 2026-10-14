"""Code-hygiene contracts (review R1, 2026-10-13).

G1 — no undefined names (pyflakes F821 class) and no UTF-8 BOM in any
tracked syncpipe source file. Found in the wild: ``canonical_runner.py``
used ``InferenceInputs`` in a local-variable annotation without importing
it (annotations are not evaluated at runtime, so the bug stayed latent
until a static checker ran).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "syncpipe"

# Type-like names commonly used in annotations; a use without an import
# (or a ``from __future__ import annotations`` string context) is an F821.
_TYPING_NAMES = {
    "List", "Dict", "Tuple", "Optional", "Sequence", "Mapping",
    "Iterable", "Iterator", "Callable", "Any", "Union",
    "InferenceInputs", "LoaderRecord", "AnalysisSpec", "SyncPipeConfig",
    "np", "pd",
}


def _imports_and_defs(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            if node.annotation is not None and isinstance(node.annotation, ast.Name):
                bound.add(node.annotation.id)  # parameter annotations bind nothing
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            bound.update(node.names)
    return bound


def _tracked_sources() -> list[pathlib.Path]:
    import subprocess
    out = subprocess.run(
        ["git", "ls-files", "syncpipe/*.py", "syncpipe/**/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [ROOT / line.strip() for line in out.splitlines() if line.strip()]


@pytest.mark.parametrize("path", _tracked_sources(), ids=lambda p: p.name)
def test_no_undefined_type_names(path: pathlib.Path) -> None:
    """Every type-like name used in code is imported or defined (F821 class)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bound = _imports_and_defs(tree)
    # Module-level import collected only if executed at module scope; walk
    # covers nested defs' own stores, so any Load-context Name not in bound
    # and not a builtin is a candidate. Restrict to the typing-like set to
    # keep false positives at zero.
    used = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = (used & _TYPING_NAMES) - bound
    assert not missing, (
        f"{path.relative_to(ROOT)} uses {sorted(missing)} without importing/"
        f"defining them (latent F821; annotations do not fail at runtime)"
    )


@pytest.mark.parametrize("path", _tracked_sources(), ids=lambda p: p.name)
def test_no_utf8_bom(path: pathlib.Path) -> None:
    """No tracked source starts with a UTF-8 BOM (U+FEFF breaks ast tooling)."""
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), (
        f"{path.relative_to(ROOT)} starts with a UTF-8 BOM"
    )

#!/usr/bin/env python
"""Pre-push verification: reproduce locally what CI will run.

CI builds the package from scratch on every push, and CI's fresh checkout
contains ONLY git-tracked files. A green local suite is therefore not
enough on two axes:

1. **Packaging**: install-time errors (license classifiers, metadata)
   only surface with a fresh editable build.
2. **Untracked-file blindness** (root cause of the 2026-10-07 double
   CI failure): tests that import untracked ``scripts/`` runners or read
   untracked governance docs pass locally but crash CI at collection.
   This is now caught BEFORE the push by running the fast layer with all
   known-untracked targets temporarily moved away (a local replica of the
   CI checkout), and also by statically scanning for imports of
   untracked modules.

Steps (fast default; ``--full`` for the complete suite)::

    1. editable install (build isolation)
    2. pytest fast layer in the NORMAL tree
    3. pytest fast layer in a SIMULATED CI checkout (untracked targets
       hidden); untracked targets are auto-discovered from
       ``git status --porcelain`` -- any file whose importer is tracked
    4. static scan: tracked test files must not import untracked modules

Exit code 0 means it is safe to push.

Usage::

    python scripts/prepush_check.py [--full] [--skip-untrack-sim]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED: {label} (exit {result.returncode})", file=sys.stderr)
        print("Do not push until this passes.", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\nPASSED: {label}")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def untracked_files() -> list[Path]:
    """All untracked files (excluding ignored artifacts), as Paths."""
    out = _git("status", "--porcelain", "--untracked-files=all")
    files = []
    for line in out.splitlines():
        if line.startswith("?? "):
            rel = line[3:].strip().strip('"')
            p = ROOT / rel
            if p.is_file():
                files.append(p)
    return files


def python_module_name(path: Path) -> str | None:
    """Dotted module name if ``path`` is importable from the repo root."""
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def scan_tracked_imports_of_untracked(untracked: list[Path]) -> list[str]:
    """Static guard: tracked .py files importing untracked modules.

    This is the fast, always-on tripwire for the 2026-10-07 failure
    class: a NEW test importing an untracked runner without a skipif
    guard. Existing guarded tests use absence-aware lazy imports, which
    this scan does not flag (they import via importlib inside a guard).

    Only module-level ``import x.y`` / ``from x.y import`` statements are
    scanned; lazy ``importlib.import_module`` calls are exempt because
    the simulation step exercises them for real.
    """
    if not untracked:
        return []
    untracked_mods = {m for m in (python_module_name(p) for p in untracked) if m}
    if not untracked_mods:
        return []
    tracked_py = [
        line.strip().strip('"')
        for line in _git("ls-files", "*.py").splitlines()
        if line.strip()
    ]
    findings = []
    import_re = re.compile(
        r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)"
    )
    for rel in tracked_py:
        p = ROOT / rel
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            m = import_re.match(line)
            if not m:
                continue
            mod = m.group(1)
            # flag only if the imported module (or its root package) is
            # an untracked module and the import is module-level (no indent)
            if not line.startswith((" ", "\t")) and any(
                mod == u or mod.startswith(u + ".") for u in untracked_mods
            ):
                findings.append(f"{rel}:{lineno}: imports untracked {mod!r}")
    return findings


def run_with_untracked_hidden(label: str, pytest_args: list[str]) -> None:
    """Move all untracked files away, run pytest, restore (even on crash).

    This is the local replica of the CI checkout: git-tracked content
    only. The 2026-10-07 failures (ModuleNotFoundError at collection in
    CI while local was green) are impossible to miss after this step.
    """
    hidden = untracked_files()
    if not hidden:
        print(f"\n=== {label} ===\n(no untracked files; identical to normal run)")
        return

    stash = Path(tempfile.mkdtemp(prefix="untrack_sim_"))
    print(f"\n=== {label} ===")
    print(f"hiding {len(hidden)} untracked file(s) in {stash}")

    moved: list[tuple[Path, Path]] = []
    try:
        for p in hidden:
            dest = stash / p.name
            # name collisions across dirs are possible; suffix by stem hash
            if dest.exists():
                dest = stash / f"{hash(str(p)) & 0xffff:x}_{p.name}"
            p.rename(dest)
            moved.append((dest, p))
        result = subprocess.run(pytest_args, cwd=ROOT)
        if result.returncode != 0:
            print(
                "\nFAILED: simulated CI checkout (untracked hidden). "
                "A tracked test depends on an untracked file. "
                "Fix: track the file, or guard the test "
                "(skipif + absence-aware lazy import + suite-health "
                "exemption), then re-run.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
    finally:
        for dest, orig in moved:
            if dest.exists():
                orig.parent.mkdir(parents=True, exist_ok=True)
                dest.rename(orig)
        try:
            stash.rmdir()
        except OSError:
            pass
    print(f"\nPASSED: {label} (restored {len(moved)} file(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the full suite instead of the fast (not slow) layer",
    )
    parser.add_argument(
        "--skip-untrack-sim",
        action="store_true",
        help="skip the simulated-CI-checkout step (NOT recommended)",
    )
    args = parser.parse_args()

    python = sys.executable
    selection = [] if args.full else ["-m", "not slow"]
    layer = "full suite" if args.full else "fast layer"

    run(
        "Step 1/4: editable install (build isolation, catches packaging errors)",
        [python, "-m", "pip", "install", "-e", ".[dev]"],
    )

    # Static tripwire BEFORE any test run: cheap, catches the failure
    # class even when the sim step would be skipped.
    findings = scan_tracked_imports_of_untracked(untracked_files())
    if findings:
        print("\nFAILED: tracked files import untracked modules:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    print("\nPASSED: Step 2/4: static scan (no tracked file imports an untracked module)")

    run(
        f"Step 3/4: pytest {layer} (normal tree)",
        [python, "-m", "pytest", "tests/", *selection, "-q"],
    )

    if not args.skip_untrack_sim:
        run_with_untracked_hidden(
            f"Step 4/4: pytest {layer} in a simulated CI checkout",
            [python, "-m", "pytest", "tests/", *selection, "-q",
             "-p", "no:cacheprovider"],
        )

    print("\nAll checks passed -- safe to push.")


if __name__ == "__main__":
    main()

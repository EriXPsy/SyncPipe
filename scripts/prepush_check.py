#!/usr/bin/env python
"""Pre-push verification: reproduce locally what CI will run.

CI builds the package from scratch on every push, so a green test suite
alone is not enough -- packaging metadata errors only surface at install
time (e.g. a license classifier rejected by newer setuptools, PEP 639).
This script runs the two checks CI performs:

1. ``pip install -e ".[dev]"``  -- fresh editable build with build
   isolation, which pulls a current setuptools; catches packaging errors.
2. ``pytest -m "not slow"``     -- the fast test layer CI runs on push.

Add ``--full`` to run the complete suite (including tests marked ``slow``)
instead of the fast layer.  Exit code 0 means it is safe to push.

Usage::

    python scripts/prepush_check.py [--full]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the full suite instead of the fast (not slow) layer",
    )
    args = parser.parse_args()

    python = sys.executable

    run(
        "Step 1/2: editable install (build isolation, catches packaging errors)",
        [python, "-m", "pip", "install", "-e", ".[dev]"],
    )

    selection = [] if args.full else ["-m", "not slow"]
    layer = "full suite" if args.full else "fast layer"
    run(
        f"Step 2/2: pytest {layer}",
        [python, "-m", "pytest", "tests/", *selection, "-q"],
    )

    print("\nAll checks passed -- safe to push.")


if __name__ == "__main__":
    main()

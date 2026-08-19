"""Regression gate for the *test suite itself*, not for product code.

This file exists because three separate `parents[1]` / `parent.parent` path bugs
silently turned guard tests into permanent skips or into cross-process
ModuleNotFoundError failures (test_api_core.py, test_inference.py,
test_features.py). Those tests *looked* green in CI while verifying nothing.
No other guard can catch that, because every other guard assumes the suite is
actually executing what it claims to execute.

Two narrow, cheap contracts are enforced:

1. Collection baseline — the suite must collect the recorded number of tests,
   split into the recorded slow / not-slow subsets.
2. Path resolvability — every module-level filesystem path in every test module
   must exist. This is the direct root cause of all three historical bugs: a
   miscomputed repo root produced a path that silently did not exist, which a
   `skipif(not path.exists())` then turned into a permanent skip.

Deliberately NOT done here: re-running the full suite in a subprocess to count
skips. This file lives under `tests/`, so such a run re-invokes this file and
recurses; it is also a ~30 minute operation, far too slow for a gate. Contract 2
attacks the same failure mode statically instead.
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

# This file lives at <repo>/tests/, so the repo root is parents[1].
# Verified below by test_repo_root_constant_is_correct rather than assumed.
REPO = pathlib.Path(__file__).resolve().parents[1]

# Recorded 2026-08-02. Any consolidation or rename MUST update both this and
# tests/README.md — changing the baseline is a reviewed act, not a side effect.
_EXPECTED_COLLECTED = 516
# 2026-08-02: 4 tests promoted from the nightly slow layer to the PR gate
# (whole-cascade summary, L2 kwarg names, L1 denominator, cross-process seed
# stability), so slow went 59 -> 55 and not-slow 447 -> 451.
# 2026-08-03: +9 tests pinning the IAAFT argsort+scatter rewrite (six in
# test_significance: rank-order equivalence incl. ties, amplitude-distribution
# preservation, determinism, RNG draw count; three in test_inference: n_workers
# bit-exact parity, pair-count integrity, invalid-value guard), so collected
# 506 -> 515 and not-slow 451 -> 460.
# 2026-08-17: the multisync -> syncpipe clean-cut rename removed the dual-
# namespace/alias tests (a `sys.meta_path` shim no longer exists), so collected
# 515 -> 505 and not-slow 460 -> 450. The prediction.py move-out (to
# experimental/) removed its unit/parity/hardening tests, so collected
# 505 -> 469 and not-slow 450 -> 414. The existence-gate rewrite to a
# second-order group surrogate test added one net test (469 -> 470,
# not-slow 414 -> 415). The autocorrelation-robustness validation added 5 tests
# (470 -> 475, not-slow 415 -> 420). The envelope exporter added 6 tests
# (475 -> 481, not-slow 420 -> 426). Robustness hardening added 3 fast tests
# (481 -> 484, not-slow 426 -> 429). Peak-duration and discriminant-validity
# validation added 12 slow tests plus one suite-health path case for the new
# module (484 -> 497, slow 55 -> 67, not-slow 429 -> 430). Explicit canonical
# endpoint/modality declarations added 4 fast contracts (497 -> 501,
# not-slow 430 -> 434). Segment-wise IAAFT added one fast contract
# (501 -> 502, not-slow 434 -> 435). Canonical schema/provenance contracts
# added 2 fast tests (502 -> 504, not-slow 435 -> 437). Typed AnalysisSpec
# immutability added one fast contract (504 -> 505, not-slow 437 -> 438).
# Prepared-observation geometry added 2 tests plus one suite-health path case
# (505 -> 508, not-slow 438 -> 441). Typed preparation exclusions added one
# fast contract (508 -> 509, not-slow 441 -> 442). Typed evidence claim
# propagation added 3 fast contracts (509 -> 512, not-slow 442 -> 445).
# v1-to-v2 migration added 3 tests plus one suite-health path case
# (512 -> 516, not-slow 445 -> 449).
_EXPECTED_SLOW = 67
_EXPECTED_NOT_SLOW = 449

# Module-level Path constants that are legitimately allowed not to exist
# (e.g. output paths written during a test run). Keyed by "module:attribute"
# so each exemption is explicit and reviewable.
_ALLOWED_MISSING_PATHS: dict[str, str] = {}


def _test_modules() -> list[str]:
    """Dotted module names for every test module in the suite."""
    names = []
    for path in sorted(REPO.joinpath("tests").rglob("test_*.py")):
        rel = path.relative_to(REPO).with_suffix("")
        names.append(".".join(rel.parts))
    return names


def test_repo_root_constant_is_correct():
    """REPO must really be the repo root, not an intermediate directory.

    Every historical path bug in this suite was exactly this mistake, and it is
    silent: the wrong directory still exists, so only the *assets underneath* go
    missing. Anchor on files that exist solely at the root.
    """
    assert (REPO / "pyproject.toml").is_file(), f"{REPO} is not the repo root"
    assert (REPO / "syncpipe").is_dir()
    assert (REPO / "scripts").is_dir()


def test_collect_count_baseline():
    """The suite must collect exactly the recorded number of tests.

    A directory reshuffle that drops a file from collection, or a bad package
    layout that breaks discovery, shows up here as a count mismatch even when
    every surviving test still passes.
    """
    assert _collected_count(()) == _EXPECTED_COLLECTED


def test_slow_split_baseline():
    """The slow / not-slow split must match the recorded baseline.

    Catches a marker accidentally added or dropped, which would let a slow
    validation test leak into the PR gate, or silently demote a PR-gate test to
    nightly-only.
    """
    assert _collected_count(("-m", "slow")) == _EXPECTED_SLOW
    assert _collected_count(("-m", "not slow")) == _EXPECTED_NOT_SLOW


@pytest.mark.parametrize("module_name", _test_modules())
def test_module_level_paths_resolve(module_name):
    """Every module-level Path in every test module must exist.

    This is the static form of the skip-count guard. A `skipif(not
    SCRIPT.exists())` whose SCRIPT was computed from the wrong repo root skips
    forever and reports success; here the non-existent path fails loudly and
    names itself.
    """
    module = importlib.import_module(module_name)
    missing = []
    for attr, value in vars(module).items():
        if attr.startswith("__") or not isinstance(value, pathlib.PurePath):
            continue
        key = f"{module_name}:{attr}"
        if key in _ALLOWED_MISSING_PATHS:
            continue
        if not pathlib.Path(value).exists():
            missing.append(f"{attr} -> {value}")
    assert not missing, (
        f"{module_name} has module-level path(s) that do not exist; a repo-root "
        f"computation is probably wrong, which silently disables any skipif "
        f"guarding on it:\n  " + "\n  ".join(missing)
    )


def _collected_count(marker_args: tuple[str, ...]) -> int:
    """Collect-only test count for the whole suite under the given markers.

    `--collect-only` never executes a test, so invoking pytest on `tests/` from
    inside `tests/` is safe here (no recursion).
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/", *marker_args,
            "--collect-only", "-q", "-p", "no:warnings", "-p", "no:cacheprovider",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = proc.stdout + proc.stderr
    line = next((l for l in reversed(out.splitlines()) if l.strip()), "")
    # Fail loudly rather than silently mis-parsing a changed summary format.
    assert "collected" in line, (
        f"could not parse pytest collect summary (rc={proc.returncode}): {line!r}"
    )
    return int(line.split("/", 1)[0].split()[0])

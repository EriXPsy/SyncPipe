"""B3 eligibility thresholds freeze — QA gate.

Frozen values (2026-07-21, confirmed with user):
  * T_DEF_MIN_WCC_POINTS = 3  — hard floor on finite WCC points per dyad
    (DECISION-04 3-point boxcar + 25-75% RISE / 50% RECOVERY fractions make
    onset+peak+recovery mathematically inseparable below 3 points).
  * N_MIN_DYADS_FDR = 10       — BH-FDR stability floor (codebase already
    treats "<10" as small-sample boundary; B4 LOO shows N>=23 all stable).

This module locks those constants in code, exercises the pure
``check_eligibility`` gate, and verifies the WARN-level NOTE injection into
``qc.run_quality_check`` (reusing the existing ``notes`` field).

QA summary (executed by this suite):
  - T_def boundary: 2 pts -> ineligible, 3 pts -> ok          [PASS]
  - n_min boundary: 9 dyads -> warn, 10 dyads -> ok            [PASS]
  - Constant value asserts + __all__ export                    [PASS]
  - QC NOTE injection (below / above / partial / absent)        [PASS]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync import feature_definitions as fd
from multisync.dataset import SynchronyDataset
from multisync.qc import run_quality_check


# ---------------------------------------------------------------------------
# T_def = 3 : minimum finite WCC sampling points per dyad
# ---------------------------------------------------------------------------

def test_t_def_below_floor_is_ineligible():
    wcc_ok, _ = fd.check_eligibility(n_wcc_points=2, n_dyads=50)
    assert wcc_ok is False, "2 WCC points must be episode-feature ineligible"


def test_t_def_at_floor_is_ok():
    wcc_ok, _ = fd.check_eligibility(n_wcc_points=3, n_dyads=50)
    assert wcc_ok is True, "exactly 3 WCC points must clear the T_def floor"


def test_t_def_above_floor_is_ok():
    wcc_ok, _ = fd.check_eligibility(n_wcc_points=50, n_dyads=50)
    assert wcc_ok is True


# ---------------------------------------------------------------------------
# n_min = 10 : minimum dyad count for meaningful BH-FDR
# ---------------------------------------------------------------------------

def test_n_min_below_floor_warns():
    _, dyads_ok = fd.check_eligibility(n_wcc_points=50, n_dyads=9)
    assert dyads_ok is False, "9 dyads must be flagged as unreliable for FDR"


def test_n_min_at_floor_is_ok():
    _, dyads_ok = fd.check_eligibility(n_wcc_points=50, n_dyads=10)
    assert dyads_ok is True, "exactly 10 dyads must clear the n_min floor"


def test_n_min_above_floor_is_ok():
    _, dyads_ok = fd.check_eligibility(n_wcc_points=50, n_dyads=176)
    assert dyads_ok is True


# ---------------------------------------------------------------------------
# Return-shape / combined logic
# ---------------------------------------------------------------------------

def test_check_eligibility_returns_two_bools():
    res = fd.check_eligibility(n_wcc_points=3, n_dyads=10)
    assert res == (True, True)
    assert isinstance(res[0], bool) and isinstance(res[1], bool)


def test_check_eligibility_both_fail():
    assert fd.check_eligibility(n_wcc_points=1, n_dyads=5) == (False, False)


# ---------------------------------------------------------------------------
# Frozen constant values + export
# ---------------------------------------------------------------------------

def test_frozen_constant_values():
    assert fd.T_DEF_MIN_WCC_POINTS == 3
    assert fd.N_MIN_DYADS_FDR == 10


def test_constants_exported_in_all():
    assert "T_DEF_MIN_WCC_POINTS" in fd.__all__
    assert "N_MIN_DYADS_FDR" in fd.__all__
    assert "check_eligibility" in fd.__all__


# ---------------------------------------------------------------------------
# QC WARN-level NOTE injection (reuses DataQualityReport.notes)
# ---------------------------------------------------------------------------

def _dataset():
    n = 600
    t = np.arange(n) / 100.0
    df = pd.DataFrame({
        "time": t,
        "eda": np.random.default_rng(0).normal(0, 1, n),
    })
    return SynchronyDataset("test", modalities={"eda": df})


def _b3_notes(report):
    return [n for n in report.notes if n.startswith("B3 eligibility")]


def test_qc_injects_both_b3_notes_below_floors():
    report = run_quality_check(
        _dataset(),
        eligibility={"n_wcc_points": 2, "n_dyads": 9},
    )
    b3 = _b3_notes(report)
    assert len(b3) == 2, f"expected 2 B3 notes, got {b3}"
    assert any("T_DEF_MIN_WCC_POINTS=3" in n for n in b3)
    assert any("N_MIN_DYADS_FDR=10" in n for n in b3)


def test_qc_no_b3_notes_above_floors():
    report = run_quality_check(
        _dataset(),
        eligibility={"n_wcc_points": 30, "n_dyads": 176},
    )
    assert _b3_notes(report) == [], "no B3 note when above both floors"


def test_qc_no_b3_notes_without_eligibility():
    report = run_quality_check(_dataset())
    assert _b3_notes(report) == [], "absent eligibility must not change behaviour"


def test_qc_partial_eligibility_only_wcc_note():
    report = run_quality_check(
        _dataset(),
        eligibility={"n_wcc_points": 1},  # no n_dyads supplied
    )
    b3 = _b3_notes(report)
    assert len(b3) == 1, f"only the WCC note should appear, got {b3}"
    assert "T_DEF_MIN_WCC_POINTS=3" in b3[0]


def test_qc_partial_eligibility_only_dyad_note():
    report = run_quality_check(
        _dataset(),
        eligibility={"n_dyads": 5},  # no n_wcc_points supplied
    )
    b3 = _b3_notes(report)
    assert len(b3) == 1, f"only the dyad note should appear, got {b3}"
    assert "N_MIN_DYADS_FDR=10" in b3[0]

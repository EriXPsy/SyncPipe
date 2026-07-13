"""Regression tests for BUG-9 (qc.py false PASS).

``_check_sampling_uniformity`` used to record an ``irregular_sampling``
detail for *mild* irregularity (ISI CV above ``max_isi_cv`` but at or below
the 0.30 hard-fail threshold) and then fall straight through to
``StageVerdict.PASS`` — a silent false PASS.  The stage must now escalate
to ``StageVerdict.WARN`` so the overall QC verdict reflects the warning.

Karpathy discipline: fail loud (Rule 12) — a detected irregularity must
never be swallowed into a PASS.
"""

import numpy as np
import pandas as pd
import pytest

from multisync.dataset import SynchronyDataset
from multisync.qc import (
    DEFAULT_CONFIG,
    StageVerdict,
    run_quality_check,
)


def _dataset_with_time(t, values=None, hz=100.0):
    n = len(t)
    if values is None:
        values = np.random.default_rng(0).normal(0, 1, n)
    df = pd.DataFrame({
        "time": np.asarray(t, dtype=float),
        "eda": np.asarray(values, dtype=float),
    })
    return SynchronyDataset("test", modalities={"eda": df})


def _sampling_stage(report):
    for st in report.stages:
        if st.stage == "sampling_uniformity":
            return st
    raise AssertionError("sampling_uniformity stage missing from report")


def test_qc_mild_irregular_sampling_warns_not_passes():
    """ISI CV > max_isi_cv but <= 0.30 must escalate to WARN, not PASS."""
    rng = np.random.default_rng(7)
    n = 600
    base = 0.01
    # multipliers ~ N(1, 0.15) -> ISI CV ~ 0.15 (between 0.10 guard and 0.30)
    isi = base * (1.0 + 0.15 * rng.normal(size=n - 1))
    isi = np.clip(isi, base * 0.2, None)  # keep strictly positive
    t = np.concatenate([[0.0], np.cumsum(isi)])
    ds = _dataset_with_time(t)
    report = run_quality_check(ds)
    st = _sampling_stage(report)
    assert st.verdict == StageVerdict.WARN
    assert any(d.get("type") == "irregular_sampling" for d in st.details)
    # overall report must also surface the warning
    assert report.overall_verdict == StageVerdict.WARN


def test_qc_highly_irregular_sampling_fails():
    """ISI CV > 0.30 must still hard-FAIL (sanity that WARN path didn't
    swallow the severe case)."""
    rng = np.random.default_rng(11)
    n = 600
    base = 0.01
    isi = base * (1.0 + 0.6 * rng.normal(size=n - 1))  # ISI CV ~ 0.6
    isi = np.clip(isi, base * 0.2, None)
    t = np.concatenate([[0.0], np.cumsum(isi)])
    ds = _dataset_with_time(t)
    report = run_quality_check(ds)
    st = _sampling_stage(report)
    assert st.verdict == StageVerdict.FAIL
    assert any(d.get("type") == "irregular_sampling" for d in st.details)


def test_qc_uniform_sampling_passes():
    """Control: genuinely uniform sampling stays PASS."""
    n = 600
    t = np.arange(n) / 100.0
    ds = _dataset_with_time(t)
    report = run_quality_check(ds)
    st = _sampling_stage(report)
    assert st.verdict == StageVerdict.PASS
    assert st.details == []

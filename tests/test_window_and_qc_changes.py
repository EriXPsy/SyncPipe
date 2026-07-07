"""Regression tests for the C (WCC window taper) and D (QC Stage 4) changes.

C — ``window_type`` parameter on ``sliding_window_wcc``:
    * ``'rect'`` MUST stay byte-identical to the legacy cumsum path
      (backward compatibility — existing results must not move).
    * ``'hann'`` / ``'hamming'`` run without error and differ from rect
      (the taper actually does something).
    * an unknown ``window_type`` raises ``ValueError``.

D — QC Stage 4 "signal integrity":
    * a flatline (long constant run) FAILs stage 4.
    * a zero-variance (constant) signal FAILs stage 4.
    * a healthy signal PASSES stage 4 and the report now carries 4 stages.
"""

import numpy as np
import pandas as pd
import pytest

from multisync.dynamic_features import sliding_window_wcc
from multisync.dataset import SynchronyDataset
from multisync.qc import (
    DEFAULT_CONFIG,
    StageVerdict,
    run_quality_check,
)


# ============================================================
# C — WCC window taper
# ============================================================

def _make_correlated(n=600, w=50, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.4, n)
    return x, y


def test_rect_matches_legacy_cumsum_path():
    """window_type='rect' must be numerically identical to the plain
    covariance cumsum used everywhere before the taper was added."""
    x, y = _make_correlated()
    rect = sliding_window_wcc(x, y, window_size=50, window_type="rect")
    # reference: naive per-window Pearson
    ref = np.array([
        np.corrcoef(x[i:i + 50], y[i:i + 50])[0, 1]
        for i in range(len(x) - 50 + 1)
    ])
    assert np.max(np.abs(rect - ref)) < 1e-9


@pytest.mark.parametrize("wt", ["hann", "hamming", "triang", "gaussian"])
def test_taper_runs_and_differs_from_rect(wt):
    x, y = _make_correlated()
    rect = sliding_window_wcc(x, y, window_size=50, window_type="rect")
    tapered = sliding_window_wcc(x, y, window_size=50, window_type=wt)
    assert np.all(np.isfinite(tapered))
    # the taper must change the estimate (not a silent no-op)
    assert np.max(np.abs(tapered - rect)) > 1e-6


def test_unknown_window_type_raises():
    x, y = _make_correlated(n=120, w=20)
    with pytest.raises(ValueError):
        sliding_window_wcc(x, y, window_size=20, window_type="bogus")


# ============================================================
# D — QC Stage 4 (signal integrity)
# ============================================================

def _dataset_with(values, hz=100.0):
    n = len(values)
    t = np.arange(n) / hz
    df = pd.DataFrame({"time": t, "eda": np.asarray(values, dtype=float)})
    return SynchronyDataset("test", modalities={"eda": df})


def _stage4(report):
    for st in report.stages:
        if st.stage == "signal_integrity":
            return st
    raise AssertionError("signal_integrity stage missing from report")


def test_qc_has_four_stages():
    ds = _dataset_with(np.random.default_rng(0).normal(0, 1, 500))
    report = run_quality_check(ds)
    assert len(report.stages) == 4
    assert {s.stage for s in report.stages} == {
        "temporal_alignment", "nan_integrity",
        "sampling_uniformity", "signal_integrity",
    }


def test_qc_flatline_fails_stage4():
    rng = np.random.default_rng(2)
    # 60% constant plateau, then a different level -> non-zero overall variance
    vals = np.concatenate([np.full(600, 0.5), rng.normal(0, 1, 400) + 5.0])
    ds = _dataset_with(vals)
    st4 = _stage4(run_quality_check(ds))
    assert st4.verdict == StageVerdict.FAIL
    assert any(d.get("type") == "flatline" for d in st4.details)


def test_qc_zero_variance_fails_stage4():
    vals = np.full(1000, 0.5)  # constant -> WCC denominator = 0
    ds = _dataset_with(vals)
    st4 = _stage4(run_quality_check(ds))
    assert st4.verdict == StageVerdict.FAIL
    assert any(d.get("type") == "zero_variance" for d in st4.details)


def test_qc_healthy_signal_passes_stage4():
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 1, 800)  # rich variation, no long constant run
    ds = _dataset_with(vals)
    st4 = _stage4(run_quality_check(ds))
    assert st4.verdict == StageVerdict.PASS
    assert st4.details == []

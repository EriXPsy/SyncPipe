"""
Tests for de-circularized morphology diagnostics (critique B, 2026-07-07).

The data-derived cluster labels must NOT be used as a prediction target,
because the clusters are themselves a function of the shape descriptors —
"predicting" them validates nothing external (circular reasoning). The
default ``diagnostics()`` path therefore OMITs ``incremental_value`` /
``matched_mean_contrast`` and records a note explaining why.

The honest external validity metric is :meth:`MorphologyAnalyzer.ari_vs_condition`,
which compares the discovered cluster structure to experimenter-provided
condition labels.

Covers
------
- ``diagnostics()`` omits circular metrics by default, explains in notes
- ``diagnostics(y_external=...)`` computes them against honest labels
- ``ari_vs_condition`` requires run_method1 and matching length
- ``to_report()`` / ``write_report()`` reflect the circularity logic
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from multisync.morphology import MorphologyAnalyzer


def _make_trace(shape: str, n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if shape == "sustained":
        w = 0.6 + 0.05 * np.sin(2 * np.pi * t / n) + rng.normal(0, 0.05, n)
    elif shape == "single_peak":
        c = n // 2
        w = 0.1 + 0.8 * np.exp(-((t - c) ** 2) / (2 * (n / 8) ** 2)) + rng.normal(0, 0.05, n)
    elif shape == "oscillatory":
        w = 0.4 + 0.45 * np.sin(2 * np.pi * t * 4 / n) + rng.normal(0, 0.05, n)
    else:
        raise ValueError(shape)
    return np.clip(w, -1, 1)


def _make_traces_and_conditions(n_per: int = 6):
    shapes = ["sustained", "single_peak", "oscillatory"]
    traces: list = []
    conds: list = []
    for i, shape in enumerate(shapes):
        for j in range(n_per):
            traces.append(_make_trace(shape, seed=10 * i + j))
            conds.append(shape)  # condition == shape group
    return traces, np.array(conds)


def _analyzer():
    traces, conds = _make_traces_and_conditions()
    analyzer = MorphologyAnalyzer(traces, hz=1.0)
    analyzer.run_method1(max_k=4, seed=42)
    return analyzer, conds


# A small feature set keeps the (inherently O(orders x features x CV))
# incremental_value metric fast; the test only checks *that* the external
# metrics are computed, not their magnitudes.
_SMALL_FEATURES = ["peak_amplitude", "dwell_time", "switching_rate"]


# --- Default path omits circular metrics -----------------------------------

def test_diagnostics_omits_circular_metrics_by_default():
    analyzer, _ = _analyzer()
    diag = analyzer.diagnostics()
    assert "incremental_drop_meansync" not in diag
    assert "incremental_keep_meansync" not in diag
    assert "matched_mean_contrast" not in diag
    # The omission must be explained, not silent.
    joined_notes = " ".join(diag["notes"]).upper()
    assert "CIRCULAR" in joined_notes


def test_diagnostics_keeps_correlation_and_vif():
    analyzer, _ = _analyzer()
    diag = analyzer.diagnostics()
    assert "correlation" in diag
    assert "vif" in diag


def test_diagnostics_computes_external_metrics_with_y_external():
    analyzer, conds = _analyzer()
    diag = analyzer.diagnostics(feature_cols=_SMALL_FEATURES, y_external=conds)
    assert "incremental_drop_meansync" in diag
    assert "incremental_keep_meansync" in diag
    assert "matched_mean_contrast" in diag


# --- ari_vs_condition (honest external validity) ---------------------------

def test_ari_vs_condition_requires_run_method1():
    traces, conds = _make_traces_and_conditions()
    analyzer = MorphologyAnalyzer(traces, hz=1.0)
    with pytest.raises(ValueError):
        analyzer.ari_vs_condition(conds)


def test_ari_vs_condition_returns_ari():
    analyzer, conds = _analyzer()
    res = analyzer.ari_vs_condition(conds)
    assert "method1_ari_vs_condition" in res
    assert isinstance(res["method1_ari_vs_condition"], float)
    assert -1.0 <= res["method1_ari_vs_condition"] <= 1.0


def test_ari_vs_condition_length_mismatch_raises():
    analyzer, conds = _analyzer()
    with pytest.raises(ValueError):
        analyzer.ari_vs_condition(conds[:-1])


# --- to_report / write_report reflect the logic ---------------------------

def test_to_report_omits_circular_without_conditions():
    analyzer, _ = _analyzer()
    analyzer.run_method2(k_range=(2, 3), seed=42)  # cache so to_report is cheap
    report = analyzer.to_report()
    assert "ari_vs_condition" not in report
    diag_report = report["diagnostics"]
    assert "incremental_drop_meansync" not in diag_report
    assert "matched_mean_contrast" not in diag_report
    joined_notes = " ".join(report["notes"]).upper()
    assert "CIRCULAR" in joined_notes


def test_to_report_includes_ari_with_conditions():
    analyzer, conds = _analyzer()
    analyzer.run_method2(k_range=(2, 3), seed=42)  # cache so to_report is cheap
    report = analyzer.to_report(condition_labels=conds)
    assert "ari_vs_condition" in report
    assert "method1_ari_vs_condition" in report["ari_vs_condition"]


def test_write_report_round_trips(tmp_path):
    analyzer, conds = _analyzer()
    analyzer.run_method2(k_range=(2, 3), seed=42)  # cache so write_report is cheap
    p = tmp_path / "morph_report.json"
    analyzer.write_report(p, condition_labels=conds)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert "ari_vs_condition" in loaded
    assert "method1_ari_vs_condition" in loaded["ari_vs_condition"]

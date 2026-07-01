"""
Tests for multisync.morphology core module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.morphology import (
    scalefree_descriptors,
    trace_shape_cluster,
    extract_episodes,
    episode_archetype_cluster,
    morphology_feature_table,
    collinearity_report,
    MorphologyAnalyzer,
)


def _make_synthetic_trace(shape: str, n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if shape == "sustained":
        w = 0.6 + 0.05 * np.sin(2 * np.pi * t / n) + rng.normal(0, 0.05, n)
    elif shape == "single_peak":
        c = n // 2
        w = 0.1 + 0.8 * np.exp(-((t - c) ** 2) / (2 * (n / 8) ** 2)) + rng.normal(0, 0.05, n)
    elif shape == "oscillatory":
        w = 0.4 + 0.45 * np.sin(2 * np.pi * t * 4 / n) + rng.normal(0, 0.05, n)
    elif shape == "asymmetric":
        rise = np.clip((t - n * 0.3) / (n * 0.05), 0, 1)
        decay = np.exp(-(t - n * 0.35) / (n * 0.4))
        decay[t < n * 0.35] = 1.0
        w = 0.1 + 0.8 * rise * decay + rng.normal(0, 0.05, n)
    else:
        raise ValueError(shape)
    return np.clip(w, -1, 1)


def test_scalefree_descriptors_basic():
    w = _make_synthetic_trace("single_peak")
    d = scalefree_descriptors(w)
    assert d is not None
    assert "skewness" in d
    assert "kurtosis" in d
    assert "peak_density" in d


def test_scalefree_descriptors_too_short():
    assert scalefree_descriptors(np.array([1.0, 2.0])) is None


def test_trace_shape_cluster():
    traces = [_make_synthetic_trace(s, seed=i) for i, s in enumerate(
        ["sustained", "single_peak", "oscillatory", "asymmetric"] * 3)]
    res = trace_shape_cluster(traces, max_k=4, seed=42)
    assert res["k_best"] is not None
    assert res["k_best"] >= 2
    assert len(res["labels"]) == len(traces)
    assert res["silhouette_best"] > -1


def test_extract_episodes():
    w = _make_synthetic_trace("single_peak")
    eps = extract_episodes(w, threshold=0.5, threshold_mode="fixed", min_len=4)
    assert isinstance(eps, list)
    assert len(eps) >= 1
    assert all(len(ep) >= 4 for ep in eps)


def test_episode_archetype_cluster():
    traces = [_make_synthetic_trace(s, seed=i) for i, s in enumerate(
        ["sustained", "single_peak", "oscillatory", "asymmetric"] * 4)]
    res = episode_archetype_cluster(traces, threshold=0.3, k_range=(2, 3), seed=42)
    assert res["n_episodes"] > 0
    assert res["waveform_k_best"] in (2, 3)
    assert res["waveform_archetypes"].shape[0] == res["waveform_k_best"]


def test_morphology_feature_table():
    traces = [_make_synthetic_trace("single_peak", seed=i) for i in range(5)]
    df = morphology_feature_table(traces, hz=1.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "shape_skewness" in df.columns
    assert "mean_synchrony" in df.columns
    assert "peak_amplitude" in df.columns


def test_collinearity_report():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "mean_synchrony": rng.normal(0, 1, 30),
        "peak_amplitude": rng.normal(0, 1, 30),
        "dwell_time": rng.normal(0, 1, 30),
    })
    corr, vif = collinearity_report(df, ["mean_synchrony", "peak_amplitude", "dwell_time"])
    assert corr.shape == (3, 3)
    assert len(vif) == 3


def test_morphology_analyzer():
    traces = [_make_synthetic_trace(s, seed=i) for i, s in enumerate(
        ["sustained", "single_peak", "oscillatory", "asymmetric"] * 4)]
    analyzer = MorphologyAnalyzer(traces, hz=1.0)
    m1 = analyzer.run_method1(max_k=4, seed=42)
    m2 = analyzer.run_method2(threshold=0.3, k_range=(2, 3), seed=42)
    df = analyzer.feature_table()
    assert m1["k_best"] is not None
    assert m2["n_episodes"] > 0
    assert len(df) == len(traces)

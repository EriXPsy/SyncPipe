import numpy as np
import pandas as pd
import pytest

from multisync.core import Dyad, DynamicAnalyzer
from multisync.dataset import SynchronyDataset
from multisync.feature_definitions import extract_features, DynamicFeatures
from multisync.surrogate import iaaft_surrogate


def test_iaaft_preserves_empirical_amplitude_distribution():
    x = np.r_[np.zeros(50), np.ones(50) * 10, np.linspace(-3, 3, 50)]
    rng = np.random.default_rng(123)
    y = iaaft_surrogate(x, rng=rng, max_iter=50)
    assert y.shape == x.shape
    assert np.allclose(np.sort(y), np.sort(x))


def test_dynamic_features_roundtrip_preserves_non_fdr_descriptors():
    wcc = np.linspace(0.0, 1.0, 80)
    f = extract_features(wcc, hz=1.0, wcc_window_sec=10.0)
    d = f.to_dict()
    rt = DynamicFeatures.from_dict(d).to_dict()
    for key in [
        "onset_latency",
        "rise_time",
        "recovery_time",
        "onset_latency_imputed",
        "rise_time_imputed",
        "recovery_time_imputed",
        "synchrony_entropy",
        "peak_amplitude",
        "mean_synchrony",
    ]:
        assert np.isclose(rt[key], d[key], equal_nan=True), key


def test_all_absolute_timestamps_align_without_mixed_error():
    base = 1_700_000_000.0
    t = base + np.arange(20, dtype=float)
    ds = SynchronyDataset(
        "abs",
        {
            "a": pd.DataFrame({"time": t, "x": np.arange(20, dtype=float)}),
            "b": pd.DataFrame({"time": t, "y": np.arange(20, dtype=float)}),
        },
    )
    ds.align(target_hz=1.0)
    assert ds._aligned


def test_mixed_absolute_relative_timestamps_fail():
    base = 1_700_000_000.0
    ds = SynchronyDataset(
        "mixed",
        {
            "a": pd.DataFrame({"time": base + np.arange(20, dtype=float), "x": np.arange(20, dtype=float)}),
            "b": pd.DataFrame({"time": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)}),
        },
    )
    with pytest.raises(ValueError, match="Timestamp type mismatch"):
        ds.align(target_hz=1.0)


def test_zscore_all_nan_remains_nan_not_zero():
    ds = SynchronyDataset(
        "nan",
        {"a": pd.DataFrame({"time": np.arange(5, dtype=float), "x": [np.nan] * 5})},
    )
    _, stats = ds.zscore()
    assert stats["a"]["x"]["status"] == "all_nan"
    assert np.isnan(ds.modalities["a"]["x"].to_numpy()).all()


def test_dynamic_analyzer_passes_surrogate_n_into_threshold_meta():
    n = 120
    t = np.arange(n, dtype=float)
    df_a = pd.DataFrame({"time": t, "x": np.sin(t / 8)})
    df_b = pd.DataFrame({"time": t, "y": np.sin(t / 8) + 0.1 * np.cos(t / 3)})
    dyad = Dyad(a=df_a, b=df_b, hz=1.0)
    dyad.align(target_hz=1.0)
    dyad.zscore()
    analyzer = DynamicAnalyzer(window_size=10, surrogate_n=7, enable_prediction=False)
    result = analyzer.fit_transform(dyad)
    assert result.threshold_meta
    assert all(meta.get("surrogate_n") == 7 for meta in result.threshold_meta.values())
    assert all(meta.get("mode") == "within_dyad_surrogate" for meta in result.threshold_meta.values())

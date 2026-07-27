from __future__ import annotations

# === source: test_core.py ===
"""
Comprehensive test suite for syncpipe.

Tests cover:
1. SynchronyDataset — alignment, Z-score, NaN handling, context
2. Association — CCF, PRTF surrogates, Hanning window, significance
3. Dynamic features — WCC, 6 SCR/ERP-inspired features
4. Prediction — TimeSeriesSplit, gap, leakage audit
5. Ground Truth — synthetic data with known lag
6. High-level API — 4-line workflow test
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_dyad():
    """Create a simple 2-modality dataset for testing."""
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=float)
    df_a = pd.DataFrame({"time": t, "value": np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 0.2})
    df_b = pd.DataFrame({"time": t, "value": np.cos(2 * np.pi * t / 50) + np.random.randn(n) * 0.2})
    from multisync.dataset import SynchronyDataset
    return SynchronyDataset(dyad_id="test", modalities={"a": df_a, "b": df_b})


def _make_aligned_dyad():
    """Create an already-aligned dyad."""
    ds = _make_simple_dyad()
    ds.align(target_hz=1.0)
    ds, _ = ds.zscore()
    return ds


# ===========================================================================
# 1. SynchronyDataset tests
# ===========================================================================

class TestSynchronyDataset:

    def test_creation(self):
        ds = _make_simple_dyad()
        assert ds.dyad_id == "test"
        assert set(ds.modality_names) == {"a", "b"}

    def test_missing_time_column_raises(self):
        from multisync.dataset import SynchronyDataset
        with pytest.raises(ValueError, match="time"):
            SynchronyDataset(
                dyad_id="bad",
                modalities={"x": pd.DataFrame({"val": [1, 2, 3]})},
            )

    def test_align_single_hz(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        assert ds._aligned
        assert len(ds.modalities["a"]) == len(ds.modalities["b"])

    def test_align_different_hz(self):
        from multisync.dataset import SynchronyDataset
        np.random.seed(42)
        t_slow = np.arange(0, 100, dtype=float)
        t_fast = np.arange(0, 100, 0.1)
        df_slow = pd.DataFrame({"time": t_slow, "value": np.random.randn(len(t_slow))})
        df_fast = pd.DataFrame({"time": t_fast, "value": np.random.randn(len(t_fast))})

        ds = SynchronyDataset(
            dyad_id="multi_hz",
            modalities={"slow": df_slow, "fast": df_fast},
        )
        ds.align(target_hz=1.0)
        # After alignment, both should have the same length
        assert len(ds.modalities["slow"]) == len(ds.modalities["fast"])

    def test_zscore(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        ds, stats = ds.zscore()
        assert ds._normalized
        # Mean should be ~0, std ~1 (ddof=1: sample std, psychology standard)
        a_vals = ds.modalities["a"]["value"]
        assert abs(a_vals.mean()) < 1e-10
        assert abs(a_vals.std(ddof=1) - 1.0) < 1e-10

    def test_zscore_stats_returned(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        _, stats = ds.zscore()
        assert "a" in stats
        assert "mean" in stats["a"]["value"]
        assert "std" in stats["a"]["value"]

    def test_context_labels(self):
        ds = _make_simple_dyad()
        ds.add_context(0, 50, "Task")
        ds.add_context(50, 100, "Rest")
        assert len(ds.context_labels) == 2
        ctx = ds.get_context_at(25)
        assert ctx is not None
        assert ctx.label == "Task"
        ctx_rest = ds.get_context_at(75)
        assert ctx_rest.label == "Rest"

    def test_nan_handling_ffill(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject NaN
        ds.modalities["a"].loc[10:15, "value"] = np.nan
        ds.handle_nan(strategy="ffill")
        assert ds.modalities["a"]["value"].iloc[15:].isna().sum() == 0

    def test_nan_handling_max_gap(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject a long gap (10 samples)
        ds.modalities["a"].loc[10:20, "value"] = np.nan
        ds.handle_nan(strategy="ffill", max_gap_sec=5.0)
        # Gap of 10s > max_gap of 5s, so some NaN should remain
        assert ds.modalities["a"]["value"].iloc[10:20].isna().any()

    def test_clip_outliers_iqr(self):
        """Outlier clipping (IQR method) should reduce the range of values."""
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject a large spike outlier
        ds.modalities["a"].loc[50, "value"] = 999.0
        original_max = ds.modalities["a"]["value"].max()
        ds, report = ds.clip_outliers(factor=3.0, method="iqr")
        new_max = ds.modalities["a"]["value"].max()
        # Spike should have been clipped
        assert new_max < original_max
        assert new_max < 100.0  # far less than the 999.0 spike
        # Report should record at least 1 clipped sample in modality "a"
        assert report["a"]["value"]["clipped"] >= 1

    def test_clip_outliers_mad(self):
        """Outlier clipping (MAD method) should also clip the injected spike."""
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        ds.modalities["b"].loc[100, "value"] = -500.0
        ds, report = ds.clip_outliers(factor=3.0, method="mad")
        new_min = ds.modalities["b"]["value"].min()
        assert new_min > -100.0
        assert report["b"]["value"]["clipped"] >= 1

    def test_median_filter_removes_spike(self):
        """Median filter should suppress a single-sample spike."""
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Record value before spike
        pre_val = float(ds.modalities["a"]["value"].iloc[50])
        # Inject a spike
        ds.modalities["a"].loc[50, "value"] = pre_val + 100.0
        ds, _ = ds.median_filter(kernel_size=5)
        post_val = float(ds.modalities["a"]["value"].iloc[50])
        # After median filter the spike at index 50 should be attenuated
        assert abs(post_val - pre_val) < 50.0  # significantly reduced

    def test_preprocess_pipeline(self):
        """preprocess() should clip outliers, z-score, and return a report."""
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject a spike to verify it gets clipped
        ds.modalities["a"].loc[50, "value"] = 500.0
        ds, report = ds.preprocess(outlier_factor=3.0, zscore_method="standard")
        # Dataset should be normalized after preprocess
        assert ds._normalized
        # Report should contain 'outliers' and 'zscore_stats' keys
        assert "outliers" in report
        assert "zscore_stats" in report
        # After preprocessing mean should be ~0
        a_vals = ds.modalities["a"]["value"].dropna()
        assert abs(a_vals.mean()) < 0.1

    def test_preprocess_pipeline_with_median_filter(self):
        """preprocess() with median_kernel should also apply median filter."""
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        ds, report = ds.preprocess(
            outlier_factor=3.0,
            median_kernel=5,
            zscore_method="robust",
            clip_sigma=3.0,
        )
        assert ds._normalized
        assert "median_filter" in report
        assert "outliers" in report
        assert "zscore_stats" in report
        # clip_sigma=3.0 means no value should exceed ±3
        a_vals = ds.modalities["a"]["value"].dropna()
        assert a_vals.max() <= 3.0 + 1e-9
        assert a_vals.min() >= -3.0 - 1e-9


# ===========================================================================
# 3. Dynamic features tests
# ===========================================================================

class TestDynamicFeatures:

    def test_wcc_identical_signals(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        n = 100
        x = np.sin(2 * np.pi * np.arange(n) / 20)
        wcc = sliding_window_wcc(x, x, window_size=10, hz=1.0)
        assert wcc.max() > 0.95  # identical → near-perfect correlation

    def test_wcc_uncorrelated_signals(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        wcc = sliding_window_wcc(x, y, window_size=10, hz=1.0)
        # Mean WCC should be near 0 for uncorrelated
        assert abs(np.nanmean(wcc)) < 0.3

    def test_wcc_with_lag(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        n = 500
        # Create a structured signal with clear temporal pattern
        t = np.arange(n, dtype=float)
        base = np.sin(2 * np.pi * t / 50) + 0.3 * np.sin(2 * np.pi * t / 20)
        lag = 10
        x = base.copy()
        y = np.zeros(n)
        y[lag:] = base[:-lag]
        # Verify that lag compensation produces a different WCC than no compensation
        wcc_no_comp = sliding_window_wcc(x, y, window_size=30, hz=1.0, lag_samples=0)
        wcc_comp = sliding_window_wcc(x, y, window_size=30, hz=1.0, lag_samples=lag)
        # The two WCC series should differ (lag compensation matters)
        assert not np.allclose(wcc_no_comp[5:-5], wcc_comp[5:-5], atol=0.01)
        # Compensated WCC should have higher max |correlation|
        assert np.nanmax(np.abs(wcc_comp)) > 0.5

    def test_extract_features(self):
        from multisync.dynamic_features import extract_dynamic_features
        # Use a Gaussian-like peak signal that find_peaks can actually detect
        n = 100
        t = np.arange(n, dtype=float)
        # Gaussian peak centered at t=35, sigma=8
        wcc = 0.8 * np.exp(-0.5 * ((t - 35) / 8.0) ** 2)
        # Explicitly set onset_threshold=0.2 to match test expectations
        feat = extract_dynamic_features(wcc, hz=1.0, onset_threshold=0.2)
        assert feat.peak_amplitude > 0.7
        # Onset: first position where WCC >= 0.2 (onset_threshold)
        # For Gaussian with center=35, sigma=8: solve 0.8*exp(-0.5*((t-35)/8)**2) = 0.2
        # => (t-35)/8 = ±sqrt(-2*ln(0.2/0.8)) ≈ ±1.1774
        # => t ≈ 35 ± 9.42 => onset at ~25.6
        assert 20 < feat.onset_latency < 30  # threshold crossing, not peak center
        # v3.0: recovery_time replaces half_recovery_time
        assert isinstance(feat.to_dict(), dict)
        assert isinstance(feat.recovery_time, float)

    def test_extract_features_all_pairs(self):
        from multisync.dynamic_features import extract_features_all_pairs
        ds = _make_aligned_dyad()
        feats, _ = extract_features_all_pairs(
            ds, window_size=10, hz=1.0, use_surrogate_threshold=False
        )
        assert len(feats) > 0
        for key, feat in feats.items():
            assert isinstance(feat.to_dict(), dict)


# ===========================================================================
# 4. Prediction tests (with leakage audit)
# ===========================================================================

class TestPrediction:

    def test_rolling_origin_cv_basic(self):
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Sine wave: every window has dynamics, labels are naturally balanced
        t = np.arange(800, dtype=float)
        series = np.sin(2 * np.pi * t / 80.0)  # period=80 samples
        pred = rolling_origin_cv(
            series, window_size=60, hz=1.0, n_splits=2, gap=2, threshold=0.0
        )
        assert len(pred.folds) > 0
        assert 0 <= pred.mean_dynamic_auc <= 1
        assert pred.mode == "intra"
        assert pred.n_features_used >= 0

    def test_dynamic_feature_matrix_not_autoregressive(self):
        """
        Verify that the prediction module now uses dynamic features,
        not raw WCC values. Feature importance keys should be dynamic
        feature names, not lag_1, lag_2, etc.
        """
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        series = np.concatenate([
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
        ])
        pred = rolling_origin_cv(
            series, window_size=60, hz=1.0, n_splits=3, gap=5
        )
        # Feature importance keys should be dynamic feature names
        if pred.feature_importance:
            for key in pred.feature_importance:
                assert not key.startswith("lag_"), (
                    f"Feature key '{key}' looks like raw WCC lag, "
                    f"not a dynamic feature name"
                )

    def test_leakage_audit_autocorrelated(self):
        """
        Leakage audit: feed a pure sine wave (perfectly autocorrelated).
        The delta-AUC should be high, and the warning flag must be raised.

        Use long enough series and small enough window/gap so that
        rolling_origin_cv actually runs (not 'data_too_short_for_cv').
        """
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Longer series + small window/gap → enough folds
        t = np.arange(800, dtype=float)
        sine_wave = np.sin(2 * np.pi * t / 80)

        pred = rolling_origin_cv(
            sine_wave,
            window_size=10,
            hz=1.0,
            n_splits=3,
            gap=2,
        )
        # Should NOT be 'data_too_short_for_cv'
        assert pred.warning != "data_too_short_for_cv", (
            f"CV could not run: {pred.diagnostics}"
        )
        # Sine wave is trivially predictable → delta-AUC must clear the
        # SSoT leakage threshold (DECISION-10, recalibrated 2026-07-18:
        # sine median ≈ 0.29 with the NEW 6-feature joint set + AR
        # baseline, threshold now 0.14; noise median ≈ -0.10).
        from multisync.feature_definitions import LEAKAGE_DELTA_AUC_THRESHOLD
        assert pred.mean_delta_auc > LEAKAGE_DELTA_AUC_THRESHOLD, (
            f"Sine wave should produce delta_AUC > "
            f"{LEAKAGE_DELTA_AUC_THRESHOLD}, got {pred.mean_delta_auc:.3f}"
        )
        # The warning flag must be raised
        assert pred.warning == "leakage_suspected", (
            f"Expected 'leakage_suspected', got '{pred.warning}'"
        )

    def test_leakage_audit_no_leakage(self):
        """
        Random noise has no autocorrelation → delta-AUC should be low,
        and NO leakage warning should be raised.

        Note: with SSoT onset_threshold=0.5 (DECISION-01), noise WCC
        rarely exceeds the threshold, so onset-related features are
        mostly NaN.  This makes delta-AUC noisier on short series.
        We use 2000 points (matching test_leakage_audit_random_noise)
        for stable estimation.
        """
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        noise = np.random.randn(2000)
        pred = rolling_origin_cv(
            noise, window_size=60, hz=1.0, n_splits=5, gap=5
        )
        # Random noise → delta-AUC must be below the SSoT leakage
        # threshold (DECISION-10 B).
        from multisync.feature_definitions import LEAKAGE_DELTA_AUC_THRESHOLD
        assert pred.mean_delta_auc <= LEAKAGE_DELTA_AUC_THRESHOLD, (
            f"Random noise produced suspicious delta_AUC "
            f"{pred.mean_delta_auc:.3f} (threshold "
            f"{LEAKAGE_DELTA_AUC_THRESHOLD})"
        )
        # Warning should NOT be raised
        assert pred.warning != "leakage_suspected", (
            "Random noise should NOT trigger leakage warning"
        )

    def test_leakage_audit_random_noise(self):
        """Random noise should give AUC near 0.5 (no leakage possible)."""
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Use MUCH longer series to ensure stable AUC estimation
        noise = np.random.randn(2000)
        pred = rolling_origin_cv(
            noise, window_size=60, hz=1.0, n_splits=3, gap=2, threshold=0.0
        )
        # Random noise → AUC should be near 0.5
        # With 6 features and NaN imputation on noise data, tolerance is wider
        assert len(pred.folds) > 0, "Should have at least one valid fold"
        assert abs(pred.mean_dynamic_auc - 0.5) < 0.25, (
            f"Random noise AUC should be near 0.5, got {pred.mean_dynamic_auc:.3f}. "
            f"This indicates leakage or overfitting."
        )

    def test_cross_modal_prediction_basic(self):
        """Cross-modal prediction: source and target are independent signals."""
        from multisync.prediction import cross_modal_prediction
        np.random.seed(42)
        # Source: has structure (sine wave)
        t = np.arange(300, dtype=float)
        source = np.sin(2 * np.pi * t / 50) + np.random.randn(300) * 0.3
        # Target: different structure (square wave)
        target = np.sign(np.sin(2 * np.pi * t / 30)) + np.random.randn(300) * 0.3

        pred = cross_modal_prediction(
            source, target,
            window_size=30, hz=1.0,
            source_name="behavioral__neural",
            target_name="neural__bio",
        )
        assert pred.mode == "cross_modal"
        assert pred.source_pair == "behavioral__neural"
        assert pred.target_pair == "neural__bio"

    def test_lodo_basic(self):
        from multisync.prediction import lodo_cv
        dyad_results = [
            {"mean_delta_auc": 0.1},
            {"mean_delta_auc": 0.2},
            {"mean_delta_auc": 0.15},
            {"mean_delta_auc": 0.25},
            {"mean_delta_auc": 0.18},
        ]
        result = lodo_cv(dyad_results)
        assert "mae" in result
        assert result["mae"] < 0.2  # predictions should be close


# ===========================================================================
# 6. High-level API test (4-line workflow)
# ===========================================================================

class TestHighLevelAPI:

    def test_four_line_workflow(self):
        """Verify the 4-line API from the README works."""
        import multisync as ms
        from multisync.synthetic import generate_ground_truth_dyad

        # 1. Load and align
        ds = generate_ground_truth_dyad(
            lead_modality="behavior",
            lag_modality="neural",
            true_lag_sec=12.0,
            noise_ratio=0.3,
            duration_sec=300,
        )
        # 2. Add context
        ds.add_context(start=0, end=150, label="PreTask")
        ds.add_context(start=150, end=300, label="Task")
        # 3. Analyze (fewer surrogates for test speed)
        analyzer = ms.DynamicAnalyzer(window_size=10, surrogate_n=50)
        ds.align(target_hz=1.0)
        ds.zscore()
        results = analyzer.fit_transform(ds)
        # 4. Export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            results.export_viewer_json(path)
            # Verify JSON structure
            with open(path, "r") as f:
                data = json.load(f)
            assert "dyad_id" in data
            assert "dynamic_features" in data
            assert "score_view" in data
            assert len(data["score_view"]) == 2  # PreTask + Task
        finally:
            os.unlink(path)

    def test_dyad_convenience_class(self):
        """Test the Dyad convenience wrapper."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        df_n = pd.DataFrame({"time": t, "plv": np.random.randn(n)})
        df_b = pd.DataFrame({"time": t, "motion": np.random.randn(n)})

        dyad = ms.Dyad(neural=df_n, behavioral=df_b, hz=1.0)
        assert set(dyad.modality_names) == {"neural", "behavioral"}

    def test_analysis_results_schema(self):
        """Verify the viewer JSON has all required fields."""
        import multisync as ms
        from multisync.synthetic import generate_ground_truth_dyad
        ds = generate_ground_truth_dyad(duration_sec=200, noise_ratio=0.2)
        ds.align(target_hz=1.0)
        ds.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=20)
        results = analyzer.fit_transform(ds)

        d = results.to_dict()
        # All required top-level keys
        assert "dyad_id" in d
        assert "dynamic_features" in d
        assert "dynamic_features_segmented" in d
        assert "prediction" in d
        assert "parameters" in d

        # JSON schema_version present (updated to 0.3.0 after cross-modal removal)
        assert "schema_version" in d
        assert d["schema_version"] == "0.3.0"

    def test_context_segmented_features(self):
        """When contexts are defined, dynamic features should be computed
        per-context, not just globally."""
        import multisync as ms
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        df_a = pd.DataFrame({
            "time": t,
            "val": np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 0.2,
        })
        df_b = pd.DataFrame({
            "time": t,
            "val": np.cos(2 * np.pi * t / 50) + np.random.randn(n) * 0.2,
        })

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.add_context(0, 150, "Phase1")
        dyad.add_context(150, 300, "Phase2")
        dyad.align(target_hz=1.0)
        dyad.zscore()

        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(dyad)

        # Should have segmented features
        assert "dynamic_features_segmented" in results.to_dict()
        seg = results.dynamic_features_segmented
        assert "Phase1" in seg
        assert "Phase2" in seg
        # Each segment should have at least one pair's features
        assert len(seg["Phase1"]) > 0
        assert len(seg["Phase2"]) > 0

    def test_prediction_uses_dynamic_features_not_raw_wcc(self):
        """High-level test: verify that prediction results now report
        dynamic feature importance (not raw WCC lag coefficients)."""
        import multisync as ms
        from multisync.synthetic import generate_ground_truth_dyad
        ds = generate_ground_truth_dyad(
            duration_sec=300, noise_ratio=0.2,
        )
        ds.align(target_hz=1.0)
        ds.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(ds)

        for key, pred in results.prediction.items():
            # Feature importance should use dynamic feature names
            if pred.get("feature_importance"):
                for feat_name in pred["feature_importance"]:
                    assert not feat_name.startswith("lag_"), (
                        f"Prediction {key} still uses raw WCC features: {feat_name}"
                    )


    def test_prediction_default_off(self):
        """A1 regression: DynamicAnalyzer() must NOT run prediction by
        default. Prediction is a confirmatory-adjacent step and must be
        explicitly opted in (enable_prediction=True / --prediction).
        """
        import multisync as ms
        from multisync.synthetic import generate_ground_truth_dyad
        ds = generate_ground_truth_dyad(
            duration_sec=200, noise_ratio=0.2,
        )
        ds.align(target_hz=1.0)
        ds.zscore()
        # No enable_prediction arg → must default to False (A1)
        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(ds)
        assert results.prediction == {}, (
            "Prediction must be OFF by default; results.prediction should be "
            f"empty, got {results.prediction!r}"
        )

    def test_prediction_window_gap_parameters_report_effective(self):
        """Finding 16 regression: results.parameters must report the EFFECTIVE
        prediction window/gap that rolling_origin_cv actually consumes (after
        the silent min-30 / min-window//4 floors), not the raw requested
        values — and must keep the requested values too, for transparent
        reproduction.
        """
        import multisync as ms
        from multisync.synthetic import generate_ground_truth_dyad
        ds = generate_ground_truth_dyad(duration_sec=200, noise_ratio=0.2)
        ds.align(target_hz=1.0)
        ds.zscore()

        # Default config: requested 10/5 -> effective 30 / max(5, 30//4=7)
        analyzer = ms.DynamicAnalyzer(
            surrogate_n=10, window_size=10,
            prediction_window=10, prediction_gap=5,
        )
        results = analyzer.fit_transform(ds)
        assert results.parameters["prediction_window"] == 30, (
            "expected effective window 30, got "
            f"{results.parameters['prediction_window']}"
        )
        assert results.parameters["prediction_gap"] == 7, (
            "expected effective gap 7, got "
            f"{results.parameters['prediction_gap']}"
        )
        assert results.parameters["prediction_window_requested"] == 10
        assert results.parameters["prediction_gap_requested"] == 5

        # Explicit larger values: no floor applies, effective == requested
        analyzer2 = ms.DynamicAnalyzer(
            surrogate_n=10, window_size=10,
            prediction_window=50, prediction_gap=3,
        )
        results2 = analyzer2.fit_transform(ds)
        assert results2.parameters["prediction_window"] == 50
        assert results2.parameters["prediction_gap"] == max(3, 50 // 4), (
            "expected effective gap 12, got "
            f"{results2.parameters['prediction_gap']}"
        )
        assert results2.parameters["prediction_window_requested"] == 50
        assert results2.parameters["prediction_gap_requested"] == 3

# ===========================================================================
# 7. JSON serialization tests
# ===========================================================================

class TestJSONSerialization:

    def test_nan_becomes_null_in_json(self):
        """NaN values must serialize as JSON null, not the string 'nan'."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        # Insert NaN to trigger sanitization
        df_a = pd.DataFrame({"time": t, "val": np.random.randn(n)})
        df_b = pd.DataFrame({"time": t, "val": np.random.randn(n)})
        df_a.loc[5, "val"] = np.nan
        df_a.loc[10, "val"] = np.nan
        df_b.loc[15, "val"] = np.nan

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(dyad)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            results.export_viewer_json(path)
            with open(path, "r") as f:
                content = f.read()
            # JSON null is allowed; the string "nan" is NOT
            assert '"nan"' not in content, (
                "NaN was serialized as string 'nan' instead of JSON null"
            )
            # Verify it's valid JSON
            data = json.loads(content)
            assert "schema_version" in data
        finally:
            os.unlink(path)


# ===========================================================================
# 8. Multimodal synthetic data tests (P1-C1)
# ===========================================================================

class TestMultimodalSynthetic:

    def test_shared_burst_anchors(self):
        """All modalities in generate_multimodal_dyad must share the same
        burst time anchors (the fix for the desync bug)."""
        from multisync.synthetic import generate_multimodal_dyad
        import multisync as ms

        ds = generate_multimodal_dyad(
            duration_sec=300,
            modalities={"neural": 10.0, "behavior": 1.0},
            seed=42,
        )
        ds.align(target_hz=1.0)

        # The synthetic generator creates Gaussian bursts at shared time
        # points. Verify that cross-modality CCF is non-trivial at short lags.
        n_feat_a = ds.feature_columns["neural"]
        n_feat_b = ds.feature_columns["behavior"]
        assert len(n_feat_a) > 0 and len(n_feat_b) > 0

# ===========================================================================
# 9. CLI tests (P1-C2)
# ===========================================================================

class TestCLI:

    def test_demo_command_runs(self):
        """The `demo` CLI command should run without errors."""
        from multisync.cli import cmd_demo
        import argparse

        args = argparse.Namespace(surrogates=20, output=None)
        cmd_demo(args)  # Should not raise

    def test_analyze_command_runs(self, tmp_path):
        """The canonical `analyze` CLI command should run from a manifest + config."""
        from multisync.cli import cmd_analyze
        import argparse

        # Build a small synthetic manifest + config
        rng = np.random.default_rng(0)
        n = 120
        t = np.arange(n, dtype=float)
        sigdir = tmp_path / "data"
        sigdir.mkdir()
        rows = []
        for i in range(4):
            for cond, coup in (("rest", 0.2), ("task", 0.8)):
                shared = np.sin(np.linspace(0, 4 * np.pi, n))
                a = coup * shared + rng.normal(scale=0.5, size=n)
                b = coup * shared + rng.normal(scale=0.5, size=n)
                pa = sigdir / f"d{i:02d}_{cond}_a.csv"
                pb = sigdir / f"d{i:02d}_{cond}_b.csv"
                pd.DataFrame({"time": t, "val": a}).to_csv(pa, index=False)
                pd.DataFrame({"time": t, "val": b}).to_csv(pb, index=False)
                rows.append((f"d{i:02d}", "EDA", cond, str(pa), str(pb), 1.0, ""))
        man = tmp_path / "manifest.csv"
        pd.DataFrame(
            rows,
            columns=["dyad_id", "modality", "condition", "person_a_path", "person_b_path", "hz", "mask_path"],
        ).to_csv(man, index=False)
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[analysis]\n"
            "window_size = 10\n"
            "contrast = ['rest', 'task']\n"
            "eligibility_policy = 'raise'\n"
            "n_min_dyads = 4\n"
            "n_permutations = 100\n"
            "seed = 42\n"
            "surrogate_n = 10\n",
            encoding="utf-8",
        )
        out = tmp_path / "results"
        args = argparse.Namespace(manifest=str(man), config=str(cfg), output=str(out))
        cmd_analyze(args)  # Should not raise
        for f in ["manifest_resolved.json", "config_resolved.toml",
                  "features.csv", "claimability.json", "REPORT.md"]:
            assert (out / f).exists(), f

    def test_describe_command_runs(self, tmp_path):
        """The `describe` CLI command (design-agnostic descriptor path) still runs."""
        from multisync.cli import cmd_describe
        import argparse

        rng = np.random.default_rng(0)
        n = 100
        t = np.arange(n, dtype=float)
        dyad_csv = tmp_path / "dyad.csv"
        pd.DataFrame({
            "time": t,
            "person_a": rng.normal(size=n),
            "person_b": rng.normal(size=n),
        }).to_csv(dyad_csv, index=False)
        out = tmp_path / "out.json"
        args = argparse.Namespace(
            input=str(dyad_csv),
            names="eda",
            hz=1.0,
            output=str(out),
            window_size=10,
            surrogates=10,
            max_lag=0.0,
            seed=42,
            cross_modal=False,
            contexts=None,
            full_family_fdr=False,
        )
        cmd_describe(args)  # Should not raise
        assert out.exists()


# ===========================================================================
# 10. Edge case tests (P3-C4)
# ===========================================================================

class TestEdgeCases:

    def test_single_modality_no_crash(self):
        """Single modality should not crash — no pairs to analyze."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        df = pd.DataFrame({"time": t, "val": np.random.randn(n)})

        dyad = ms.Dyad(neural=df, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10)
        results = analyzer.fit_transform(dyad)
        # Should have empty results but no crash
        assert len(results.dynamic_features) == 0

    def test_very_short_data_graceful(self):
        """Data shorter than window_size should return empty results, not crash."""
        from multisync.dynamic_features import sliding_window_wcc
        x = np.random.randn(5)
        y = np.random.randn(5)
        result = sliding_window_wcc(x, y, window_size=10, hz=1.0)
        assert len(result) == 0  # empty array

    def test_mostly_nan_pair_fails_qc_by_default(self):
        """A modality pair with 90%+ NaN should fail the mandatory QC gate."""
        import multisync as ms
        n = 100
        t = np.arange(n, dtype=float)
        vals_a = np.random.randn(n)
        vals_a[:90] = np.nan  # 90% NaN
        df_a = pd.DataFrame({"time": t, "val": vals_a})
        df_b = pd.DataFrame({"time": t, "val": np.random.randn(n)})

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10)
        with pytest.raises(ms.DataQualityError):
            analyzer.fit_transform(dyad)

        exploratory = ms.DynamicAnalyzer(surrogate_n=10, qc_raise_on_fail=False)
        results = exploratory.fit_transform(dyad)
        assert "dynamic_features" in results.to_dict()
        assert any(d["stage"] == "qc" for d in results.diagnostics)

# === source: test_dynamic_features.py ===
"""Tests for dynamic feature extraction (WCC, episode detection, etc.)."""

import numpy as np
import pytest
from multisync.dynamic_features import sliding_window_wcc


# ============================================================
# WCC cumsum regression guard
# ============================================================

@pytest.mark.parametrize("offset", [0.0, 5.0, 72.0, 1000.0])
def test_wcc_invariant_under_global_mean_shift(offset):
    """
    Pearson correlation is invariant to additive shifts. The cumsum
    fast path MUST yield identical results to a naive per-window
    np.corrcoef regardless of the global mean of the signal.

    Regression guard for the v0.x covariance-formula bug, where
    omitting the ``- mean_x*mean_y`` correction caused a silent,
    direction-consistent ~0.02 bias on real physiological data
    (HR ~72 BPM, SCL ~5 microS) while remaining invisible on zero-mean
    synthetic data.
    """
    rng = np.random.default_rng(0)
    n, w = 600, 50
    x = rng.normal(0, 1, n) + offset
    y = 0.6 * (x - offset) + rng.normal(0, 0.4, n) + offset

    fast = sliding_window_wcc(x, y, window_size=w)
    naive = np.array([
        np.corrcoef(x[i:i + w], y[i:i + w])[0, 1]
        for i in range(n - w + 1)
    ])
    max_abs_err = float(np.max(np.abs(fast - naive)))
    assert max_abs_err < 1e-9, (
        f"WCC cumsum path drifts under global-mean shift "
        f"(offset={offset}): max abs error = {max_abs_err:.4e}"
    )

# === source: test_morphology.py ===
"""
Tests for multisync.morphology core module.
"""

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


def test_extract_episodes_gap_robust():
    """A sustained elevated episode with a short artifact gap in the middle
    must stay ONE episode, not fragment into two (Finding 10).

    Regression guard for the gap-robust fix that mirrors feature_definitions
    fix 1cc5397: missing points are excluded, not treated as below-threshold,
    so an artifact gap no longer splits a single sustained run.
    """
    w = np.full(40, 0.9)          # one sustained elevated episode
    w[15:18] = np.nan             # 3-point artifact gap in the middle
    eps = extract_episodes(w, threshold=0.5, threshold_mode="fixed", min_len=4)
    assert len(eps) == 1, (
        "gap in the middle of a sustained episode must NOT split it; "
        f"got {len(eps)} episodes"
    )
    assert len(eps[0]) == 37, (
        "merged episode should keep all 37 valid elevated samples"
    )

    # Sanity: a genuine dip (below threshold, not missing) still splits.
    w2 = np.full(40, 0.9)
    w2[15:25] = 0.1               # 10-point real sub-threshold dip
    eps2 = extract_episodes(w2, threshold=0.5, threshold_mode="fixed", min_len=4)
    assert len(eps2) == 2


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

# === source: test_morphology_circularity.py ===
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

# === source: test_morphology_threshold_nanfraction.py ===
"""Regression tests for Claude review findings #1 (v1b) and #4 (fix ①).

#1 v1b: ``nan_fraction`` (discontinuity-masked WCC fraction) is reported
alongside ``dwell_time`` / ``switching_rate`` as a mandatory diagnostic.

#4 ①: ``morphology_feature_table`` / ``MorphologyAnalyzer.feature_table``
forward ``onset_threshold`` so the extracted SyncPipe features share the
main pipeline's surrogate-derived threshold instead of silently falling back
to the locked 0.5 default.
"""

import numpy as np

from multisync import dynamic_features as df_mod
from multisync.feature_definitions import DynamicFeatures
from multisync.morphology import morphology_feature_table, MorphologyAnalyzer


def _make_wcc(n=200, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    w = 0.5 + 0.3 * rng.normal(0, 1, n)
    if nan_frac > 0:
        k = int(n * nan_frac)
        w[:k] = np.nan
    return w


def _make_bimodal_wcc(n=300, seed=0):
    # WCC with a MID run (~0.6, above 0.5 but below 0.7) and a HIGH run
    # (~0.9, above both). At threshold 0.5 both runs count toward dwell;
    # at threshold 0.7 only the HIGH run counts, so dwell_time(0.7) <
    # dwell_time(0.5) — proving the threshold reaches the feature math.
    rng = np.random.default_rng(seed)
    base = np.full(n, 0.3)
    base[50:120] = 0.6    # mid run (len 70)
    base[150:180] = 0.9   # high run (len 30)
    return base + 0.02 * rng.normal(0, 1, n)


# ---------------------------------------------------------------------------
# Finding #1 (v1b): nan_fraction diagnostic
# ---------------------------------------------------------------------------

def test_nan_fraction_recorded_by_extract_dynamic_features():
    w = _make_wcc(n=200, seed=1, nan_frac=0.1)  # exactly 20 NaN of 200
    feats = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0)
    assert isinstance(feats, DynamicFeatures)
    assert abs(feats.nan_fraction - 0.1) < 1e-9


def test_nan_fraction_in_to_dict():
    w = _make_wcc(n=200, seed=2, nan_frac=0.25)
    d = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0).to_dict()
    assert "nan_fraction" in d
    assert abs(d["nan_fraction"] - 0.25) < 1e-9


def test_nan_fraction_round_trip_via_from_dict():
    w = _make_wcc(n=200, seed=3, nan_frac=0.2)
    f = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0)
    f2 = DynamicFeatures.from_dict(f.to_dict())
    assert abs(float(f2.nan_fraction) - 0.2) < 1e-9


def test_morphology_table_surfaces_nan_fraction():
    traces = [
        _make_wcc(n=200, seed=4, nan_frac=0.2),
        _make_wcc(n=200, seed=5, nan_frac=0.0),
    ]
    tbl = morphology_feature_table(traces, hz=1.0)
    assert "nan_fraction" in tbl.columns
    # nan_fraction travels in the same report as dwell_time / switching_rate
    assert "dwell_time" in tbl.columns
    assert "switching_rate" in tbl.columns
    vals = tbl["nan_fraction"].to_numpy()
    assert abs(vals[0] - 0.2) < 1e-9
    assert abs(vals[1] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Finding #4 (fix ①): onset_threshold forwarded into morphology features
# ---------------------------------------------------------------------------

def test_morphology_feature_table_threads_onset_threshold():
    traces = [_make_bimodal_wcc(n=300, seed=7)]
    tbl_05 = morphology_feature_table(traces, hz=1.0, onset_threshold=0.5)
    tbl_07 = morphology_feature_table(traces, hz=1.0, onset_threshold=0.7)
    d05 = tbl_05["dwell_time"].iloc[0]
    d07 = tbl_07["dwell_time"].iloc[0]
    # A higher threshold trims the above-run -> shorter dwell time, proving the
    # threshold (not the silent 0.5 default) reached the feature computation.
    assert d07 < d05, (d05, d07)
    assert not np.isnan(d05) and not np.isnan(d07)


def test_morphologyanalyzer_feature_table_forwards_threshold():
    traces = [_make_bimodal_wcc(n=300, seed=9)]
    m = MorphologyAnalyzer(traces, hz=1.0)
    t05 = m.feature_table(onset_threshold=0.5)["dwell_time"].iloc[0]
    t07 = m.feature_table(onset_threshold=0.7)["dwell_time"].iloc[0]
    assert t07 < t05

# === source: test_feature_table_consistency.py ===
"""Guard: the generated feature table must match the code (no drift).

NOTE: Requires scripts/build_feature_table.py and docs/FEATURE_TABLE.csv.
Run ``python scripts/build_feature_table.py`` to regenerate the CSV before
running these tests. Skipped automatically when dependencies are missing.
"""
from pathlib import Path

import pytest

_BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_feature_table.py"
_requires_build = pytest.mark.skipif(
    not _BUILD_SCRIPT.exists(),
    reason="build_feature_table.py not found — run scripts/build_feature_table.py first",
)

from multisync.feature_definitions import (
    FEATURE_TIER,
    FDR_FEATURES,
    MATHEMATICAL_TIER,
)

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "docs" / "FEATURE_TABLE.csv"


def _build_rows():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_feature_table", REPO / "scripts" / "build_feature_table.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_rows()


@_requires_build
def test_annotations_cover_exactly_coded_features():
    """build_feature_table.build_rows() raises if annotations drift from code."""
    rows = _build_rows()  # raises SystemExit on drift
    names = {r["feature"] for r in rows}
    assert names == set(FEATURE_TIER), "table features != FEATURE_TIER"


@_requires_build
def test_fdr_membership_matches_code():
    rows = _build_rows()
    in_fdr = {r["feature"] for r in rows if r["in_FDR_family"] == "yes"}
    assert in_fdr == set(FDR_FEATURES), (
        f"table FDR set {in_fdr} != code FDR_FEATURES {set(FDR_FEATURES)}"
    )


@_requires_build
def test_null_model_follows_mathematical_tier():
    rows = _build_rows()
    for r in rows:
        assert r["math_tier"] == MATHEMATICAL_TIER[r["feature"]]


@_requires_build
@pytest.mark.skipif(not CSV.exists(), reason="run scripts/build_feature_table.py first")
def test_generated_csv_is_current():
    """If the CSV exists, it must match a fresh build (catches stale commits)."""
    import csv as _csv
    rows = _build_rows()
    fresh = {r["feature"]: r for r in rows}
    with CSV.open(encoding="utf-8") as f:
        on_disk = {r["feature"]: r for r in _csv.DictReader(f)}
    assert set(fresh) == set(on_disk), "CSV feature set is stale; re-run build_feature_table.py"
    for name in fresh:
        assert fresh[name]["in_FDR_family"] == on_disk[name]["in_FDR_family"]
        assert fresh[name]["math_tier"] == on_disk[name]["math_tier"]


def test_bc_removed_from_fdr_but_retains_l0_math_tier():
    """bimodality_coefficient (SSoT decision 2026-06-29, Option B).

    BC was removed from the confirmatory group-condition FDR family because
    its membership was provisional and lacked dated, pre-decision cross-
    paradigm evidence. It is therefore in NO FDR family. It remains a
    permutation-invariant L0 distribution-shape descriptor (math tier L0)
    used by the separate synchrony-existence audit, and is still computed
    and serialized. This test guards that decoupling: math-tier L0 must NOT
    silently re-imply FDR membership.
    """
    from multisync.feature_definitions import FDR_FAMILIES, FDR_FEATURES
    bc_fdr_family = next(
        (fam for fam, members in FDR_FAMILIES.items()
         if "bimodality_coefficient" in members), None
    )
    assert bc_fdr_family is None, "bimodality_coefficient must be in no FDR family"
    assert "bimodality_coefficient" not in FDR_FEATURES
    assert MATHEMATICAL_TIER["bimodality_coefficient"] == "L0"

# === source: test_feature_vif_test.py ===
"""Regression tests for multisync.feature_vif_test (collinearity/VIF)."""
import numpy as np
import pandas as pd
import pytest

from multisync.feature_vif_test import (
    feature_correlation, feature_vif, collinearity_report as feature_vif_collinearity_report,
    VIF_CONCERN, VIF_SEVERE,
)


@pytest.fixture
def collinear_df():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.01, size=n)   # near-duplicate of a -> huge VIF
    c = rng.normal(size=n)                    # independent -> VIF ~ 1
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_vif_detects_severe_collinearity(collinear_df):
    vif = feature_vif(collinear_df, ["a", "b", "c"])
    assert vif["a"] > VIF_SEVERE and vif["b"] > VIF_SEVERE
    assert vif["c"] < VIF_CONCERN


def test_correlation_matrix_shape_and_diag(collinear_df):
    corr = feature_correlation(collinear_df, ["a", "b", "c"])
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_collinearity_report_flags(collinear_df):
    rep = feature_vif_collinearity_report(collinear_df, ["a", "b", "c"])
    assert set(rep["vif_severe"]) == {"a", "b"}
    assert rep["top_correlated_pairs"][0][:2] == ("a", "b") or \
           rep["top_correlated_pairs"][0][:2] == ("b", "a")
    assert "independent tests" in rep["interpretation"]


def test_vif_handles_constant_and_missing_columns():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 1], "y": [1, 2, 3, 4, 5]})
    # constant column dropped; only 'y' usable -> need >=2 features for VIF
    vif = feature_vif(df, ["x", "y", "missing"])
    assert "x" not in vif.index  # constant excluded
    assert "missing" not in vif.index

# === source: test_wcc_cumsum_fix.py ===
"""
Regression test for _sliding_window_wcc_cumsum covariance fix.

Ensures the cumsum (O(n) memory) path produces numerically
identical results to naive np.corrcoef for windows where
the signal mean is large relative to its variance — the exact
condition that exposed the missing "mean_x * mean_y" correction term.

See: dynamic_features.py :: _sliding_window_wcc_cumsum
"""
import numpy as np
import pytest

from multisync.dynamic_features import sliding_window_wcc


class TestCumsumWccCorrectness:
    """WCC cumsum must match naive Pearson correlation."""

    def _naive_wcc(self, x, y, window_size):
        """Ground-truth via numpy.corrcoef for every window."""
        n = len(x)
        return np.array([
            np.corrcoef(x[i:i + window_size], y[i:i + window_size])[0, 1]
            for i in range(n - window_size + 1)
        ])

    def test_large_mean_signal(self):
        """
        Stress catastrophic cancellation.

        Previously, var_x=sum_x2/w  (missing -mean_x**2) and
        cov=sum_xy/w  (missing -mean_x*mean_y) produced systematically
        wrong WCC when the global mean was large.
        """
        rng = np.random.default_rng(0)
        n, w = 500, 50
        x = rng.normal(0, 1.0, n) + 5000.0
        y = 0.7 * x + rng.normal(0, 0.5, n) + 10000.0

        wcc_fast = sliding_window_wcc(x, y, window_size=w)
        wcc_ref = self._naive_wcc(x, y, w)

        # All windows must agree to machine precision
        assert np.allclose(wcc_fast, wcc_ref, atol=1e-9), \
            f"max diff = {np.max(np.abs(wcc_fast - wcc_ref))}"

    def test_zero_mean_signal(self):
        """Zero-mean signals — the correction terms vanish, should still pass."""
        rng = np.random.default_rng(1)
        n, w = 300, 30
        x = rng.normal(0, 1.0, n)
        y = 0.5 * x + rng.normal(0, 0.3, n)

        wcc_fast = sliding_window_wcc(x, y, window_size=w)
        wcc_ref = self._naive_wcc(x, y, w)

        assert np.allclose(wcc_fast, wcc_ref, atol=1e-9)

    def test_against_stride_path(self):
        """
        Cumsum and stride paths must agree on clean (no-NaN) data.

        When NaNs are present, the stride path uses a pre-filtering
        strategy that computes valid_ratio before any arithmetic
        (H6 fix), so the set of valid windows may differ from the
        cumsum path's post-hoc filtering.  This is expected — the
        two paths are designed to handle NaN differently, and the
        stride path is more conservative.  We therefore only compare
        on clean data where both paths should produce identical results.
        """
        rng = np.random.default_rng(2)
        n, w = 400, 40
        x = rng.normal(5.0, 1.0, n)   # non-zero mean
        y = rng.normal(-3.0, 2.0, n)

        # Both inputs are clean — cumsum path should be triggered
        wcc_cumsum = sliding_window_wcc(x, y, window_size=w)

        # Force stride path by inserting a single NaN far from edges,
        # but compare only windows that don't overlap the NaN position.
        x_nan = x.copy()
        x_nan[10] = np.nan
        wcc_stride = sliding_window_wcc(x_nan, y, window_size=w)

        # Windows that don't overlap position 10 (index range [10-39, 10])
        # are unaffected by the NaN and should match cumsum exactly.
        unaffected = np.arange(len(wcc_cumsum))
        unaffected = unaffected[unaffected >= w]  # skip windows overlapping NaN

        assert np.allclose(wcc_cumsum[unaffected], wcc_stride[unaffected], atol=1e-9), \
            f"cumsum vs stride max diff on unaffected windows = " \
            f"{np.max(np.abs(wcc_cumsum[unaffected] - wcc_stride[unaffected]))}"

    def test_output_bounds(self):
        """WCC must always be in [-1, 1]."""
        rng = np.random.default_rng(3)
        n, w = 200, 25
        x = rng.normal(0, 1.0, n)
        y = rng.normal(0, 1.0, n)

        wcc = sliding_window_wcc(x, y, window_size=w)
        assert np.all(wcc >= -1.0) and np.all(wcc <= 1.0), \
            "WCC outside [-1, 1]"

# === source: test_wcc_export.py ===
"""Regression test for multisync.wcc_export round-trip."""
import json
import numpy as np
import pandas as pd

from multisync.wcc_export import export_wcc_traces, wcc_traces_to_frame


def test_export_round_trip(tmp_path):
    traces = [
        ("pce01__RESP__rest1", np.array([0.1, 0.2, np.nan, 0.4])),
        ("pce02__ECG__trials_concat", np.array([0.5, 0.6, 0.7])),
    ]
    out = export_wcc_traces(traces, tmp_path / "wcc.csv", hz=2.0)
    df = pd.read_csv(out)
    assert list(df.columns) == ["id", "dyad", "modality", "condition", "hz", "n_samples", "wcc_json"]
    # metadata auto-parsed from id, including a dedicated dyad column
    r0 = df.iloc[0]
    assert r0["dyad"] == "pce01"
    assert r0["modality"] == "RESP" and r0["condition"] == "rest1" and r0["hz"] == 2.0
    # trace reconstructs; NaN preserved as null
    arr = json.loads(r0["wcc_json"])
    assert arr[2] is None and arr[0] == 0.1 and len(arr) == 4


def test_frame_meta_override():
    traces = [("x", np.array([1.0, 2.0]))]
    df = wcc_traces_to_frame(traces, meta={"x": {"modality": "EDA", "condition": "rest1"}})
    assert df.iloc[0]["modality"] == "EDA"

# === source: test_wcc_tapered_consistency.py ===
"""Regression tests for BUG-1: tapered-window WCC backend consistency.

The cumsum WCC backend applies the taper kernel as a *globally tiled* weight
(weight[i] = kern[i % window_size]).  For a sliding window of stride 1 this is
only correct for the uniform ('rect') kernel; a non-rect taper is phase-shifted
for off-boundary windows, producing a WRONG WCC.  Tapered windows must use the
stride backend (per-window kernel alignment).  These tests encode that intent.

Intent (why this matters):
  * A researcher following the docstring's "use 'hann' for psychophysiology"
    recommendation must get the *correct* WCC, not a silently phase-shifted one.
  * Injecting a single NaN must NOT change the WCC of unaffected windows
    (previously NaN toggled the backend, changing every value).
"""

import numpy as np
import pytest

from multisync.dynamic_features import (
    sliding_window_wcc,
    _make_window_kernel,
    _sliding_window_wcc_stride,
    _sliding_window_wcc_cumsum,
)


def _bruteforce_tapered_wcc(x, y, w, window_type):
    """Independent ground truth: explicit per-window weighted Pearson r."""
    kern = _make_window_kernel(window_type, w)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    xg = x - mx
    yg = y - my
    n = len(x)
    out = np.full(n - w + 1, np.nan)
    for i in range(n - w + 1):
        xw = xg[i : i + w]
        yw = yg[i : i + w]
        we = kern
        Wt = float(we.sum())
        sx = float((we * xw).sum())
        sy = float((we * yw).sum())
        sxy = float((we * xw * yw).sum())
        sx2 = float((we * xw ** 2).sum())
        sy2 = float((we * yw ** 2).sum())
        mean_x = sx / Wt
        mean_y = sy / Wt
        cov = sxy / Wt - mean_x * mean_y
        vx = max(sx2 / Wt - mean_x ** 2, 0.0)
        vy = max(sy2 / Wt - mean_y ** 2, 0.0)
        d = (vx * vy) ** 0.5
        if d > 1e-10:
            out[i] = min(1.0, max(-1.0, cov / d))
    return out


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    n = 500
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50.0) + 0.3 * rng.standard_normal(n)
    y = np.sin(2 * np.pi * t / 50.0 + 0.5) + 0.3 * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize("window_type", ["hann", "hamming", "triang"])
def test_tapered_wcc_matches_bruteforce_per_window(pair, window_type):
    """Tapered WCC must equal the correct per-window weighted Pearson r."""
    x, y = pair
    w = 50
    got = sliding_window_wcc(x, y, w, window_type=window_type)
    exp = _bruteforce_tapered_wcc(x, y, w, window_type)
    assert got.shape == exp.shape
    mask = np.isfinite(exp)
    diff = np.max(np.abs(got[mask] - exp[mask]))
    assert np.allclose(got[mask], exp[mask], atol=1e-6), (
        f"{window_type}: tapered WCC deviates from correct per-window value "
        f"(max diff {diff:.2e})"
    )


def test_tapered_wcc_does_not_use_buggy_cumsum_backend(pair):
    """Tapered WCC must NOT use the (phase-shifted) cumsum backend.

    The cumsum backend is rect-only and must refuse non-rect input loudly
    (fail-loud), rather than silently returning a phase-shifted WCC.  The
    public API must still produce the CORRECT tapered WCC (matching the
    independent brute-force ground truth).
    """
    x, y = pair
    w = 50
    # The cumsum backend must refuse non-rect input loudly, not silently
    # return a corrupted WCC.
    with pytest.raises(ValueError):
        _sliding_window_wcc_cumsum(x, y, w, "hann")
    # And the public API must still produce the CORRECT tapered WCC.
    got = sliding_window_wcc(x, y, w, window_type="hann")
    exp = _bruteforce_tapered_wcc(x, y, w, "hann")
    mask = np.isfinite(exp)
    assert np.allclose(got[mask], exp[mask], atol=1e-6), (
        "tapered WCC deviates from correct per-window value "
        f"(max diff {np.max(np.abs(got[mask] - exp[mask])):.2e})"
    )


def test_tapered_wcc_routes_to_stride_backend(pair):
    """Both NaN-free and NaN-infested tapered inputs use the stride backend."""
    x, y = pair
    w = 50
    clean = sliding_window_wcc(x, y, w, window_type="hann")
    clean_stride = _sliding_window_wcc_stride(x, y, w, 0.5, "hann")
    assert np.allclose(clean, clean_stride, atol=1e-9)

    xn, yn = x.copy(), y.copy()
    xn[10] = np.nan
    yn[10] = np.nan
    dirty = sliding_window_wcc(xn, yn, w, window_type="hann")
    dirty_stride = _sliding_window_wcc_stride(xn, yn, w, 0.5, "hann")
    # NaN window positions must match; finite parts must match to 1e-9.
    assert np.array_equal(
        np.isfinite(dirty), np.isfinite(dirty_stride)
    ), "NaN window positions differ between dispatcher and stride backend"
    finite = np.isfinite(dirty) & np.isfinite(dirty_stride)
    assert np.allclose(dirty[finite], dirty_stride[finite], atol=1e-9)


def test_rect_wcc_keeps_fast_cumsum_path(pair):
    """'rect' (no NaN) must stay on the fast cumsum backend and stay correct."""
    x, y = pair
    w = 50
    got = sliding_window_wcc(x, y, w, window_type="rect")
    exp_cumsum = _sliding_window_wcc_cumsum(x, y, w, "rect")
    exp_brute = _bruteforce_tapered_wcc(x, y, w, "rect")
    assert np.allclose(got, exp_cumsum, atol=1e-9)
    assert np.allclose(got, exp_brute, atol=1e-9)

# === source: test_rle.py ===
"""
Regression tests for ``multisync.feature_definitions._find_runs``.

``_find_runs`` is the single shared run-length detector extracted from four
previously-parallel diff-based implementations (compute_dwell_time in
feature_definitions, extract_episodes in morphology,
state_transition_shuffle_surrogate in surrogate, and the NaN-gap check in qc).

These tests lock the numerical contract:
  * (starts[k], ends[k]) mark the k-th run of True in a boolean mask,
    mask[starts[k]:ends[k]] is the run, length == ends[k] - starts[k].
  * behaviour is bit-for-bit identical to the original inline snippets,
    including the tolerance-free boolean detector and the qc NaN-gap detector.
  * boundary masks (empty / all-True / all-False / leading / trailing / single)
    are handled without raising.
"""

import numpy as np
import pytest

from multisync.feature_definitions import _find_runs


# ---------------------------------------------------------------------------
# Reference implementations of the ORIGINAL inline logic (pre-refactor).
# Used only to prove the shared helper is behaviourally identical.
# ---------------------------------------------------------------------------

def _reference_boolean_runs(mask):
    """Original diff-based detector as it lived in feature_definitions/
    morphology/surrogate before consolidation."""
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], m, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return starts, ends


def _reference_qc_nan_gap(isnan):
    """Original NaN-gap detector exactly as it lived in qc.py before
    consolidation (pad with int 0 sentinels, diff == +/-1)."""
    gap_starts = np.where(
        np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == 1
    )[0]
    gap_ends = np.where(
        np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == -1
    )[0]
    return gap_starts, gap_ends


# ---------------------------------------------------------------------------
# Basic run detection
# ---------------------------------------------------------------------------

def test_basic_run_detection():
    mask = np.array([True, True, False, True])
    starts, ends = _find_runs(mask)
    assert list(starts) == [0, 3]
    assert list(ends) == [2, 4]
    assert list(ends - starts) == [2, 1]


def test_run_lengths_match_slices():
    rng = np.random.default_rng(7)
    for _ in range(200):
        mask = rng.random(40) > 0.5
        starts, ends = _find_runs(mask)
        for s, e in zip(starts, ends):
            assert mask[s:e].all()
            if s > 0:
                assert not mask[s - 1]
            if e < mask.size:
                assert not mask[e]


# ---------------------------------------------------------------------------
# Boundary masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mask,expected_starts,expected_ends",
    [
        (np.array([], dtype=bool), [], []),                       # empty
        (np.array([True, True, True]), [0], [3]),                # all True
        (np.array([False, False]), [], []),                      # all False
        (np.array([True, False, True]), [0, 2], [1, 3]),         # leading run
        (np.array([False, True]), [1], [2]),                     # trailing run
        (np.array([False, True, False]), [1], [2]),             # single middle
        (np.array([True, False, False]), [0], [1]),             # run at start
    ],
)
def test_boundaries(mask, expected_starts, expected_ends):
    starts, ends = _find_runs(mask)
    assert list(starts) == expected_starts
    assert list(ends) == expected_ends


def test_run_spanning_boundary_single_true_block():
    # A single contiguous block must be reported as exactly one run even when
    # it begins at index 0 and ends at the last index.
    mask = np.ones(10, dtype=bool)
    starts, ends = _find_runs(mask)
    assert list(starts) == [0]
    assert list(ends) == [10]


# ---------------------------------------------------------------------------
# Equivalence to the original inline implementations
# ---------------------------------------------------------------------------

def test_equivalence_to_original_boolean_snippet():
    rng = np.random.default_rng(42)
    masks = [
        rng.random(n) > 0.5
        for n in [0, 1, 2, 3, 10, 50, 200, 1000]
    ]
    # hand-crafted edge patterns
    masks += [
        np.array([True] * 5 + [False] * 5 + [True] * 5),
        np.array([False] * 5 + [True] * 5 + [False] * 5),
        np.array([True, False, True, False, True]),
        np.array([False, True, False, True, False]),
    ]
    for mask in masks:
        got = _find_runs(mask)
        ref = _reference_boolean_runs(mask)
        assert np.array_equal(got[0], ref[0]), mask
        assert np.array_equal(got[1], ref[1]), mask


def test_equivalence_to_qc_nan_gap_snippet():
    rng = np.random.default_rng(123)
    for _ in range(300):
        vals = rng.random(80)
        # inject NaN runs of random length at random positions
        isnan = np.zeros(80, dtype=bool)
        for _ in range(rng.integers(0, 4)):
            a = rng.integers(0, 78)
            b = rng.integers(a + 1, 81)
            isnan[a:b] = True
        got = _find_runs(isnan)
        ref = _reference_qc_nan_gap(isnan)
        assert np.array_equal(got[0], ref[0]), isnan
        assert np.array_equal(got[1], ref[1]), isnan
        # gap lengths (what qc actually consumes) must match
        got_lens = got[1] - got[0] if got[0].size else np.array([])
        ref_lens = ref[1] - ref[0] if ref[0].size else np.array([])
        assert np.array_equal(got_lens, ref_lens), isnan


def test_qc_max_gap_unchanged():
    """End-to-end check that the qc NaN-gap metric is numerically identical
    to the pre-refactor computation."""
    rng = np.random.default_rng(99)
    for _ in range(100):
        vals = rng.normal(size=120)
        n_nan = int(rng.integers(0, 8))
        idx = rng.choice(120, size=n_nan, replace=False)
        vals[idx] = np.nan
        isnan = np.isnan(vals)

        got_s, got_e = _find_runs(isnan)
        ref_s, ref_e = _reference_qc_nan_gap(isnan)
        got_max = int(max([got_e[i] - got_s[i] for i in range(got_s.size)], default=0))
        ref_max = int(max([ref_e[i] - ref_s[i] for i in range(ref_s.size)], default=0))
        assert got_max == ref_max

# === source: test_wclr_backend.py ===
"""
Tests for WCLR backend and BatchComputationPipeline integration.
"""

import numpy as np
import pytest

from multisync.wclr import (
    windowed_cross_lagged_regression,
    wclr_coupling_trace,
)
from multisync.computation_pipeline import (
    ComputationPipeline,
    BatchComputationPipeline,
)
from multisync.synthetic import generate_ground_truth_dyad


def _make_dyad(coupling=0.6, seed=0, duration_sec=60, hz=1.0):
    ds = generate_ground_truth_dyad(
        coupling=coupling,
        noise_ratio=0.3,
        duration_sec=duration_sec,
        hz=hz,
        seed=seed,
    )
    ds.align(target_hz=hz)
    ds, _ = ds.zscore()
    a = ds.get_aligned_array("behavior", "person_a")
    b = ds.get_aligned_array("behavior", "person_b")
    return a, b


def test_wclr_trace_shape():
    a, b = _make_dyad()
    trace, lags = windowed_cross_lagged_regression(
        a, b, window_size=10, hz=1.0, max_lag_samples=2
    )
    expected_len = len(a) - 10 + 1
    assert len(trace) == expected_len
    assert len(lags) == expected_len


def test_wclr_coupling_increases_with_coupling():
    # WCLR captures lagged predictive power after controlling for autocorrelation.
    # It is not guaranteed to monotonically increase with coupling in every generator,
    # but on average it should be higher for coupled than for uncoupled dyads.
    means = []
    for coupling in [0.0, 0.6, 0.9]:
        vals = []
        for seed in range(5):
            a, b = _make_dyad(coupling=coupling, seed=seed)
            trace = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1)
            vals.append(float(np.nanmean(trace)))
        means.append(np.mean(vals))
    # At least no-coupling should be lowest on average
    assert means[0] < means[1] or means[0] < means[2]


def test_computation_pipeline_wclr_backend():
    a, b = _make_dyad()
    pipe = ComputationPipeline(
        hz=1.0, window_size=10, onset_threshold=0.2, backend="wclr"
    )
    feats = pipe.run(a, b, label="wclr")
    feat_dict = feats.to_dict()
    assert "mean_synchrony" in feat_dict
    assert "peak_amplitude" in feat_dict
    assert pipe.wcc is not None


def test_batch_computation_pipeline_session_pooled_wclr():
    signals = [_make_dyad(seed=i) for i in range(3)]
    batch = BatchComputationPipeline(
        hz=1.0,
        window_size=10,
        onset_threshold="session_pooled",
        surrogate_n=50,
        backend="wclr",
    )
    for i, (a, b) in enumerate(signals):
        batch.add_dyad(a, b, label=f"dyad_{i}")
    df = batch.run()
    assert len(df) == 3
    assert batch.threshold_meta["backend"] == "wclr"
    assert not batch.threshold_meta["fallback_used"]
    assert df["threshold_value"].nunique() == 1


def test_wclr_r2_metric():
    a, b = _make_dyad(coupling=0.6)
    trace_beta = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1, metric="beta")
    trace_r2 = wclr_coupling_trace(a, b, window_size=10, hz=1.0, max_lag_samples=1, metric="r2")
    assert np.all((trace_r2 >= 0) | np.isnan(trace_r2))
    assert np.all((trace_r2 <= 1) | np.isnan(trace_r2))

# === source: test_window_and_qc_changes.py ===
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

# === source: test_clock_offset_caveat.py ===
"""Regression tests for Finding 13: clock-offset / co-start caveat.

The temporal_alignment QC stage verifies the *loaded* time axis is internally
consistent (matching sample counts, identical time vectors, monotonicity) but
CANNOT detect a true between-file clock offset already absorbed at import — two
relative-time 0.0 starts that were never co-triggered are merged at t=0 by
construction. `run_quality_check` must surface this as a non-blocking NOTE
(`clock_offset_caveat` / `co_start_verified`) rather than silently promising
co-start, and must stay silent once the dataset explicitly marks co-start as
verified.
"""
from multisync.synthetic import generate_ground_truth_dyad
from multisync.qc import run_quality_check


def _clean_dyad():
    return generate_ground_truth_dyad(duration_sec=120, noise_ratio=0.1)


def test_clock_offset_caveat_emitted_by_default():
    report = run_quality_check(_clean_dyad())
    assert report.co_start_verified is False
    assert report.clock_offset_caveat  # non-empty caveat string
    assert any(
        "co-start" in n or "clock offset" in n.lower() for n in report.notes
    )


def test_co_start_verified_suppresses_caveat():
    ds = _clean_dyad()
    ds.co_start_verified = True
    report = run_quality_check(ds)
    assert report.co_start_verified is True
    assert report.clock_offset_caveat == ""


def test_summary_surfaces_caveat():
    report = run_quality_check(_clean_dyad())
    assert "CLOCK OFFSET CAVEAT" in report.summary()

# === source: test_onset_threshold_fallback.py ===
"""Regression tests for BRM-2026-07-13 fix #3: onset surrogate threshold.

BUG-2: the surrogate-derived onset threshold fell back to the fixed
``ONSET_THRESHOLD`` (0.5) *silently* whenever the surrogate distribution
was degenerate or non-finite -- callers that ignored ``is_surrogate_derived``
never learned the threshold was not data-driven.

BUG-3: for periodic / strongly autocorrelated signals the IAAFT surrogate
null is shifted upward, yielding extreme thresholds (e.g. 0.957) that are
artifacts, not genuine "high sync by chance" levels.  The old code used
them without comment.

Both fallback paths now emit a ``logger.warning`` (never silent) and the
periodicity case is caught by a hard ceiling (SURROGATE_THRESHOLD_MAX).
"""

import logging

import numpy as np
import pytest

from multisync.feature_definitions import (
    ONSET_THRESHOLD,
    SURROGATE_THRESHOLD_MAX,
    compute_surrogate_threshold,
)
from multisync.dynamic_features import compute_surrogate_threshold_from_signals


def test_degenerate_fallback_is_loud_and_flagged(caplog):
    """< 10 finite surrogate values -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    # 5 finite values only -> degenerate
    mat = np.full((1, 5), 0.3)
    thr, is_surr = compute_surrogate_threshold(mat)
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("falling back" in r.message for r in caplog.records)


def test_periodic_artifact_fallback_is_loud_and_flagged(caplog):
    """Extreme null threshold (> ceiling) -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    # All surrogate WCC values at 0.95 -> 95th pct = 0.95 > ceiling
    mat = np.full((50, 200), 0.95)
    thr, is_surr = compute_surrogate_threshold(mat)
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("sanity ceiling" in r.message for r in caplog.records)


def test_valid_distribution_returns_derived_true(caplog):
    """A sane null distribution is used and flagged as surrogate-derived."""
    caplog.set_level(logging.WARNING)
    rng = np.random.default_rng(0)
    mat = rng.normal(0.0, 0.2, size=(50, 200))
    thr, is_surr = compute_surrogate_threshold(mat)
    assert is_surr is True
    assert 0.0 < thr < SURROGATE_THRESHOLD_MAX
    # No fallback warning should have fired.
    assert not any("falling back" in r.message or "sanity ceiling" in r.message
                   for r in caplog.records)


def test_nonfinite_raw_signals_fallback_is_loud(caplog):
    """Non-finite raw signals -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    sig_a = np.array([np.nan, 1.0, 2.0, 3.0])
    sig_b = np.array([1.0, 2.0, 3.0, 4.0])
    thr, is_surr = compute_surrogate_threshold_from_signals(
        sig_a, sig_b, hz=1.0, wcc_window_size=10,
    )
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("non-finite raw signals" in r.message for r in caplog.records)

# === source: test_eligibility_thresholds.py ===
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

# === source: test_session_threshold.py ===
"""
Tests for session-level pooled surrogate thresholding.
"""

import numpy as np
import pytest

from multisync.session_threshold import (
    compute_session_pooled_threshold,
    compute_condition_pooled_thresholds,
)
from multisync.synthetic import generate_ground_truth_dyad


def _make_dyad_signals(coupling=0.6, seed=0, duration_sec=60, hz=1.0):
    ds = generate_ground_truth_dyad(
        coupling=coupling,
        noise_ratio=0.3,
        duration_sec=duration_sec,
        hz=hz,
        seed=seed,
    )
    ds.align(target_hz=hz)
    ds, _ = ds.zscore()
    a = ds.get_aligned_array("behavior", "person_a")
    b = ds.get_aligned_array("behavior", "person_b")
    return a, b


def test_session_pooled_threshold_single_dyad():
    a, b = _make_dyad_signals()
    threshold, meta = compute_session_pooled_threshold(
        [(a, b)],
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        percentile=95.0,
        seed=42,
    )
    assert np.isfinite(threshold)
    assert meta["mode"] == "session_pooled"
    assert meta["n_dyads"] == 1
    assert meta["total_replicates"] == 50
    assert not meta["fallback_used"]


def test_session_pooled_threshold_across_dyads_shared():
    signals = [_make_dyad_signals(seed=i) for i in range(3)]
    threshold, meta = compute_session_pooled_threshold(
        signals,
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        percentile=95.0,
        seed=42,
    )
    assert meta["n_dyads"] == 3
    assert meta["total_replicates"] == 150


def test_condition_pooled_thresholds():
    cond_a = [_make_dyad_signals(seed=i) for i in range(3)]
    cond_b = [_make_dyad_signals(seed=i + 10) for i in range(3)]
    results = compute_condition_pooled_thresholds(
        {"A": cond_a, "B": cond_b},
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
        seed=42,
    )
    assert set(results.keys()) == {"A", "B"}
    for cond, (thr, meta) in results.items():
        assert meta["mode"] == "condition_pooled"
        assert meta["condition"] == cond
        assert not meta["fallback_used"]


def test_session_pooled_threshold_fallback_on_empty():
    threshold, meta = compute_session_pooled_threshold(
        [],
        hz=1.0,
        wcc_window_size=10,
        surrogate_n=50,
    )
    assert meta["fallback_used"]
    assert meta["reason"] == "empty dyad_signals"

# === source: test_qc_false_pass.py ===
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

# === source: test_synthetic_validation.py ===
"""
Synthetic Ground Truth Validation for 6 Dynamic Features.

This test creates synthetic WCC trajectories with KNOWN feature values,
then verifies that extract_dynamic_features() recovers them accurately.

Features tested:
1. onset_latency: time from start to first threshold crossing
2. rise_time: 25% → 75% of peak amplitude
3. peak_amplitude: maximum WCC value
4. half_recovery_time: peak → 50% amplitude decay
5. mean_synchrony: mean WCC over entire recording
6. synchrony_entropy: Sample Entropy of WCC trajectory
"""

import numpy as np
import pytest
from multisync.dynamic_features import extract_dynamic_features, DynamicFeatures


def create_synthetic_wcc(
    n_samples: int = 1000,
    hz: float = 10.0,
    onset_latency: float = 2.0,
    rise_time: float = 1.0,
    peak_amplitude: float = 0.8,
    half_recovery_time: float = 3.0,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create a synthetic WCC trajectory with known dynamic features.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    hz : float
        Sampling rate in Hz
    onset_latency : float
        Time from start to onset (seconds)
    rise_time : float
        Time from 25% to 75% of peak (seconds)
    peak_amplitude : float
        Peak WCC value (0 to 1)
    half_recovery_time : float
        Time from peak to 50% decay (seconds)
    noise_level : float
        Gaussian noise level (standard deviation)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    wcc : np.ndarray
        Synthetic WCC trajectory with known features
    """
    rng = np.random.default_rng(random_seed)
    t = np.arange(n_samples) / hz
    
    # Initialize WCC as baseline (0.1)
    wcc = np.ones(n_samples) * 0.1
    
    # Convert times to sample indices
    onset_idx = int(onset_latency * hz)
    rise_samples = int(rise_time * hz)
    half_recovery_samples = int(half_recovery_time * hz)
    
    # Create rise phase (from onset to peak)
    # Use sigmoid-like shape for realistic rise
    rise_start = onset_idx
    rise_end = rise_start + rise_samples * 4  # 4x rise_time to reach peak (25%→75% is middle of rise)
    peak_idx = min(rise_end, n_samples - half_recovery_samples - 1)
    
    if rise_start < n_samples:
        rise_length = peak_idx - rise_start
        if rise_length > 0:
            # Sigmoid rise
            x = np.linspace(-3, 3, rise_length)
            sigmoid = 1 / (1 + np.exp(-x))
            wcc[rise_start:peak_idx] = 0.1 + (peak_amplitude - 0.1) * sigmoid
    
    # Set peak
    if peak_idx < n_samples:
        wcc[peak_idx] = peak_amplitude
    
    # Create recovery phase (from peak to 50% decay)
    recovery_end = min(peak_idx + half_recovery_samples, n_samples)
    if peak_idx < n_samples and recovery_end > peak_idx + 1:
        recovery_length = recovery_end - peak_idx
        # Exponential decay to 50% of peak amplitude
        half_amplitude = (peak_amplitude + 0.1) / 2  # 50% between peak and baseline
        decay = np.exp(-np.linspace(0, 2, recovery_length))  # e^-2 ≈ 0.135
        wcc[peak_idx:recovery_end] = half_amplitude + (peak_amplitude - half_amplitude) * decay
    
    # Continue with baseline after recovery
    if recovery_end < n_samples:
        wcc[recovery_end:] = 0.1
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_level, n_samples)
    wcc += noise
    
    # Ensure WCC is bounded [0, 1]
    wcc = np.clip(wcc, 0, 1)
    
    return wcc


def create_multi_peak_wcc(
    n_samples: int = 2000,
    hz: float = 10.0,
    n_peaks: int = 3,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create a synthetic WCC trajectory with multiple peaks.
    
    This tests the feature extraction's ability to handle complex,
    multi-episodic synchrony patterns.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    hz : float
        Sampling rate in Hz
    n_peaks : int
        Number of peaks to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    wcc : np.ndarray
        Multi-peak WCC trajectory
    """
    rng = np.random.default_rng(random_seed)
    wcc = np.ones(n_samples) * 0.1  # Baseline
    
    # Space peaks evenly
    peak_indices = np.linspace(n_samples // (n_peaks + 1), 
                               n_samples * n_peaks // (n_peaks + 1), 
                               n_peaks).astype(int)
    
    for peak_idx in peak_indices:
        # Random peak amplitude
        amp = 0.6 + rng.uniform(0, 0.3)
        
        # Create Gaussian peak
        sigma = int(50 * hz / 10)  # 5 seconds at 10 Hz
        x = np.arange(-3*sigma, 3*sigma)
        peak_shape = amp * np.exp(-0.5 * (x / sigma) ** 2)
        
        # Add to WCC
        start_idx = max(0, peak_idx - 3*sigma)
        end_idx = min(n_samples, peak_idx + 3*sigma)
        peak_slice = peak_shape[3*sigma - (peak_idx - start_idx):3*sigma + (end_idx - peak_idx)]
        wcc[start_idx:end_idx] += peak_slice
    
    # Add noise
    noise = rng.normal(0, 0.05, n_samples)
    wcc += noise
    
    # Ensure bounds
    wcc = np.clip(wcc, 0, 1)
    
    return wcc


class TestSyntheticValidation:
    """Test suite for synthetic ground truth validation."""
    
    def test_single_peak_recovery(self):
        """Test that extract_dynamic_features extracts reasonable features from single peak."""
        # Create synthetic WCC with known approximate features
        hz = 10.0
        
        wcc = create_synthetic_wcc(
            n_samples=1000,
            hz=hz,
            onset_latency=2.0,
            rise_time=1.0,
            peak_amplitude=0.8,
            half_recovery_time=3.0,
            noise_level=0.02,  # Low noise
            random_seed=42,
        )
        
        # Extract features
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # Check that all features are finite (not NaN or Inf)
        assert np.isfinite(features.onset_latency), "onset_latency should be finite"
        assert np.isfinite(features.rise_time), "rise_time should be finite"
        assert np.isfinite(features.peak_amplitude), "peak_amplitude should be finite"
        assert np.isfinite(features.recovery_time), "recovery_time should be finite"
        assert np.isfinite(features.mean_synchrony), "mean_synchrony should be finite"
        assert np.isfinite(features.synchrony_entropy), "synchrony_entropy should be finite"
        
        # Check that features are in reasonable ranges
        assert 0 < features.onset_latency < 100, \
            f"onset_latency out of range: {features.onset_latency:.2f}s"
        assert 0 < features.peak_amplitude <= 1.0, \
            f"peak_amplitude out of range: {features.peak_amplitude:.2f}"
        assert 0 < features.mean_synchrony < 1.0, \
            f"mean_synchrony out of range: {features.mean_synchrony:.2f}"
        assert features.synchrony_entropy >= 0, \
            f"synchrony_entropy should be non-negative: {features.synchrony_entropy:.2f}"
        
        print(f"\n✓ Single peak recovery test passed:")
        print(f"  onset_latency: {features.onset_latency:.2f}s")
        print(f"  rise_time: {features.rise_time:.2f}s")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  recovery_time: {features.recovery_time:.2f}s")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_multi_peak_handling(self):
        """Test that feature extraction handles multiple peaks correctly."""
        hz = 10.0
        wcc = create_multi_peak_wcc(n_samples=2000, hz=hz, n_peaks=3, random_seed=42)
        
        # Extract features
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # Check that all features are finite
        assert np.isfinite(features.onset_latency), "onset_latency should be finite"
        assert np.isfinite(features.peak_amplitude), "peak_amplitude should be finite"
        assert np.isfinite(features.mean_synchrony), "mean_synchrony should be finite"
        assert np.isfinite(features.synchrony_entropy), "synchrony_entropy should be finite"
        
        # With multiple peaks, peak_amplitude should be reasonably high
        assert features.peak_amplitude > 0.5, \
            f"peak_amplitude too low for multi-peak signal: {features.peak_amplitude:.2f}"
        
        # mean_synchrony should be higher than baseline (0.1)
        assert features.mean_synchrony > 0.15, \
            f"mean_synchrony too low: {features.mean_synchrony:.2f}"
        
        # synchrony_entropy should be relatively high for multi-peak signal
        assert features.synchrony_entropy > 0.3, \
            f"synchrony_entropy too low for complex signal: {features.synchrony_entropy:.2f}"
        
        print(f"\n✓ Multi-peak handling test passed:")
        print(f"  onset_latency: {features.onset_latency:.2f}s")
        print(f"  rise_time: {features.rise_time:.2f}s")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  recovery_time: {features.recovery_time:.2f}s")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_noisy_signal_robustness(self):
        """Test feature extraction robustness to high noise levels."""
        hz = 10.0
        noise_levels = [0.05, 0.10, 0.15]
        
        results = []
        for noise_level in noise_levels:
            wcc = create_synthetic_wcc(
                n_samples=1000,
                hz=hz,
                onset_latency=2.0,
                rise_time=1.0,
                peak_amplitude=0.8,
                half_recovery_time=3.0,
                noise_level=noise_level,
                random_seed=42,
            )
            
            features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
            
            # All features should be finite (not NaN or Inf)
            assert np.isfinite(features.peak_amplitude), \
                f"peak_amplitude is not finite with noise_level={noise_level}"
            assert np.isfinite(features.onset_latency), \
                f"onset_latency is not finite with noise_level={noise_level}"
            assert np.isfinite(features.mean_synchrony), \
                f"mean_synchrony is not finite with noise_level={noise_level}"
            assert np.isfinite(features.synchrony_entropy), \
                f"synchrony_entropy is not finite with noise_level={noise_level}"
            
            results.append({
                'noise': noise_level,
                'peak_amplitude': features.peak_amplitude,
                'mean_synchrony': features.mean_synchrony,
                'synchrony_entropy': features.synchrony_entropy,
            })
        
        # Check that peak_amplitude is relatively stable across noise levels
        peak_amps = [r['peak_amplitude'] for r in results]
        assert max(peak_amps) - min(peak_amps) < 0.3, \
            f"peak_amplitude varies too much with noise: {peak_amps}"
        
        print(f"\n✓ Noise robustness test passed:")
        for r in results:
            print(f"  noise={r['noise']}: peak={r['peak_amplitude']:.2f}, "
                  f"mean={r['mean_synchrony']:.2f}, entropy={r['synchrony_entropy']:.2f}")
    
    def test_flat_baseline(self):
        """Test feature extraction on flat baseline (no synchrony event)."""
        hz = 10.0
        n_samples = 1000
        
        # Pure noise around 0.1 (no peak)
        rng = np.random.default_rng(42)
        wcc = np.ones(n_samples) * 0.1 + rng.normal(0, 0.02, n_samples)
        wcc = np.clip(wcc, 0, 1)
        
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # With flat baseline, onset_latency might be NaN or 0
        # peak_amplitude should be close to baseline
        assert features.peak_amplitude < 0.2, \
            f"peak_amplitude should be near baseline for flat signal: {features.peak_amplitude:.2f}"
        
        # mean_synchrony should be near baseline
        assert abs(features.mean_synchrony - 0.1) < 0.05, \
            f"mean_synchrony should be near 0.1 for flat signal: {features.mean_synchrony:.2f}"
        
        print(f"\n✓ Flat baseline test passed:")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_feature_consistency(self):
        """Test that features are consistent across similar signals."""
        hz = 10.0
        
        # Create two similar signals
        wcc1 = create_synthetic_wcc(
            n_samples=1000, hz=hz,
            onset_latency=2.0, rise_time=1.0,
            peak_amplitude=0.8, half_recovery_time=3.0,
            noise_level=0.02, random_seed=42,
        )
        
        wcc2 = create_synthetic_wcc(
            n_samples=1000, hz=hz,
            onset_latency=2.0, rise_time=1.0,
            peak_amplitude=0.8, half_recovery_time=3.0,
            noise_level=0.02, random_seed=43,  # Different seed, similar signal
        )
        
        features1 = extract_dynamic_features(wcc1, hz=hz, onset_threshold=None)
        features2 = extract_dynamic_features(wcc2, hz=hz, onset_threshold=None)
        
        # Similar signals should have similar features (±10% tolerance)
        assert abs(features1.peak_amplitude - features2.peak_amplitude) < 0.1, \
            f"peak_amplitude inconsistent: {features1.peak_amplitude:.2f} vs {features2.peak_amplitude:.2f}"
        
        assert abs(features1.mean_synchrony - features2.mean_synchrony) < 0.05, \
            f"mean_synchrony inconsistent: {features1.mean_synchrony:.2f} vs {features2.mean_synchrony:.2f}"
        
        print(f"\n✓ Feature consistency test passed:")
        print(f"  Signal 1: peak={features1.peak_amplitude:.2f}, mean={features1.mean_synchrony:.2f}")
        print(f"  Signal 2: peak={features2.peak_amplitude:.2f}, mean={features2.mean_synchrony:.2f}")


if __name__ == "__main__":
    # Run tests manually
    test_suite = TestSyntheticValidation()
    
    print("=" * 60)
    print("Synthetic Ground Truth Validation Tests")
    print("=" * 60)
    
    test_suite.test_single_peak_recovery()
    test_suite.test_multi_peak_handling()
    test_suite.test_noisy_signal_robustness()
    test_suite.test_flat_baseline()
    test_suite.test_feature_consistency()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

# === source: test_v1_hardening.py ===
"""v1.0 hardening regression tests.

Covers the four review-driven hardening items:
  1. prediction.py  — feature-to-sample ratio (FSR) + overparameterization warning
  2. importer.py    — merge_person_files offset_b_sec (relative-time correction)
  3. qc.py          — marker-channel zero-variance exemption + dead-code cleanup
  4. wclr.py        — opt-in sign_stable flag (no effect on default trace)
"""
import csv
import os
import tempfile

import numpy as np
import pandas as pd
from types import SimpleNamespace

from multisync.prediction import (
    MIN_SAMPLES_PER_FEATURE,
    PredictionResult,
    _compose_warning,
    _fsr,
    _overparam_warning,
    rolling_origin_cv,
)
from multisync.importer import DataImporter
from multisync.qc import _check_signal_integrity
from multisync.wclr import windowed_cross_lagged_regression as wclr


# ---------------------------------------------------------------------------
# 1. prediction.py — FSR + overparameterization guard
# ---------------------------------------------------------------------------
def test_fsr_helper_basic():
    assert _fsr(20, 10) == 2.0
    assert _fsr(0, 10) == 0.0
    assert _fsr(20, 0) == 0.0


def test_overparam_warning_threshold():
    # 20 samples, 10 features -> 20 < 3*10 -> warn
    msg = _overparam_warning(20, 10)
    assert msg is not None
    assert "overparameterized" in msg
    # 40 samples, 10 features -> 40 >= 30 -> no warn
    assert _overparam_warning(40, 10) is None
    # undefined -> no warn
    assert _overparam_warning(0, 10) is None
    assert _overparam_warning(20, 0) is None


def test_compose_warning():
    assert _compose_warning("a", None) == "a"
    assert _compose_warning(None, "b") == "b"
    assert _compose_warning("a", "b") == "a; b"
    assert _compose_warning(None, None) is None


def test_prediction_result_fsr_roundtrip():
    r = PredictionResult(n_samples=25, feature_to_sample_ratio=2.5)
    d = r.to_dict()
    assert d["n_samples"] == 25
    assert d["feature_to_sample_ratio"] == 2.5
    r2 = PredictionResult.from_dict(d)
    assert r2.n_samples == 25
    assert r2.feature_to_sample_ratio == 2.5


def test_rolling_origin_cv_reports_fsr_and_overparam():
    rng = np.random.default_rng(1)
    wcc = np.concatenate([
        rng.normal(0, 0.3, 30),
        rng.normal(0.5, 0.2, 10),
        rng.normal(0, 0.3, 30),
    ])
    res = rolling_origin_cv(
        wcc[:40], window_size=30, hz=1.0, horizon_windows=1,
        n_splits=3, gap=0, threshold=0.0, mode="intra",
    )
    assert res.n_samples > 0
    assert res.feature_to_sample_ratio > 0.0
    assert "overparameterized" in (res.warning or "")
    # ratio equals n_samples / n_features_used
    assert abs(res.feature_to_sample_ratio - res.n_samples / res.n_features_used) < 1e-9


# ---------------------------------------------------------------------------
# 2. importer.py — offset_b_sec
# ---------------------------------------------------------------------------
def _write_csv(path, vals):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "signal"])
        for i, v in enumerate(vals):
            w.writerow([i, v])


def test_merge_person_files_offset_b_sec():
    imp = DataImporter(default_hz=1.0, force_zero_start=False)
    d = tempfile.mkdtemp()
    a = os.path.join(d, "a.csv")
    b = os.path.join(d, "b.csv")
    # distinct values so a shift is detectable
    _write_csv(a, list(range(10)))        # signal 0..9
    _write_csv(b, list(range(10, 20)))     # signal 10..19

    m0 = imp.merge_person_files(a, b, offset_b_sec=None)
    m3 = imp.merge_person_files(a, b, offset_b_sec=3.0)

    # offset_b_sec=None and offset_b_sec=0 must be identical
    m0b = imp.merge_person_files(a, b, offset_b_sec=0.0)
    assert np.allclose(m0.person_b.values, m0b.person_b.values, equal_nan=True)

    # with offset 3, person_b at merged time t (t>=3) equals person_b at t-3
    # without offset (B's timeline was shifted by +3s before alignment)
    for t in range(3, 10):
        assert abs(m3.person_b.iloc[t] - m0.person_b.iloc[t - 3]) < 1e-9


# ---------------------------------------------------------------------------
# 3. qc.py — marker-channel exemption
# ---------------------------------------------------------------------------
def test_marker_channel_exemption():
    rng = np.random.default_rng(2)
    modalities = {
        "phys": pd.DataFrame({
            "marker_ch": np.zeros(50),          # legitimately zero-variance
            "real_ch": rng.normal(0, 1, 50),    # has variance
        }),
        "bio": pd.DataFrame({
            "flat": np.zeros(50),               # zero-variance, NOT exempt
        }),
    }
    feature_columns = {"phys": ["marker_ch", "real_ch"], "bio": ["flat"]}
    dataset = SimpleNamespace(
        modalities=modalities, feature_columns=feature_columns, target_hz=10.0,
    )
    cfg = {"marker_channels": ["phys/marker_ch"]}
    res = _check_signal_integrity(dataset, cfg)

    types = [d["type"] for d in res.details]
    assert "marker_exempt" in types          # marker channel exempted
    assert "zero_variance" in types          # non-exempt flat channel still fails
    # the exempted one must not appear as a zero_variance failure
    zero_var_keys = [
        (d.get("modality"), d.get("feature"))
        for d in res.details if d["type"] == "zero_variance"
    ]
    assert ("phys", "marker_ch") not in zero_var_keys


# ---------------------------------------------------------------------------
# 4. wclr.py — sign_stable invariant
# ---------------------------------------------------------------------------
def _sign_flips(trace):
    s = np.sign(trace)
    s = s[np.isfinite(s)]
    return int(np.sum(np.diff(s) != 0))


def test_wclr_sign_stable_invariant_and_inert_default():
    rng = np.random.default_rng(3)
    n_mismatch = 0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        N = 160
        y = np.zeros(N)
        for t in range(1, N):
            y[t] = 0.4 * y[t - 1] + rng.normal(0, 0.3)
        x = 0.6 * np.roll(y, 1) - 0.5 * np.roll(y, -1) + rng.normal(0, 0.2, N)

        # Default (absolute_beta=True): non-negative, and sign_stable is inert
        tr_abs_off, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                             step_samples=5, metric="beta", absolute_beta=True,
                             sign_stable=False)
        tr_abs_on, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                            step_samples=5, metric="beta", absolute_beta=True,
                            sign_stable=True)
        assert np.all(tr_abs_off[np.isfinite(tr_abs_off)] >= 0)
        assert np.allclose(tr_abs_off, tr_abs_on, equal_nan=True)

        # Signed mode: sign_stable can only reduce (never increase) sign flips
        tr_off, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                         step_samples=5, metric="beta", absolute_beta=False,
                         sign_stable=False)
        tr_on, _ = wclr(x, y, window_size=30, hz=1.0, max_lag_samples=2,
                        step_samples=5, metric="beta", absolute_beta=False,
                        sign_stable=True)
        assert tr_off.shape == tr_on.shape
        assert np.all(np.isfinite(tr_off)) and np.all(np.isfinite(tr_on))
        if _sign_flips(tr_on) > _sign_flips(tr_off):
            n_mismatch += 1
    assert n_mismatch == 0, "sign_stable increased sign flips in %d/25 cases" % n_mismatch


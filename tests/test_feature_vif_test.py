"""Regression tests for multisync.feature_vif_test (collinearity/VIF)."""
import numpy as np
import pandas as pd
import pytest

from multisync.feature_vif_test import (
    feature_correlation, feature_vif, collinearity_report,
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
    rep = collinearity_report(collinear_df, ["a", "b", "c"])
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

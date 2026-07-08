"""
Tests for the full-family FDR option (critique A, 2026-07-07).

Reviewer-proofing: instead of only testing the 3 pre-registered FDR-family
features, the full 12-feature set can be entered into a single BH-FDR step
via ``full_family_fdr=True``. The pre-registered core family remains the
primary endpoint; the full family is a strictly more conservative
supplementary check that neutralises the "cherry-picking 3/12" critique.

Covers
------
- SSoT API: ``feature_definitions.get_fdr_features`` / ``ALL_FEATURES``
- ``feature_pipeline.get_fdr_features`` is a thin pass-through to the SSoT
- ``inference_pipeline.test_l2_condition`` / ``run_full_cascade`` thread the
  flag and test 3 vs 12 features accordingly
- ``summarize()`` / ``cascade_summary`` reflect the family
  ("FDR-family" vs "all-features")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.feature_definitions import (
    ALL_FEATURES,
    FDR_FEATURES,
    get_fdr_features,
)
from multisync.feature_pipeline import get_fdr_features as fp_get_fdr_features
from multisync.inference_pipeline import InferencePipeline

N_FEATURES = 12
FDR_N = 3


def _full_feature_df(n_dyads: int = 8, seed: int = 0) -> pd.DataFrame:
    """A DataFrame containing ALL 12 SyncPipe features + dyad/condition cols.

    Every feature is finite so dyad-paired permutation keeps both conditions
    per dyad (we are testing *feature counts*, not missing-data handling).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        for cond in ("rest", "task"):
            row = {"dyad_id": f"dyad_{i}", "condition": cond}
            for f in ALL_FEATURES:
                shift = 0.0 if cond == "rest" else 0.3
                row[f] = float(rng.normal(shift, 0.3))
            rows.append(row)
    return pd.DataFrame(rows)


# --- A1: SSoT API -----------------------------------------------------------

def test_all_features_has_twelve_members():
    assert isinstance(ALL_FEATURES, tuple)
    assert len(ALL_FEATURES) == N_FEATURES
    # FDR features are a strict subset of the full family
    assert set(FDR_FEATURES).issubset(set(ALL_FEATURES))


def test_get_fdr_features_default_is_three():
    assert get_fdr_features() == list(FDR_FEATURES)
    assert get_fdr_features(False) == list(FDR_FEATURES)
    assert len(get_fdr_features(False)) == FDR_N


def test_get_fdr_features_full_family_is_all():
    assert get_fdr_features(True) == list(ALL_FEATURES)
    assert len(get_fdr_features(True)) == N_FEATURES


def test_feature_pipeline_mirrors_ssot():
    """feature_pipeline.get_fdr_features must be a thin pass-through to the SSoT."""
    assert fp_get_fdr_features(False) == get_fdr_features(False)
    assert fp_get_fdr_features(True) == get_fdr_features(True)


# --- A2/A3: inference threading --------------------------------------------

def test_l2_default_tests_only_fdr_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(n_permutations=200, contrast=("rest", "task"))
    assert res["n_tested"] == FDR_N
    assert len(res["per_feature"]) == FDR_N
    assert set(f.feature for f in res["per_feature"]) == set(FDR_FEATURES)


def test_l2_full_family_tests_all_twelve():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(
        n_permutations=200, contrast=("rest", "task"), full_family_fdr=True
    )
    assert res["n_tested"] == N_FEATURES
    assert len(res["per_feature"]) == N_FEATURES
    assert set(f.feature for f in res["per_feature"]) == set(ALL_FEATURES)


def test_l2_full_family_ignored_when_feature_cols_explicit():
    """When feature_cols is supplied, full_family_fdr must have no effect."""
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(
        feature_cols=["peak_amplitude"],
        n_permutations=200,
        contrast=("rest", "task"),
        full_family_fdr=True,
    )
    assert res["n_tested"] == 1
    assert set(f.feature for f in res["per_feature"]) == {"peak_amplitude"}


def test_summarize_reflects_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    pipe.test_l2_condition(n_permutations=200, contrast=("rest", "task"))
    assert "FDR-family" in pipe.summarize()

    pipe2 = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    pipe2.test_l2_condition(
        n_permutations=200, contrast=("rest", "task"), full_family_fdr=True
    )
    assert "all-features" in pipe2.summarize()


def test_run_full_cascade_carries_full_family_flag():
    """The flag must propagate to L2 inside run_full_cascade.

    An empty wcc_dict skips the (slow) L0/L1 surrogate loops, isolating the
    L2 threading check.
    """
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.run_full_cascade(
        raw_signals_dict={},
        wcc_dict={},
        wcc_window_size=20,
        n_permutations=200,
        full_family_fdr=True,
    )
    assert res["l2_results"]["n_tested"] == N_FEATURES
    assert "all-features" in res["cascade_summary"]


def test_run_full_cascade_default_is_fdr_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.run_full_cascade(
        raw_signals_dict={},
        wcc_dict={},
        wcc_window_size=20,
        n_permutations=200,
    )
    assert res["l2_results"]["n_tested"] == FDR_N
    assert "FDR-family" in res["cascade_summary"]

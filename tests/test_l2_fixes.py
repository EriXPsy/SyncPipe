"""Regression tests for L2 between-condition fixes:

- `cohens_d` field renamed to `perm_effect_size` (it is observed_diff /
  SD(null), NOT classical Cohen's d).  The old `cohens_d` attribute is a
  deprecated property that warns.
- Small-n (n_dyads <= 12) now uses *exact* enumeration of all 2^n sign
  flips, giving an honest discrete p-value resolution (1/(2^n + 1)),
  not the spurious 1/(n_permutations + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.validation.l2_between_condition import between_condition_fdr


def _small_df(n_dyads=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        a, b = rng.normal(1.0, 0.3), rng.normal(0.5, 0.3)
        rows.append({"dyad_label": f"d{i}", "condition": "A", "peak_amplitude": a})
        rows.append({"dyad_label": f"d{i}", "condition": "B", "peak_amplitude": b})
    return pd.DataFrame(rows)


def test_perm_effect_size_field_and_summary_column():
    df = _small_df(n_dyads=6)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=1
    )
    r = res["per_feature"][0]
    # New canonical name exists and is a float.
    assert hasattr(r, "perm_effect_size")
    assert isinstance(r.perm_effect_size, float)
    # summary_df column renamed (no longer "cohens_d").
    assert "perm_effect_size" in res["summary_df"].columns
    assert "cohens_d" not in res["summary_df"].columns


def test_cohens_d_attribute_deprecated():
    df = _small_df(n_dyads=6)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=1
    )
    r = res["per_feature"][0]
    with pytest.warns(DeprecationWarning):
        val = r.cohens_d
    # The deprecated alias returns the same value as the canonical field.
    assert val == r.perm_effect_size


def test_small_n_exact_discrete_p_resolution():
    """With n_dyads=4, the exact null has 2^4 = 16 points, so p_raw must be
    a multiple of 1/17 (honest resolution), NOT 1/10001."""
    df = _small_df(n_dyads=4)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=2
    )
    p_raw = res["per_feature"][0].p_raw
    # p_raw = (n_ge + 1) / (16 + 1); check it lands on the 1/17 grid.
    scaled = p_raw * 17.0
    assert abs(scaled - round(scaled)) < 1e-9, f"p_raw={p_raw} not on 1/17 grid"
    assert 0.0 <= p_raw <= 1.0


def test_large_n_still_runs():
    """n_dyads > 12 takes the Monte-Carlo path and returns finite results."""
    df = _small_df(n_dyads=20, seed=5)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=2000, seed=3
    )
    assert res["n_dyads"] == 20
    assert np.isfinite(res["per_feature"][0].p_raw)

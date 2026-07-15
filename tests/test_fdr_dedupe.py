"""Regression tests for the #18 FDR de-duplication guard.

`dedupe_fdr_input` refuses (or merges) duplicate endpoint/feature keys
BEFORE BH-FDR runs, so a key entered twice can never silently inflate the
test count ``m``.  `apply_fdr` is the single canonical FDR entry (dedupe ->
BH-FDR -> keyed result) that must agree with the canonical
`batch._bh_fdr_correction`.  The guard is also wired into the two FDR call
sites (`between_condition_fdr`, `apply_bh_fdr_within_noise`) and must fail
loud on duplicate keys there too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.batch import _bh_fdr_correction, apply_fdr, dedupe_fdr_input
from multisync.validation.l2_between_condition import between_condition_fdr
from multisync.validation.pgt1_intensity import apply_bh_fdr_within_noise


def test_dedupe_preserves_unique():
    names, vals = dedupe_fdr_input(["a", "b", "c"], [0.1, 0.2, 0.3])
    assert names == ["a", "b", "c"]
    assert vals == [0.1, 0.2, 0.3]


def test_dedupe_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        dedupe_fdr_input(["a", "b", "a"], [0.01, 0.02, 0.03])


def test_dedupe_first_keeps_first():
    names, vals = dedupe_fdr_input(
        ["a", "a", "b"], [0.5, 0.1, 0.2], on_duplicate="first"
    )
    assert names == ["a", "b"]
    assert vals == [0.5, 0.2]


def test_dedupe_max_keeps_largest_p():
    names, vals = dedupe_fdr_input(
        ["a", "a", "b"], [0.5, 0.1, 0.2], on_duplicate="max"
    )
    assert names == ["a", "b"]
    assert vals == [0.5, 0.2]  # max(0.5, 0.1)


def test_apply_fdr_dedupes_matches_bh():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=20)
    res = apply_fdr([f"f{i}" for i in range(20)], list(p), alpha=0.05)
    adj, rej = _bh_fdr_correction(list(p), alpha=0.05)
    for i in range(20):
        assert abs(res[f"f{i}"]["p_fdr"] - adj[i]) < 1e-12
        assert res[f"f{i}"]["significant"] == rej[i]
        assert abs(res[f"f{i}"]["p_raw"] - p[i]) < 1e-12


def test_apply_fdr_raises_on_dup():
    with pytest.raises(ValueError, match="duplicate"):
        apply_fdr(["a", "a"], [0.01, 0.02])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        dedupe_fdr_input(["a", "b"], [0.1])


def _small_l2_df(n_dyads=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        a, b = rng.normal(1.0, 0.3), rng.normal(0.5, 0.3)
        rows.append({"dyad_label": f"d{i}", "condition": "A", "peak_amplitude": a})
        rows.append({"dyad_label": f"d{i}", "condition": "B", "peak_amplitude": b})
    return pd.DataFrame(rows)


def test_l2_guard_rejects_duplicate_feature_cols():
    """between_condition_fdr must fail loud (not inflate m) on a duplicated
    feature column."""
    df = _small_l2_df(n_dyads=4)
    with pytest.raises(ValueError, match="duplicate"):
        between_condition_fdr(
            df,
            feature_cols=["peak_amplitude", "peak_amplitude"],
            n_permutations=1000,
            seed=1,
        )


def test_pgt1_guard_rejects_duplicate_p_columns():
    """apply_bh_fdr_within_noise must fail loud on a duplicated p-column."""
    df = pd.DataFrame({"p_a": [0.01], "p_b": [0.4]})
    with pytest.raises(ValueError, match="duplicate"):
        apply_bh_fdr_within_noise(
            df, feature_p_columns=["p_a", "p_a"], q=0.05
        )

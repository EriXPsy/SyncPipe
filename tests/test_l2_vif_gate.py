"""
VIF gate integration test (gstack #14).

The L2 between-condition test now attaches a ``vif_gate`` diagnostic to its
result: when FDR-family features are severely collinear the gate must FLAG
it (warn + report) rather than silently drop pre-registered features.
"""
import numpy as np
import pandas as pd

from multisync.validation.l2_between_condition import between_condition_fdr


def _make_df(rng, n, extra=None):
    base = {
        "dyad_label": [f"d{i}" for i in range(n)] * 2,
        "condition": ["A"] * n + ["B"] * n,
        "peak_amplitude": rng.normal(0.5, 0.1, n * 2),
        "dwell_time": rng.normal(10.0, 2.0, n * 2),
        "switching_rate": rng.normal(3.0, 1.0, n * 2),
    }
    if extra:
        base.update(extra)
    return pd.DataFrame(base)


def test_vif_gate_present_and_passes_for_independent_features():
    rng = np.random.default_rng(0)
    df = _make_df(rng, 14)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude", "dwell_time", "switching_rate"],
        n_permutations=100, seed=1,
    )
    assert "vif_gate" in res
    assert res["vif_gate"]["passed"] is True
    assert res["vif_gate"]["skipped"] is False


def test_vif_gate_flags_severe_collinearity():
    rng = np.random.default_rng(1)
    # build a peak_amplitude column, then inject an EXACT copy of it so the
    # two FDR-family features are perfectly collinear (VIF -> inf -> severe)
    pa = rng.normal(0.5, 0.1, 28)
    df = _make_df(rng, 14, extra={"pa_dup": pa})
    df["pa_dup"] = df["peak_amplitude"]  # force exact collinearity
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude", "pa_dup"],
        n_permutations=100, seed=1,
    )
    assert "vif_gate" in res
    assert res["vif_gate"]["passed"] is False
    assert "pa_dup" in res["vif_gate"]["vif_severe"]

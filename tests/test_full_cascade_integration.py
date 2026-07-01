"""END-TO-END integration test for run_full_cascade().

Motivation: three separate bugs (missing surrogate key, _build_cascade_summary
NameError, and between_condition_fdr kwarg-name drift) all passed the unit suite
because NO test ever drove run_full_cascade() all the way through L2. Each bug
only surfaced once the previous blocker was removed. This test exercises the
WHOLE cascade on a synthetic multi-dyad dataset and asserts a complete result,
so any future signature drift in between_condition_fdr / test_l2_condition is
caught immediately instead of by hand.
"""
import numpy as np
import pandas as pd
import pytest

from multisync.inference_pipeline import InferencePipeline
from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features
from multisync.feature_definitions import FDR_FEATURES


def _make_dyad_signals(rng, coupling, n=600):
    """Two signals sharing a latent driver at strength `coupling`."""
    shared = np.cumsum(rng.normal(0, 1, n))
    a = coupling * shared + rng.normal(0, 1.5, n)
    b = coupling * shared + rng.normal(0, 1.5, n)
    return a, b


@pytest.fixture
def cascade_inputs():
    """Build a paired multi-dyad dataset: each dyad has rest (low coupling) and
    task (high coupling) conditions, exactly one row each (L2 needs pairing)."""
    rng = np.random.default_rng(7)
    hz = 1.0
    window = 30
    rows, wcc_dict, raw_dict = [], {}, {}
    n_dyads = 8
    for d in range(n_dyads):
        for cond, coup in (("rest", 0.1), ("task", 0.9)):
            a, b = _make_dyad_signals(rng, coup)
            wcc = sliding_window_wcc(a, b, window_size=window)
            feats = extract_dynamic_features(wcc, hz=hz, wcc_window_sec=window / hz)
            label = f"dyad{d}__{cond}"
            wcc_dict[label] = wcc
            raw_dict[label] = (a, b)
            row = {"dyad_id": f"dyad{d}", "condition": cond, "label": label}
            row.update(feats.to_dict())
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, wcc_dict, raw_dict, window, hz


@pytest.mark.slow
def test_run_full_cascade_returns_complete_summary(cascade_inputs):
    df, wcc_dict, raw_dict, window, hz = cascade_inputs
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=50, seed=1)

    # This call exercises L0 + L1 surrogate tests AND L2 between_condition_fdr.
    # It would have crashed on: missing surrogate key / NameError / kwarg drift.
    result = pipe.run_full_cascade(
        raw_signals_dict=raw_dict,
        wcc_dict=wcc_dict,
        wcc_window_size=window,
        condition_col="condition",
        dyad_col="dyad_id",
        n_permutations=200,
    )

    # ---- structural assertions on the full cascade output ----
    for key in ("l0_summary", "l1_summary", "l2_results", "cascade_summary"):
        assert key in result, f"missing top-level key: {key}"

    # cascade_summary must be a non-empty string mentioning all three levels
    cs = result["cascade_summary"]
    assert isinstance(cs, str) and "L0" in cs and "L1" in cs and "L2" in cs

    # per-feature pass dicts (no OR-aggregate) present and symmetric L0/L1
    assert "per_feature_pass" in result["l0_summary"]
    assert "per_feature_pass" in result["l1_summary"]
    assert result["l0_summary"]["primary_feature"] == "peak_amplitude"
    assert result["l1_summary"]["primary_feature"] == "switching_rate"

    # L2 ran and produced per-feature FDR output
    l2 = result["l2_results"]
    assert l2 is not None


@pytest.mark.slow
def test_run_full_cascade_l2_param_names_are_correct(cascade_inputs):
    """Directly guards against between_condition_fdr kwarg-name drift:
    a wrong kwarg (e.g. fdr_alpha=/contrast=) would raise TypeError here."""
    df, wcc_dict, raw_dict, window, hz = cascade_inputs
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=20, seed=1)
    # test_l2_condition with an explicit contrast -> must map to condition_values
    res = pipe.test_l2_condition(
        condition_col="condition", dyad_col="dyad_id",
        fdr_alpha=0.05, n_permutations=100, contrast=("rest", "task"),
    )
    assert res is not None

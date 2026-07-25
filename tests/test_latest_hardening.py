"""Regression tests for the latest SyncPipe hardening patch."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multisync.computation_pipeline import ComputationPipeline
from multisync.core import DynamicAnalyzer
from multisync.inference_pipeline import InferencePipeline
from multisync.pipeline_bridge import _as_array, records_to_inference_inputs
from multisync.validation.l2_between_condition import between_condition_fdr


def test_nonzero_lag_fails_loud_and_cli_default_is_zero():
    from multisync.cli import build_parser
    args = build_parser().parse_args(["analyze", "-i", "a.csv,b.csv"])
    assert args.max_lag == 0.0
    with pytest.raises(ValueError, match="zero-lag WCC only"):
        DynamicAnalyzer(max_lag_sec=1.0)


def test_computation_output_records_observation_opportunity():
    x = np.sin(np.linspace(0, 20, 100))
    pipe = ComputationPipeline(hz=1.0, window_size=10)
    pipe.load_signals(x, x)
    pipe.compute_wcc()
    pipe.extract_features()
    row = pipe.to_dataframe().iloc[0]
    assert row["n_signal_samples"] == 100
    assert row["n_wcc_points"] == 91
    assert row["n_valid_wcc_points"] == 91
    assert row["valid_wcc_fraction"] == pytest.approx(1.0)


def test_wclr_applies_discontinuity_mask():
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    y = rng.normal(size=120)
    dm = np.ones(120, dtype=bool)
    dm[60] = False
    pipe = ComputationPipeline(hz=1.0, window_size=10, backend="wclr")
    pipe.load_signals(x, y, discontinuity_mask=dm)
    trace = pipe.compute_wcc()
    assert np.isnan(trace[51:61]).all()


def test_bridge_rejects_ambiguous_signal_dataframe():
    with pytest.raises(ValueError, match="value.*exactly one|candidates"):
        _as_array(pd.DataFrame({"time": [0, 1], "eda": [1, 2], "resp": [2, 3]}))


def test_bridge_rejects_mixed_hz():
    rec = SimpleNamespace(
        dyad_label="d1", modality="EDA", condition="A", target_hz=2.0,
        incomplete=False,
        person_a=pd.DataFrame({"time": [0, 1, 2], "value": [1., 2., 3.]}),
        person_b=pd.DataFrame({"time": [0, 1, 2], "value": [1., 2., 3.]}),
    )
    with pytest.raises(ValueError, match="target_hz"):
        records_to_inference_inputs([rec], hz=1.0, window_size=2)


def test_observation_guard_rejects_unequal_wcc_opportunity():
    rows = []
    for i in range(4):
        for cond, n_wcc in (("A", 91), ("B", 181)):
            rows.append({
                "dyad_id": f"d{i}", "condition": cond,
                "peak_amplitude": .5, "dwell_time": 1.,
                "switching_rate": 2., "n_wcc_points": n_wcc,
            })
    with pytest.raises(ValueError, match="Observation opportunity"):
        between_condition_fdr(
            pd.DataFrame(rows), dyad_col="dyad_id",
            condition_values=("A", "B"), observation_policy="raise",
        )


def test_undefinedness_gate_blocks_confirmatory_claim():
    rows = []
    for i in range(6):
        rows.extend([
            {"dyad_id": f"d{i}", "condition": "A", "dwell_time": 1.},
            {"dyad_id": f"d{i}", "condition": "B", "dwell_time": np.nan},
        ])
    result = between_condition_fdr(
        pd.DataFrame(rows), dyad_col="dyad_id",
        condition_values=("A", "B"), feature_cols=["dwell_time"],
        undefined_policy="gate",
    )
    item = result["per_feature"][0]
    assert item.claimable is False
    assert item.significant_05 is False
    assert item.definedness_status == "informative_undefinedness"


def test_global_modality_fdr_is_declared():
    rows = []
    for i in range(4):
        for mod in ("EDA", "RESP"):
            for cond, val in (("A", .6), ("B", .4)):
                rows.append({
                    "dyad_id": f"d{i}", "modality": mod,
                    "condition": cond, "peak_amplitude": val,
                    "dwell_time": val, "switching_rate": val,
                })
    result = InferencePipeline(pd.DataFrame(rows), seed=1).test_l2_by_modality(
        contrast=("A", "B"), n_permutations=20, fdr_scope="global",
    )
    assert {x["fdr_scope"] for x in result.values()} == {"global_modality_feature"}


def test_pairing_policy_is_explicit_in_manifest():
    import pandas as pd
    from multisync.core import Dyad
    ds = Dyad(hz=1.0, eda=pd.DataFrame({"time": np.arange(40.), "person_a": np.sin(np.arange(40.)), "person_b": np.cos(np.arange(40.))}))
    ds.align(target_hz=1.0).zscore()
    result = DynamicAnalyzer(window_size=5, surrogate_n=2, run_qc=False).fit_transform(ds)
    assert result.parameters["pairing_policy"] == "same_modality"
    assert result.parameters["effective_lag_sec"] == 0.0


def test_observation_guard_detects_within_cell_trial_length_variation():
    rows = [
        {"dyad_id": "d0", "condition": "A", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d0", "condition": "A", "n_wcc_points": 120, "peak_amplitude": .5},
        {"dyad_id": "d0", "condition": "B", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d1", "condition": "A", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d1", "condition": "B", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d2", "condition": "A", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d2", "condition": "B", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d3", "condition": "A", "n_wcc_points": 90, "peak_amplitude": .5},
        {"dyad_id": "d3", "condition": "B", "n_wcc_points": 90, "peak_amplitude": .5},
    ]
    with pytest.raises(ValueError, match="Observation opportunity"):
        between_condition_fdr(
            pd.DataFrame(rows), dyad_col="dyad_id",
            feature_cols=["peak_amplitude"], condition_values=("A", "B"),
            observation_policy="raise",
        )

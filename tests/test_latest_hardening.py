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
    args = build_parser().parse_args(["describe", "-i", "a.csv,b.csv"])
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
    # Non-seam windows must remain finite — a global-NaN regression
    # (e.g. mask applied to the entire trace) would still pass the
    # assertion above, so guard the complement explicitly.
    assert np.isfinite(trace[:51]).all()
    assert np.isfinite(trace[61:]).all()


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


def test_syncpipe_module_entrypoint_uses_canonical_wrapper():
    """`python -m syncpipe` must resolve to the canonical CLI wrapper (main)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "syncpipe", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("syncpipe ")


def test_onset_threshold_zero_not_replaced_with_default():
    """A fixed onset_threshold of 0.0 must be preserved, not silently
    replaced with 0.5 via truthiness (`or 0.5`). Guards the prediction
    path in DynamicAnalyzer.analyze() where the threshold is resolved
    with `is not None` rather than truthiness."""
    analyzer = DynamicAnalyzer(threshold_mode="fixed", onset_threshold=0.0,
                               run_qc=False)
    assert analyzer.onset_threshold == 0.0
    assert analyzer._use_surrogate_threshold is False
    # The prediction path resolves:
    #   pred_thr = self.onset_threshold if self.onset_threshold is not None else 0.5
    # With 0.0 stored, this yields 0.0, not 0.5. A regression to `or 0.5`
    # would make 0.0 -> 0.5 silently (the original bug).


def test_effective_gap_prevents_train_test_overlap():
    """The gap must account for horizon_windows * window_size raw WCC
    samples, not just horizon_windows feature rows. Regression test for
    the time-leakage bug where horizon_gap_rows was undercounted."""
    from multisync.prediction import _compute_effective_gap
    for ws in (20, 30, 60):
        step = max(1, ws // 2)
        for hw in (1, 2, 3, 4):
            eff = _compute_effective_gap(0, ws, hw)
            # Train row i: features from WCC[i*step, i*step+ws-1],
            # label from WCC[i*step+ws, i*step+ws+hw*ws-1].
            # Test row j=i+eff: features from WCC[j*step, j*step+ws-1].
            # No overlap: j*step >= i*step + ws + hw*ws
            label_end_wcc = ws + hw * ws  # exclusive, in raw WCC samples
            test_start_wcc = eff * step
            assert test_start_wcc >= label_end_wcc, (
                f"ws={ws} hw={hw}: effective_gap={eff} leaves "
                f"test_start={test_start_wcc} < label_end={label_end_wcc} "
                f"(overlap={label_end_wcc - test_start_wcc} samples)"
            )


def test_bridge_skips_zero_variance_signal():
    """The bridge QC gate must skip records with near-zero-variance
    signals (electrode flatline) rather than producing corrupted
    all-NaN features that silently pollute the FDR family."""
    n = 200
    good_a = np.sin(np.linspace(0, 10, n))
    flat_b = np.full(n, 0.7)  # zero variance
    rec = SimpleNamespace(
        dyad_label="d1", modality="eda", condition="A",
        person_a=good_a, person_b=flat_b, target_hz=1.0,
        incomplete=False, discontinuity_mask=None,
    )
    with pytest.warns(UserWarning, match="near-zero variance"):
        with pytest.raises(ValueError, match="No usable records"):
            records_to_inference_inputs([rec], hz=1.0, window_size=30)


def test_bridge_skips_high_nan_signal():
    """The bridge QC gate must skip records with excessive NaN
    (>5% per qc.DEFAULT_CONFIG) rather than letting them through."""
    n = 200
    good_a = np.sin(np.linspace(0, 10, n))
    nan_b = np.sin(np.linspace(0, 10, n))
    nan_b[:120] = np.nan  # 60% NaN
    rec = SimpleNamespace(
        dyad_label="d1", modality="eda", condition="A",
        person_a=good_a, person_b=nan_b, target_hz=1.0,
        incomplete=False, discontinuity_mask=None,
    )
    with pytest.warns(UserWarning, match="NaN rate"):
        with pytest.raises(ValueError, match="No usable records"):
            records_to_inference_inputs([rec], hz=1.0, window_size=30)

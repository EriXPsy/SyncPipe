"""Tests for the v1 audited evidence-chain inference API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multisync.dynamic_features import extract_dynamic_features, sliding_window_wcc
from multisync.inference_pipeline import InferencePipeline


def _signals(seed: int, n: int = 180, coupling: float = 0.7):
    rng = np.random.default_rng(seed)
    shared = np.sin(np.linspace(0, 8 * np.pi, n)) + 0.3 * rng.normal(size=n)
    a = coupling * shared + rng.normal(scale=0.6, size=n)
    b = coupling * shared + rng.normal(scale=0.6, size=n)
    return a, b


def _feature_df():
    rows = []
    hz = 1.0
    window = 20
    for i in range(4):
        for cond, coup in (("rest", 0.2), ("task", 0.8)):
            a, b = _signals(100 + i * 10 + (cond == "task"), coupling=coup)
            wcc = sliding_window_wcc(a, b, window_size=window, hz=hz)
            feats = extract_dynamic_features(wcc, hz=hz, wcc_window_sec=window / hz)
            row = {"dyad_id": f"dyad_{i}", "condition": cond}
            row.update(feats.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def test_audited_inference_api_runs_all_steps():
    df = _feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=10, seed=1)

    raw = {"dyad_0": _signals(1)}
    existence = pipe.run_synchrony_existence_audit(raw, wcc_window_size=20)
    assert existence["step"] == "synchrony_existence_audit"
    assert existence["n_pairs"] == 1

    cohort = {f"dyad_{i}": _signals(i) for i in range(3)}
    design = pipe.run_design_control_audit(
        cohort,
        wcc_window_size=20,
        n_pseudo_per_dyad=1,
        shift_lags_sec=(-40.0, 40.0),
    )
    assert design["audit"] == "design_controls"
    assert "feature_summary" in design

    segments = [
        ("seg1", _signals(10, n=60)[0], _signals(10, n=60)[1]),
        ("seg2", _signals(11, n=60)[0], _signals(11, n=60)[1]),
        ("seg3", _signals(12, n=60)[0], _signals(12, n=60)[1]),
    ]
    across = pipe.run_across_stimulus_shuffle_audit(
        segments,
        wcc_window_size=10,
        n_shuffles=8,
    )
    assert across["step"] == "across_stimulus_shuffle_audit"
    assert across["n_segments"] == 3

    group = pipe.run_group_condition_inference(n_permutations=20, contrast=("rest", "task"))
    assert group is not None


def test_run_audited_evidence_chain_returns_summary():
    df = _feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=8, seed=2)
    raw = {"dyad_0": _signals(20)}
    cohort = {f"dyad_{i}": _signals(30 + i) for i in range(3)}
    result = pipe.run_audited_evidence_chain(
        raw,
        wcc_window_size=20,
        design_signal_pairs=cohort,
        n_permutations=20,
    )
    assert result["evidence_chain_version"] == "v1"
    assert "Synchrony-existence" in result["summary"]
    assert result["synchrony_existence"]["n_pairs"] == 1
    assert result["design_controls"] is not None
    assert result["group_condition_inference"] is not None

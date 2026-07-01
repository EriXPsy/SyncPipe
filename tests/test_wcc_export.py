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

"""Execute pooled WCC calls to verify window-definition propagation."""
import numpy as np
import pytest

import syncpipe.session_threshold as thresholds


@pytest.mark.parametrize("entry", ["session", "modality", "condition"])
def test_pooled_helpers_execute_requested_window(monkeypatch, entry):
    original = thresholds.sliding_window_wcc
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs.get("window_type", "rect"))
        return original(*args, **kwargs)

    monkeypatch.setattr(thresholds, "sliding_window_wcc", record)
    rng = np.random.default_rng(9)
    signals = [(rng.normal(size=80), rng.normal(size=80))]
    kwargs = dict(hz=1., wcc_window_size=20, surrogate_n=3,
                  surrogate_method="ft", window_type="hann")
    if entry == "session":
        _, meta = thresholds.compute_session_pooled_threshold(signals, **kwargs)
    elif entry == "modality":
        result = thresholds.compute_session_pooled_thresholds_by_modality(
            signals, ["EDA"], return_meta=True, **kwargs)
        _, meta = result["EDA"]
    else:
        result = thresholds.compute_condition_pooled_thresholds({"task": signals}, **kwargs)
        _, meta = result["task"]
    assert seen == ["hann"] * 3
    assert meta["window_type"] == "hann"


def test_pooled_default_preserves_rect():
    rng = np.random.default_rng(12)
    signals = [(rng.normal(size=80), rng.normal(size=80))]
    kwargs = dict(hz=1., wcc_window_size=20, surrogate_n=3, surrogate_method="ft")
    default = thresholds.compute_session_pooled_threshold(signals, **kwargs)
    explicit = thresholds.compute_session_pooled_threshold(signals, window_type="rect", **kwargs)
    assert default == explicit


def test_bridge_executes_same_hann_window_for_pooled_null(monkeypatch):
    from types import SimpleNamespace
    from syncpipe.pipeline_bridge import records_to_inference_inputs

    original = thresholds.sliding_window_wcc
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs.get("window_type", "rect"))
        return original(*args, **kwargs)

    monkeypatch.setattr(thresholds, "sliding_window_wcc", record)
    rng = np.random.default_rng(17)
    rec = SimpleNamespace(dyad_label="d1", modality="EDA", condition="task",
                          person_a=rng.normal(size=100), person_b=rng.normal(size=100),
                          target_hz=1., incomplete=False)
    inputs = records_to_inference_inputs([rec], hz=1., window_size=20, window_type="hann")
    assert inputs.thresholds_by_modality is not None
    assert len(seen) == 200
    assert set(seen) == {"hann"}

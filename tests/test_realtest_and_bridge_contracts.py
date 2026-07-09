"""Contract tests for the real-data loader + pipeline-bridge path.

These are the "fuses" that Blind Spot B of the v1.0 review asked for:
the loader -> bridge -> inference chain is exactly where the
``discontinuity_mask`` single-sided-drop bug lived, yet it had *zero*
dedicated tests.  A later refactor of WCC defaults or the loaders could
silently corrupt the numbers behind the paper without any test catching
it.  These tests pin the contract so such regressions fail loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multisync.pipeline_bridge import records_to_inference_inputs
from multisync.realtest.gordon_2025 import (
    GordonDyadCondition,
    gordon_record_to_multisync_dyad,
)
from multisync.realtest.lerique_2024 import (
    LeriqueDyadCondition,
    lerique_record_to_multisync_dyad,
)

N = 50
HZ = 5.0
T = np.arange(N) / HZ


def _lerique_rec(mask: np.ndarray) -> LeriqueDyadCondition:
    a = pd.DataFrame({"time": T, "value": np.sin(T)})
    b = pd.DataFrame({"time": T, "value": np.cos(T)})
    return LeriqueDyadCondition(
        dyad_id="pce01__ecg__trials_concat",
        dyad_label="pce01",
        modality="ecg",
        condition="trials_concat",
        person_a=a,
        person_b=b,
        target_hz=HZ,
        n_samples=N,
        duration_sec=N / HZ,
        incomplete=False,
        discontinuity_mask=mask,
    )


def _gordon_rec() -> GordonDyadCondition:
    a = pd.DataFrame({
        "time": T,
        "motion_intensity": np.sin(T),
        "R": np.ones(N),
        "theta": np.zeros(N),
    })
    b = pd.DataFrame({
        "time": T,
        "motion_intensity": np.cos(T),
        "R": np.ones(N),
        "theta": np.zeros(N),
    })
    return GordonDyadCondition(
        dyad_id="p1_p2__exp1",
        pair_label="p1_p2",
        condition="exp1",
        person_a=a,
        person_b=b,
        target_hz=HZ,
        n_samples=N,
        duration_sec=N / HZ,
    )


# ---------------------------------------------------------------------------
# Lerique loader: the discontinuity_mask must survive into the Dyad
# ---------------------------------------------------------------------------
def test_lerique_loader_carries_discontinuity_mask():
    mask = np.zeros(N, dtype=bool)
    mask[10:20] = True  # arbitrary internal-segment flag
    rec = _lerique_rec(mask)
    dyad = lerique_record_to_multisync_dyad(rec)
    assert dyad.discontinuity_mask is not None
    assert np.array_equal(dyad.discontinuity_mask, mask)


def test_lerique_loader_rejects_incomplete():
    rec = _lerique_rec(np.ones(N, dtype=bool))
    rec = rec.__class__(**{**rec.__dict__, "incomplete": True})
    with pytest.raises(ValueError):
        lerique_record_to_multisync_dyad(rec)


# ---------------------------------------------------------------------------
# Gordon loader: runs and yields the expected cross-person modalities
# ---------------------------------------------------------------------------
def test_gordon_loader_produces_dyad():
    dyad = gordon_record_to_multisync_dyad(_gordon_rec())
    assert "motion_intensity_a" in dyad.modalities
    assert "motion_intensity_b" in dyad.modalities


# ---------------------------------------------------------------------------
# pipeline_bridge: no KeyError + mask propagates into InferenceInputs
# (re-guards the prior single-sided mask-drop bug)
# ---------------------------------------------------------------------------
def test_bridge_propagates_discontinuity_mask():
    mask = np.zeros(N, dtype=bool)
    mask[5:15] = True
    rec = _lerique_rec(mask)
    inputs = records_to_inference_inputs(
        [rec], hz=HZ, window_size=20,
    )
    assert inputs.discontinuity_mask is not None
    key = "pce01__ecg__trials_concat"
    assert key in inputs.discontinuity_mask
    assert np.array_equal(inputs.discontinuity_mask[key], mask)
    assert not inputs.features_df.empty


def test_bridge_handles_missing_mask_gracefully():
    # A record whose discontinuity_mask attribute is None must not crash
    # and must round-trip as None in the per-key map.
    rec = SimpleNamespace(
        dyad_label="pce02",
        modality="ecg",
        condition="rest1",
        target_hz=HZ,
        incomplete=False,
        person_a=np.sin(T),
        person_b=np.cos(T),
        discontinuity_mask=None,
    )
    inputs = records_to_inference_inputs([rec], hz=HZ, window_size=20)
    key = "pce02__ecg__rest1"
    assert inputs.discontinuity_mask is not None
    assert inputs.discontinuity_mask.get(key) is None

"""Shared prepared-observation contracts across computation and null stages."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from syncpipe.pipeline_bridge import records_to_inference_inputs
from syncpipe.preparation import PreparedObservation
from syncpipe.session_threshold import compute_session_pooled_threshold


def test_prepared_geometry_is_immutable_and_window_aligned():
    a = np.arange(100, dtype=float)
    b = a.copy()
    a[40] = np.nan
    source = np.ones(100, dtype=bool)
    source[70] = False
    obs = PreparedObservation.from_signals(
        key="d1__EDA__task", dyad_id="d1", modality="EDA",
        condition="task", hz=1.0, signal_a=a, signal_b=b,
        discontinuity_mask=source,
    )
    assert obs.geometry.segments == ((0, 40), (41, 70), (71, 100))
    expected = sum(max(0, length - 10 + 1) for length in (40, 29, 29))
    assert int(obs.geometry.window_mask(10).sum()) == expected
    assert obs.signal_a.flags.writeable is False
    assert obs.geometry.analysis_mask.flags.writeable is False


def test_prepared_cohort_carries_typed_exclusions():
    signal = np.sin(np.linspace(0, 10, 100))
    good = SimpleNamespace(
        dyad_label="d1", modality="EDA", condition="task", target_hz=1.0,
        incomplete=False, person_a=signal, person_b=signal,
        discontinuity_mask=None,
    )
    incomplete = SimpleNamespace(
        dyad_label="d2", modality="EDA", condition="task", target_hz=1.0,
        incomplete=True, person_a=signal, person_b=signal,
        discontinuity_mask=None,
    )
    inputs = records_to_inference_inputs(
        [good, incomplete], hz=1.0, window_size=20, onset_threshold=0.5,
    )
    exclusions = inputs.prepared_cohort.exclusions
    assert len(exclusions) == 1
    assert exclusions[0].code == "incomplete_record"
    assert exclusions[0].stage == "preparation"
    assert exclusions[0].to_dict()["claim_effect"] == "observation_excluded"


def test_pooled_threshold_uses_finite_segments_instead_of_dropping_dyad():
    rng = np.random.default_rng(2)
    a = rng.normal(size=140)
    b = rng.normal(size=140)
    a[70] = np.nan
    geometry_obs = PreparedObservation.from_signals(
        key="d1__EDA__task", dyad_id="d1", modality="EDA",
        condition="task", hz=1.0, signal_a=a, signal_b=b,
    )
    threshold, meta = compute_session_pooled_threshold(
        [(a, b)], hz=1.0, wcc_window_size=20,
        surrogate_n=10, seed=3,
        discontinuity_masks=[geometry_obs.geometry.analysis_mask],
    )
    assert np.isfinite(threshold)
    assert meta["n_dyads_used"] == 1
    assert meta["n_dyads_excluded_nonfinite"] == 0
    assert meta["n_dyads_excluded_no_eligible_segments"] == 0

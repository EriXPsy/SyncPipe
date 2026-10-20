"""Segment seams are excluded by design; they must not be mistaken for missing data.

With several short trials the windows that would straddle a seam are a fixed share of the series
(about ``(window - 1) / trial_length``). Counting them toward ``max_nan_ratio`` made the whole
feature set undefined for any multi-trial series with ``window / trial_length`` above ~0.2, while the
log blamed "NaN ratio" as if the data were poor.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.dynamic_feature_extraction import extract_dynamic_features
from syncpipe.pair_pipeline import compute_pair_pipeline


def _trace(n=200, gap_every=50, gap_len=19, seed=0):
    rng = np.random.default_rng(seed)
    wcc = np.clip(0.4 + 0.3 * np.sin(np.arange(n) / 9.0) + 0.1 * rng.normal(size=n), -0.99, 0.99)
    gap = np.zeros(n, dtype=bool)
    for start in range(gap_every, n, gap_every):
        gap[start:start + gap_len] = True
    wcc[gap] = np.nan
    return wcc, gap


def test_by_design_gaps_do_not_trigger_the_missing_data_guard():
    wcc, gap = _trace()
    assert 1 - np.isfinite(wcc).mean() > 0.2                                   # would exceed max_nan_ratio
    without = extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0, gap_policy="segment")
    assert np.isnan(without.mean_synchrony)                                    # the old, over-eager behaviour
    declared = extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0, gap_policy="segment", expected_gap=gap)
    assert np.isfinite(declared.mean_synchrony) and declared.nan_fraction == 0.0


def test_genuine_missing_data_still_trips_the_guard_when_seams_are_declared():
    wcc, gap = _trace()
    rng = np.random.default_rng(1)
    eligible = np.flatnonzero(~gap)
    wcc[rng.choice(eligible, size=int(0.4 * eligible.size), replace=False)] = np.nan
    out = extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0, gap_policy="segment", expected_gap=gap)
    assert np.isnan(out.mean_synchrony) and out.nan_fraction > 0.3


def test_omitting_expected_gap_changes_nothing():
    wcc, _ = _trace()
    wcc[:] = np.nan_to_num(wcc, nan=0.2)
    a = extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0)
    b = extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0, expected_gap=None)
    assert a.to_dict() == b.to_dict()


def test_expected_gap_must_match_the_trace_shape():
    wcc, gap = _trace()
    with pytest.raises(ValueError, match="same shape"):
        extract_dynamic_features(wcc, hz=1.0, wcc_window_sec=20.0, expected_gap=gap[:-1])


def test_a_multi_trial_pair_keeps_its_structure_features_end_to_end():
    """Three 60 s trials at window 20: ~25% of windows straddle a seam, yet the features must be defined."""
    rng = np.random.default_rng(3)
    n, window = 180, 20
    shared = np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, mode="same")[4:4 + n]
    a = shared + 0.7 * np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, mode="same")[4:4 + n]
    b = shared + 0.7 * np.convolve(rng.normal(size=n + 8), np.ones(5) / 5, mode="same")[4:4 + n]
    mask = np.ones(n, dtype=bool)
    mask[60] = mask[120] = False
    result = compute_pair_pipeline(a, b, hz=1.0, window_size=window, discontinuity_mask=mask)
    assert 1 - np.isfinite(result.wcc).mean() > 0.2
    assert np.isfinite(result.features.mean_synchrony) and np.isfinite(result.features.peak_amplitude)

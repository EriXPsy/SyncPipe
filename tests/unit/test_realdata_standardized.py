"""Contracts for the bounded descriptive real-data runner."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.run_realdata_standardized import (
    UNITS, validate_signals, geometry_diagnostics, independent_segment_reference,
)
from syncpipe.pair_pipeline import compute_pair_pipeline
from syncpipe.preparation import resolve_signal_geometry


def record():
    frame = pd.DataFrame({'time': np.arange(60) / 2, 'value': np.sin(np.arange(60))})
    return SimpleNamespace(person_a=frame, person_b=frame.copy(), n_samples=60, target_hz=2.)


def test_verified_feature_units():
    assert UNITS['dwell_time'] == 'seconds'
    assert UNITS['first_peak_time'] == 'seconds'
    assert UNITS['switching_rate'] == 'transitions_per_minute'


def test_common_axis_and_mask_accepted():
    validate_signals(record(), 'value', np.ones(60, dtype=bool))


@pytest.mark.parametrize('defect', ['different_time', 'compressed_gap', 'mask', 'nonfinite'])
def test_invalid_signal_contract_rejected(defect):
    rec = record()
    mask = np.ones(60, dtype=bool)
    if defect == 'different_time':
        rec.person_b.loc[5, 'time'] += .1
    elif defect == 'compressed_gap':
        rec.person_a.loc[5, 'time'] += .1
        rec.person_b.loc[5, 'time'] += .1
    elif defect == 'mask':
        mask = mask[:-1]
    else:
        rec.person_b.loc[5, 'value'] = np.inf
    with pytest.raises(AssertionError):
        validate_signals(rec, 'value', mask)


@pytest.mark.parametrize('split', [False, True])
def test_reference_matches_independent_pairs_and_no_cross_boundary(split):
    rng = np.random.default_rng(81)
    a = rng.normal(size=180)
    b = .6*a + rng.normal(size=180)
    mask = np.ones(180, dtype=bool)
    if split:
        mask[90] = False
    diagnostics = geometry_diagnostics(a, b, mask, 30)
    assert diagnostics['raw_missing_fraction'] == 0
    refs = independent_segment_reference(a, b, mask, 30, 1.)
    full = compute_pair_pipeline(a, b, hz=1., window_size=30, discontinuity_mask=mask)
    eligible = resolve_signal_geometry(a, b, mask).window_mask(30)
    assert np.isnan(full.wcc[~eligible]).all()
    for item in refs:
        s, e = item['start'], item['end']
        result = compute_pair_pipeline(a[s:e], b[s:e], hz=1., window_size=30,
            discontinuity_mask=np.ones(e-s, dtype=bool))
        np.testing.assert_allclose(full.wcc[s:e-29], result.wcc, atol=1e-12)
        for key, value in item['features'].items():
            if value is None:
                assert np.isnan(result.features_dict[key])
            else:
                assert value == result.features_dict[key]
    if not split:
        assert diagnostics['design_invalid_window_fraction'] == 0
        assert refs[0]['features']['mean_synchrony'] == full.features.mean_synchrony


@pytest.mark.parametrize('hide_missing', [False, True])
def test_missing_not_exempted_by_mask(hide_missing):
    a = np.sin(np.arange(180))
    b = a.copy()
    a[45] = np.nan
    mask = np.ones(180, dtype=bool)
    if hide_missing:
        mask[45] = False
    diagnostics = geometry_diagnostics(a, b, mask, 30)
    assert diagnostics['raw_missing_fraction'] == pytest.approx(1/180)
    assert all(r['reason'] == 'raw_nonfinite_no_mask_exemption'
        for r in independent_segment_reference(a, b, mask, 30, 1.))


@pytest.mark.parametrize('length,all_masked', [(20, False), (33, False), (100, True)])
def test_short_or_all_invalid_segments(length, all_masked):
    a = np.sin(np.arange(length))
    mask = np.full(length, not all_masked, dtype=bool)
    refs = independent_segment_reference(a, a, mask, 30, 1.)
    assert not refs or all(r['status'] == 'excluded' for r in refs)
    diagnostics = geometry_diagnostics(a, a, mask, 30)
    assert diagnostics['context_segments_below_3window'] == (0 if all_masked else 1)


def test_boundary_gate_retained_and_context_trials_ineligible():
    a = np.sin(np.arange(180))
    mask = np.ones(180, dtype=bool)
    mask[[60, 120]] = False
    diagnostics = geometry_diagnostics(a, a, mask, 30)
    assert diagnostics['raw_missing_fraction'] == 0
    assert diagnostics['design_invalid_window_fraction'] > .2
    assert diagnostics['context_segments_below_3window'] == 3
    result = compute_pair_pipeline(a, a, hz=1., window_size=30, discontinuity_mask=mask)
    assert np.isnan(result.features.mean_synchrony)
    assert all(not r['context_eligible'] for r in independent_segment_reference(a, a, mask, 30, 1.))

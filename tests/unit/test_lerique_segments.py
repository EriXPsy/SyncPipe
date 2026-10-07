"""Diagnostic alternatives, not a change to the frozen production loader."""

from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_lerique_segments.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason=("compare_lerique_segments.py is untracked in the published repository; "
            "these contracts apply only to full working checkouts"),
)

import numpy as np
import pytest

import importlib

if _SCRIPT.exists():
    segment_reference = importlib.import_module(
        "scripts.compare_lerique_segments").segment_reference
else:
    segment_reference = None
from syncpipe.realtest import lerique_2024 as L


def signal(modality):
    t = np.arange(60000) / 1000.
    if modality == 'ECG':
        return np.exp(-((t % .8 - .4) / .015) ** 2).astype(np.float32)
    return (np.sin(2*np.pi*.2*t) + .1*np.sin(2*np.pi*.07*t)).astype(np.float32)


@pytest.mark.parametrize('modality', L.MODALITIES)
def test_segment_reference_later_perturbation_independence(modality):
    if modality == 'ECG':
        pytest.importorskip('neurokit2')
    first = signal(modality)
    later = first.copy()
    altered = later + 30*np.exp(-np.arange(len(later))/3000)
    original, mask = segment_reference([first, later], modality)
    perturbed, altered_mask = segment_reference([first, altered], modality)
    np.testing.assert_array_equal(original[:60][mask[:60]], perturbed[:60][mask[:60]])
    np.testing.assert_array_equal(mask[:60], altered_mask[:60])
    assert not mask[60]
    assert mask[:60].all()


@pytest.mark.parametrize('modality', L.MODALITIES)
def test_single_segment_reference_is_bit_exact(modality):
    if modality == 'ECG':
        pytest.importorskip('neurokit2')
    raw = signal(modality)
    actual, am = segment_reference([raw], modality)
    expected, em = L._PREPROC_DISPATCH[modality](raw, 1000., 1.)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(am, em)


@pytest.mark.parametrize('modality', ['EDA', 'RESP'])
def test_frozen_concat_has_preboundary_influence(modality):
    first = signal(modality)
    second = first.copy()
    changed = second + 30*np.exp(-np.arange(len(second))/3000)
    boundary = np.ones(120000, dtype=bool)
    boundary[60000] = False
    original, mask = L._PREPROC_DISPATCH[modality](np.r_[first, second], 1000., 1., boundary)
    altered, _ = L._PREPROC_DISPATCH[modality](np.r_[first, changed], 1000., 1., boundary)
    assert np.max(np.abs(original[:60][mask[:60]] - altered[:60][mask[:60]])) > 1e-6


def test_ecg_segment_failure_retains_geometry():
    pytest.importorskip('neurokit2')
    raw = signal('ECG')
    output, mask = segment_reference([raw, np.zeros_like(raw)], 'ECG')
    assert len(output) == 120
    assert mask[:60].all()
    assert not mask[60:].any()


def test_noninteger_grid_requires_explicit_decision():
    with pytest.raises(ValueError, match='Noninteger segment grid'):
        segment_reference([np.zeros(60001)], 'EDA')

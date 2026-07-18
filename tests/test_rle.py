"""
Regression tests for ``multisync.feature_definitions._find_runs``.

``_find_runs`` is the single shared run-length detector extracted from four
previously-parallel diff-based implementations (compute_dwell_time in
feature_definitions, extract_episodes in morphology,
state_transition_shuffle_surrogate in surrogate, and the NaN-gap check in qc).

These tests lock the numerical contract:
  * (starts[k], ends[k]) mark the k-th run of True in a boolean mask,
    mask[starts[k]:ends[k]] is the run, length == ends[k] - starts[k].
  * behaviour is bit-for-bit identical to the original inline snippets,
    including the tolerance-free boolean detector and the qc NaN-gap detector.
  * boundary masks (empty / all-True / all-False / leading / trailing / single)
    are handled without raising.
"""

import numpy as np
import pytest

from multisync.feature_definitions import _find_runs


# ---------------------------------------------------------------------------
# Reference implementations of the ORIGINAL inline logic (pre-refactor).
# Used only to prove the shared helper is behaviourally identical.
# ---------------------------------------------------------------------------

def _reference_boolean_runs(mask):
    """Original diff-based detector as it lived in feature_definitions/
    morphology/surrogate before consolidation."""
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], m, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return starts, ends


def _reference_qc_nan_gap(isnan):
    """Original NaN-gap detector exactly as it lived in qc.py before
    consolidation (pad with int 0 sentinels, diff == +/-1)."""
    gap_starts = np.where(
        np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == 1
    )[0]
    gap_ends = np.where(
        np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == -1
    )[0]
    return gap_starts, gap_ends


# ---------------------------------------------------------------------------
# Basic run detection
# ---------------------------------------------------------------------------

def test_basic_run_detection():
    mask = np.array([True, True, False, True])
    starts, ends = _find_runs(mask)
    assert list(starts) == [0, 3]
    assert list(ends) == [2, 4]
    assert list(ends - starts) == [2, 1]


def test_run_lengths_match_slices():
    rng = np.random.default_rng(7)
    for _ in range(200):
        mask = rng.random(40) > 0.5
        starts, ends = _find_runs(mask)
        for s, e in zip(starts, ends):
            assert mask[s:e].all()
            if s > 0:
                assert not mask[s - 1]
            if e < mask.size:
                assert not mask[e]


# ---------------------------------------------------------------------------
# Boundary masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mask,expected_starts,expected_ends",
    [
        (np.array([], dtype=bool), [], []),                       # empty
        (np.array([True, True, True]), [0], [3]),                # all True
        (np.array([False, False]), [], []),                      # all False
        (np.array([True, False, True]), [0, 2], [1, 3]),         # leading run
        (np.array([False, True]), [1], [2]),                     # trailing run
        (np.array([False, True, False]), [1], [2]),             # single middle
        (np.array([True, False, False]), [0], [1]),             # run at start
    ],
)
def test_boundaries(mask, expected_starts, expected_ends):
    starts, ends = _find_runs(mask)
    assert list(starts) == expected_starts
    assert list(ends) == expected_ends


def test_run_spanning_boundary_single_true_block():
    # A single contiguous block must be reported as exactly one run even when
    # it begins at index 0 and ends at the last index.
    mask = np.ones(10, dtype=bool)
    starts, ends = _find_runs(mask)
    assert list(starts) == [0]
    assert list(ends) == [10]


# ---------------------------------------------------------------------------
# Equivalence to the original inline implementations
# ---------------------------------------------------------------------------

def test_equivalence_to_original_boolean_snippet():
    rng = np.random.default_rng(42)
    masks = [
        rng.random(n) > 0.5
        for n in [0, 1, 2, 3, 10, 50, 200, 1000]
    ]
    # hand-crafted edge patterns
    masks += [
        np.array([True] * 5 + [False] * 5 + [True] * 5),
        np.array([False] * 5 + [True] * 5 + [False] * 5),
        np.array([True, False, True, False, True]),
        np.array([False, True, False, True, False]),
    ]
    for mask in masks:
        got = _find_runs(mask)
        ref = _reference_boolean_runs(mask)
        assert np.array_equal(got[0], ref[0]), mask
        assert np.array_equal(got[1], ref[1]), mask


def test_equivalence_to_qc_nan_gap_snippet():
    rng = np.random.default_rng(123)
    for _ in range(300):
        vals = rng.random(80)
        # inject NaN runs of random length at random positions
        isnan = np.zeros(80, dtype=bool)
        for _ in range(rng.integers(0, 4)):
            a = rng.integers(0, 78)
            b = rng.integers(a + 1, 81)
            isnan[a:b] = True
        got = _find_runs(isnan)
        ref = _reference_qc_nan_gap(isnan)
        assert np.array_equal(got[0], ref[0]), isnan
        assert np.array_equal(got[1], ref[1]), isnan
        # gap lengths (what qc actually consumes) must match
        got_lens = got[1] - got[0] if got[0].size else np.array([])
        ref_lens = ref[1] - ref[0] if ref[0].size else np.array([])
        assert np.array_equal(got_lens, ref_lens), isnan


def test_qc_max_gap_unchanged():
    """End-to-end check that the qc NaN-gap metric is numerically identical
    to the pre-refactor computation."""
    rng = np.random.default_rng(99)
    for _ in range(100):
        vals = rng.normal(size=120)
        n_nan = int(rng.integers(0, 8))
        idx = rng.choice(120, size=n_nan, replace=False)
        vals[idx] = np.nan
        isnan = np.isnan(vals)

        got_s, got_e = _find_runs(isnan)
        ref_s, ref_e = _reference_qc_nan_gap(isnan)
        got_max = int(max([got_e[i] - got_s[i] for i in range(got_s.size)], default=0))
        ref_max = int(max([ref_e[i] - ref_s[i] for i in range(ref_s.size)], default=0))
        assert got_max == ref_max

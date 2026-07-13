"""Regression tests for BRM-2026-07-13 fix #3: onset surrogate threshold.

BUG-2: the surrogate-derived onset threshold fell back to the fixed
``ONSET_THRESHOLD`` (0.5) *silently* whenever the surrogate distribution
was degenerate or non-finite -- callers that ignored ``is_surrogate_derived``
never learned the threshold was not data-driven.

BUG-3: for periodic / strongly autocorrelated signals the IAAFT surrogate
null is shifted upward, yielding extreme thresholds (e.g. 0.957) that are
artifacts, not genuine "high sync by chance" levels.  The old code used
them without comment.

Both fallback paths now emit a ``logger.warning`` (never silent) and the
periodicity case is caught by a hard ceiling (SURROGATE_THRESHOLD_MAX).
"""

import logging

import numpy as np
import pytest

from multisync.feature_definitions import (
    ONSET_THRESHOLD,
    SURROGATE_THRESHOLD_MAX,
    compute_surrogate_threshold,
)
from multisync.dynamic_features import compute_surrogate_threshold_from_signals


def test_degenerate_fallback_is_loud_and_flagged(caplog):
    """< 10 finite surrogate values -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    # 5 finite values only -> degenerate
    mat = np.full((1, 5), 0.3)
    thr, is_surr = compute_surrogate_threshold(mat)
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("falling back" in r.message for r in caplog.records)


def test_periodic_artifact_fallback_is_loud_and_flagged(caplog):
    """Extreme null threshold (> ceiling) -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    # All surrogate WCC values at 0.95 -> 95th pct = 0.95 > ceiling
    mat = np.full((50, 200), 0.95)
    thr, is_surr = compute_surrogate_threshold(mat)
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("sanity ceiling" in r.message for r in caplog.records)


def test_valid_distribution_returns_derived_true(caplog):
    """A sane null distribution is used and flagged as surrogate-derived."""
    caplog.set_level(logging.WARNING)
    rng = np.random.default_rng(0)
    mat = rng.normal(0.0, 0.2, size=(50, 200))
    thr, is_surr = compute_surrogate_threshold(mat)
    assert is_surr is True
    assert 0.0 < thr < SURROGATE_THRESHOLD_MAX
    # No fallback warning should have fired.
    assert not any("falling back" in r.message or "sanity ceiling" in r.message
                   for r in caplog.records)


def test_nonfinite_raw_signals_fallback_is_loud(caplog):
    """Non-finite raw signals -> (0.5, False) with a WARNING."""
    caplog.set_level(logging.WARNING)
    sig_a = np.array([np.nan, 1.0, 2.0, 3.0])
    sig_b = np.array([1.0, 2.0, 3.0, 4.0])
    thr, is_surr = compute_surrogate_threshold_from_signals(
        sig_a, sig_b, hz=1.0, wcc_window_size=10,
    )
    assert thr == ONSET_THRESHOLD
    assert is_surr is False
    assert any("non-finite raw signals" in r.message for r in caplog.records)

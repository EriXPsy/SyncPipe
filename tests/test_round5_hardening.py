"""Round-5 hardening tests.

Covers the two audited code-level refinements shipped with v1.1.0:

- ``compute_synchrony_entropy`` fixed-range variant (cross-dyad
  comparability) alongside the default data-adaptive range;
- per-feature salted sign-flip seeds used by the design-control audit
  (``design_controls._paired_signflip_p_upper`` call sites).
"""
import zlib

import numpy as np

from syncpipe.design_controls import _paired_signflip_p_upper
from syncpipe.feature_definitions import compute_synchrony_entropy


def test_entropy_adaptive_range_is_not_cross_dyad_comparable():
    """Two traces with the same distribution shape on different ranges get
    (near-)equal adaptive entropies — the documented caveat that motivated
    the fixed-range variant."""
    rng = np.random.default_rng(7)
    wide = rng.uniform(0.0, 0.8, size=600)
    narrow = rng.uniform(0.0, 0.4, size=600)
    adaptive_gap = abs(
        compute_synchrony_entropy(wide) - compute_synchrony_entropy(narrow)
    )
    assert adaptive_gap < 0.05


def test_entropy_fixed_range_restores_cross_dyad_comparability():
    """With ``fixed_range=True`` both traces are histogrammed over [-1, 1],
    so different spans yield different, directly comparable values."""
    rng = np.random.default_rng(7)
    wide = rng.uniform(0.0, 0.8, size=600)
    narrow = rng.uniform(0.0, 0.4, size=600)
    fixed_wide = compute_synchrony_entropy(wide, fixed_range=True)
    fixed_narrow = compute_synchrony_entropy(narrow, fixed_range=True)
    assert fixed_wide > fixed_narrow  # wider span -> more populated bins
    # Same input, same value: the variant is deterministic.
    assert compute_synchrony_entropy(wide, fixed_range=True) == fixed_wide


def test_signflip_seed_salting_is_deterministic_and_feature_stable():
    """The design-control audit salts each feature's sign-flip seed with a
    stable label hash: the same salt must reproduce the same p-value, and
    distinct feature labels must map to distinct salts."""
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.4, 1.0, size=30)  # >12 dyads -> Monte-Carlo path

    def salted(label: str) -> int:
        return 42 + zlib.crc32(label.encode("utf-8")) % 100000

    p1 = _paired_signflip_p_upper(deltas, seed=salted("peak_amplitude"))
    assert p1 == _paired_signflip_p_upper(deltas, seed=salted("peak_amplitude"))
    assert 0.0 <= p1 <= 1.0
    assert salted("peak_amplitude") != salted("dwell_time")

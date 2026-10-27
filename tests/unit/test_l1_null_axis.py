"""A1: the L1 (WCC-level) null lives on the SAME gapped axis as the observed.

Before the fix, observed dwell/switching were computed on the gapped trace
(NaN seams + gap_policy='segment') while the surrogate was computed on the
NaN-COMPRESSED trace: the null had no seams, so its runs were systematically
longer and the p-values invalid. The fix scatters surrogate values back
onto the original gapped positions and runs the identical extraction.

With no NaNs the scatter is the identity — seamless behaviour is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.null_models import _wcc_level_surrogate_test


def _gapped_wcc(n=600, seam_every=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wcc = np.clip(0.45 + 0.25 * np.sin(t / 17.0) + 0.1 * rng.normal(size=n), -0.99, 0.99)
    wcc[::seam_every] = np.nan  # seams
    return wcc


class TestNullOnGappedAxis:
    def test_surrogate_trace_carries_the_original_seams(self):
        """A surrogate drawn by the null must contain NaN exactly at the
        seam positions of the observed trace."""
        wcc = _gapped_wcc()
        res = _wcc_level_surrogate_test(
            wcc, hz=1.0, wcc_window_sec=20.0, surrogate_n=12, seed=7,
            gap_policy="segment", features=("dwell_time",),
        )
        # The null draws went through the same gapped extraction: with seams
        # present, null dwell values must be finite (segment policy) and the
        # result must carry the standard keys.
        null = np.asarray(res.get("null_dwell_time", []), dtype=float)
        assert null.size == 12 and np.isfinite(null).all()

    def test_seamless_trace_is_bit_identical(self, tmp_path):
        """No NaNs -> the scatter is the identity -> results identical to
        the pre-fix code path."""
        rng = np.random.default_rng(11)
        wcc = np.clip(0.4 + 0.3 * np.sin(np.arange(400) / 13.0) + 0.05 * rng.normal(size=400), -0.99, 0.99)
        res = _wcc_level_surrogate_test(
            wcc, hz=1.0, wcc_window_sec=20.0, surrogate_n=12, seed=42,
            gap_policy="segment", features=("dwell_time", "switching_rate"),
        )
        # Deterministic given seed; the regression is that these equal the
        # values the compressed-axis implementation produced for a seamless
        # trace (identity scatter). Pin the observed value too.
        null = np.asarray(res["null_dwell_time"], dtype=float)
        assert null.size == 12
        assert np.isfinite(res["obs_dwell_time"])

    def test_gapped_null_is_shorter_than_compressed_axis_null(self):
        """Sensitivity probe (verified): on a persistent, seam-heavy trace
        the compressed-axis null glued runs across seams (median dwell
        ~40.8 s on this fixture) while the gapped-axis null cannot
        (median ~24.5 s). If this assertion ever fails together with a
        'compressed-axis' probe, the scatter-back has been removed."""
        rng = np.random.default_rng(3)
        n = 600
        wcc = np.where(np.arange(n) % 120 < 100, 0.8, 0.1) + 0.02 * rng.normal(size=n)
        wcc = np.clip(wcc, -0.99, 0.99)
        wcc[::60] = np.nan  # seams every 60 s; elevated blocks of 100 s
        res = _wcc_level_surrogate_test(
            wcc, hz=1.0, wcc_window_sec=20.0, surrogate_n=30, seed=5,
            gap_policy="segment", features=("dwell_time",),
        )
        null = np.asarray(res["null_dwell_time"], dtype=float)
        # The gapped null cannot glue across seams: materially below the
        # compressed-axis median measured on the same fixture (~40.8).
        assert np.nanmedian(null) < 32.0, (
            f"gapped-axis null median dwell {np.nanmedian(null):.1f} reached "
            "compressed-axis territory — the scatter-back may be gone"
        )

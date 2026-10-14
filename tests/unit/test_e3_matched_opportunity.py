"""E3 matched-opportunity arms (review R2, 2026-10-13; ported from the
Claude P0 changeset after dialectical review).

The time-shift control used to compare the full-length real pair with a
shorter-overlap shifted pair. ``peak_amplitude`` is a maximum, so the
window-count loss alone lowered the shifted arm and manufactured
"real > shifted": null-world FPR 15% at 180 s continuous and 87.5% on ten
60-s trials (window 30). The fix compares a real arm cropped to the same
span and skips lags whose shifted arm keeps < 80% of the matched real
arm's valid windows.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.design_controls import (
    MIN_SHIFT_WINDOW_RATIO,
    design_control_audit,
)


def _independent_pairs(n_dyads: int, n: int, seed: int):
    """Null-world cohort: independent AR(1) pairs, no coupling."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for d in range(n_dyads):
        def ar1():
            e = rng.normal(size=n) * np.sqrt(1 - 0.9 ** 2)
            x = np.empty(n)
            x[0] = rng.normal()
            for t in range(1, n):
                x[t] = 0.9 * x[t - 1] + e[t]
            return x
        pairs[f"d{d:02d}"] = (ar1(), ar1())
    return pairs


class TestMatchedOpportunityArms:
    def test_opportunity_fields_present(self):
        pairs = _independent_pairs(6, 1800, seed=1)
        out = design_control_audit(
            pairs, hz=1.0, window_size=20, shift_lags_sec=(-30.0, 30.0), feature_names=("peak_amplitude",),
        )
        opp = out["time_shift_opportunity"]
        assert opp["min_window_ratio"] == MIN_SHIFT_WINDOW_RATIO
        assert opp["n_lag_arms_used"] >= 1
        assert opp["n_lag_arms_used"] + opp["n_lag_arms_skipped_low_opportunity"] >= 1

    @staticmethod
    def _segmented_mask(n: int, seg_len: int) -> np.ndarray:
        """True inside segments; False on every segment boundary sample.
        Windows straddling a boundary are invalid by design."""
        m = np.ones(n, dtype=bool)
        m[::seg_len] = False  # boundary at each segment start
        return m

    def test_shift_delta_uses_matched_arms_not_full_length(self):
        """With segmented data (the regime the defect lived in) and a lag
        comparable to the segment length, the shifted arm loses most
        valid windows; every lag must be skipped and the delta reported
        NaN (E3 not adjudicated) instead of a manufactured positive."""
        seg_len, n = 60, 600
        rng = np.random.default_rng(2)

        def ar1():
            e = rng.normal(size=n) * np.sqrt(1 - 0.9 ** 2)
            x = np.empty(n)
            x[0] = rng.normal()
            for t in range(1, n):
                x[t] = 0.9 * x[t - 1] + e[t]
            return x

        pairs, masks = {}, {}
        for d in range(3):
            pairs[f"d{d:02d}"] = (ar1(), ar1())
            masks[f"d{d:02d}"] = self._segmented_mask(n, seg_len)
        out = design_control_audit(
            pairs, hz=1.0, window_size=30, shift_lags_sec=(250.0,),
            feature_names=("peak_amplitude",),
            discontinuity_masks=masks,
        )
        fs = out["feature_summary"]["peak_amplitude"]
        assert out["time_shift_opportunity"]["n_lag_arms_used"] == 0, (
            "a 250-s shift over 60-s segments must lose >80% of windows"
        )
        assert np.isnan(fs["real_minus_time_shift_mean"])
        assert np.isnan(fs["p_real_gt_time_shift"])

    def test_null_world_fpr_controlled_short_records(self):
        """Slow calibration: independent pairs, short (180 s) records —
        the regime where the old comparison inflated to ~15%."""
        rng = np.random.default_rng(20261013)
        n_cohorts = 20
        rejections = 0
        for rep in range(n_cohorts):
            pairs = _independent_pairs(8, 180, seed=int(rng.integers(1 << 30)))
            out = design_control_audit(
                pairs, hz=1.0, window_size=20,     shift_lags_sec=(-60.0, -45.0, -30.0, 30.0, 45.0, 60.0),
                feature_names=("peak_amplitude",),
                seed=rep,
            )
            p = out["feature_summary"]["peak_amplitude"]["p_real_gt_time_shift"]
            if np.isfinite(p) and p < 0.05:
                rejections += 1
        # Binomial(20, 0.05): P(X >= 4) ~ 1.6%; allow up to 3.
        assert rejections <= 3, (
            f"E3 null-world rejections {rejections}/{n_cohorts} exceed the "
            "binomial tolerance — matched-opportunity arms are leaking"
        )

"""Audit-round-6 regression tests (adversarial-review fixes, 2026-09-13).

M1 — smoothed_wcc masked-convolution revision:
  * edge peaks are no longer attenuated by zero-padding,
  * a NaN seam does not poison neighbouring smoothing windows,
  * interior positions remain bit-identical to the legacy boxcar mean.

M8 — sample-rate constant: window_sec= path scales the kernel with hz so
the physical smoothing duration is constant across sampling rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syncpipe.feature_definitions import (
    ONSET_THRESHOLD,
    compute_peak_amplitude,
    smoothed_wcc,
)


class TestSmoothedWccEdgeExact:
    def test_edge_peak_not_attenuated(self):
        # Dominant episode AT the trace edge: legacy zero-padding pulled the
        # smoothed edge value toward 0; masked convolution with symmetric
        # truncation returns the exact 2-sample finite mean at BOTH edges.
        w = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        sm = smoothed_wcc(w)
        assert sm[0] == pytest.approx((0.6 + 0.1) / 2.0)  # 2-sample window
        assert sm[-1] == pytest.approx((0.1 + 0.1) / 2.0)  # symmetric right
        # ...and the peak stays where it is, undamped
        val, idx = compute_peak_amplitude(sm)
        assert idx == 0
        assert val == pytest.approx(0.35)

    def test_legacy_convolve_attenuated_this_case(self):
        # Documents the defect being fixed: the legacy formula gives a LOWER
        # edge value than the edge-exact mean for a positive edge peak.
        w = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        legacy = np.convolve(w, np.ones(3) / 3.0, mode="same")
        sm = smoothed_wcc(w)
        assert sm[0] > legacy[0]

    def test_nan_does_not_poison_neighbours(self):
        w = np.array([0.2, np.nan, 0.2, 0.5, 0.5, 0.5, 0.2])
        sm = smoothed_wcc(w)
        # Positions adjacent to the NaN must stay finite...
        assert np.isfinite(sm[0])
        assert np.isfinite(sm[2])
        # ...and the NaN itself may be either NaN (empty window impossible
        # here) or a valid finite-excluded mean — with a 3-window centred on
        # index 1 there ARE finite neighbours, so it must be finite.
        assert np.isfinite(sm[1])

    def test_all_nan_window_yields_nan(self):
        # With symmetric truncation an edge window can still reach finite
        # samples; a NaN is only produced where the truncated support has NO
        # finite sample — e.g. a NaN run at least as long as the window in
        # the trace interior.
        w = np.array([0.5, np.nan, np.nan, np.nan, 0.5])
        sm = smoothed_wcc(w)
        assert np.isnan(sm[2])  # 3-window centred on index 2: all NaN

    def test_interior_bit_identical_to_legacy(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(-1, 1, 200)
        sm = smoothed_wcc(w)
        legacy = np.convolve(w, np.ones(3) / 3.0, mode="same")
        # Interior (both neighbours finite) positions must be identical.
        np.testing.assert_allclose(sm[1:-1], legacy[1:-1], rtol=0, atol=1e-15)

    def test_window_of_one_is_identity(self):
        w = np.array([0.1, 0.9, np.nan, 0.4])
        np.testing.assert_array_equal(smoothed_wcc(w, window=1), w)

    def test_even_window_truncates_left(self):
        # Document the alignment convention: for even windows the support is
        # [i - (w-1)//2, ... + w) = truncated-left, deterministic.
        w = np.array([1.0, 2.0, 3.0])
        sm = smoothed_wcc(w, window=2)
        assert sm[0] == pytest.approx(1.5)
        assert sm[1] == pytest.approx(2.5)
        assert sm[2] == pytest.approx(3.0)  # right-edge truncated window


class TestSmoothedWccWindowSec:
    def test_window_sec_scales_with_hz(self):
        w = np.ones(50)
        # 0.3 s at 10 Hz -> 3 samples; at 1 Hz -> 1 sample (identity)
        sm10 = smoothed_wcc(w, window_sec=0.3, hz=10.0)
        sm1 = smoothed_wcc(w, window_sec=0.3, hz=1.0)
        assert np.allclose(sm10, 1.0)
        assert np.allclose(sm1, 1.0)  # identity still all-ones
        # Distinguishable bandwidth: a single spike spreads only over the
        # physical duration, not the sample count.
        spike = np.zeros(50)
        spike[10] = 1.0
        s10 = smoothed_wcc(spike, window_sec=0.3, hz=10.0)  # w=3
        s1 = smoothed_wcc(spike, window_sec=0.3, hz=1.0)    # w=1
        assert np.count_nonzero(np.isclose(s10, 1.0 / 3)) >= 1
        assert np.allclose(s1, spike)  # w=1 leaves the spike untouched

    def test_window_sec_overrides_window(self):
        w = np.array([1.0, 2.0, 3.0])
        sm = smoothed_wcc(w, window=3, window_sec=1.0, hz=1.0)
        # window_sec=1 s at 1 Hz -> window=1 -> identity, ignoring window=3
        np.testing.assert_allclose(sm, w)


class TestPeakAmplitudeUnchangedContract:
    def test_peak_amplitude_on_clean_trace_matches_legacy(self):
        rng = np.random.default_rng(1)
        w = rng.uniform(-0.5, 0.8, 300)
        val_new, idx_new = compute_peak_amplitude(smoothed_wcc(w))
        legacy = np.convolve(w, np.ones(3) / 3.0, mode="same")
        val_old = float(np.max(legacy[1:-1]))
        assert val_new == pytest.approx(val_old, abs=1e-12)


# ---------------------------------------------------------------------------
# M2 — design-control mask symmetry: a short mask must fail loud, never
# silently disable seam gating for one arm only.
# ---------------------------------------------------------------------------

class TestDesignControlMaskFailLoud:
    """A mask shorter than its signal must raise somewhere — never silently
    disable seam gating for one arm only.

    Contract-level test (M2): the public API validates mask lengths up front
    (mask must equal the dyad's A-signal length); the internal ``_crop``
    branch additionally fails loud if ever reached with a short mask, so the
    old silent ``None`` path cannot be resurrected by relaxing the upstream
    validation.
    """

    @staticmethod
    def _pairs(n=200, seed=7):
        rng = np.random.default_rng(seed)
        return {
            "d1": (rng.normal(size=n), rng.normal(size=n)),
            "d2": (rng.normal(size=n), rng.normal(size=n)),
        }

    def test_short_mask_raises_never_silent(self):
        from syncpipe.design_controls import design_control_audit
        pairs = self._pairs()
        # d2's mask is SHORTER than its signal -> must raise (upfront
        # validator), not silently drop the mask for any arm.
        bad_masks = {"d1": np.ones(200, bool), "d2": np.ones(150, bool)}
        with pytest.raises(ValueError, match="discontinuity mask"):
            design_control_audit(
                pairs, hz=1.0, window_size=10,
                shift_lags_sec=(30.0,),
                discontinuity_masks=bad_masks,
            )

    def test_full_length_masks_still_work(self):
        from syncpipe.design_controls import design_control_audit
        pairs = self._pairs()
        masks = {k: np.ones(200, bool) for k in pairs}
        out = design_control_audit(
            pairs, hz=1.0, window_size=10,
            shift_lags_sec=(30.0,),
            discontinuity_masks=masks,
        )
        assert out["audit"] == "design_controls"
        assert out["discontinuity_masks_applied"] is True


# ---------------------------------------------------------------------------
# M3 — L0 existence audit covers synchrony_entropy; declared-not-audited
# labels stay in sync with what the audit actually tests.
# ---------------------------------------------------------------------------

class TestL0AuditEntropyCoverage:
    def test_entropy_keys_present_and_valid(self):
        from syncpipe.dynamic_features import _signal_level_surrogate_test
        rng = np.random.default_rng(11)
        n = 600
        # Coupled signals: y = 0.6 x + noise -> WCC well above IAAFT null.
        x = np.cumsum(rng.normal(size=n))
        y = 0.6 * x + rng.normal(size=n) * 0.5
        wcc = np.tanh(x * y * 0.02)  # crude bounded coupling proxy
        wcc = np.clip(wcc, -0.99, 0.99)
        res = _signal_level_surrogate_test(
            x, y, wcc, hz=4.0, surrogate_n=30, seed=42,
            wcc_window_size=10,
        )
        assert "p_synchrony_entropy" in res
        assert np.isfinite(res["p_synchrony_entropy"])
        assert res["null_synchrony_entropy"].size > 0
        assert res["n_valid_synchrony_entropy"] > 0
        assert "synchrony_entropy" in res["per_feature_significant"]
        assert "synchrony_entropy" in res["surrogate_precision"]

    def test_empty_result_has_entropy_shape(self):
        from syncpipe.dynamic_features import _empty_result
        empty = _empty_result("too short")
        assert empty["p_synchrony_entropy"] == 1.0
        assert empty["null_synchrony_entropy"].size == 0

    def test_declared_not_audited_labels_are_consistent(self):
        from syncpipe.feature_definitions import MATHEMATICAL_TIER
        from syncpipe.dynamic_features import _NULL_MODEL_L0
        audited = {"mean_synchrony", "peak_amplitude",
                   "bimodality_coefficient", "synchrony_entropy"}
        # (1) Everything the audit tests must be plain-L0 and in the null set.
        for name in audited:
            assert MATHEMATICAL_TIER.get(name) == "L0"
            assert name in _NULL_MODEL_L0
        # (2) The null-model set must be exactly the audited set (no member
        # that no test consumes, no silent gaps after the M3 fix).
        assert set(_NULL_MODEL_L0) == audited
        # (3) Declared-not-audited labels must exist exactly on the two
        # descriptive L0 features that no null is defined for.
        labelled = {
            n for n, t in MATHEMATICAL_TIER.items() if "not audited" in t
        }
        assert labelled == {"fraction_above_threshold", "peak_abs_amplitude"}
        # ...and those must NOT claim plain "L0" (which now implies audited).
        assert labelled.isdisjoint(audited)


# ---------------------------------------------------------------------------
# M5 — L1 gap_policy passthrough: observed features honour the caller's
# policy instead of forced NaN-compression (merge_valid semantics).
# ---------------------------------------------------------------------------

class TestL1GapPolicyPassthrough:
    @staticmethod
    def _gappy_wcc():
        # Deterministic layout where the two policies genuinely differ:
        # one long elevated run [10, 50) split by a 2-sample NaN seam at
        # [30, 32).  merge_valid bridges the seam (one run of 38); segment
        # splits it into two runs (20 and 18) with a lower mean.
        w = np.full(80, 0.2)
        w[10:50] = 0.8
        w[30:32] = np.nan
        return w

    def test_reference_policies_differ_on_this_layout(self):
        from syncpipe.feature_definitions import compute_dwell_time
        w = self._gappy_wcc()
        d_merge = compute_dwell_time(w, hz=1.0, threshold=0.5,
                                     gap_policy="merge_valid")
        d_seg = compute_dwell_time(w, hz=1.0, threshold=0.5,
                                   gap_policy="segment")
        assert d_merge == pytest.approx(38.0)   # bridged single run
        assert d_seg == pytest.approx(19.0)     # mean(20, 18)

    def test_l1_observed_tracks_caller_policy(self):
        from syncpipe.dynamic_features import _wcc_level_surrogate_test
        w = self._gappy_wcc()
        res_seg = _wcc_level_surrogate_test(
            w, hz=1.0, surrogate_n=5, seed=1, wcc_window_sec=30.0,
            gap_policy="segment",
        )
        res_merge = _wcc_level_surrogate_test(
            w, hz=1.0, surrogate_n=5, seed=1, wcc_window_sec=30.0,
            gap_policy=None,  # legacy default -> merge_valid
        )
        assert res_seg["obs_dwell_time"] == pytest.approx(19.0)
        assert res_merge["obs_dwell_time"] == pytest.approx(38.0)

    def test_default_unchanged_for_clean_traces(self):
        from syncpipe.dynamic_features import _wcc_level_surrogate_test
        rng = np.random.default_rng(2)
        w = rng.uniform(0.2, 0.9, 120)  # no NaN
        res = _wcc_level_surrogate_test(
            w, hz=1.0, surrogate_n=5, seed=1, wcc_window_sec=30.0,
        )
        assert res["applicable"] is True
        assert np.isfinite(res["obs_dwell_time"])


# ---------------------------------------------------------------------------
# M6 — multi-trial aggregation estimand governance.
# ---------------------------------------------------------------------------

class TestL2AggregationEstimand:
    @staticmethod
    def _dup_table(seed=9, n_dyads=12):
        import pandas as pd
        rng = np.random.default_rng(seed)
        rows = []
        for d in range(n_dyads):
            for cond in ("A", "B"):
                for trial in range(3):
                    rows.append({
                        "dyad": f"d{d}",
                        "condition": cond,
                        "peak_amplitude": rng.uniform(0.3, 0.9),
                        "dwell_time": rng.uniform(1.0, 5.0),
                    })
        return pd.DataFrame(rows)

    def test_mean_default_warns_on_extremum_duplicates(self):
        from syncpipe.validation.l2_between_condition import between_condition_fdr
        df = self._dup_table()
        with pytest.warns(UserWarning, match="estimand"):
            res = between_condition_fdr(
                df, condition_col="condition", dyad_col="dyad",
                feature_cols=["peak_amplitude"],
                condition_values=("A", "B"), n_permutations=200,
                observation_col=None, n_min_dyads=4,
            )
        assert res["max_feature_aggregation"] == "mean"

    def test_max_aggregation_takes_extremum(self):
        from syncpipe.validation.l2_between_condition import between_condition_fdr
        df = self._dup_table()
        # No warning under 'max' (estimand is the extremum itself).
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            res = between_condition_fdr(
                df, condition_col="condition", dyad_col="dyad",
                feature_cols=["peak_amplitude"],
                condition_values=("A", "B"), n_permutations=200,
                observation_col=None, n_min_dyads=4,
                max_feature_aggregation="max",
            )
        # observed_diff must be a median of per-dyad MAX values: recompute
        # reference directly and compare.
        ref = (
            df.groupby(["dyad", "condition"])["peak_amplitude"].max()
            .unstack("condition")
        )
        ref_diff = float(np.median(ref["A"].to_numpy() - ref["B"].to_numpy()))
        got = [r for r in res["per_feature"]
               if r.feature == "peak_amplitude"][0].observed_diff
        assert got == pytest.approx(ref_diff, rel=1e-12)


# ---------------------------------------------------------------------------
# M7 — design-control default-threshold governance warning.
# ---------------------------------------------------------------------------

class TestDesignControlThresholdWarning:
    def test_fixed_half_warns_when_structure_features_requested(self, caplog):
        from syncpipe.inference_pipeline import InferencePipeline
        rng = np.random.default_rng(4)
        feats = pd.DataFrame({
            "dyad_id": [f"d{i}" for i in range(12)],
            "condition": ["A"] * 12,
            "peak_amplitude": rng.uniform(0.2, 0.8, 12),
        })
        pipe = InferencePipeline(feats, hz=1.0)
        pairs = {
            f"d{i}": (rng.normal(size=150), rng.normal(size=150))
            for i in range(3)
        }
        import logging
        with caplog.at_level(logging.WARNING, logger="syncpipe.inference_pipeline"):
            pipe.run_design_control_audit(pairs, wcc_window_size=10)
        assert any("pooled surrogate threshold" in r.message for r in caplog.records)

    def test_explicit_threshold_no_warning(self, caplog):
        from syncpipe.inference_pipeline import InferencePipeline
        rng = np.random.default_rng(4)
        feats = pd.DataFrame({
            "dyad_id": [f"d{i}" for i in range(12)],
            "condition": ["A"] * 12,
            "peak_amplitude": rng.uniform(0.2, 0.8, 12),
        })
        pipe = InferencePipeline(feats, hz=1.0)
        pairs = {
            f"d{i}": (rng.normal(size=150), rng.normal(size=150))
            for i in range(3)
        }
        import logging
        with caplog.at_level(logging.WARNING, logger="syncpipe.inference_pipeline"):
            pipe.run_design_control_audit(
                pairs, wcc_window_size=10, threshold=0.62
            )
        assert not any("pooled surrogate threshold" in r.message
                       for r in caplog.records)


# ---------------------------------------------------------------------------
# M4 — terminology + narrative contracts.
# ---------------------------------------------------------------------------

class TestAuditMTerminology:
    def test_no_preregistration_claim_in_ssot(self):
        """The SSoT must not claim external pre-registration (no OSF/
        registry carrier exists for the frozen endpoint)."""
        import syncpipe.feature_definitions as fd
        import inspect
        src = inspect.getsource(fd)
        # The docstrings must use the honest wording...
        assert "Frozen a-priori PRIMARY endpoint" in src
        # ...and must no longer open with the pre-registration claim.
        assert '"""Pre-registered PRIMARY endpoint' not in src
        assert '"""Pre-registered primary modalities' not in src

    def test_cascade_narrative_is_cohort_level(self):
        from syncpipe.inference_pipeline import _build_cascade_summary
        # Zero-pass case must use audit phrasing, not dyad certainty.
        text = _build_cascade_summary(0, 10, 0, 10, {})
        assert "audit was positive" in text
        assert "dyads show above-chance synchrony" not in text
        assert "dyads show structured" not in text
        # High-pass case keeps the cohort framing and points at the gate.
        text2 = _build_cascade_summary(8, 10, 5, 10, {})
        assert "cohort-level" in text2
        assert "second-order group existence gate" in text2

"""E2/E3 modality-aware consumption (review R4c, 2026-10-15).

``InferencePipeline.run_design_control_audit`` now partitions signal
pairs by modality (pseudo-partners are same-modality only), and
``evidence.builder._design_p`` combines per-modality p values with
any-of-k (min(1, k * min p)) when the pooled summary is suppressed.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.evidence.builder import _design_p


class TestDesignPAnyOfK:
    def test_pooled_summary_used_when_present(self):
        design = {
            "feature_summary": {"peak_amplitude": {"p_real_gt_pseudo": 0.03}},
        }
        assert _design_p(design, "peak_amplitude", "p_real_gt_pseudo") == pytest.approx(0.03)

    def test_any_of_k_over_per_group_when_pooled_suppressed(self):
        """Two modalities, min p = 0.02 -> any-of-k = 2 * 0.02 = 0.04."""
        design = {
            "feature_summary": {},  # suppressed (>1 group)
            "per_group": {
                "ECG": {"peak_amplitude": {"p_real_gt_pseudo": 0.02}},
                "EDA": {"peak_amplitude": {"p_real_gt_pseudo": 0.30}},
            },
        }
        assert _design_p(design, "peak_amplitude", "p_real_gt_pseudo") == pytest.approx(0.04)

    def test_any_of_k_caps_at_one(self):
        design = {
            "feature_summary": {},
            "per_group": {
                "ECG": {"peak_amplitude": {"p_real_gt_pseudo": 0.90}},
                "EDA": {"peak_amplitude": {"p_real_gt_pseudo": 0.80}},
            },
        }
        assert _design_p(design, "peak_amplitude", "p_real_gt_pseudo") == 1.0

    def test_all_nan_yields_nan(self):
        design = {"feature_summary": {}, "per_group": {"ECG": {"peak_amplitude": {}}}}
        assert np.isnan(_design_p(design, "peak_amplitude", "p_real_gt_pseudo"))

    def test_single_empty_group_partition_returns_nan(self):
        """Unpartitioned audits use feature_summary; a per_group keyed only
        by '' (single implicit group, pooled suppressed) has no usable p."""
        design = {"feature_summary": {}, "per_group": {"": {"peak_amplitude": {}}}}
        assert np.isnan(_design_p(design, "peak_amplitude", "p_real_gt_pseudo"))

"""Gate A direction fixes (review R4, 2026-10-14; ported from the Claude
P0 changeset after dialectical review).

Change 1 — the existence gate is ONE-SIDED (upper). The question it asks
is "does POSITIVE co-fluctuation exceed the independent-signal null?",
which is an upper-tail question. The legacy two-sided policy let a peak
significantly BELOW the null pass the gate (anti-phase coupled pairs:
obs 0.681 vs null 0.807, old gate PASS — the defect that motivated this
change). The lower tail is reported descriptively in ``direction``; the
mirror endpoint on -WCC in ``antiphase`` (exploratory); neither decides.

Gate key ``min_attainable_two_sided_p`` is now ``min_attainable_p``
(1/(n+1), one-sided).
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.inference_pipeline import _existence_gate_by_modality


def _result(obs, null, direction=None):
    base = {
        "observed": {"peak_amplitude": obs},
        "null_peak_amplitude": np.asarray(null, dtype=float),
        "per_feature_significant": {"peak_amplitude": False},
    }
    if direction is not None:
        base["direction_resolved"] = {"peak_amplitude": direction}
    return base


class TestGateIsOneSided:
    def test_gate_does_not_pass_when_peak_is_significantly_below_null(self):
        """The motivating defect: anti-phase coupled pairs used to PASS."""
        # All dyads far BELOW their null -> lower tail tiny, upper tail ~1.
        results = {
            f"d{i}__ECG": _result(0.60 + 0.01 * i, np.linspace(0.78, 0.82, 40))
            for i in range(12)
        }
        gate = _existence_gate_by_modality(results, primary_modalities=("ECG",))
        mod = gate["per_modality"]["ECG"]
        assert mod["p_group"] > 0.99, "upper-tail p must be ~1 when obs << null"
        assert mod["p_group"] == pytest.approx(1.0, abs=0.01)
        assert gate["primary_pass"] is False, (
            "a peak significantly BELOW the null must not pass the existence gate"
        )
        # Lower tail is reported, not deciding.
        assert mod["direction"]["p_group_lower"] < 0.05  # floor 1/41 ≈ 0.024

    def test_gate_passes_when_peak_above_null(self):
        results = {
            f"d{i}__ECG": _result(0.85, np.linspace(0.30, 0.50, 40))
            for i in range(12)
        }
        gate = _existence_gate_by_modality(results, primary_modalities=("ECG",))
        mod = gate["per_modality"]["ECG"]
        assert mod["tail"] == "upper"
        assert mod["p_group"] == pytest.approx(1 / 41)
        assert gate["primary_pass"] is True

    def test_min_attainable_p_is_one_sided(self):
        """Floor is 1/(n+1), not 2/(n+1)."""
        results = {
            f"d{i}__ECG": _result(0.5, np.linspace(0.3, 0.6, 39))
            for i in range(6)
        }
        gate = _existence_gate_by_modality(results, primary_modalities=("ECG",))
        assert gate["per_modality"]["ECG"]["min_attainable_p"] == pytest.approx(1 / 40)

    def test_direction_and_antiphase_reported_descriptively(self):
        """direction split and the anti-phase mirror ride along, exploratory."""
        results = {
            "d0__ECG": _result(
                0.85, np.linspace(0.30, 0.50, 40),
                direction={"p_upper": 0.02, "p_lower": 0.9,
                           "significant_upper": True, "significant_lower": False},
            ),
        }
        gate = _existence_gate_by_modality(results, primary_modalities=("ECG",))
        mod = gate["per_modality"]["ECG"]
        assert mod["direction"]["n_dyads_sig_upper"] == 1  # dyad p_upper=0.02 < alpha
        assert mod["direction"]["reading"].startswith("descriptive")
        assert mod["antiphase"]["tier"] == "exploratory"

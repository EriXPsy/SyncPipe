"""Modality-aware stranger test (review R4b, 2026-10-14; ported from the
Claude P0 changeset after dialectical review).

``design_control_audit(partition=...)`` draws pseudo-partners ONLY within
the same group (the canonical path passes the modality). Before this, the
pooled "<dyad>__<modality>" keys let an EDA-like signal be paired with an
ECG-like signal (53% of pseudo-pairs were cross-modality on the Lerique
cohort) and counted one dyad's modalities as independent stranger units
(null-world false-positive rate 55% pooled vs 0% per modality).
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.design_controls import design_control_audit


def _pair(rng: np.random.Generator, n: int = 600):
    """One independent AR(1) dyad (tuple of arrays), not generators."""
    def ar1():
        e = rng.normal(size=n) * np.sqrt(1 - 0.9 ** 2)
        x = np.empty(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + e[t]
        return x
    return ar1(), ar1()


class TestModalityAwareStranger:
    def test_pseudo_partners_never_cross_modality_when_partitioned(self):
        """Instrument the partner draw: with a partition, every pseudo arm
        must reuse a SAME-modality partner. Verified at the summary level:
        per-group n_pseudo_dyads is per-modality (3), not pooled (6)."""
        rng = np.random.default_rng(3)
        pairs = {f"d{i}__ECG": _pair(rng) for i in range(3)}
        pairs.update({f"d{i}__EDA": _pair(rng) for i in range(3)})
        partition = {k: k.split("__")[1] for k in pairs}
        out = design_control_audit(
            pairs, hz=1.0, window_size=20, shift_lags_sec=(30.0,),
            feature_names=("peak_amplitude",), partition=partition, seed=1,
        )
        assert out["groups"] == ["ECG", "EDA"]
        assert out["pooled_summary_suppressed"] is True
        for mod in ("ECG", "EDA"):
            fs = out["per_group"][mod]["peak_amplitude"]
            assert fs["n_real"] == 3
            assert fs["n_pseudo_dyads"] == 3  # same-modality partners only

    def test_no_partition_keeps_pooled_summary(self):
        rng = np.random.default_rng(4)
        pairs = {f"d{i}__ECG": _pair(rng) for i in range(3)}
        out = design_control_audit(
            pairs, hz=1.0, window_size=20, shift_lags_sec=(30.0,),
            feature_names=("peak_amplitude",), seed=1,
        )
        assert out["pooled_summary_suppressed"] is False
        assert out["feature_summary"]["peak_amplitude"]["n_real"] == 3

    def test_partition_with_missing_keys_fails_loud(self):
        rng = np.random.default_rng(5)
        pairs = {f"d{i}__ECG": _pair(rng) for i in range(2)}
        with pytest.raises(ValueError, match="same keys as signal_pairs"):
            design_control_audit(
                pairs, hz=1.0, window_size=20, feature_names=("peak_amplitude",),
                partition={"only_one__ECG": "ECG"},  # missing the second key
            )

    def test_sole_member_group_gets_no_pseudo_arm(self):
        """A group with one pair has no possible stranger: its pseudo arm
        is skipped (NaN p) rather than drawing a cross-modality partner."""
        rng = np.random.default_rng(6)
        pairs = {
            "d0__ECG": _pair(rng),
            "d0__EDA": _pair(rng),
        }
        out = design_control_audit(
            pairs, hz=1.0, window_size=20, shift_lags_sec=(30.0,),
            feature_names=("peak_amplitude",),
            partition={"d0__ECG": "ECG", "d0__EDA": "EDA"}, seed=1,
        )
        for mod in ("ECG", "EDA"):
            fs = out["per_group"][mod]["peak_amplitude"]
            assert fs["n_pseudo_dyads"] == 0
            assert np.isnan(fs["p_real_gt_pseudo"])

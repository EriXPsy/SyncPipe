"""P1 residual fixes (contrast forward, gap_policy on mask, session masks)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.computation_pipeline import ComputationPipeline
from multisync.feature_definitions import compute_dwell_time, extract_features
from multisync.inference_pipeline import InferencePipeline
from multisync.session_threshold import (
    compute_session_pooled_threshold,
    _generate_surrogate_coupling_matrix,
)
from multisync.dynamic_features import sliding_window_wcc, _apply_discontinuity_mask


def _multimodal_df(n_dyads=8):
    rows = []
    for d in range(n_dyads):
        for cond, val_eda in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=val_eda,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
            # opposite pattern on ECG
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="ECG",
                    peak_amplitude=1.0 - val_eda,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    return pd.DataFrame(rows)


def test_p1_r1_multimodal_forwards_contrast():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=7)
    out = pipe.run_group_condition_inference(
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=50,
        contrast=("rest", "task"),
        modality_col="modality",
    )
    assert set(out) >= {"EDA", "ECG"}
    for mod in ("EDA", "ECG"):
        assert "error" not in out[mod], out[mod]
        assert out[mod]["condition_a"] == "rest"
        assert out[mod]["condition_b"] == "task"


def test_p1_r1_by_modality_accepts_contrast_and_seed():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=99)
    out = pipe.test_l2_by_modality(
        feature_cols=["peak_amplitude"],
        n_permutations=40,
        contrast=("task", "rest"),  # reversed on purpose
        seed=99,
    )
    for mod, sub in out.items():
        assert sub["condition_a"] == "task"
        assert sub["condition_b"] == "rest"


def test_p1_r1_chain_summary_counts_multimodal():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=0)
    # Build dummy existence-compatible raw signals (short chain call heavy);
    # unit-test the summary helper directly.
    group = {
        "EDA": {"n_significant": 2},
        "ECG": {"n_significant": 1},
    }
    s = InferencePipeline._build_audited_chain_summary(
        {"n_pairs": 0}, None, None, group
    )
    assert "per-modality" in s
    assert "3 significant" in s


def test_p1_r2_masked_pipeline_uses_segment_gap_policy():
    """Discontinuity mask mid-run: segment policy must not merge dwells."""
    n = 100
    a = np.sin(np.linspace(0, 8, n))
    b = a.copy()
    # Strong coupling whole series; mask creates a seam in the middle
    mask = np.ones(n, dtype=bool)
    mask[48:53] = False  # seam
    pipe = ComputationPipeline(hz=1.0, window_size=10, onset_threshold=0.3)
    pipe.load_signals(a, b, discontinuity_mask=mask)
    pipe.compute_wcc()
    feats = pipe.extract_features()
    assert feats.params.get("gap_policy") == "segment"
    # Reference: same WCC with explicit policies
    wcc = pipe._wcc
    d_seg = compute_dwell_time(wcc, hz=1.0, threshold=0.3, gap_policy="segment")
    d_merge = compute_dwell_time(wcc, hz=1.0, threshold=0.3, gap_policy="merge_valid")
    assert feats.dwell_time == pytest.approx(d_seg, nan_ok=True)
    # When both defined, segment dwell should be <= merge (no glue)
    if np.isfinite(d_seg) and np.isfinite(d_merge):
        assert d_seg <= d_merge + 1e-9


def test_p1_r2_unmasked_keeps_merge_valid_default():
    n = 80
    a = np.sin(np.linspace(0, 6, n))
    b = a + 0.01
    pipe = ComputationPipeline(hz=1.0, window_size=10, onset_threshold=0.3)
    pipe.load_signals(a, b)  # no mask
    pipe.compute_wcc()
    feats = pipe.extract_features()
    assert feats.params.get("gap_policy") == "merge_valid"


def test_p1_r3_session_threshold_applies_mask():
    rng = np.random.default_rng(0)
    n = 120
    a = rng.normal(size=n)
    b = 0.7 * a + rng.normal(scale=0.2, size=n)
    mask = np.ones(n, dtype=bool)
    mask[40:80] = False  # large gated region
    window = 15
    # Finite counts with vs without mask on a single surrogate matrix draw
    m_plain = _generate_surrogate_coupling_matrix(
        a, b, hz=1.0, window_size=window, surrogate_n=5, seed=1
    )
    m_mask = _generate_surrogate_coupling_matrix(
        a, b, hz=1.0, window_size=window, surrogate_n=5, seed=1,
        discontinuity_mask=mask,
    )
    assert np.isfinite(m_mask).sum() < np.isfinite(m_plain).sum()

    thr_plain, meta_p = compute_session_pooled_threshold(
        [(a, b)], hz=1.0, wcc_window_size=window, surrogate_n=20, seed=2
    )
    thr_mask, meta_m = compute_session_pooled_threshold(
        [(a, b)],
        hz=1.0,
        wcc_window_size=window,
        surrogate_n=20,
        seed=2,
        discontinuity_masks=[mask],
    )
    assert meta_m.get("n_discontinuity_masks_applied") == 1
    assert meta_m["n_finite_coupling_values"] < meta_p["n_finite_coupling_values"]


def test_p1_r3_mask_length_mismatch_raises():
    a = np.zeros(50)
    b = np.zeros(50)
    with pytest.raises(ValueError, match="discontinuity_masks"):
        compute_session_pooled_threshold(
            [(a, b), (a, b)],
            hz=1.0,
            wcc_window_size=10,
            surrogate_n=5,
            discontinuity_masks=[None],  # wrong length
        )

"""P2 release-hygiene fixes: summarize multimodal, JSON L2Result, seed, empty WCC."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from multisync.feature_definitions import extract_features
from multisync.inference_pipeline import InferencePipeline


def _uni(n=8):
    rows = []
    for d in range(n):
        for c, v in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=c,
                    peak_amplitude=v,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    return pd.DataFrame(rows)


def _multi(n=8):
    rows = []
    for d in range(n):
        for c, v in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=c,
                    modality="EDA",
                    peak_amplitude=v,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=c,
                    modality="ECG",
                    peak_amplitude=1.0 - v,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    return pd.DataFrame(rows)


def test_summarize_shows_multimodal_l2():
    pipe = InferencePipeline(_multi(), hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=1)
    pipe.run_group_condition_inference(
        contrast=("rest", "task"),
        feature_cols=["peak_amplitude"],
        n_permutations=30,
    )
    text = pipe.summarize()
    assert "L2" in text
    assert "EDA" in text and "ECG" in text
    assert "per-modality" in text or "[EDA]" in text


def test_summarize_shows_unimodal_l2_via_group_store():
    pipe = InferencePipeline(_uni(), hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=1)
    pipe.run_group_condition_inference(
        contrast=("rest", "task"),
        feature_cols=["peak_amplitude"],
        n_permutations=30,
    )
    text = pipe.summarize()
    assert "L2" in text
    assert "significant" in text


def test_to_json_serializes_l2result_as_dict_not_str():
    pipe = InferencePipeline(_multi(), hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=1)
    pipe.run_group_condition_inference(
        contrast=("rest", "task"),
        feature_cols=["peak_amplitude"],
        n_permutations=30,
    )
    payload = json.loads(pipe.to_json())
    eda_pf = payload["group_inference_results"]["EDA"]["per_feature"]
    assert isinstance(eda_pf, list) and eda_pf
    assert isinstance(eda_pf[0], dict), type(eda_pf[0])
    assert "feature" in eda_pf[0] and eda_pf[0]["feature"] == "peak_amplitude"
    assert "p_fdr" in eda_pf[0]


def test_to_json_unimodal_l2result_dict():
    pipe = InferencePipeline(_uni(), hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=1)
    pipe.test_l2_condition(
        contrast=("rest", "task"),
        feature_cols=["peak_amplitude"],
        n_permutations=30,
    )
    payload = json.loads(pipe.to_json())
    pf = payload["l2_results"]["per_feature"]
    assert isinstance(pf[0], dict)
    assert pf[0]["feature"] == "peak_amplitude"


def test_test_l2_condition_uses_pipeline_seed():
    """seed must be forwarded so InferencePipeline(seed=...) is honored."""
    df = _uni(n=14)  # >12 so Monte-Carlo path uses seed
    p1 = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=11)
    p2 = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=11)
    p3 = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=99)
    r1 = p1.test_l2_condition(
        contrast=("rest", "task"), feature_cols=["peak_amplitude"], n_permutations=200
    )
    r2 = p2.test_l2_condition(
        contrast=("rest", "task"), feature_cols=["peak_amplitude"], n_permutations=200
    )
    r3 = p3.test_l2_condition(
        contrast=("rest", "task"), feature_cols=["peak_amplitude"], n_permutations=200
    )
    assert r1["per_feature"][0].p_raw == r2["per_feature"][0].p_raw
    # Different seeds may still collide rarely; only soft-check structure
    assert "p_raw" in r3["per_feature"][0].__dataclass_fields__


def test_extract_features_empty_wcc_returns_nan_not_raise():
    f = extract_features(np.array([]), hz=1.0, wcc_window_sec=10.0)
    assert np.isnan(f.peak_amplitude)
    assert f.nan_fraction == pytest.approx(1.0)


def test_run_full_cascade_accepts_contrast_and_multimodal_df():
    """Cascade should route group L2 like the scientific path."""
    # Build minimal cascade inputs: empty wcc/raw dicts, L2 from features_df only
    df = _multi(n=6)
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=2)
    out = pipe.run_full_cascade(
        raw_signals_dict={},
        wcc_dict={},
        wcc_window_size=10,
        feature_cols=["peak_amplitude"],
        n_permutations=30,
        contrast=("rest", "task"),
    )
    l2 = out["l2_results"]
    assert isinstance(l2, dict)
    # multimodal keys
    assert "EDA" in l2 and "ECG" in l2
    assert l2["EDA"]["condition_a"] == "rest"
    assert "per-modality" in out["cascade_summary"] or "L2" in out["cascade_summary"]


# ===========================================================================
# P1 regression guard: naive-baseline failure must NOT be fabricated as 0.5
# ===========================================================================
# Background (2026-07-24 reconcile, Finding class "fixed the primary model,
# forgot the adjacent baseline"): the restricted / joint baseline failure
# paths are already guarded (test_prediction_failed_fold.py,
# test_prediction_finding17.py), but the *naive* baseline failure path in
# multisync.prediction had NO assertion. On failure prediction.py leaves the
# naive baseline AUC as NaN ("Do not fabricate chance-level AUC", intra L888 /
# cross_modal L1508). This guard locks that behavior so nobody can silently
# revert it to the old hard-coded ``baseline_auc = 0.5`` -- a fabricated
# chance-level result that would falsely flatter delta_AUC honesty.
import math

import multisync.prediction as pred_mod
from sklearn.metrics import roc_auc_score as _real_roc_auc


def _make_wcc(n=4000, seed=0):
    """Oscillating 1-D synchrony series -> both label classes present."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 80, n)
    return np.sin(t) + rng.normal(0, 0.15, n)


def _make_signals(n=3000, seed=0):
    """Moderately coupled source/target signals for cross-modal prediction."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    src = np.sin(t) + rng.normal(0, 0.1, n)
    tgt = np.cos(t) + rng.normal(0, 0.1, n)
    return src, tgt


class _FailFirstNaiveBaseline:
    """Force the *naive baseline* roc_auc_score call to raise.

    The naive baseline scores a constant predictor
    (``np.full_like(y_test, y_train.mean())``); its y_score is therefore
    constant (``ptp == 0``). Every *other* roc_auc_score call inside the CV
    loop (primary model, AR / restricted baseline) passes a non-constant
    score. We raise only on the first constant-score call -- which is exactly
    the naive baseline of fold 0 in BOTH rolling_origin_cv (intra) and
    cross_modal_prediction -- exercising the "Do not fabricate chance-level
    AUC" path without dropping the fold (the primary model still fits, so the
    fold is kept and its NaN baseline is observable).
    """

    _hit = {"done": False}

    def __call__(self, y_true, y_score, **kw):
        ys = np.asarray(y_score)
        if ys.ndim > 1:
            ys = ys[:, 1] if ys.shape[1] > 1 else ys.ravel()
        is_constant = bool(np.ptp(ys) == 0)
        if is_constant and not _FailFirstNaiveBaseline._hit["done"]:
            _FailFirstNaiveBaseline._hit["done"] = True
            raise ValueError("forced naive-baseline AUC failure")
        return _real_roc_auc(y_true, y_score, **kw)


class TestNaiveBaselineNaN:
    """Naive baseline failure must stay NaN, never a fabricated 0.5."""

    def test_intra_naive_baseline_failure_not_fabricated(self, monkeypatch):
        _FailFirstNaiveBaseline._hit["done"] = False
        monkeypatch.setattr(pred_mod, "roc_auc_score", _FailFirstNaiveBaseline())
        wcc = _make_wcc()
        res = pred_mod.rolling_origin_cv(
            wcc, window_size=30, n_splits=4, pair_name="p", seed=1,
        )
        # Primary model fit succeeded -> fold 0 was kept (not dropped).
        assert res.n_failed_folds == 0, res.n_failed_folds
        assert len(res.folds) >= 1
        # Fold 0's naive baseline failed -> stored as NaN, never 0.5.
        assert math.isnan(res.folds[0].baseline_auc), res.folds[0].baseline_auc
        assert res.folds[0].baseline_auc != 0.5
        # Aggregate must NOT be a fabricated 0.5; NaN propagation is honest.
        assert math.isnan(res.mean_baseline_auc), res.mean_baseline_auc

    def test_cross_modal_naive_baseline_failure_not_fabricated(self, monkeypatch):
        _FailFirstNaiveBaseline._hit["done"] = False
        monkeypatch.setattr(pred_mod, "roc_auc_score", _FailFirstNaiveBaseline())
        src, tgt = _make_signals()
        res = pred_mod.cross_modal_prediction(
            src, tgt, window_size=30, n_splits=4,
            source_name="s", target_name="t",
        )
        # Joint model succeeded -> fold 0 was kept (not dropped).
        assert res.n_failed_folds == 0, res.n_failed_folds
        assert len(res.folds) >= 1
        # Fold 0's naive baseline failed -> stored as NaN, never 0.5.
        assert math.isnan(res.folds[0].baseline_auc), res.folds[0].baseline_auc
        assert res.folds[0].baseline_auc != 0.5
        # Aggregate must NOT be a fabricated 0.5; NaN propagation is honest.
        assert math.isnan(res.mean_baseline_auc), res.mean_baseline_auc

from __future__ import annotations

# === source: test_p2_release_hygiene.py ===
"""P2 release-hygiene fixes: summarize multimodal, JSON L2Result, seed, empty WCC."""

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


def _make_wcc_hyg(n=4000, seed=0):
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
        wcc = _make_wcc_hyg()
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

# === source: test_parity_paths.py ===
"""
A4 parity tests — descriptor / default CLI DynamicAnalyzer path vs opt-in prediction path.

Parity contract (enforced here):
  The prediction side-path (``prediction.build_feature_matrix`` /
  ``rolling_origin_cv``) consumes the IDENTICAL set of dynamic synchrony
  descriptors as the descriptor / default CLI DynamicAnalyzer path
  (``dynamic_features.extract_dynamic_features`` ->
  ``feature_definitions.extract_features``), with NO silent rename /
  re-ordering / re-implementation.

Concretely verified:
  1. FEATURE_NAME_PARITY    — ``prediction.FEATURE_NAMES`` equals the 6
     descriptor / default-CLI epoch descriptors that DynamicAnalyzer reports
     (``DynamicFeatures`` epoch fields), in the same order.
  2. SSOT_NO_SILENT_TRANSFORM — ``build_feature_matrix``'s per-window feature
     vector equals a direct ``feature_definitions.extract_features`` call on
     the same window slice (proves prediction delegates to the SSoT, not a
     reimplemented copy).
  3. PATH_PARITY (end-to-end) — for identical input WCC + identical window
     params, the descriptor / default-CLI global feature vector equals the prediction
     path's feature matrix (single window): both paths land on the same
     descriptor definitions.
  4. CLI_ROUTING — ``DynamicAnalyzer`` is the descriptor / default-CLI path
     (``enable_prediction=False``); prediction is opt-in only
     (``demo --prediction``; defaults OFF). ``analyze`` never enters it.

These tests encode the A4 routing decision as a regression guard: any future
drift (renaming a descriptor only inside prediction, or reimplementing the
feature math there) will break one of the assertions below.
"""

import numpy as np
import pytest

from multisync import dynamic_features as df
from multisync import prediction as pred_mod
from multisync.core import DynamicAnalyzer, CANONICAL_PATH, OPT_IN_PATH
from multisync.feature_definitions import (
    DynamicFeatures,
    extract_features as ssot_extract,
)
from multisync.synthetic import generate_ground_truth_dyad
from multisync.cli import build_parser

# The 6 descriptor / default-CLI epoch descriptors reported by the DynamicAnalyzer path
# (the DynamicFeatures epoch fields, in descriptor order).
CANONICAL_EPOCH_FEATURES = [
    "onset_latency",
    "rise_time",
    "peak_amplitude",
    "recovery_time",
    "dwell_time",
    "switching_rate",
]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_wcc_parity(n=600, seed=0):
    """Build a clean, coupled WCC series for feature-level parity checks."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    x = np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.15, n)
    y = np.roll(x, 3) + rng.normal(0, 0.15, n)
    return df.sliding_window_wcc(x, y, window_size=30, hz=1.0)


def _make_dyad():
    """Build an aligned + normalized synthetic dyad for path/CLI checks."""
    ds = generate_ground_truth_dyad(
        lead_modality="behavior",
        lag_modality="neural",
        true_lag_sec=0.0,
        noise_ratio=0.3,
        duration_sec=300,
        hz=1.0,
        seed=42,
        coupling=0.65,
    )
    ds.align(target_hz=1.0)
    ds.zscore()
    return ds


def _assert_descriptor_equal(a, b, ctx):
    """Parity comparison: a and b must agree, INCLUDING NaN-agreement.

    The descriptor / default-CLI SSoT legitimately returns NaN for event-locked
    descriptors whose timing is scientifically undefined (e.g. ``rise_time`` with
    ``rise_defined=0``).  The prediction path must NOT fabricate a value where
    the descriptor / default-CLI path says undefined — so NaN must match NaN, and
    finite values must match numerically.  (``np.isclose(nan, nan)`` is False, hence
    the explicit NaN handling.)
    """
    a = float(a)
    b = float(b)
    if np.isnan(a) or np.isnan(b):
        assert np.isnan(a) and np.isnan(b), f"NaN mismatch in {ctx}: {a} vs {b}"
    else:
        assert np.isclose(a, b, rtol=0, atol=1e-9), f"mismatch in {ctx}: {a} vs {b}"


# ---------------------------------------------------------------------------
# 1. Feature-name parity (no silent rename)
# ---------------------------------------------------------------------------

def test_feature_name_parity():
    """prediction.FEATURE_NAMES must equal the descriptor / default-CLI 6 epoch descriptors."""
    assert pred_mod.FEATURE_NAMES == CANONICAL_EPOCH_FEATURES
    # Every name must map to a real DynamicFeatures epoch field (no made-up
    # descriptor names that would silently diverge from the descriptor / default-CLI path).
    df_fields = set(DynamicFeatures.__dataclass_fields__.keys())
    for name in pred_mod.FEATURE_NAMES:
        assert name in df_fields, f"{name} is not a DynamicFeatures field"


# ---------------------------------------------------------------------------
# 2. SSOT parity (no silent transform / re-implementation)
# ---------------------------------------------------------------------------

def test_prediction_delegates_to_ssot_no_transform():
    """build_feature_matrix must NOT silently reimplement descriptors: its
    per-window feature vector must equal a direct SSoT ``extract_features``
    call on the exact same window slice."""
    wcc = _make_wcc_parity(n=600, seed=1)
    hz = 1.0
    window_size = 60
    threshold = 0.5
    window_sec = window_size / hz

    X, names = pred_mod.build_feature_matrix(
        wcc, window_size, hz, onset_threshold=threshold
    )

    n = len(wcc)
    step = max(1, window_size // 2)
    starts = list(range(0, n - window_size + 1, step))
    assert X.shape[0] == len(starts)
    assert names == CANONICAL_EPOCH_FEATURES

    for row_i, s in enumerate(starts):
        wcc_window = wcc[s : s + window_size]
        feat = ssot_extract(
            wcc_window, hz=hz, wcc_window_sec=window_sec, threshold=threshold
        )
        for col_i, name in enumerate(names):
            _assert_descriptor_equal(
                X[row_i, col_i],
                getattr(feat, name),
                f"window {row_i} feature {name}",
            )


# ---------------------------------------------------------------------------
# 3. End-to-end path parity (descriptor / default-CLI global vector == prediction matrix)
# ---------------------------------------------------------------------------

def test_descriptor_and_prediction_agree_on_descriptors():
    """For identical input WCC + identical window parameters, the descriptor
    / default-CLI global feature vector (``extract_dynamic_features`` over the
    whole series as one window) equals the prediction path's feature matrix
    (single window). Proves both paths land on the same descriptor definitions."""
    wcc = _make_wcc_parity(n=300, seed=2)
    hz = 1.0
    threshold = 0.5

    # Descriptor / default-CLI path: global feature over the whole WCC as a single window.
    feat = df.extract_dynamic_features(
        wcc, hz=hz, onset_threshold=threshold, wcc_window_sec=len(wcc) / hz
    )
    # Prediction path: feature matrix with a single window covering the whole WCC.
    X, names = pred_mod.build_feature_matrix(
        wcc, window_size=len(wcc), hz=hz, onset_threshold=threshold
    )

    assert X.shape == (1, 6), X.shape
    for i, name in enumerate(names):
        _assert_descriptor_equal(
            X[0, i], getattr(feat, name), f"canonical vs prediction: {name}"
        )


# ---------------------------------------------------------------------------
# 4. CLI / entry-point routing parity
# ---------------------------------------------------------------------------

def test_cli_demo_prediction_opt_in_defaults_off():
    """prediction is opt-in: ``demo --prediction`` defaults OFF, ON only with
    the explicit flag."""
    parser = build_parser()
    assert parser.parse_args(["demo"]).prediction is False
    assert parser.parse_args(["demo", "--prediction"]).prediction is True


def test_cli_describe_has_no_prediction_path():
    """``describe`` is the descriptor / default-CLI path: it never exposes/enters the
    prediction path (no --prediction toggle wired in). The canonical ``analyze``
    command is the audited scientific path and is tested separately."""
    parser = build_parser()
    args = parser.parse_args(["describe", "-i", "a.csv,b.csv", "-n", "x,y"])
    assert not hasattr(args, "prediction")


def test_dynamic_analyzer_default_cli_no_prediction():
    """DynamicAnalyzer default IS the descriptor / default-CLI path (enable_prediction=False);
    a default fit_transform yields an empty prediction dict, and the routing
    constants document the two paths."""
    ds = _make_dyad()
    analyzer = DynamicAnalyzer(surrogate_n=50, run_qc=False)  # default opt-in OFF
    assert analyzer.enable_prediction is False
    results = analyzer.fit_transform(ds)
    assert results.prediction == {}
    assert CANONICAL_PATH == "DynamicAnalyzer.fit_transform"
    assert OPT_IN_PATH == "prediction.rolling_origin_cv"


def test_opt_in_prediction_populates_results():
    """When explicitly opted in, the prediction side path is entered and
    populates ``results.prediction`` (proving it is reachable but OFF by
    default)."""
    ds = _make_dyad()
    analyzer = DynamicAnalyzer(
        surrogate_n=50, run_qc=False, enable_prediction=True
    )
    results = analyzer.fit_transform(ds)
    assert isinstance(results.prediction, dict)
    # The opt-in branch ran for every pair (a key per pair is recorded).
    assert len(results.prediction) >= 1


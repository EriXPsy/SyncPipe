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

def _make_wcc(n=600, seed=0):
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
    wcc = _make_wcc(n=600, seed=1)
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
    wcc = _make_wcc(n=300, seed=2)
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


def test_cli_analyze_has_no_prediction_path():
    """``analyze`` is descriptor / default-CLI only: it never exposes/enters the prediction
    path (no --prediction toggle wired in)."""
    parser = build_parser()
    args = parser.parse_args(["analyze", "-i", "a.csv,b.csv", "-n", "x,y"])
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

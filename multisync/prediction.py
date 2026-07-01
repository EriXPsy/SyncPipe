"""
Cross-modal prediction: rolling-origin time-series CV with temporal-leakage safeguards.

Uses joint model [source features + target features] vs restricted model [target only]
to measure *incremental* predictive value of synchrony dynamics beyond autocorrelation.

Key design choices:
- TimeSeriesSplit with physical-time-aware gap (prevents sliding-window leakage)
- AR baseline (mean_synchrony per window) as methodological floor
- LEAKAGE_DELTA_AUC_THRESHOLD = 0.30 flags potential data leakage
- Ablation: joint model without mean_synchrony verifies shape features have
  independent predictive power

References
----------
Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and
  Practice. OTexts. https://otexts.com/fpp3/
Koul, A., Grossman, S., & Feldman, R. (2023). Parent-infant synchrony:
  A bio-behavioral model. Current Opinion in Psychology, 52, 101637.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import logging
import warnings

from .feature_definitions import ONSET_THRESHOLD

LEAKAGE_DELTA_AUC_THRESHOLD: float = 0.30
"""Threshold for ``mean_delta_auc`` above which prediction pipeline flags
``warning="leakage_suspected"``.  Calibrated via sine wave (perfectly
autocorrelated, delta ≈ 0.37) vs random noise (delta ≈ 0).  Provides ~0.07
margin on both sides.  Re-tuning requires Reversal Protocol (DECISION_LOG.md)."""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature names — DECISION-10 (METHODOLOGY_LOCK_IN.md)
# ---------------------------------------------------------------------------
# The joint prediction model uses the **7 FDR-family features**
# (Core + Conditional tiers; DECISION-09, family size = 7).
# synchrony_entropy is in the FDR family but excluded from the
# prediction model due to multicollinearity (ρ ≈ -0.94 with
# mean_synchrony; DECISION-13).
#
# ``mean_synchrony`` is **not** in the FDR family — it is extracted on its
# own dedicated channel via :func:`_extract_mean_synchrony_per_window`
# and used exclusively as the AR baseline (restricted model)
# predictor, so that the joint model's ``delta_auc`` measures
# incremental value over "current synchrony level predicts future
# synchrony" (the methodological floor).
#
# ``synchrony_entropy`` is exploratory (DECISION-09) and excluded from
# the prediction path entirely.

FEATURE_NAMES = [
    "onset_latency",
    "rise_time",
    "peak_amplitude",
    "recovery_time",
    "dwell_time",
    "switching_rate",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Result of one CV fold.

    For **intra** mode, ``dynamic_auc`` is the model AUC.
    For **cross_modal** mode, ``joint_auc`` is the joint-model AUC,
    ``dynamic_auc`` is kept as an alias for backward compatibility, and
    ``ar_baseline_auc`` stores the restricted-model AUC.
    """
    fold_id: int
    train_size: int
    test_size: int
    dynamic_auc: float              # intra: model AUC | cross_modal: = joint_auc (alias)
    baseline_auc: float             # naive baseline (constant prediction)
    delta_auc: float                # dynamic_auc - max(baseline, AR)
    ar_baseline_auc: float = 0.5    # intra: AR baseline | cross_modal: restricted model AUC
    joint_auc: float = float("nan") # cross-modal only: joint model AUC
    # Ablation: joint model without mean_synchrony
    ablation_auc: float = float("nan")
    ablation_delta_auc: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "fold_id": self.fold_id,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "dynamic_auc": float(self.dynamic_auc),
            "joint_auc": float(self.joint_auc) if not np.isnan(self.joint_auc) else float(self.dynamic_auc),
            "baseline_auc": float(self.baseline_auc),
            "ar_baseline_auc": float(self.ar_baseline_auc),
            "delta_auc": float(self.delta_auc),
        }
        if not np.isnan(self.ablation_auc):
            d["ablation_auc"] = float(self.ablation_auc)
            d["ablation_delta_auc"] = float(self.ablation_delta_auc)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FoldResult":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(
            fold_id=int(d["fold_id"]),
            train_size=int(d["train_size"]),
            test_size=int(d["test_size"]),
            dynamic_auc=float(d["dynamic_auc"]),
            baseline_auc=float(d["baseline_auc"]),
            delta_auc=float(d["delta_auc"]),
            ar_baseline_auc=float(d.get("ar_baseline_auc", 0.5)),
            joint_auc=float(d.get("joint_auc", float("nan"))),
            ablation_auc=float(d.get("ablation_auc", float("nan"))),
            ablation_delta_auc=float(d.get("ablation_delta_auc", float("nan"))),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FoldResult":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


@dataclass
class PredictionResult:
    """Aggregated prediction results for one analysis.

    For **intra** mode:
        mean_dynamic_auc = model AUC
        mean_joint_auc = NaN (not applicable)

    For **cross_modal** mode:
        mean_joint_auc = joint model (source+target) AUC
        mean_dynamic_auc = alias for mean_joint_auc (backward compatible)
        mean_ar_baseline_auc = restricted model (target-only) AUC
        feature_importance = all 20 features (source_XXX + target_XXX)
        source_feature_importance = only source_XXX features (researcher-facing)
    """
    source_pair: str = ""  # e.g., "behavior_value__neural_value"
    target_pair: str = ""  # same as source for intra-pair prediction
    mode: str = "intra"    # "intra" or "cross_modal"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    source_feature_importance: Dict[str, float] = field(default_factory=dict)  # source_XXX only
    mean_dynamic_auc: float = 0.5    # intra: model AUC | cross_modal: alias for joint
    mean_joint_auc: float = float("nan")  # cross-modal only: joint model AUC
    mean_baseline_auc: float = 0.5
    mean_ar_baseline_auc: float = 0.5
    mean_delta_auc: float = 0.0
    # --- Ablation results (cross_modal only) ---
    mean_ablation_auc: float = float("nan")
    mean_ablation_delta_auc: float = float("nan")
    mean_ablation_cost: float = float("nan")  # joint_auc - ablation_auc
    ablation_conclusion: str = ""  # human-readable summary
    # --- Confidence intervals (bootstrap) ---
    dynamic_auc_ci: Optional[Tuple[float, float]] = None   # [lower, upper]
    delta_auc_ci: Optional[Tuple[float, float]] = None     # [lower, upper]
    folds: List[FoldResult] = field(default_factory=list)
    warning: Optional[str] = None  # e.g. "leakage suspected"
    n_features_used: int = 0  # how many features were non-NaN
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # VIF, multicollinearity, etc.

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_pair": self.source_pair,
            "target_pair": self.target_pair,
            "mode": self.mode,
            "feature_importance": {k: float(v) for k, v in self.feature_importance.items()},
            "source_feature_importance": {k: float(v) for k, v in self.source_feature_importance.items()},
            "mean_dynamic_auc": float(self.mean_dynamic_auc),
            "mean_joint_auc": float(self.mean_joint_auc) if not np.isnan(self.mean_joint_auc) else None,
            "mean_baseline_auc": float(self.mean_baseline_auc),
            "mean_ar_baseline_auc": float(self.mean_ar_baseline_auc),
            "mean_delta_auc": float(self.mean_delta_auc),
            "warning": self.warning,
            "n_features_used": self.n_features_used,
            "diagnostics": self.diagnostics,
            "folds": [f.to_dict() for f in self.folds],
        }
        if not np.isnan(self.mean_ablation_auc):
            d["mean_ablation_auc"] = float(self.mean_ablation_auc)
            d["mean_ablation_delta_auc"] = float(self.mean_ablation_delta_auc)
            d["mean_ablation_cost"] = float(self.mean_ablation_cost)
            d["ablation_conclusion"] = self.ablation_conclusion
        if self.dynamic_auc_ci is not None:
            d["dynamic_auc_ci"] = [float(self.dynamic_auc_ci[0]), float(self.dynamic_auc_ci[1])]
        if self.delta_auc_ci is not None:
            d["delta_auc_ci"] = [float(self.delta_auc_ci[0]), float(self.delta_auc_ci[1])]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionResult":
        """Deserialize from a dict (inverse of to_dict)."""
        folds = [FoldResult.from_dict(f) for f in d.get("folds", [])]
        dynamic_auc_ci = None
        if "dynamic_auc_ci" in d and d["dynamic_auc_ci"] is not None:
            ci = d["dynamic_auc_ci"]
            dynamic_auc_ci = (float(ci[0]), float(ci[1]))
        delta_auc_ci = None
        if "delta_auc_ci" in d and d["delta_auc_ci"] is not None:
            ci = d["delta_auc_ci"]
            delta_auc_ci = (float(ci[0]), float(ci[1]))
        raw_joint = d.get("mean_joint_auc")
        return cls(
            source_pair=d.get("source_pair", ""),
            target_pair=d.get("target_pair", ""),
            mode=d.get("mode", "intra"),
            feature_importance={k: float(v) for k, v in d.get("feature_importance", {}).items()},
            source_feature_importance={k: float(v) for k, v in d.get("source_feature_importance", {}).items()},
            mean_dynamic_auc=float(d.get("mean_dynamic_auc", 0.5)),
            mean_joint_auc=float(raw_joint) if raw_joint is not None else float("nan"),
            mean_baseline_auc=float(d.get("mean_baseline_auc", 0.5)),
            mean_ar_baseline_auc=float(d.get("mean_ar_baseline_auc", 0.5)),
            mean_delta_auc=float(d.get("mean_delta_auc", 0.0)),
            mean_ablation_auc=float(d.get("mean_ablation_auc", float("nan"))),
            mean_ablation_delta_auc=float(d.get("mean_ablation_delta_auc", float("nan"))),
            mean_ablation_cost=float(d.get("mean_ablation_cost", float("nan"))),
            ablation_conclusion=d.get("ablation_conclusion", ""),
            dynamic_auc_ci=dynamic_auc_ci,
            delta_auc_ci=delta_auc_ci,
            warning=d.get("warning"),
            n_features_used=int(d.get("n_features_used", 0)),
            diagnostics=d.get("diagnostics", {}),
            folds=folds,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PredictionResult":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Feature matrix builder — the core fix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    wcc: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    onset_threshold: float = ONSET_THRESHOLD,
    exclude_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dynamic-feature matrix from a WCC time series.

    Instead of using raw WCC values as features (which is just autoregression),
    this function computes dynamic features *within each* sliding window of
    the WCC series.  Each row of the output matrix is a feature vector
    describing the dynamics of that window.

    Parameters
    ----------
    wcc : 1-D array
        WCC time series (output of sliding_window_wcc).
    window_size : int
        Number of WCC samples per feature-extraction window.
    hz : float
        Sampling rate of WCC.
    onset_threshold : float
        WCC threshold for onset detection.
    exclude_features : list of str, optional
        Feature names to exclude from the matrix (e.g., ["mean_synchrony"]
        for ablation studies).  If None, all features are included.

    Returns
    -------
    X : 2-D array (n_windows, n_features)
        Feature matrix. Rows with insufficient valid data are NaN.
    feature_names : list of str
        Names of the features (columns of X).
    """
    from .feature_definitions import extract_features as _ssot_extract

    n = len(wcc)
    step = max(1, window_size // 2)  # 50% overlap
    starts = list(range(0, n - window_size + 1, step))

    # Determine which features to include
    if exclude_features:
        exclude_set = set(exclude_features)
        keep_mask = [name not in exclude_set for name in FEATURE_NAMES]
        active_names = [name for name, keep in zip(FEATURE_NAMES, keep_mask) if keep]
        n_features = len(active_names)
        active_indices = [i for i, keep in enumerate(keep_mask) if keep]
    else:
        active_names = list(FEATURE_NAMES)
        n_features = len(FEATURE_NAMES)
        active_indices = list(range(len(FEATURE_NAMES)))

    if not starts:
        return np.empty((0, n_features)), active_names

    X = np.full((len(starts), n_features), np.nan)

    for i, s in enumerate(starts):
        wcc_window = wcc[s : s + window_size]
        feat = _ssot_extract(wcc_window, hz=hz, wcc_window_sec=1.0, threshold=onset_threshold)
        # DECISION-10: joint feature matrix uses the v1 dynamic descriptor
        # features.  mean_synchrony is intentionally absent here; it is
        # extracted separately by ``_extract_mean_synchrony_per_window``
        # for the AR baseline.
        all_values = [
            feat.onset_latency,
            feat.rise_time,
            feat.peak_amplitude,
            feat.recovery_time,
            feat.dwell_time,
            feat.switching_rate,
        ]
        X[i] = [all_values[j] for j in active_indices]

    return X, active_names


def _extract_mean_synchrony_per_window(
    wcc: np.ndarray,
    window_size: int,
    hz: float,
    onset_threshold: float,
) -> np.ndarray:
    """Extract the per-window ``mean_synchrony`` channel.

    DECISION-10 (METHODOLOGY_LOCK_IN.md): ``mean_synchrony`` is removed
    from the joint feature matrix (which now contains only the 6 epoch
    dynamic descriptors) but is retained as the **AR baseline
    (restricted model) predictor** — it lets ``delta_auc`` measure
    incremental value beyond "current synchrony level predicts future
    synchrony", which is the methodological floor for any synchrony
    dynamics model.

    Window slicing here exactly mirrors :func:`_wcc_to_feature_matrix`
    so that the returned column indexes row-by-row with the joint X.
    """
    from .feature_definitions import extract_features as _ssot_extract

    n = len(wcc)
    step = max(1, window_size // 2)
    starts = list(range(0, n - window_size + 1, step))
    if not starts:
        return np.empty(0, dtype=float)

    out = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        wcc_window = wcc[s : s + window_size]
        feat = _ssot_extract(wcc_window, hz=hz, wcc_window_sec=1.0, threshold=onset_threshold)
        out[i] = float(feat.mean_synchrony)
    return out


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def _create_binary_label_from_wcc(
    wcc: np.ndarray,
    window_size: int,
    step: int,
    horizon_windows: int = 1,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary labels from a WCC series aligned to feature windows.

    For each feature window starting at position s, the label is 1 if the
    mean WCC in the next horizon_windows worth of WCC samples exceeds threshold.

    Parameters
    ----------
    wcc : 1-D array
        WCC time series.
    window_size : int
        Feature window size (same as used in build_feature_matrix).
    step : int
        Step size (same as used in build_feature_matrix).
    horizon_windows : int
        Number of future windows to average for the label.
    threshold : float
        Label threshold.

    Returns
    -------
    y : 1-D array of binary labels.
    valid_mask : 1-D bool array (True where label could be computed).
    """
    n = len(wcc)
    starts = list(range(0, n - window_size + 1, step))

    y = np.full(len(starts), np.nan)
    valid_mask = np.zeros(len(starts), dtype=bool)

    # The "future" WCC region starts at s + window_size
    for i, s in enumerate(starts):
        future_start = s + window_size
        future_end = min(future_start + horizon_windows * window_size, n)
        if future_end <= future_start:
            continue
        future_wcc = wcc[future_start:future_end]
        if np.isnan(future_wcc).sum() > len(future_wcc) * 0.5:
            continue
        future_mean = np.nanmean(future_wcc)
        y[i] = 1.0 if future_mean > threshold else 0.0
        valid_mask[i] = True

    return y, valid_mask


# ---------------------------------------------------------------------------
# Rolling-origin CV with dynamic features
# ---------------------------------------------------------------------------

def rolling_origin_cv(
    wcc: np.ndarray,
    window_size: int = 30,
    hz: float = 1.0,
    horizon_windows: int = 1,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    onset_threshold: float = ONSET_THRESHOLD,
    max_iter: int = 200,
    pair_name: str = "",
    mode: str = "intra",
) -> PredictionResult:
    """
    Rolling-origin time-series CV using DYNAMIC FEATURES (not raw WCC).

    This is the corrected prediction pipeline:
    1. Build feature matrix: each sliding window of WCC -> 10 dynamic features.
    2. Create binary labels from future WCC windows.
    3. Train LogisticRegression on features, compare against naive baseline.

    The *gap* parameter enforces a buffer zone between the last training
    sample and the first test sample, preventing temporal leakage.

    Parameters
    ----------
    wcc : 1-D array
        Continuous synchrony time series (WCC output).
    window_size : int
        Window size for both feature extraction and label creation.
        Default 30 (larger than old default 10 to ensure meaningful
        dynamic feature extraction within each window).
    hz : float
        Sampling rate of WCC.
    horizon_windows : int
        Number of future windows whose mean determines the label.
    n_splits : int
        Number of CV folds.
    gap : int
        Gap (buffer) between train and test sets, in samples (feature rows).
    threshold : float
        Label threshold for "high synchrony".
    onset_threshold : float
        WCC threshold for onset detection in dynamic features.
    max_iter : int
        Max iterations for LogisticRegression.
    pair_name : str
        Human-readable name for this pair (for result metadata).
    mode : str
        "intra" or "cross_modal".

    Returns
    -------
    PredictionResult
    """
    diagnostics: Dict[str, Any] = {}

    # Auto-adjust window_size if too small for reliable feature extraction.
    # Record this as a structured diagnostic rather than emitting a warning.
    n = len(wcc)
    min_window = min(60, n // 4)
    if window_size < min_window:
        diagnostics["window_size_auto_adjusted"] = True
        diagnostics["window_size_requested"] = int(window_size)
        diagnostics["window_size_effective"] = int(min_window)
        window_size = min_window

    # 1. Build feature matrix
    step = max(1, window_size // 2)
    X, feature_names = build_feature_matrix(
        wcc, window_size, hz, onset_threshold
    )
    # DECISION-10: extract mean_synchrony on its own channel for AR
    # baseline (restricted model).  Same window slicing as X, so it
    # row-aligns with X before / after the both_valid mask is applied.
    mean_sync_col = _extract_mean_synchrony_per_window(
        wcc, window_size, hz, onset_threshold
    )

    # 2. Create labels
    y, valid_mask = _create_binary_label_from_wcc(
        wcc, window_size, step, horizon_windows, threshold
    )

    # Align: only keep rows where both (a) the label is valid AND
    # (b) the feature row is not *structurally* empty.
    all_nan_rows = np.all(np.isnan(X), axis=1)
    both_valid = valid_mask & ~all_nan_rows
    X = X[both_valid]
    if mean_sync_col.size > 0:
        mean_sync_col = mean_sync_col[both_valid]
    y = y[both_valid].astype(int)

    # Imputation strategy depends on feature semantics:
    # DECISION-10: with the v1 dynamic descriptor set, duration
    # features are onset_latency (0), rise_time (1), recovery_time (3),
    # and dwell_time (4).  peak_amplitude (2) and switching_rate (5)
    # are bounded-range / rate quantities and use the 0-fill default.
    DURATION_FEATURE_IDX = {0, 1, 3, 4}  # onset, rise, recovery, dwell
    window_duration = window_size / hz

    if len(X) > 0:
        n_non_nan = (~np.isnan(X)).sum(axis=1)
        for col_idx in DURATION_FEATURE_IDX:
            X[np.isnan(X[:, col_idx]), col_idx] = window_duration
        X = np.nan_to_num(X, nan=0.0)
        avg_features_used = int(np.mean(n_non_nan))
    else:
        avg_features_used = 0
    # mean_sync_col: bounded-range quantity → 0-fill is the natural
    # neutral imputation (matches the legacy ``np.nan_to_num`` behaviour
    # used when mean_synchrony lived inside X).
    if mean_sync_col.size > 0:
        mean_sync_col = np.nan_to_num(mean_sync_col, nan=0.0)

    if len(y) < max(3, n_splits):  # Lowered from max(5, n_splits) to accommodate smaller windows
        logger.debug(
            "Insufficient samples: len(y)=%d, n_splits=%d", len(y), n_splits
        )
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=avg_features_used,
            diagnostics={},
        )

    class_counts = np.bincount(y)
    if len(class_counts) < 2 or min(class_counts) < 3:
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="class_imbalance",
            n_features_used=avg_features_used,
            diagnostics={},
        )

    # --- Hard check: refuse to run if data is insufficient for CV ---
    n_samples = len(X)
    est_test_size = max(1, n_samples // (n_splits + 1))
    min_gap = n_samples - est_test_size * n_splits - 1

    if min_gap < 0 or gap > min_gap:
        # Data too short for requested CV parameters - HARD FAIL
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="data_too_short_for_cv",
            n_features_used=0,
            diagnostics={
                "error": "insufficient_data_for_cv",
                "n_samples": n_samples,
                "n_splits_requested": n_splits,
                "gap_requested": gap,
                "min_samples_needed": (n_splits + 1) * (1 + gap) + n_splits,
            },
        )

    # --- Physical-time-aware gap ---
    effective_gap = _compute_effective_gap(gap, window_size, horizon_windows)
    if effective_gap > gap:
        diagnostics.update({
            "gap_auto_adjusted": True,
            "gap_requested": gap,
            "gap_effective": effective_gap,
        })
        logger.info(
            "Gap auto-adjusted from %d to %d (window_size=%d, horizon=%d)",
            gap, effective_gap, window_size, horizon_windows,
        )

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=effective_gap)

    folds: List[FoldResult] = []
    feature_coefs_sum = np.zeros(len(feature_names))
    valid_folds = 0

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(test_idx) < 3:  # Lowered from 5 to accommodate smaller windows
            continue

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Hard check: skip fold if test set has only one class
        if len(np.unique(y_test)) < 2:
            continue

        # Hard check: skip fold if train set has only one class
        if len(np.unique(y_train)) < 2:
            continue

        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=max(max_iter, 500),
                class_weight="balanced",
            )
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            dynamic_auc = roc_auc_score(y_test, y_prob)
            feature_coefs_sum += clf.coef_[0]
            valid_folds += 1
        except Exception as e:
            # Skip this fold - do NOT pretend AUC=0.5
            logger.debug("Fold %d failed: %s", fold_id, e)
            continue

        # Baseline: predict constant = training set positive rate
        # This is a NAIVE baseline (not using any features)
        try:
            baseline_prob = np.full_like(y_test, y_train.mean())
            baseline_auc = roc_auc_score(y_test, baseline_prob)
        except Exception:
            baseline_auc = 0.5

        # AR baseline (restricted model): use mean_synchrony only.
        # This measures whether the full epoch-feature model adds
        # predictive value beyond "current synchrony level predicts
        # future synchrony", the minimum bar for any synchrony dynamics
        # model.  Keeps intra-mode delta_auc comparable to cross-modal.
        #
        # DECISION-10: mean_synchrony is a *diagnostic* (not part of the
        # FDR family / FEATURE_NAMES).  It rides on its own row-aligned
        # channel (mean_sync_col) extracted with the same windowing as X.
        #
        # NOTE: Single-feature AR baseline should NOT use L1
        # regularization.  L1 can shrink the only coefficient to 0,
        # collapsing the model to intercept-only (AUC = 0.5) and making
        # delta_AUC meaningless.
        ar_auc = baseline_auc  # fallback if mean_synchrony unavailable
        try:
            if mean_sync_col.size == 0:
                raise ValueError("mean_sync_col unavailable")
            X_train_ar = mean_sync_col[train_idx].reshape(-1, 1)
            X_test_ar = mean_sync_col[test_idx].reshape(-1, 1)
            # Scale on the AR channel alone (do NOT reuse the
            # epoch-feature scaler — different column semantics).
            scaler_ar = StandardScaler()
            X_train_ar_scaled = scaler_ar.fit_transform(X_train_ar)
            X_test_ar_scaled = scaler_ar.transform(X_test_ar)
            # Single feature -> no regularization needed.
            # sklearn 1.8+: solver=lbfgs + large C disables regularization
            clf_ar = LogisticRegression(
                solver="lbfgs",
                C=1e12,  # effectively no regularization
                max_iter=max(max_iter, 500),
                class_weight="balanced",
            )
            clf_ar.fit(X_train_ar_scaled, y_train)
            ar_auc = roc_auc_score(
                y_test, clf_ar.predict_proba(X_test_ar_scaled)[:, 1]
            )
        except ValueError:
            pass  # mean_synchrony unavailable — fall back to naive baseline
        except Exception:
            ar_auc = 0.5

        folds.append(FoldResult(
            fold_id=fold_id,
            train_size=len(train_idx),
            test_size=len(test_idx),
            dynamic_auc=dynamic_auc,
            baseline_auc=baseline_auc,
            ar_baseline_auc=ar_auc,
            delta_auc=dynamic_auc - max(baseline_auc, ar_auc),
        ))

    if valid_folds == 0:
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="no_valid_folds",
            n_features_used=avg_features_used,
            diagnostics=diagnostics,
        )

    mean_dynamic = np.mean([f.dynamic_auc for f in folds])
    mean_baseline = np.mean([f.baseline_auc for f in folds])
    mean_delta = np.mean([f.delta_auc for f in folds])

    avg_coefs = feature_coefs_sum / valid_folds
    importance = {
        feature_names[i]: float(avg_coefs[i]) for i in range(len(feature_names))
    }

    warning = None
    if mean_delta > LEAKAGE_DELTA_AUC_THRESHOLD:
        warning = "leakage_suspected"

    return PredictionResult(
        source_pair=pair_name,
        target_pair=pair_name,
        mode=mode,
        feature_importance=importance,
        mean_dynamic_auc=mean_dynamic,
        mean_baseline_auc=mean_baseline,
        mean_delta_auc=mean_delta,
        folds=folds,
        warning=warning,
        n_features_used=avg_features_used,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Cross-modal prediction: incremental predictive validity test
# ---------------------------------------------------------------------------

def _compute_effective_gap(
    gap: int,
    window_size: int,
    horizon_windows: int,
) -> int:
    """Compute a physical-time-aware gap for TimeSeriesSplit.

    With 50% overlapping sliding windows, adjacent feature rows share data.
    The gap must be large enough that train and test windows do not overlap
    in the underlying signal, and that the label's future horizon does not
    leak into the test features.

    Parameters
    ----------
    gap : int
        User-requested gap (in feature-row units).
    window_size : int
        Feature extraction window size (in raw samples).
    horizon_windows : int
        Number of future windows used for label creation.

    Returns
    -------
    effective_gap : int
        At least max(gap, ceil(window_size / step) + horizon_windows).
    """
    step = max(1, window_size // 2)
    # Rows needed to skip one full non-overlapping window
    min_physical_gap_rows = int(np.ceil(window_size / step))
    # Rows needed to skip the label's future horizon
    horizon_gap_rows = horizon_windows
    return max(gap, min_physical_gap_rows + horizon_gap_rows)


def cross_modal_prediction(
    source_wcc: np.ndarray,
    target_wcc: np.ndarray,
    window_size: int = 30,
    hz: float = 1.0,
    horizon_windows: int = 1,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    onset_threshold: float = ONSET_THRESHOLD,
    max_iter: int = 200,
    source_name: str = "",
    target_name: str = "",
) -> PredictionResult:
    """
    Cross-modal prediction: incremental predictive validity test.

    Tests whether SOURCE pair's dynamic features provide *incremental*
    predictive value for TARGET pair's future synchrony, beyond what
    TARGET's own past dynamic features can predict.

    Methodology (nested model comparison):
    1. **Restricted model**: [target_features] -> target_future_label
       (autoregressive baseline — how well does target predict itself?)
    2. **Joint model**: [source_features + target_features] -> target_future_label
       (does source add information on top of target's own history?)
    3. **delta_auc** = joint_auc - max(naive_baseline, restricted_auc)

    A positive and significant delta_auc indicates that source provides
    incremental predictive value — i.e., source dynamics contain precursor
    information not already captured by target's own autocorrelation.

    This is NOT a Granger causality test (which requiresVAR model comparison
    in continuous time).  It is an incremental predictive validity test
    inspired by Granger's principle of incremental information.

    Ablation experiment:
    The joint model is also run WITHOUT ``mean_synchrony`` to verify that
    temporal-shape features (onset, build-up, peak, breakdown) have
    independent predictive power beyond the synchrony level itself.

    Parameters
    ----------
    source_wcc : 1-D array
        WCC time series of the source modality pair (features come from here).
    target_wcc : 1-D array
        WCC time series of the target modality pair (labels come from here).
    window_size : int
        Window size for feature extraction and label creation.
    hz : float
        Sampling rate.
    horizon_windows : int
        Number of future windows for label creation.
    n_splits : int
        Number of CV folds.
    gap : int
        Minimum buffer gap between train and test (in feature-row units).
        The actual gap may be larger to account for window overlap and
        label horizon (see _compute_effective_gap).
    threshold : float
        Label threshold.
    onset_threshold : float
        WCC onset threshold for dynamic features.
    max_iter : int
        Max iterations for LogisticRegression.
    source_name : str
        Name of source pair.
    target_name : str
        Name of target pair.

    Returns
    -------
    PredictionResult with mode="cross_modal", including ablation results.
    """
    step = max(1, window_size // 2)

    # Build features from SOURCE (v1 dynamic descriptor set)
    X_source, source_feature_names = build_feature_matrix(
        source_wcc, window_size, hz, onset_threshold
    )

    # Build features from TARGET (v1 dynamic descriptor set,
    # for restricted model)
    X_target, target_feature_names = build_feature_matrix(
        target_wcc, window_size, hz, onset_threshold
    )

    # Build labels from TARGET
    y, valid_mask = _create_binary_label_from_wcc(
        target_wcc, window_size, step, horizon_windows, threshold
    )

    # DECISION-10 X1: extract mean_synchrony as an *external channel*
    # for each pair, mirroring the intra-mode rolling_origin_cv design.
    # Used both as the AR signal in the joint model and as the
    # "ablation handle" — the joint model includes both pairs' AR
    # signals, the ablation removes only the SOURCE AR signal to test
    # whether the temporal-shape features of the source carry
    # incremental predictive value beyond its overall synchrony level
    # (Granger-style "does source-shape add beyond source-level?").
    source_mean_sync_col = _extract_mean_synchrony_per_window(
        source_wcc, window_size, hz, onset_threshold
    )
    target_mean_sync_col = _extract_mean_synchrony_per_window(
        target_wcc, window_size, hz, onset_threshold
    )

    # Align lengths (feature matrices may differ in length due to WCC lengths)
    min_rows = min(
        len(X_source),
        len(y),
        len(X_target),
        len(source_mean_sync_col),
        len(target_mean_sync_col),
    )
    if min_rows < 20:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_ar_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=0,
            diagnostics={},
        )

    X_source = X_source[:min_rows]
    y = y[:min_rows]
    valid_mask = valid_mask[:min_rows]
    X_target = X_target[:min_rows]
    source_mean_sync_col = source_mean_sync_col[:min_rows]
    target_mean_sync_col = target_mean_sync_col[:min_rows]

    # Only keep rows where both (a) the label is valid AND
    # (b) the feature row is not *structurally* empty.
    all_nan_rows = np.all(np.isnan(X_source), axis=1)
    both_valid = valid_mask & ~all_nan_rows
    X_source = X_source[both_valid]
    y = y[both_valid].astype(int)
    X_target = X_target[both_valid]
    source_mean_sync_col = source_mean_sync_col[both_valid]
    target_mean_sync_col = target_mean_sync_col[both_valid]

    # Imputation strategy depends on feature semantics:
    # DECISION-10: with the v1 dynamic descriptor set, duration
    # features are onset_latency (0), rise_time (1), recovery_time (3),
    # and dwell_time (4).  peak_amplitude (2) and switching_rate (5)
    # are bounded-range / rate quantities and use the 0-fill default.
    #   Duration features: NaN means "event not detected" — impute with
    #   window duration (conservative upper bound).
    #   Rate/Amplitude features: NaN -> 0.0 (no event -> zero rate/amp).
    DURATION_FEATURE_IDX = {0, 1, 3, 4}  # onset, rise, recovery, dwell
    window_duration = window_size / hz

    for X_mat in (X_source, X_target):
        if len(X_mat) > 0:
            for col_idx in DURATION_FEATURE_IDX:
                if col_idx < X_mat.shape[1]:
                    X_mat[np.isnan(X_mat[:, col_idx]), col_idx] = window_duration
            np.nan_to_num(X_mat, copy=False, nan=0.0)

    # mean_sync channels: bounded-range diagnostics — 0-fill is the
    # natural neutral imputation (consistent with intra-mode).
    if source_mean_sync_col.size > 0:
        source_mean_sync_col = np.nan_to_num(source_mean_sync_col, nan=0.0)
    if target_mean_sync_col.size > 0:
        target_mean_sync_col = np.nan_to_num(target_mean_sync_col, nan=0.0)

    avg_features_used = 0
    if len(X_source) > 0:
        avg_features_used = int(np.mean((~np.isnan(X_source)).sum(axis=1)))

    if len(y) < 20:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_ar_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=avg_features_used,
            diagnostics={},
        )

    class_counts = np.bincount(y)
    if len(class_counts) < 2 or min(class_counts) < 3:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_ar_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="class_imbalance",
            n_features_used=avg_features_used,
            diagnostics={},
        )

    # --- Multicollinearity check on source features ---
    diagnostics = {}
    high_corr_pairs = []
    try:
        corr_matrix = np.corrcoef(X_source.T)
        for i in range(len(source_feature_names)):
            for j in range(i + 1, len(source_feature_names)):
                if abs(corr_matrix[i, j]) > 0.9:
                    high_corr_pairs.append(
                        (source_feature_names[i], source_feature_names[j],
                         float(corr_matrix[i, j]))
                    )
        if high_corr_pairs:
            diagnostics["multicollinearity"] = True
            diagnostics["high_corr_pairs"] = high_corr_pairs
            logger.warning(
                f"High multicollinearity detected: {len(high_corr_pairs)} feature pairs "
                f"with |r| > 0.9. Consider removing redundant features."
            )
    except Exception:
        pass

    # --- Physical-time-aware gap ---
    effective_gap = _compute_effective_gap(gap, window_size, horizon_windows)
    if effective_gap > gap:
        diagnostics["gap_auto_adjusted"] = True
        diagnostics["gap_requested"] = gap
        diagnostics["gap_effective"] = effective_gap
        logger.info(
            "Gap auto-adjusted from %d to %d (window_size=%d, horizon=%d)",
            gap, effective_gap, window_size, horizon_windows,
        )

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=effective_gap)

    # DECISION-10 X1: model column layouts.
    #
    #   restricted  = [target_6epoch ⊕ target_mean_sync]
    #   joint       = [source_6epoch ⊕ target_6epoch
    #                  ⊕ source_mean_sync ⊕ target_mean_sync]
    #   ablation    = [source_6epoch ⊕ target_6epoch
    #                  ⊕ target_mean_sync]   (drop ONLY source AR)
    #
    # ablation answers: "does the source's *temporal shape* add
    # incremental predictive value beyond (target's everything)
    # AND (source's overall synchrony level)?"
    joint_feature_names = (
        [f"source_{n}" for n in source_feature_names]
        + [f"target_{n}" for n in target_feature_names]
        + ["source_mean_synchrony", "target_mean_synchrony"]
    )
    abl_joint_names = (
        [f"source_{n}" for n in source_feature_names]
        + [f"target_{n}" for n in target_feature_names]
        + ["target_mean_synchrony"]  # source AR dropped
    )
    restricted_feature_names = (
        [f"target_{n}" for n in target_feature_names]
        + ["target_mean_synchrony"]
    )

    folds: List[FoldResult] = []
    joint_coefs_sum = np.zeros(len(joint_feature_names))
    abl_coefs_sum = np.zeros(len(abl_joint_names))
    valid_folds = 0

    # Reshape AR channels into 2-D column vectors once for hstack reuse.
    source_ar = source_mean_sync_col.reshape(-1, 1) if source_mean_sync_col.size else np.zeros((len(X_source), 1))
    target_ar = target_mean_sync_col.reshape(-1, 1) if target_mean_sync_col.size else np.zeros((len(X_target), 1))

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X_source)):
        if len(test_idx) < 3:
            continue

        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue
        if len(np.unique(y_train)) < 2:
            continue

        # --- Slices for this fold ---
        X_source_train = X_source[train_idx]
        X_source_test = X_source[test_idx]
        X_target_train = X_target[train_idx]
        X_target_test = X_target[test_idx]
        src_ar_train = source_ar[train_idx]
        src_ar_test = source_ar[test_idx]
        tgt_ar_train = target_ar[train_idx]
        tgt_ar_test = target_ar[test_idx]

        # --- Joint features: hstack([source, target, src_ar, tgt_ar]) ---
        X_joint_train = np.hstack(
            (X_source_train, X_target_train, src_ar_train, tgt_ar_train)
        )
        X_joint_test = np.hstack(
            (X_source_test, X_target_test, src_ar_test, tgt_ar_test)
        )

        # --- Ablation joint features: drop SOURCE AR only ---
        X_abl_joint_train = np.hstack(
            (X_source_train, X_target_train, tgt_ar_train)
        )
        X_abl_joint_test = np.hstack(
            (X_source_test, X_target_test, tgt_ar_test)
        )

        # --- Restricted features: target_6epoch + target_ar ---
        X_restricted_train = np.hstack((X_target_train, tgt_ar_train))
        X_restricted_test = np.hstack((X_target_test, tgt_ar_test))

        # Standardize (separate scaler per model to avoid leakage)
        scaler_joint = StandardScaler()
        scaler_restricted = StandardScaler()
        scaler_abl = StandardScaler()

        try:
            X_joint_train_s = scaler_joint.fit_transform(X_joint_train)
            X_joint_test_s = scaler_joint.transform(X_joint_test)
            X_target_train_s = scaler_restricted.fit_transform(X_restricted_train)
            X_target_test_s = scaler_restricted.transform(X_restricted_test)
            X_abl_train_s = scaler_abl.fit_transform(X_abl_joint_train)
            X_abl_test_s = scaler_abl.transform(X_abl_joint_test)
        except Exception:
            continue

        # --- Restricted model (target shape + target AR) ---
        try:
            clf_restricted = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=max(max_iter, 500),
                class_weight="balanced",
            )
            clf_restricted.fit(X_target_train_s, y_train)
            restricted_auc = roc_auc_score(
                y_test, clf_restricted.predict_proba(X_target_test_s)[:, 1]
            )
        except Exception:
            restricted_auc = 0.5

        # --- Joint model (source + target + both ARs) ---
        joint_auc = 0.5
        try:
            clf_joint = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=max(max_iter, 500),
                class_weight="balanced",
            )
            clf_joint.fit(X_joint_train_s, y_train)
            joint_auc = roc_auc_score(
                y_test, clf_joint.predict_proba(X_joint_test_s)[:, 1]
            )
            joint_coefs_sum += clf_joint.coef_[0]
            valid_folds += 1
        except Exception:
            pass

        # --- Ablation joint model (drop SOURCE AR only) ---
        ablation_auc = float("nan")
        try:
            clf_abl = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=max(max_iter, 500),
                class_weight="balanced",
            )
            clf_abl.fit(X_abl_train_s, y_train)
            ablation_auc = roc_auc_score(
                y_test, clf_abl.predict_proba(X_abl_test_s)[:, 1]
            )
            abl_coefs_sum += clf_abl.coef_[0]
        except Exception:
            pass

        # --- Naive baseline (constant prediction) ---
        try:
            baseline_prob = np.full_like(y_test, y_train.mean())
            baseline_auc = roc_auc_score(y_test, baseline_prob)
        except Exception:
            baseline_auc = 0.5

        # Delta = joint - max(naive, restricted)
        delta_auc = joint_auc - max(baseline_auc, restricted_auc)

        # Ablation delta
        if not np.isnan(ablation_auc):
            ablation_delta = ablation_auc - max(baseline_auc, restricted_auc)
        else:
            ablation_delta = float("nan")

        folds.append(FoldResult(
            fold_id=fold_id,
            train_size=len(train_idx),
            test_size=len(test_idx),
            dynamic_auc=joint_auc,
            baseline_auc=baseline_auc,
            ar_baseline_auc=restricted_auc,
            delta_auc=delta_auc,
            joint_auc=joint_auc,
            ablation_auc=ablation_auc,
            ablation_delta_auc=ablation_delta,
        ))

    if valid_folds == 0:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_ar_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="no_valid_folds",
            n_features_used=avg_features_used,
            diagnostics=diagnostics,
        )

    mean_joint = np.mean([f.dynamic_auc for f in folds])
    mean_baseline = np.mean([f.baseline_auc for f in folds])
    mean_restricted = np.mean([f.ar_baseline_auc for f in folds])
    mean_delta = np.mean([f.delta_auc for f in folds])

    # Bootstrap 95% confidence intervals
    dynamic_vals = [f.dynamic_auc for f in folds]
    delta_vals = [f.delta_auc for f in folds]
    dynamic_auc_ci = _bootstrap_ci(dynamic_vals)
    delta_auc_ci = _bootstrap_ci(delta_vals)

    # Feature importance from joint model (all 20 features)
    avg_coefs = joint_coefs_sum / valid_folds
    importance = {
        joint_feature_names[i]: float(avg_coefs[i])
        for i in range(len(joint_feature_names))
    }

    # Source-only feature importance (researcher-facing: which source
    # features have the strongest incremental predictive power after
    # controlling for target's own history?)
    source_importance = {
        k: float(v) for k, v in importance.items() if k.startswith("source_")
    }

    # Ablation summary
    ablation_aucs = [f.ablation_auc for f in folds if not np.isnan(f.ablation_auc)]
    ablation_delta_vals = [f.ablation_delta_auc for f in folds if not np.isnan(f.ablation_delta_auc)]
    mean_ablation_auc = float(np.mean(ablation_aucs)) if ablation_aucs else float("nan")
    mean_ablation_delta = float(np.mean(ablation_delta_vals)) if ablation_delta_vals else float("nan")

    # Ablation cost: how much does dropping the SOURCE AR signal hurt?
    # delta_cost ≈ 0 → source temporal-shape features alone explain the
    #                  incremental value (target shape + target AR + the
    #                  *source AR signal* are not needed beyond shape).
    # delta_cost > 0.05 → source's overall synchrony level carries
    #                     substantial unique predictive value beyond its
    #                     temporal shape (Granger-style level vs shape).
    mean_ablation_cost = (
        float(mean_joint - mean_ablation_auc)
        if not np.isnan(mean_ablation_auc)
        else float("nan")
    )

    # Ablation conclusion based on bootstrap CI (not arbitrary threshold).
    # CI lower bound > 0 is the minimum standard for claiming "source
    # temporal-shape features retain predictive value after accounting
    # for source synchrony level".
    ablation_delta_ci = _bootstrap_ci(ablation_delta_vals) if ablation_delta_vals else None

    if not np.isnan(mean_ablation_delta):
        if ablation_delta_ci is not None and ablation_delta_ci[0] > 0.0:
            ablation_conclusion = (
                f"Strict validation passed: source temporal-shape "
                f"features (v1 dynamic descriptor set) retain "
                f"significant incremental predictive value beyond "
                f"target's full model and source's overall synchrony "
                f"level (ablation delta_AUC = {mean_ablation_delta:.3f}, "
                f"95% CI = [{ablation_delta_ci[0]:.3f}, {ablation_delta_ci[1]:.3f}], "
                f"CI lower bound > 0). Ablation cost (dropping source "
                f"AR signal) = {mean_ablation_cost:.3f}."
            )
        else:
            ci_str = (
                f", 95% CI = [{ablation_delta_ci[0]:.3f}, {ablation_delta_ci[1]:.3f}]"
                if ablation_delta_ci is not None
                else ""
            )
            ablation_conclusion = (
                f"Validation failed: dropping the SOURCE AR signal "
                f"eliminates robust incremental predictive value of "
                f"source temporal-shape features (ablation delta_AUC = "
                f"{mean_ablation_delta:.3f}{ci_str}). Prediction is "
                f"predominantly driven by source synchrony level, not "
                f"shape. Ablation cost = {mean_ablation_cost:.3f}."
            )
    else:
        ablation_conclusion = "Ablation experiment failed (no valid folds)."

    warning = None
    if mean_delta > LEAKAGE_DELTA_AUC_THRESHOLD:
        warning = "leakage_suspected"
    elif mean_delta < -0.2:
        warning = "restricted_baseline_dominates"

    return PredictionResult(
        source_pair=source_name,
        target_pair=target_name,
        mode="cross_modal",
        feature_importance=importance,
        source_feature_importance=source_importance,
        mean_dynamic_auc=mean_joint,
        mean_joint_auc=mean_joint,
        mean_baseline_auc=mean_baseline,
        mean_ar_baseline_auc=mean_restricted,
        mean_delta_auc=mean_delta,
        mean_ablation_auc=mean_ablation_auc,
        mean_ablation_delta_auc=mean_ablation_delta,
        mean_ablation_cost=mean_ablation_cost,
        ablation_conclusion=ablation_conclusion,
        dynamic_auc_ci=dynamic_auc_ci,
        delta_auc_ci=delta_auc_ci,
        folds=folds,
        warning=warning,
        n_features_used=avg_features_used,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for CV metrics
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """
    Percentile bootstrap 95% confidence interval.

    Resamples fold-level AUCs with replacement B times, computes the
    mean of each bootstrap sample, and returns the [2.5%, 97.5%]
    percentiles.

    Parameters
    ----------
    values : list of float
        Per-fold metric values (e.g., dynamic_auc from each fold).
    n_boot : int
        Number of bootstrap resamples.  Default 1000.
    alpha : float
        Significance level.  Default 0.05 → 95% CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) tuple or None if fewer than 3 values.

    Notes
    -----
    **Small-sample caveat:** When ``len(values) < 10`` (e.g., the typical
    5-fold CV), the percentile bootstrap CI may *underestimate* the true
    interval width because the resampling support is limited to at most
    ``len(values)**len(values)`` distinct means.  The returned CI remains
    more honest than a normal approximation (``mean ± 1.96*SE``), but
    should be interpreted as a *lower bound* on uncertainty.
    """
    if len(values) < 3:
        return None

    if len(values) < 10:
        warnings.warn(
            f"Bootstrap CI computed on only {len(values)} values — "
            "the confidence interval may underestimate the true "
            "uncertainty.  Consider increasing n_splits for more "
            "reliable estimation.",
            UserWarning,
            stacklevel=3,
        )

    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[b] = np.mean(sample)

    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ---------------------------------------------------------------------------
# CV feasibility checker — public API for frontend pre-validation
# ---------------------------------------------------------------------------

def check_cv_feasibility(
    n_samples: int,
    window_size: int = 30,
    step: int = 15,
    n_splits: int = 5,
    gap: int = 0,
) -> Dict[str, Any]:
    """
    Check if data length is sufficient for TimeSeriesSplit CV.

    Frontend should call this BEFORE allowing the user to click "Run".
    If not feasible, the response includes safe parameter suggestions.

    Parameters
    ----------
    n_samples : int
        Number of samples in the feature matrix (after alignment/imputation).
    window_size : int
        Window size used for feature extraction.
    step : int
        Step size between windows (typially window_size // 2).
    n_splits : int
        Requested number of CV splits.
    gap : int
        Requested gap between train and test.

    Returns
    -------
    dict with keys:
      - feasible: bool
      - n_samples: int
      - n_splits_requested: int
      - gap_requested: int
      - min_samples_needed: int
      - max_splits_safe: int
      - max_gap_safe: int
      - suggestion: str (human-readable suggestion for the user)
    """
    # Number of windows (feature rows)
    n_windows = max(0, (n_samples - window_size) // step + 1)
    feasible = True
    suggestions = []

    # Check: n_windows >= n_splits * 2 + gap
    # Each split needs at least 1 train + 1 test window, plus gap
    min_samples = (n_splits + 1) * (1 + gap) + n_splits

    if n_windows < 3:
        feasible = False
        suggestions.append(
            f"Data too short: only {n_windows} windows available. "
            f"Need at least 3 windows for any CV. "
            f"Try increasing Hz or decreasing window_size."
        )
    elif n_windows < min_samples:
        feasible = False
        # Suggest safe n_splits
        safe_splits = max(2, n_windows // (2 + gap) - 1)
        safe_gap = max(0, n_windows // (n_splits + 1) - 2)
        suggestions.append(
            f"Data too short for {n_splits}-fold CV with gap={gap}. "
            f"You have {n_windows} windows; need {min_samples}. "
            f"Suggested: reduce to {safe_splits}-fold CV, "
            f"or use Leave-One-Segment-Out (LOSO) for small datasets."
        )
    else:
        suggestions.append("CV parameters are feasible.")

    # Compute safe max values
    max_splits_safe = max(2, n_windows - 2) if n_windows >= 3 else 0
    max_gap_safe = max(0, n_windows - 3) if n_windows >= 3 else 0

    return {
        "feasible": feasible,
        "n_samples": n_samples,
        "n_windows": n_windows,
        "n_splits_requested": n_splits,
        "gap_requested": gap,
        "min_samples_needed": min_samples,
        "max_splits_safe": max_splits_safe,
        "max_gap_safe": max_gap_safe,
        "suggestion": " ".join(suggestions),
    }


# ---------------------------------------------------------------------------
# Group consistency check — validity check for dyad-level metrics
# ---------------------------------------------------------------------------

def check_group_consistency(
    dyad_results: List[Dict],
    target_key: str = "mean_delta_auc",
) -> Dict:
    """
    Check whether dyad-level metrics are consistent across the group.

    This is a **validity check**, not a predictive model.  It measures
    whether each dyad's metric (e.g., delta_AUC) is close to the
    group mean — analogous to a crude intraclass correlation check.

    The method holds out one dyad, computes the mean of the remaining
    dyads, and compares to the held-out dyad's actual value.  A low
    MAE and high correlation indicate that the metric is consistent
    across dyads (not driven by a single outlier).

    **Naming note:** Previously called ``lodo_cv`` ("Leave-One-Dyad-Out
    Cross-Validation").  Renamed because this is NOT a cross-dyad predictive
    model (it does not train on N-1 dyads and test on the Nth).  It is a
    group-level consistency check.  ``lodo_cv`` is kept as a backward-compatible
    alias.

    Parameters
    ----------
    dyad_results : list of dict
        Each dict must contain *target_key*.
    target_key : str
        Which metric to evaluate.

    Returns
    -------
    dict with keys: predictions, actuals, mae, correlation.
    """
    if len(dyad_results) < 3:
        return {
            "error": "need at least 3 dyads for group consistency check",
            "predictions": [],
            "actuals": [],
        }

    values = np.array([d[target_key] for d in dyad_results])
    predictions = []
    actuals = []

    for i in range(len(dyad_results)):
        others = np.delete(values, i)
        pred = np.mean(others)
        predictions.append(float(pred))
        actuals.append(float(values[i]))

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = float(np.mean(np.abs(predictions - actuals)))
    residuals = actuals - predictions

    # Pearson correlation (handle zero-variance)
    if np.std(actuals) > 0 and np.std(predictions) > 0:
        corr = float(np.corrcoef(actuals, predictions)[0, 1])
    else:
        corr = 0.0

    return {
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "mae": mae,
        "correlation": corr,
    }


# Backward-compatible alias
lodo_cv = check_group_consistency


# ---------------------------------------------------------------------------
# Threshold sensitivity analysis — parameter robustness
# ---------------------------------------------------------------------------

def threshold_sensitivity(
    wcc: np.ndarray,
    window_size: int = 30,
    hz: float = 1.0,
    horizon_windows: int = 1,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    onset_thresholds: Optional[List[float]] = None,
    max_iter: int = 200,
    pair_name: str = "",
) -> Dict[str, Any]:
    """Sweep onset_threshold values and collect per-threshold delta_AUC.

    This enables researchers to demonstrate that SyncPipe's dynamic features
    are robust to the onset_threshold hyperparameter — a common reviewer
    concern.  In a paper, plot onset_threshold (X-axis, in SD units for
    Z-scored data) vs delta_AUC (Y-axis).  Robustness is confirmed when
    delta_AUC does not show a cliff-drop across the threshold range.

    Parameters
    ----------
    wcc : 1-D array
        WCC time series.
    window_size : int
        Window size for feature extraction and label creation.
    hz : float
        Sampling rate.
    horizon_windows : int
        Number of future windows for label creation.
    n_splits : int
        Number of CV folds.
    gap : int
        Minimum buffer gap between train and test.
    threshold : float
        Label threshold for "high synchrony".
    onset_thresholds : list of float, optional
        Threshold values to sweep.  If None, uses a sensible default range
        of [-0.5, 0.0, 0.2, 0.3, 0.5, 0.7, 1.0] (in SD units for Z-scored
        WCC).  These represent deviations from the mean synchrony level.
    max_iter : int
        Max iterations for LogisticRegression.
    pair_name : str
        Human-readable pair name.

    Returns
    -------
    dict with keys:
      - onset_thresholds: list of float (the swept values)
      - delta_aucs: list of float (mean delta_AUC per threshold)
      - dynamic_aucs: list of float (mean model AUC per threshold)
      - baseline_aucs: list of float (mean baseline AUC per threshold)
      - robust: bool (True if delta_AUC range < 0.15 across thresholds)
      - robust_range: (min_delta_auc, max_delta_auc)
      - note: str (interpretation guidance)
    """
    if onset_thresholds is None:
        # Anchor = 0.5, swept over [0.3, 0.7].  Zero/negative thresholds are
        # excluded because they make every positive WCC value count as
        # "synchrony" and inflate dwell/switching to noise.
        onset_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    for ot in onset_thresholds:
        res = rolling_origin_cv(
            wcc=wcc,
            window_size=window_size,
            hz=hz,
            horizon_windows=horizon_windows,
            n_splits=n_splits,
            gap=gap,
            threshold=threshold,
            onset_threshold=ot,
            max_iter=max_iter,
            pair_name=pair_name,
        )
        results.append(res)

    delta_aucs = [r.mean_delta_auc for r in results]
    dynamic_aucs = [r.mean_dynamic_auc for r in results]
    baseline_aucs = [r.mean_baseline_auc for r in results]

    delta_range = max(delta_aucs) - min(delta_aucs) if delta_aucs else float("inf")
    robust = delta_range < 0.15

    if robust:
        note = (
            f"delta_AUC range = {delta_range:.3f} across {len(onset_thresholds)} "
            f"threshold values. Model is robust to onset_threshold selection."
        )
    else:
        note = (
            f"delta_AUC range = {delta_range:.3f} — consider reporting the "
            f"optimal threshold or using a default of 0.3 SD."
        )

    return {
        "onset_thresholds": onset_thresholds,
        "delta_aucs": [float(v) for v in delta_aucs],
        "dynamic_aucs": [float(v) for v in dynamic_aucs],
        "baseline_aucs": [float(v) for v in baseline_aucs],
        "robust": robust,
        "robust_range": (
            float(min(delta_aucs)) if delta_aucs else 0.0,
            float(max(delta_aucs)) if delta_aucs else 0.0,
        ),
        "note": note,
    }

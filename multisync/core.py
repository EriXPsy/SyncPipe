"""
Target: From raw data to Viewer-ready JSON.

    import multisync as ms

    # 1. Load and align data
    dyad = ms.Dyad(neural=df_neural, bio_hrv=df_hrv, behavioral=df_motion, hz=1.0)
    # 2. Add context labels
    dyad.add_context(start=0, end=300, label="Task")
    # 3. Analyze
    analyzer = ms.DynamicAnalyzer(window_size=10, surrogate_n=5000)
    results = analyzer.fit_transform(dyad)
    # 4. Export
    results.export_viewer_json("viewer_payload.json")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset import SynchronyDataset
from .dynamic_features import (
    DynamicFeatures,
    extract_dynamic_features,
    extract_features_all_pairs,
    extract_features_segmented,
    sliding_window_wcc,
)
from .feature_definitions import ONSET_THRESHOLD
from .prediction import FoldResult, PredictionResult, rolling_origin_cv
from .qc import DataQualityError, run_quality_check


class Dyad(SynchronyDataset):
    """
    User-friendly dyad container.

    Accepts modality DataFrames as keyword arguments.  Each keyword becomes
    the modality name.

    Parameters
    ----------
    hz : float
        Default target sampling rate for alignment.
    **modalities : DataFrame
        Modality name → DataFrame mapping.
    """

    def __init__(self, hz: float = 1.0, **modalities: pd.DataFrame) -> None:
        # Extract dyad_id if provided as a string; otherwise use default.
        # Always remove it from modalities to prevent add_modality() from
        # treating it as a DataFrame.
        dyad_id = modalities.pop("dyad_id", "dyad_01")
        if not isinstance(dyad_id, str):
            dyad_id = "dyad_01"
        super().__init__(dyad_id=dyad_id)
        self._default_hz = hz
        for name, df in modalities.items():
            self.add_modality(name, df)

    def align(self, target_hz: Optional[float] = None, **kwargs) -> "Dyad":
        hz = target_hz or self._default_hz
        super().align(target_hz=hz, **kwargs)
        return self

    def zscore(self, method: str = "standard", clip_sigma=None):
        return super().zscore(method=method, clip_sigma=clip_sigma)


# ---------------------------------------------------------------------------
# Analysis results container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResults:
    """Complete analysis output — ready for Viewer JSON export."""
    dyad_id: str
    # Dynamic features (global, per pair)
    dynamic_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Dynamic features (segmented by context)
    dynamic_features_segmented: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Threshold metadata — per-pair {"threshold": float, "mode": "within_dyad_surrogate"|"fixed", ...}
    threshold_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Prediction (nested by modality pair key, e.g. "neural__behavioral")
    prediction: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Context / Score view
    score_view: List[Dict[str, Any]] = field(default_factory=list)
    # Diagnostics — structured log of skipped/failed computations.
    # Each entry: {"stage": str, "pair": str, "reason": str, "detail": dict}
    # This replaces silent logger-only drops so the frontend can render
    # a "Data Exclusion Report" panel instead of showing empty results.
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            # schema_version tracks this JSON output STRUCTURE, not the
            # package version (1.0.0). Bump only when the serialized schema
            # below changes in a breaking way.
            "schema_version": "0.3.0",
            "dyad_id": self.dyad_id,
            "dynamic_features": self.dynamic_features,
            "dynamic_features_segmented": self.dynamic_features_segmented,
            "threshold_meta": self.threshold_meta,
            "prediction": self.prediction,
            "score_view": self.score_view,
            "diagnostics": self.diagnostics,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisResults":
        """Deserialize from a dict (inverse of to_dict)."""
        # Coercion helper — DRY replacement for 4 inline isinstance chains
        def _coerce(obj: Any, ty: type) -> Any:
            return obj if isinstance(obj, ty) else ty.from_dict(obj)

        def _coerce_dict(
            src: Dict[str, Any], ty: type
        ) -> Dict[str, Any]:
            return {
                k: _coerce(v, ty) for k, v in src.items()
                if isinstance(v, (dict, ty))
            }

        dyn_feat = _coerce_dict(d.get("dynamic_features", {}), DynamicFeatures)
        dyn_seg: Dict[str, Dict[str, DynamicFeatures]] = {}
        for label, pairs in d.get("dynamic_features_segmented", {}).items():
            dyn_seg[label] = _coerce_dict(pairs, DynamicFeatures)

        pred = _coerce_dict(d.get("prediction", {}), PredictionResult)

        return cls(
            dyad_id=d.get("dyad_id", "unknown"),
            dynamic_features=dyn_feat,
            dynamic_features_segmented=dyn_seg,
            prediction=pred,
            score_view=d.get("score_view", []),
            diagnostics=d.get("diagnostics", []),
            parameters=d.get("parameters", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResults":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

    def export_viewer_json(self, filepath: str) -> str:
        """
        Export viewer-ready JSON.

        This JSON is the single decoupling bridge between Python Core
        and React Viewer.  The Viewer do ZERO computation — all
        statistics, p-values, peaks, and graph edges are pre-computed.

        Schema (schema_version tracks this output STRUCTURE, not the
        package version):
        {
            "schema_version": "0.3.0",
            "dyad_id": "pair_01",
            "dynamic_features": {"behavior__neural": {...}},
            "prediction": {"neural_behavioral": {...}},
            "score_view": [{"start_sec": 0, "end_sec": 300,
                           "label": "Task", "mean_sync": 0.45}],
            "diagnostics": [{"stage": "prediction", "pair": "neural__behavioral",
                             "reason": "segment_too_short", "detail": {...}}]
        }
        """
        d = self.to_dict()

        # Replace NaN/Inf with None (JSON null) before serialization.
        # Also convert numpy scalars and arrays to native Python types so
        # json.dump never raises TypeError on np.float64 / np.ndarray.
        def _sanitize(obj: Any) -> Any:
            # Numpy scalars → native Python
            if isinstance(obj, (np.floating, np.complexfloating)):
                v = float(obj)
                return None if (np.isnan(v) or np.isinf(v)) else v
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return _sanitize(obj.tolist())
            if isinstance(obj, float):
                return None if (np.isnan(obj) or np.isinf(obj)) else obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        d = _sanitize(d)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return str(path)


# ---------------------------------------------------------------------------
# DynamicAnalyzer — the main workhorse
# ---------------------------------------------------------------------------

class DynamicAnalyzer:
    """
   The full analysis pipeline.

    Parameters
    ---------T-
    window_size : int
        Sliding window size in samples (for WCC and dynamic features).
    surrogate_n : int
        Number of IAAFT/PRTF surrogates for cascade significance testing.
    max_lag_sec : float
        Maximum cross-correlation lag in seconds.
    alpha : float
        Significance threshold for surrogate testing.
    seed : int
        Random seed for reproducibility.
    onset_threshold : float
        WCC threshold for first synchrony epoch onset detection.
    prediction_window : int
        Window size for prediction features (in samples).
    prediction_horizon : int
        Horizon for prediction labels (in samples).
    prediction_gap : int
        Gap between train and test in prediction CV (in samples).
    enable_prediction : bool, default True
        If False, skip the prediction CV stages (step 4 + step 5 of
        ``fit_transform``). Use this for descriptive-only workflows
        (surrogate controls, dose-response checks, trial-level slope
        tests) where the rolling-origin CV + LogisticRegression cost
        is wasted compute. Static / dynamic results are unaffected;
        ``results.prediction`` will be an empty dict.
    threshold_mode : {"within_dyad", "fixed"}, default "within_dyad"
        ``"within_dyad"`` computes a per-pair signal-level IAAFT threshold
        for descriptive/existence workflows. ``"fixed"`` uses
        ``onset_threshold`` (or the default 0.5).  Session-pooled thresholds
        are intentionally handled by ``BatchComputationPipeline`` because one
        ``DynamicAnalyzer`` instance only sees one dyad.
    run_qc : bool, default True
        Run the 3-stage quality gate before feature extraction.
    qc_raise_on_fail : bool, default True
        Raise ``DataQualityError`` on QC FAIL. Set False only for exploratory
        inspection where a failed QC report should be exported instead of
        blocking analysis.
    """

    def __init__(
        self,
        window_size: int = 10,
        surrogate_n: int = 5000,
        max_lag_sec: float = 30.0,
        alpha: float = 0.05,
        seed: int = 42,
        onset_threshold: Optional[float] = None,
        prediction_window: int = 10,
        prediction_horizon: int = 5,
        prediction_gap: int = 5,
        enable_prediction: bool = True,
        threshold_mode: str = "within_dyad",
        run_qc: bool = True,
        qc_raise_on_fail: bool = True,
        qc_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.window_size = window_size
        self.surrogate_n = surrogate_n
        self.max_lag_sec = max_lag_sec
        self.alpha = alpha
        self.seed = seed
        self.enable_prediction = bool(enable_prediction)

        threshold_mode = str(threshold_mode).lower()
        if threshold_mode == "session_pooled":
            raise ValueError(
                "DynamicAnalyzer operates on one dyad and cannot compute a "
                "session-pooled threshold. Use BatchComputationPipeline "
                "(onset_threshold='session_pooled') for between-dyad/group "
                "comparability."
            )
        if threshold_mode not in ("within_dyad", "fixed"):
            raise ValueError(
                "threshold_mode must be 'within_dyad' or 'fixed' "
                f"for DynamicAnalyzer, got {threshold_mode!r}."
            )
        if onset_threshold is not None:
            threshold_mode = "fixed"

        self.threshold_mode = threshold_mode
        if threshold_mode == "fixed":
            self.onset_threshold = (
                float(onset_threshold) if onset_threshold is not None else ONSET_THRESHOLD
            )
            self._use_surrogate_threshold = False
        else:
            # Within-dyad/per-pair signal-level IAAFT threshold. This is for
            # descriptive/existence workflows; group-comparable episode features
            # should be computed with BatchComputationPipeline's pooled threshold.
            self.onset_threshold = None
            self._use_surrogate_threshold = True

        self.run_qc = bool(run_qc)
        self.qc_raise_on_fail = bool(qc_raise_on_fail)
        self.qc_config = qc_config
        self.prediction_window = prediction_window
        self.prediction_horizon = prediction_horizon
        self.prediction_gap = prediction_gap

    
    def _iter_pairs(self, dataset):
        """
        Generate (src_key, name_a, name_b, col_a, col_b, x, y)
        for all valid modality+feature column pairs.
        """
        names = dataset.modality_names
        feat_cols = dataset.feature_columns
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue
                        src_key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                        yield src_key, name_a, name_b, col_a, col_b, x, y

    def fit_transform(self, dataset: SynchronyDataset) -> AnalysisResults:
        """
        Run the complete analysis pipeline on an aligned+normalized dataset.

        Steps:
        1. WCC + 6 dynamic features for each modality pair (global).
        2. WCC + 6 dynamic features segmented by context/conditions (if contexts exist).
        3. Cascade analysis (CCF + IAAFT/PRTF surrogate testing).
        4. Prediction window analysis (Rolling Origin CV with dynamic features).
        5. Cross-modal prediction (if 3+ modalities: source pair → target pair).
        6. Score view (context-based synchrony summaries).
        7. Package everything into AnalysisResults.

        Parameters
        ----------
        dataset : SynchronyDataset
            Must already be aligned and normalized.

        Returns
        -------
        AnalysisResults — viewer-ready output.
        """
        if not dataset._aligned:
            raise ValueError("Dataset must be aligned. Call dataset.align() first.")
        if not dataset._normalized:
            raise ValueError(
                "Dataset must be Z-score normalized. Call dataset.zscore() first."
            )

        hz = dataset.target_hz
        wcc_window_sec = self.window_size / hz if hz > 0 else 1.0

        qc_report = None
        if self.run_qc:
            qc_report = run_quality_check(
                dataset,
                config=self.qc_config,
                raise_on_fail=False,
            )
            if not qc_report.passed and self.qc_raise_on_fail:
                raise DataQualityError(qc_report.summary())

        # Determine threshold mode for parameter reporting.  A DynamicAnalyzer
        # threshold is either within-dyad/per-pair surrogate-derived or fixed;
        # session-pooled thresholds live in BatchComputationPipeline.
        _thr_mode = "within_dyad_surrogate" if self._use_surrogate_threshold else "fixed"
        _thr_value = self.onset_threshold  # None if within-dyad surrogate, float if fixed

        results = AnalysisResults(
            dyad_id=dataset.dyad_id,
            parameters={
                "window_size": self.window_size,
                "surrogate_n": self.surrogate_n,
                "max_lag_sec": self.max_lag_sec,
                "alpha": self.alpha,
                "seed": self.seed,
                "onset_threshold": _thr_value,
                "onset_threshold_mode": _thr_mode,
                "threshold_scope": self.threshold_mode,
                "prediction_window": self.prediction_window,
                "prediction_horizon": self.prediction_horizon,
                "prediction_gap": self.prediction_gap,
                "hz": hz,
                "qc": qc_report.to_dict() if qc_report is not None else None,
            },
        )
        if qc_report is not None and qc_report.overall_verdict != "PASS":
            results.diagnostics.append({
                "stage": "qc",
                "pair": "all",
                "reason": qc_report.overall_verdict,
                "detail": qc_report.to_dict(),
            })

        # 1. Dynamic features (global)
        feat_dict, threshold_meta = extract_features_all_pairs(
            dataset,
            window_size=self.window_size,
            hz=hz,
            onset_threshold=self.onset_threshold,
            wcc_window_sec=wcc_window_sec,
            use_surrogate_threshold=self._use_surrogate_threshold,
            surrogate_n=self.surrogate_n,
            surrogate_seed=self.seed,
        )
        results.dynamic_features = {k: v.to_dict() for k, v in feat_dict.items()}
        results.threshold_meta = threshold_meta

        # 2. Dynamic features (context-segmented)
        if dataset.context_labels:
            seg_dict, seg_threshold_meta = extract_features_segmented(
                dataset,
                window_size=self.window_size,
                hz=hz,
                onset_threshold=self.onset_threshold,
                wcc_window_sec=wcc_window_sec,
                use_surrogate_threshold=self._use_surrogate_threshold,
                surrogate_n=self.surrogate_n,
                surrogate_seed=self.seed,
            )
            results.dynamic_features_segmented = {
                label: {pair: feat.to_dict() for pair, feat in pairs.items()}
                for label, pairs in seg_dict.items()
            }

        # 3. Prediction window analysis (dynamic features, not raw WCC)
        # Use a larger window for feature extraction (need enough data
        # within each window to compute meaningful dynamic features).
        #
        # NOTE on ``enable_prediction``: when False, we still build
        # ``wcc_cache`` (needed by Step 6 score view) but skip the
        # rolling-origin CV + LogisticRegression calls, which dominate
        # per-pair runtime. Step 5 (cross-modal prediction) is also
        # skipped entirely when disabled.
        pred_window = max(self.prediction_window, 30)

        names = dataset.modality_names

        # Cache WCC sequences for score view (#6) and reuse
        wcc_cache: Dict[str, np.ndarray] = {}
        for src_key, name_a, name_b, col_a, col_b, x, y in self._iter_pairs(dataset):
            wcc = sliding_window_wcc(
                x, y, self.window_size, hz
            )
            wcc_cache[src_key] = wcc

            if not self.enable_prediction:
                continue

            pred_thr = threshold_meta.get(src_key, {}).get("threshold", 0.5) if self._use_surrogate_threshold else (self.onset_threshold or 0.5)

            pred = rolling_origin_cv(
                wcc,
                window_size=pred_window,
                hz=hz,
                n_splits=5,
                gap=max(self.prediction_gap, pred_window // 4),
                pair_name=src_key,
                mode="intra",
                onset_threshold=pred_thr,
            )
            # Record result even if no folds (warning shown in diagnostics)
            pred_entry = {
                "modality_a": name_a,
                "modality_b": name_b,
                "mode": "intra",
                "mean_dynamic_auc": pred.mean_dynamic_auc,
                "mean_baseline_auc": pred.mean_baseline_auc,
                "mean_delta_auc": pred.mean_delta_auc,
                "feature_importance": pred.feature_importance,
                "warning": pred.warning,
                "n_features_used": pred.n_features_used,
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "dynamic_auc": f.dynamic_auc,
                        "baseline_auc": f.baseline_auc,
                        "delta_auc": f.delta_auc,
                    }
                    for f in pred.folds
                ],
            }
            results.prediction[src_key] = pred_entry
            if not pred.folds and pred.warning:
                results.diagnostics.append({
                    "stage": "prediction",
                    "pair": src_key,
                    "reason": pred.warning,
                    "detail": {"n_folds": 0},
                })

        # 5. Score view (context-based synchrony summaries)
        if dataset.context_labels:
            t_vec = dataset.time_vector()
            wcc_offset = (self.window_size - 1) / (2.0 * hz)
            for ctx in dataset.context_labels:
                mask = (t_vec >= ctx.start_sec) & (t_vec < ctx.end_sec)
                if not mask.any():
                    continue
                # Map the time-based mask to WCC indices.
                wcc_indices = np.where(mask)[0]
                local_sync_vals = []
                for key, wcc in wcc_cache.items():
                    # Only use WCC indices that fall within valid WCC range
                    valid_idx = wcc_indices[
                        (wcc_indices >= 0) & (wcc_indices < len(wcc))
                    ]
                    if len(valid_idx) == 0:
                        continue
                    local_wcc = wcc[valid_idx]
                    local_mean = np.nanmean(local_wcc)
                    if not np.isnan(local_mean):
                        local_sync_vals.append(local_mean)
                mean_sync = float(np.mean(local_sync_vals)) if local_sync_vals else 0.0
                results.score_view.append({
                    "start_sec": ctx.start_sec,
                    "end_sec": ctx.end_sec,
                    "label": ctx.label,
                    "score": ctx.score,
                    "mean_sync": mean_sync,
                })

        return results

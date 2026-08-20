"""Turn loaded study records into checked inputs for calculation and testing.

This module keeps the preparation details out of the user-facing runner. It
builds the feature table, raw-pair views, comparison-pair views, shared masks,
and preparation diagnostics from one checked cohort.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .computation_pipeline import ComputationPipeline
from .feature_definitions import FDR_FEATURES, ONSET_THRESHOLD, REFERENCE_FEATURE
from .qc import DEFAULT_CONFIG as _QC_CONFIG
from .preparation import PreparedCohort, PreparedObservation, PreparationExclusion
from .session_threshold import compute_session_pooled_thresholds_by_modality


@dataclass
class InferenceInputs:
    """Assembled inputs for :class:`syncpipe.inference_pipeline.InferencePipeline`."""

    features_df: pd.DataFrame
    raw_signals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    design_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]
    discontinuity_mask: Optional[Dict[str, np.ndarray]] = None
    wcc_traces: Optional[Dict[str, np.ndarray]] = None
    condition_col: str = "condition"
    dyad_col: str = "dyad_id"
    thresholds_by_modality: Optional[Dict[str, float]] = None
    prepared_cohort: Optional[PreparedCohort] = None
    preparation_diagnostics: Optional[Dict[str, Dict[str, object]]] = None


def _as_array(df_or_series) -> Optional[np.ndarray]:
    """Extract a 1-D float array from a DataFrame column or Series."""
    if df_or_series is None:
        return None
    if isinstance(df_or_series, pd.DataFrame):
        # Never guess between time and multiple signal columns.
        if "value" in df_or_series.columns:
            signal_col = "value"
        else:
            candidates = [
                c for c in df_or_series.select_dtypes(include=[np.number]).columns
                if c not in {"time", "timestamp"}
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Signal DataFrame must contain a numeric 'value' column "
                    "or exactly one numeric non-time signal column; "
                    f"candidates={candidates}."
                )
            signal_col = candidates[0]
        arr = df_or_series[signal_col].to_numpy(dtype=float)
    elif isinstance(df_or_series, pd.Series):
        arr = df_or_series.to_numpy(dtype=float)
    else:
        arr = np.asarray(df_or_series, dtype=float)
    return arr


def records_to_inference_inputs(
    records: Sequence[Any],
    *,
    hz: float,
    window_size: int,
    onset_threshold: Union[float, str] = "session_pooled",
    design_condition: Optional[str] = None,
    condition_col: str = "condition",
    dyad_col: str = "dyad_id",
    feature_cols: Optional[Sequence[str]] = None,
    window_type: str = "rect",
    initial_exclusions: Sequence[PreparationExclusion] = (),
) -> InferenceInputs:
    """Convert loader records into the three-pipeline-ready inputs.

    Parameters
    ----------
    records : sequence of loader records
        Each record must expose: ``dyad_label`` (str), ``modality`` (str),
        ``condition`` (str), ``person_a`` / ``person_b`` (DataFrame with a
        numeric signal column, or 1-D array), ``target_hz`` (float),
        ``incomplete`` (bool, optional).  ``LeriqueDyadCondition`` satisfies
        this contract exactly.
    hz, window_size :
        Forwarded to :class:`ComputationPipeline`.
    onset_threshold : float or str
        Onset threshold strategy. ``"session_pooled"`` (default) computes one
        IAAFT surrogate threshold *per modality* across the whole dataset, so
        every dyad of a modality shares a threshold calibrated to that
        modality's null distribution (slow/smooth EDA vs fast/spiky ECG get
        different thresholds). A fixed ``float`` is forwarded unchanged
        (sensitivity analysis / paper reproduction). The resolved per-modality
        thresholds are returned in ``InferenceInputs.thresholds_by_modality``.
    window_type : str
        WCC window taper (``'rect'`` default, or ``'hann'``/``'hamming'``/
        ``'triang'``/``'gaussian'``).  Forwarded to :class:`ComputationPipeline`.
    design_condition : str, optional
        Which condition's signal pair is used for the pseudo-pair / time-shift
        design controls (the condition whose coupling you want to challenge).
        If ``None``, the *last* condition seen per (dyad, modality) is used.
    condition_col, dyad_col :
        Column names written into ``features_df`` so they match the
        InferencePipeline's default ``condition_col`` / ``dyad_col``.
    feature_cols : sequence of str, optional
        Subset of features to keep in ``features_df``.  Defaults to the FDR
        family + reference feature.

    Returns
    -------
    InferenceInputs
        ``features_df``, ``raw_signals``, ``design_pairs`` + the column names.
    """
    if feature_cols is None:
        feature_cols = list(FDR_FEATURES) + list(REFERENCE_FEATURE)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError(f"hz must be finite and positive, got {hz!r}")
    if int(window_size) != window_size or window_size < 2:
        raise ValueError(f"window_size must be an integer >= 2, got {window_size!r}")
    feature_cols = list(feature_cols)
    if isinstance(onset_threshold, str) and onset_threshold != "session_pooled":
        raise ValueError(
            "onset_threshold string must be 'session_pooled' or a numeric value."
        )
    if not isinstance(onset_threshold, str):
        threshold_value = float(onset_threshold)
        if not np.isfinite(threshold_value) or not -1.0 <= threshold_value <= 1.0:
            raise ValueError("numeric onset_threshold must be finite and lie in [-1, 1]")

    # ------------------------------------------------------------------
    # Pass 1: parse + collect usable entries (skip incomplete / too-short).
    # We collect first so the canonical "session_pooled" onset threshold can
    # be computed per-modality across the WHOLE dataset before any per-record
    # computation pipeline runs.
    # ------------------------------------------------------------------
    entries: List[Dict[str, Any]] = []
    exclusions: List[PreparationExclusion] = list(initial_exclusions)

    def _exclude(rec, code: str, detail: str, stage: str = "preparation") -> None:
        dyad = str(getattr(rec, "dyad_label", "unknown"))
        mod = str(getattr(rec, "modality", "unknown"))
        cond = str(getattr(rec, "condition", "unknown"))
        exclusions.append(PreparationExclusion(
            key=f"{dyad}__{mod}__{cond}", dyad_id=dyad, modality=mod,
            condition=cond, code=code, stage=stage, detail=detail,
        ))

    for rec in records:
        if getattr(rec, "incomplete", False):
            _exclude(rec, "incomplete_record", "loader marked record incomplete")
            continue
        a = _as_array(getattr(rec, "person_a", None))
        b = _as_array(getattr(rec, "person_b", None))
        if a is None or b is None:
            _exclude(rec, "missing_signal", "person_a or person_b signal is missing")
            continue
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
            raise ValueError(
                "Each record must contain two one-dimensional, equal-length "
                f"signals; got {getattr(a, 'shape', None)} and {getattr(b, 'shape', None)}."
            )
        if a.size < window_size:
            _exclude(
                rec, "signal_too_short",
                f"{a.size} samples cannot form window_size={window_size}",
            )
            continue
        dyad = str(rec.dyad_label)
        mod = str(rec.modality)
        cond = str(rec.condition)
        key = f"{dyad}__{mod}__{cond}"
        if any(e["key"] == key for e in entries):
            raise ValueError(f"Duplicate record key {key!r}; expected one record per dyad/modality/condition.")
        rec_hz = float(getattr(rec, "target_hz", hz))
        if not np.isfinite(rec_hz) or rec_hz <= 0 or not np.isclose(rec_hz, hz):
            raise ValueError(
                f"Record {key!r} target_hz={rec_hz} does not match bridge hz={hz}. "
                "Resample explicitly before pooling thresholds."
            )
        rec_mask = getattr(rec, "discontinuity_mask", None)
        mask = np.asarray(rec_mask, dtype=bool) if rec_mask is not None else None
        if mask is not None and (mask.ndim != 1 or mask.size != a.size):
            raise ValueError(
                f"Record {key!r} discontinuity_mask must have length {a.size}."
            )

        # Quality gate: mirror run_quality_check's nan_integrity (max_nan_rate)
        # and signal_integrity (min_signal_std) stages. The bridge works with
        # raw arrays, not a SynchronyDataset, so we inline the two checks that
        # would otherwise let dirty data produce corrupted features silently.
        # Records that would FAIL QC are skipped with a loud warning.
        _nan_limit = _QC_CONFIG["max_nan_rate"]
        _std_floor = _QC_CONFIG["min_signal_std"]
        _skip_reason = None
        _skip_code = None
        for _label, _sig in (("person_a", a), ("person_b", b)):
            _inf_count = int(np.isinf(_sig).sum())
            if _inf_count:
                raise ValueError(
                    f"Record {key!r} {_label} contains {_inf_count} infinite "
                    "value(s). Replace invalid infinities during preprocessing; "
                    "only explicitly governed NaN gaps are supported."
                )
            _nan_rate = float(np.isnan(_sig).mean())
            if _nan_rate > _nan_limit:
                _skip_code = "high_nan_rate"
                _skip_reason = (
                    f"{_label} NaN rate {_nan_rate:.1%} (>{_nan_limit:.0%}, "
                    "nan_integrity FAIL)"
                )
                break
            _finite = _sig[np.isfinite(_sig)]
            if _finite.size >= 2 and float(np.std(_finite)) < _std_floor:
                _skip_code = "near_zero_variance"
                _skip_reason = (
                    f"{_label} near-zero variance (std<{_std_floor}, "
                    "signal_integrity FAIL)"
                )
                break
        if _skip_reason is not None:
            _exclude(rec, _skip_code or "qc_failure", _skip_reason, stage="qc")
            warnings.warn(
                f"Record {key!r} skipped by bridge QC gate: {_skip_reason}. "
                "Fix sensor dropout/flatline before pooling, or exclude "
                "this record from the loader output.",
                UserWarning,
                stacklevel=2,
            )
            continue

        prepared = PreparedObservation.from_signals(
            key=key, dyad_id=dyad, modality=mod, condition=cond,
            hz=rec_hz, signal_a=a, signal_b=b,
            discontinuity_mask=mask,
        )
        entries.append({
            "observation": prepared,
            "a": prepared.signal_a,
            "b": prepared.signal_b,
            "dyad": dyad,
            "mod": mod,
            "cond": cond,
            "key": key,
            "rec_hz": rec_hz,
            # Legacy name retained at the API edge; this is now the shared
            # analysis mask (source discontinuities AND joint finite samples).
            "mask": prepared.geometry.analysis_mask,
        })

    if not entries:
        codes = sorted({item.code for item in exclusions})
        raise ValueError(
            "No usable records after preparation; exclusion codes="
            f"{codes}. Check loader filters, QC and durations."
        )

    prepared_cohort = PreparedCohort(
        tuple(e["observation"] for e in entries), tuple(exclusions)
    )

    # ------------------------------------------------------------------
    # Canonical default: per-modality pooled IAAFT onset threshold.
    # Every dyad of a modality shares one threshold, calibrated to that
    # modality's null (slow/smooth EDA vs fast/spiky ECG get different
    # thresholds). A fixed numeric onset_threshold is forwarded unchanged
    # (sensitivity analysis / paper reproduction).
    # ------------------------------------------------------------------
    thresholds_by_modality: Optional[Dict[str, float]] = None
    if isinstance(onset_threshold, str) and onset_threshold == "session_pooled":
        dyad_signals = [(e["a"], e["b"]) for e in entries]
        modalities = [e["mod"] for e in entries]
        masks = [e["mask"] for e in entries]
        thresholds_by_modality = compute_session_pooled_thresholds_by_modality(
            dyad_signals,
            modalities,
            hz=hz,
            wcc_window_size=window_size,
            discontinuity_masks=masks,
        )

    # ------------------------------------------------------------------
    # Pass 2: per-record computation pipeline. For "session_pooled" we apply
    # each record's own modality threshold; otherwise the caller's numeric
    # threshold is forwarded unchanged.
    # ------------------------------------------------------------------
    frames: List[pd.DataFrame] = []
    raw_signals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    discontinuity_mask: Dict[str, Optional[np.ndarray]] = {}
    design_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    wcc_traces: Dict[str, np.ndarray] = {}

    for e in entries:
        a, b = e["a"], e["b"]
        dyad, mod, cond, key = e["dyad"], e["mod"], e["cond"], e["key"]
        if thresholds_by_modality is not None:
            thr = thresholds_by_modality.get(mod, ONSET_THRESHOLD)
        else:
            thr = float(onset_threshold)

        # --- Stage 2: computation pipeline (load -> WCC -> features) ---
        pipe = ComputationPipeline(
            hz=e["rec_hz"],
            window_size=window_size,
            onset_threshold=thr,
            window_type=window_type,
        )
        pipe.load_signals(a, b, discontinuity_mask=e["mask"])
        pipe.compute_wcc()
        wcc_traces[key] = pipe._wcc
        pipe.extract_features()
        row = pipe.to_dataframe()
        row[dyad_col] = dyad
        row["modality"] = mod
        row[condition_col] = cond
        missing_features = [c for c in feature_cols if c not in row.columns]
        if missing_features:
            raise ValueError(f"Requested features not produced: {missing_features}")
        metadata_cols = [
            "n_signal_samples", "n_wcc_points", "n_valid_wcc_points",
            "valid_wcc_fraction", "wcc_observation_sec",
        ]
        keep = [dyad_col, "modality", condition_col] + metadata_cols + list(feature_cols)
        keep = [c for i, c in enumerate(keep) if c not in keep[:i]]
        frames.append(row[keep])

        # --- raw signals for the existence audit (every condition) ---
        raw_signals[key] = (a, b)

        # --- design pairs (one condition per dyad__modality) ---
        if design_condition is None or cond == design_condition:
            design_pairs[f"{dyad}__{mod}"] = (a, b)
        discontinuity_mask[key] = e["mask"]

    if design_condition is not None:
        expected = {(e["dyad"], e["mod"]) for e in entries}
        observed = {
            (e["dyad"], e["mod"])
            for e in entries
            if f"{e['dyad']}__{e['mod']}" in design_pairs
        }
        if expected - observed:
            raise ValueError(
                f"design_condition={design_condition!r} missing for dyad/modality "
                f"pairs: {sorted(expected - observed)}"
            )

    features_df = pd.concat(frames, ignore_index=True)
    return InferenceInputs(
        features_df=features_df,
        raw_signals=raw_signals,
        design_pairs=design_pairs,
        discontinuity_mask=discontinuity_mask or None,
        wcc_traces=wcc_traces or None,
        condition_col=condition_col,
        dyad_col=dyad_col,
        thresholds_by_modality=thresholds_by_modality,
        prepared_cohort=prepared_cohort,
        preparation_diagnostics=prepared_cohort.diagnostics(window_size),
    )

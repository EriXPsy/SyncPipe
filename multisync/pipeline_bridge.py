"""
Pipeline bridge — connect the data layer to the three SyncPipe pipelines.

The three pipelines

    feature_pipeline.py     (Pipeline 1: consult / select features)
    computation_pipeline.py (Pipeline 2: load -> WCC -> features -> DataFrame)
    inference_pipeline.py   (Pipeline 3: audited evidence chain)

are intentionally thin and decoupled.  This module is the *missing seam* a
reviewer needs: it turns a list of loader records (anything shaped like
``multisync.realtest.lerique_2024.LeriqueDyadCondition`` — i.e. with
``dyad_label``, ``modality``, ``condition``, ``person_a``/``person_b``
DataFrames, ``target_hz``, ``duration_sec``, ``incomplete``) into the exact
inputs the computation and inference pipelines expect:

    features_df  : one row per (dyad, modality, condition), with the columns
                   the InferencePipeline needs (``dyad_id``, ``modality``,
                   ``condition``) plus every extracted feature.
    raw_signals  : keyed ``"<dyad>__<modality>__<condition>"`` -> (sig_a, sig_b)
                   for the synchrony-existence audit.
    design_pairs : keyed ``"<dyad>__<modality>"`` -> (sig_a, sig_b) taken from
                   ``design_condition`` (the condition whose coupling you want
                   to audit against mismatched partners / shifted alignment).

Column-name contract (must match InferencePipeline defaults)
----------------------------------------------------------------
    features_df columns : dyad_id, modality, condition, <features...>
    condition_col        = "condition"
    dyad_col             = "dyad_id"
    (InferencePipeline default args use exactly these names, so the bridge
     emits them — no manual renaming required downstream.)

Usage
-----
    from multisync.pipeline_bridge import records_to_inference_inputs

    features_df, raw_signals, design_pairs = records_to_inference_inputs(
        records, hz=1.0, window_size=30, onset_threshold="session_pooled",
        design_condition="trials_concat",
    )

    pipe = InferencePipeline(features_df, hz=1.0, wcc_window_sec=30.0)
    chain = pipe.run_audited_evidence_chain(
        raw_signals=raw_signals, wcc_window_size=30,
        design_signal_pairs=design_pairs,
        condition_col="condition", dyad_col="dyad_id",
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .computation_pipeline import ComputationPipeline
from .feature_definitions import FDR_FEATURES, ONSET_THRESHOLD, REFERENCE_FEATURE
from .session_threshold import compute_session_pooled_thresholds_by_modality


@dataclass
class InferenceInputs:
    """Assembled inputs for :class:`multisync.inference_pipeline.InferencePipeline`."""

    features_df: pd.DataFrame
    raw_signals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    design_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]
    discontinuity_mask: Optional[Dict[str, np.ndarray]] = None
    condition_col: str = "condition"
    dyad_col: str = "dyad_id"
    thresholds_by_modality: Optional[Dict[str, float]] = None


def _as_array(df_or_series) -> Optional[np.ndarray]:
    """Extract a 1-D float array from a DataFrame column or Series."""
    if df_or_series is None:
        return None
    if isinstance(df_or_series, pd.DataFrame):
        # Prefer a column literally named "value"; else the first numeric col.
        if "value" in df_or_series.columns:
            arr = df_or_series["value"].to_numpy(dtype=float)
        else:
            num = df_or_series.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                return None
            arr = num.iloc[:, 0].to_numpy(dtype=float)
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

    # ------------------------------------------------------------------
    # Pass 1: parse + collect usable entries (skip incomplete / too-short).
    # We collect first so the canonical "session_pooled" onset threshold can
    # be computed per-modality across the WHOLE dataset before any per-record
    # computation pipeline runs.
    # ------------------------------------------------------------------
    entries: List[Dict[str, Any]] = []
    for rec in records:
        if getattr(rec, "incomplete", False):
            continue
        a = _as_array(getattr(rec, "person_a", None))
        b = _as_array(getattr(rec, "person_b", None))
        if a is None or b is None or a.size < window_size or b.size < window_size:
            continue
        dyad = str(rec.dyad_label)
        mod = str(rec.modality)
        cond = str(rec.condition)
        key = f"{dyad}__{mod}__{cond}"
        rec_mask = getattr(rec, "discontinuity_mask", None)
        entries.append({
            "a": a.astype(float),
            "b": b.astype(float),
            "dyad": dyad,
            "mod": mod,
            "cond": cond,
            "key": key,
            "rec_hz": float(getattr(rec, "target_hz", hz)),
            "mask": np.asarray(rec_mask, dtype=bool) if rec_mask is not None else None,
        })

    if not entries:
        raise ValueError(
            "No usable records: every record was incomplete, missing a person, "
            "or shorter than window_size. Check the loader filters / durations."
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
        pipe.extract_features()
        row = pipe.to_dataframe()
        row[dyad_col] = dyad
        row["modality"] = mod
        row[condition_col] = cond
        # Keep only requested features + the join keys (drop the verbose
        # duplicate metadata that load_signals() may have stored).
        keep = [dyad_col, "modality", condition_col] + [
            c for c in feature_cols if c in row.columns
        ]
        keep = [c for i, c in enumerate(keep) if c not in keep[:i]]  # dedupe
        frames.append(row[keep])

        # --- raw signals for the existence audit (every condition) ---
        raw_signals[key] = (a, b)

        # --- design pairs (one condition per dyad__modality) ---
        if design_condition is None or cond == design_condition:
            design_pairs[f"{dyad}__{mod}"] = (a, b)
        discontinuity_mask[key] = e["mask"]

    features_df = pd.concat(frames, ignore_index=True)
    return InferenceInputs(
        features_df=features_df,
        raw_signals=raw_signals,
        design_pairs=design_pairs,
        discontinuity_mask=discontinuity_mask or None,
        condition_col=condition_col,
        dyad_col=dyad_col,
        thresholds_by_modality=thresholds_by_modality,
    )

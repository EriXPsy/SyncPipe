"""
Dynamic feature extraction — operationalizing synchrony dynamics.

Theoretical Framework
--------------------
SyncPipe conceptualizes interpersonal synchrony through a
morphology-aware dimensional model (INTENSITY / STRUCTURE / TIMING;
see ``docs/FEATURE_TABLE.md``).  Synchrony Epochs can take
multiple forms — single-peak, oscillatory, sustained, asymmetric decay
and features are classified into three tiers (CORE /
CONDITIONAL / REFERENCE) reflecting cross-morphology robustness.

Compatibility facade + cross-pair orchestration.  Feature math lives in
:mod:`syncpipe.feature_definitions` (SSoT); WCC computation lives in
:mod:`syncpipe.wcc`, and the surrogate / null models in
:mod:`syncpipe.null_models`.  This module re-exports those objects for
backward compatibility and provides the cross-pair orchestration entry
points :func:`extract_features_all_pairs` / :func:`extract_features_segmented`,
which delegate to the SSoT.

Attribution
----------
Some features are **inspired by** the theoretical framework proposed in Gordon, I., Tomashin, A., & Mayo, O. (2024). A Theory of Flexible
Multimodal Synchrony. *Psychological Review*, 132(3), 680–718. https://doi.org/10.1037/rev0000495

References
----------
Bassett, D. S., Wymbs, N. F., Porter, M. A., et al. (2011). Dynamic reconfiguration of human brain networks during learning. *PNAS*, 108(18), 7641–7646.
Benedek, M., & Kaernbach, C. (2010). A continuous measure of phasic electrodermal activity. *Journal of Neuroscience Methods*, 190(1), 80–91.
Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
Gordon, I., Tomashin, A., & Mayo, O. (2025). A theory of flexible multimodal synchrony. *Psychological Review*, 132(3), 680–718.
Kelso, J. A. S. (1995). *Dynamic Patterns*. MIT Press.
Luck, S. J. (2014). *An Introduction to the Event-Related Potential Technique* (2nd ed.). MIT Press.
Schreiber, T., & Schmitz, A. (2000). Surrogate time series. *Physica D*, 142(3-4), 346–382.
Tognoli, E., & Kelso, J. A. S. (2014). The metastable brain. *Neuron*, 81(1), 35–48.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .dataset import SynchronyDataset

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal.windows import get_window
import logging
import warnings

from .surrogate import iaaft_surrogate, ft_surrogate, prtf_surrogate  # noqa: F401  # re-export
from .preparation import resolve_signal_geometry

# Feature math lives in feature_definitions (SSoT); re-export DynamicFeatures
# for backward-compatible imports (syncpipe.DynamicFeatures, core.py, etc.)
from .feature_definitions import (
    DynamicFeatures,
    extract_features as _ssot_extract_features,
    compute_surrogate_threshold,
    ONSET_THRESHOLD,
    SURROGATE_THRESHOLD_PERCENTILE,
    SWITCHING_HYSTERESIS_DELTA,
)

logger = logging.getLogger(__name__)

from .wcc import (
    _make_window_kernel,
    _sliding_window_wcc_cumsum,
    _sliding_window_wcc_stride,
    sliding_window_wcc,
    _apply_discontinuity_mask,
    sliding_window_wcc_masked,
)
from .null_models import (SURROGATE_MEM_GUARD_BYTES, _NULL_MODEL_L0, _NULL_MODEL_L1, _empty_result, _prepare_iaaft_segments, _segmentwise_wcc, _signal_level_surrogate_test, _wcc_level_surrogate_test, wcc_surrogate_test, compute_surrogate_threshold_from_signals)

# Onset threshold — locked in feature_definitions (DECISION-01)
# ---------------------------------------------------------------------------

from .dynamic_feature_extraction import extract_dynamic_features


from .pairing import pairing_policy, iter_dyad_pairs


def extract_features_all_pairs(
    dataset: "SynchronyDataset",  # noqa: F821
    window_size: int = 10,
    hz: float = 1.0,
    onset_threshold: Optional[float] = None,
    onset_k: float = 2.0,
    wcc_window_sec: float = 1.0,
    use_surrogate_threshold: bool = True,
    surrogate_n: int = 200,
    surrogate_seed: int = 42,
    discontinuity_mask: Optional[np.ndarray] = None,
    cross_modal: bool = False,
) -> Tuple[Dict[str, DynamicFeatures], Dict[str, Dict[str, Any]]]:
    """
    Compute WCC + dynamic features for all modality pairs.

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned and normalized.
    window_size : int
        WCC window size in samples.
    hz : float
        Sampling rate.
    onset_threshold : float or None
        Explicit WCC threshold. When ``None`` (default) and
        ``use_surrogate_threshold=True``, a per-dyad IAAFT surrogate-derived
        threshold (95th percentile) is computed automatically. When
        ``use_surrogate_threshold=False``, falls back to ``ONSET_THRESHOLD``
        (0.5) for sensitivity analysis.
    onset_k : float
        DEPRECATED since v1.0.0. Ignored; included for signature compat.
    wcc_window_sec : float
        WCC window duration in seconds (DECISION-02). Default 1.0.
    use_surrogate_threshold : bool
        If True (default), compute per-dyad surrogate-derived threshold.
        Set to False for fixed-threshold sensitivity analysis.
    surrogate_n : int
        Number of IAAFT surrogates for threshold computation (default 200).
    surrogate_seed : int
        RNG seed for surrogate threshold reproducibility.
    discontinuity_mask : np.ndarray or None
        Per-sample validity mask (True = internal to a segment). Windows
        spanning a discontinuity are set NaN in the WCC so features skip
        them. If None, read from ``dataset.discontinuity_mask``.

    Returns
    -------
    Tuple[Dict[str, DynamicFeatures], Dict[str, Dict[str, Any]]]
        ``(features, threshold_meta)`` where ``threshold_meta`` maps each
        pair key to ``{"threshold": float,
        "mode": "within_dyad_surrogate"|"fixed",
        "is_surrogate_derived": bool}``.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names
    results: Dict[str, DynamicFeatures] = {}
    threshold_meta: Dict[str, Dict[str, Any]] = {}
    dm = discontinuity_mask if discontinuity_mask is not None else getattr(
        dataset, "discontinuity_mask", None
    )

    for src_key, name_a, name_b, col_a, col_b, x, y in iter_dyad_pairs(
        dataset, cross_modal=cross_modal
    ):
        key = src_key

        # --- Resolve threshold ---
        if use_surrogate_threshold:
            thr, is_surr = compute_surrogate_threshold_from_signals(
                x, y,
                hz=hz,
                wcc_window_size=window_size,
                surrogate_n=surrogate_n,
                seed=surrogate_seed,
                discontinuity_mask=dm,
            )
            threshold_meta[key] = {
                "threshold": thr,
                "mode": "within_dyad_surrogate",
                "scope": "within_dyad",
                "is_surrogate_derived": is_surr,
                "surrogate_n": surrogate_n,
                "surrogate_percentile": SURROGATE_THRESHOLD_PERCENTILE,
            }
        else:
            thr = (
                ONSET_THRESHOLD
                if onset_threshold is None
                else float(onset_threshold)
            )
            threshold_meta[key] = {
                "threshold": thr,
                "mode": "fixed",
                "scope": "fixed",
                "is_surrogate_derived": False,
            }

        wcc = sliding_window_wcc_masked(
            x, y, window_size, hz, discontinuity_mask=dm
        )
        feat = extract_dynamic_features(
            wcc, hz, thr, onset_k,
            wcc_window_sec=wcc_window_sec,
            gap_policy="segment" if dm is not None else None,
        )
        results[key] = feat

    return results, threshold_meta


def extract_features_segmented(
    dataset: "SynchronyDataset",  # noqa: F821
    window_size: int = 10,
    hz: float = 1.0,
    onset_threshold: Optional[float] = None,
    onset_k: float = 2.0,
    max_nan_ratio: float = 0.2,
    wcc_window_sec: float = 1.0,
    use_surrogate_threshold: bool = True,
    surrogate_n: int = 200,
    surrogate_seed: int = 42,
    discontinuity_mask: Optional[np.ndarray] = None,
    cross_modal: bool = False,
) -> Tuple[Dict[str, Dict[str, DynamicFeatures]], Dict[str, Dict[str, Any]]]:
    """
    Compute WCC + dynamic features per CONTEXT segment.

    Surrogate-derived thresholds are computed once per dyad from
    full-length raw signals, then shared across all context segments
    (cross-condition comparability; see docs/METHOD_LOG.md).

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned, normalized, and have context_labels set.
    window_size : int
        WCC window size in samples.
    hz : float
        Sampling rate.
    onset_threshold : float or None
        Explicit threshold override. When None and
        ``use_surrogate_threshold=True``, per-dyad IAAFT surrogate-derived
        thresholds are computed automatically. When
        ``use_surrogate_threshold=False``, falls back to
        ``ONSET_THRESHOLD`` (0.5).
    onset_k : float
        DEPRECATED since v1.0.0. Ignored; signature compat.
    max_nan_ratio : float
        Maximum NaN fraction in a segment pair. Default 0.2.
    wcc_window_sec : float
        WCC window duration in seconds (DECISION-02). Default 1.0.
    use_surrogate_threshold : bool
        If True (default), compute per-dyad surrogate-derived threshold
        from full-length signals and share across all segments.
    surrogate_n : int
        Number of IAAFT surrogates (default 200).
    surrogate_seed : int
        RNG seed for threshold reproducibility.
    discontinuity_mask : np.ndarray or None
        Per-sample validity mask (True = internal to a segment). Windows
        spanning a discontinuity are set NaN in the WCC so features skip
        them. If None, read from ``dataset.discontinuity_mask``.

    Returns
    -------
    Tuple[Dict, Dict]
        ``(segmented_features, threshold_meta)``.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names
    t_vec = dataset.time_vector()
    dm = discontinuity_mask if discontinuity_mask is not None else getattr(
        dataset, "discontinuity_mask", None
    )

    # --- Pre-compute per-dyad thresholds from full-length signals ---
    dyad_thresholds: Dict[str, float] = {}
    threshold_meta: Dict[str, Dict[str, Any]] = {}

    for src_key, name_a, name_b, col_a, col_b, x, y in iter_dyad_pairs(
        dataset, cross_modal=cross_modal
    ):
        key = src_key
        if use_surrogate_threshold:
            thr, is_surr = compute_surrogate_threshold_from_signals(
                x, y,
                hz=hz,
                wcc_window_size=window_size,
                surrogate_n=surrogate_n,
                seed=surrogate_seed,
                discontinuity_mask=dm,
            )
            dyad_thresholds[key] = thr
            threshold_meta[key] = {
                "threshold": thr,
                "mode": "within_dyad_surrogate",
                "scope": "within_dyad",
                "is_surrogate_derived": is_surr,
                "surrogate_n": surrogate_n,
                "surrogate_percentile": SURROGATE_THRESHOLD_PERCENTILE,
            }
        else:
            thr = (
                ONSET_THRESHOLD
                if onset_threshold is None
                else float(onset_threshold)
            )
            dyad_thresholds[key] = thr
            threshold_meta[key] = {
                "threshold": thr,
                "mode": "fixed",
                "scope": "fixed",
                "is_surrogate_derived": False,
            }

    segments: List[Tuple[str, float, float]] = []
    if dataset.context_labels:
        for ctx in dataset.context_labels:
            segments.append((ctx.label, ctx.start_sec, ctx.end_sec))
    else:
        if len(t_vec) > 0:
            segments.append(("full", t_vec[0], t_vec[-1]))

    if not segments:
        return {}, threshold_meta

    results: Dict[str, Dict[str, DynamicFeatures]] = {}

    for label, start_sec, end_sec in segments:
        mask = (t_vec >= start_sec) & (t_vec < end_sec)
        min_seg_len = 3 * window_size
        if mask.sum() < min_seg_len:
            logger.warning(
                "Context '%s': segment too short (%d samples < %d = 3×window_size). "
                "Skipping.",
                label, int(mask.sum()), min_seg_len,
            )
            results[label] = {}
            continue

        seg_results: Dict[str, DynamicFeatures] = {}
        for src_key, name_a, name_b, col_a, col_b, x, y in iter_dyad_pairs(
            dataset, cross_modal=cross_modal
        ):
            key = src_key

            x_seg = x[mask]
            y_seg = y[mask]
            # Segment slice of the signal-resolution mask.
            seg_dm = dm[mask] if dm is not None else None

            valid_ratio = (
                ~np.isnan(x_seg) & ~np.isnan(y_seg)
            ).sum() / len(x_seg)
            if valid_ratio < (1.0 - max_nan_ratio):
                continue

            wcc = sliding_window_wcc_masked(
                x_seg, y_seg, window_size, hz, discontinuity_mask=seg_dm
            )
            if len(wcc) < 5:
                continue

            thr = dyad_thresholds.get(key, ONSET_THRESHOLD)
            feat = extract_dynamic_features(
                wcc, hz, thr, onset_k, max_nan_ratio,
                wcc_window_sec=wcc_window_sec,
                gap_policy="segment" if seg_dm is not None else None,
            )
            seg_results[key] = feat

        results[label] = seg_results

    return results, threshold_meta

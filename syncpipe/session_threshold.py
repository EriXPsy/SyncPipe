"""
Session-level pooled surrogate thresholding.

Provides a single threshold shared across all dyads in a session (or condition),
addressing the cross-dyad comparability problem of per-dyad surrogate thresholds.

**Canonical default — per-modality pooled.** :func:`compute_session_pooled_thresholds_by_modality`
pools the surrogate null *within* each modality and returns one threshold per
modality; this is the normative v1 onset-threshold default (cross-modal
comparability preserved while each modality's threshold is calibrated to its own
null).  The session/condition helpers below are OPTIONAL granularities.

Rationale
---------
Per-dyad thresholds adapt to each dyad's own null distribution. This is useful for
within-dyad inference, but makes between-dyad feature values (e.g. dwell_time,
switching_rate) hard to compare because the "episode" definition differs per dyad.

A session-level pooled threshold:
1. generates surrogates for every dyad in the session,
2. pools all finite surrogate coupling values across dyads and replicates,
3. returns a single percentile-based threshold.

This preserves the null-hypothesis grounding of surrogate thresholds while giving
all dyads the same threshold, making group-level comparisons meaningful.

Two modes are supported:
- ``session`` : one threshold for the whole session (default for cross-condition
  comparability; Task A in the surrogate threshold design docs).
- ``condition`` : one threshold per condition, computed by pooling only the
  dyads/segments belonging to that condition (sensitivity analysis).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .dynamic_features import sliding_window_wcc, _apply_discontinuity_mask
from .feature_definitions import (
    compute_surrogate_threshold,
    ONSET_THRESHOLD,
    SURROGATE_THRESHOLD_MAX,
)
from .surrogate import iaaft_surrogate, ft_surrogate
from .wclr import wclr_coupling_trace

logger = logging.getLogger(__name__)

# Memory guard (gstack OOM #6): compute_session_pooled_threshold materializes a
# pooled surrogate matrix of shape (n_dyads * surrogate_n, n_coupling_points)
# float64. Under a high surrogate_n (default 5000 in core.DynamicAnalyzer) and
# many dyads this can silently OOM. Warn loudly (fail loud) *before* allocating
# instead of crashing the process. This does NOT change behaviour for normal
# inputs and does NOT alter the production default surrogate_n=5000.
SURROGATE_POOLED_MEM_GUARD_BYTES = 512 * 1024 * 1024  # 512 MiB


__all__ = [
    "compute_session_pooled_threshold",
    "compute_session_pooled_thresholds_by_modality",
    "compute_condition_pooled_thresholds",
]


def _generate_surrogate_coupling_matrix(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    hz: float,
    window_size: int,
    surrogate_n: int,
    seed: int,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    discontinuity_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate (surrogate_n, n_coupling_points) matrix of surrogate coupling values.

    Parameters
    ----------
    backend : {"wcc", "wclr"}
        If "wcc", compute sliding-window cross-correlation on surrogate pairs.
        If "wclr", compute windowed cross-lagged regression on surrogate pairs.
    discontinuity_mask : np.ndarray of bool or None
        Optional per-sample boundary mask (signal resolution). When set, the
        same seam windows are NaN-gated on every surrogate coupling trace as
        on the observed WCC (P1-R3), so the pooled null threshold is not
        inflated by discontinuity-spanning windows.
    """
    rng = np.random.default_rng(seed)
    _gen = iaaft_surrogate if surrogate_method == "iaaft" else ft_surrogate

    surrogate_couplings: List[np.ndarray] = []
    for _ in range(surrogate_n):
        a_surr = _gen(sig_a, rng)
        b_surr = _gen(sig_b, rng)
        if backend == "wclr":
            coup_s = wclr_coupling_trace(
                a_surr, b_surr,
                window_size=window_size,
                hz=hz,
                max_lag_samples=wclr_max_lag_samples,
            )
            # WCLR path: mask application is best-effort on length match only.
            if discontinuity_mask is not None:
                coup_s = _apply_discontinuity_mask(
                    coup_s, discontinuity_mask, window_size
                )
        else:
            coup_s = sliding_window_wcc(
                a_surr, b_surr,
                window_size=window_size,
                hz=hz,
            )
            if discontinuity_mask is not None:
                coup_s = _apply_discontinuity_mask(
                    coup_s, discontinuity_mask, window_size
                )
        surrogate_couplings.append(coup_s)

    return np.vstack(surrogate_couplings)


def compute_session_pooled_threshold(
    dyad_signals: List[Tuple[np.ndarray, np.ndarray]],
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = 95.0,
    seed: int = 42,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    fallback_threshold: float = ONSET_THRESHOLD,
    discontinuity_masks: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[float, Dict]:
    """Compute a single surrogate threshold pooled across all dyads.

    Parameters
    ----------
    dyad_signals : list of (sig_a, sig_b) tuples
        All dyad signal pairs in the session. Each pair must have the same
        length and a common sampling rate.
    hz : float
        Sampling rate (Hz).
    wcc_window_size : int
        WCC/WCLR window size in samples.
    surrogate_n : int
        Number of surrogates per dyad. Total surrogate replicates =
        ``len(dyad_signals) * surrogate_n``.
    percentile : float
        Quantile of the pooled surrogate coupling distribution (default 95).
    seed : int
        Base RNG seed. Per-dyad seeds are derived as ``seed + i`` so results
        are reproducible even when dyads are reordered.
    surrogate_method : {"iaaft", "ft"}
        Surrogate method. IAAFT (default) preserves spectrum and amplitude
        distribution; FT preserves only spectrum.
    backend : {"wcc", "wclr"}
        Computational backend used to compute the coupling trace on surrogate
        pairs. Must match the backend used for the observed analysis.
    wclr_max_lag_samples : int
        Max lag for WCLR backend (ignored for WCC).
    fallback_threshold : float
        Threshold to return if the pooled surrogate distribution is degenerate
        (fewer than 10 finite values).

    Returns
    -------
    Tuple[float, Dict]
        ``(threshold, meta)`` where ``meta`` contains:
        - ``mode``: "session_pooled"
        - ``n_dyads``: number of dyads that actually contributed
        - ``n_dyads_input``: total number of dyads passed in
        - ``n_dyads_used``: number of dyads that contributed (same as n_dyads)
        - ``n_dyads_excluded_nonfinite``: dyads excluded due to NaN/Inf
        - ``n_dyads_excluded_length_mismatch``: dyads excluded due to length mismatch
        - ``surrogate_n_per_dyad``: surrogates per dyad
        - ``total_replicates``: total number of surrogate coupling series
        - ``n_finite_coupling_values``: number of finite coupling values pooled
        - ``percentile``: percentile used
        - ``surrogate_method``: "iaaft" or "ft"
        - ``backend``: "wcc" or "wclr"
        - ``fallback_used``: whether the fallback threshold was used
    """
    if not dyad_signals:
        return fallback_threshold, {
            "mode": "session_pooled",
            "fallback_used": True,
            "reason": "empty dyad_signals",
            "n_dyads_input": 0,
            "n_dyads_used": 0,
        }

    if discontinuity_masks is not None and len(discontinuity_masks) != len(dyad_signals):
        raise ValueError(
            "discontinuity_masks must be None or a sequence with the same "
            f"length as dyad_signals (got {len(discontinuity_masks)} masks for "
            f"{len(dyad_signals)} dyads)."
        )

    pooled_values: List[np.ndarray] = []
    n_excluded_nonfinite = 0
    n_excluded_length_mismatch = 0
    n_masks_applied = 0
    for i, (sig_a, sig_b) in enumerate(dyad_signals):
        sig_a = np.asarray(sig_a, dtype=float)
        sig_b = np.asarray(sig_b, dtype=float)
        if not (np.all(np.isfinite(sig_a)) and np.all(np.isfinite(sig_b))):
            n_excluded_nonfinite += 1
            continue
        if len(sig_a) != len(sig_b):
            n_excluded_length_mismatch += 1
            continue
        mask_i = None if discontinuity_masks is None else discontinuity_masks[i]
        if mask_i is not None:
            n_masks_applied += 1
        coup_matrix = _generate_surrogate_coupling_matrix(
            sig_a, sig_b, hz, wcc_window_size,
            surrogate_n=surrogate_n,
            seed=seed + i,
            surrogate_method=surrogate_method,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
            discontinuity_mask=mask_i,
        )
        pooled_values.append(coup_matrix)

    n_excluded_total = n_excluded_nonfinite + n_excluded_length_mismatch
    if n_excluded_total:
        logger.warning(
            "compute_session_pooled_threshold: excluded %d/%d dyad(s) from "
            "threshold pooling (%d non-finite, %d length-mismatch). The "
            "pooled threshold reflects only the remaining %d dyad(s).",
            n_excluded_total, len(dyad_signals), n_excluded_nonfinite,
            n_excluded_length_mismatch, len(dyad_signals) - n_excluded_total,
        )

    if not pooled_values:
        return fallback_threshold, {
            "mode": "session_pooled",
            "fallback_used": True,
            "reason": "no dyads produced finite surrogate coupling values",
            "n_dyads_input": len(dyad_signals),
            "n_dyads_used": 0,
            "n_dyads_excluded_nonfinite": n_excluded_nonfinite,
            "n_dyads_excluded_length_mismatch": n_excluded_length_mismatch,
        }

    # Memory guard (gstack OOM #6): estimate the pooled surrogate matrix size
    # before allocating. Warn loudly instead of silently OOM-ing if it would
    # exceed the guard budget. Purely a warning — computation is unchanged for
    # normal inputs.
    if pooled_values:
        # NOTE: dyad WCC traces have VARIABLE lengths (different session
        # durations across conditions, e.g. Lerique rest1 vs trials_concat),
        # so each coup_matrix has a different number of columns and a naive
        # np.vstack raises ValueError. The pooled threshold is a single
        # percentile over ALL surrogate coupling values irrespective of which
        # WCC timepoint they came from (compute_surrogate_threshold flattens
        # internally), so we flatten + concatenate per dyad instead.
        _total_values = sum(int(m.size) for m in pooled_values)
        _est = _total_values * pooled_values[0].dtype.itemsize
        if _est > SURROGATE_POOLED_MEM_GUARD_BYTES:
            logger.warning(
                "compute_session_pooled_threshold: pooled surrogate matrix "
                "would allocate ~%.1f MiB (%d dyads x surrogate_n=%d x variable "
                "points). This may OOM; consider lowering surrogate_n or "
                "chunking dyads.",
                _est / (1024 * 1024), len(pooled_values), surrogate_n,
            )
    pooled = np.concatenate([m.ravel() for m in pooled_values])
    threshold, is_surrogate = compute_surrogate_threshold(pooled, percentile=percentile)

    meta = {
        "mode": "session_pooled",
        # NOTE: n_dyads means "dyads that actually contributed to the pooled
        # threshold" (== n_dyads_used), NOT the number passed in. See
        # n_dyads_input for the original count.
        "n_dyads": len(pooled_values),
        "n_dyads_input": len(dyad_signals),
        "n_dyads_used": len(pooled_values),
        "n_dyads_excluded_nonfinite": n_excluded_nonfinite,
        "n_dyads_excluded_length_mismatch": n_excluded_length_mismatch,
        "surrogate_n_per_dyad": surrogate_n,
        "total_replicates": sum(int(m.shape[0]) for m in pooled_values),
        "n_finite_coupling_values": int(np.isfinite(pooled).sum()),
        "percentile": percentile,
        "surrogate_method": surrogate_method,
        "backend": backend,
        "fallback_used": not is_surrogate,
        "n_discontinuity_masks_applied": n_masks_applied,
    }

    # Label the *cause* of any fallback so the per-modality warning below (and
    # any downstream consumer of ``meta``) is not mislabeled "degenerate null"
    # when the real cause is the periodicity / strong-autocorrelation ceiling
    # (BUG-3).  ``compute_surrogate_threshold`` returns only
    # (threshold, is_surrogate_derived); the same pooled values it inspected are
    # re-checked here to set an accurate ``reason``.
    if not is_surrogate:
        finite = pooled[np.isfinite(pooled)]
        if finite.size < 10:
            meta["reason"] = "degenerate_null"
        else:
            derived = float(np.percentile(finite, percentile))
            meta["reason"] = (
                "periodicity_ceiling"
                if derived > SURROGATE_THRESHOLD_MAX
                else "degenerate_null"
            )
    else:
        meta["reason"] = "surrogate_derived"

    return threshold, meta


def compute_session_pooled_thresholds_by_modality(
    dyad_signals: List[Tuple[np.ndarray, np.ndarray]],
    modalities: List[str],
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = 95.0,
    seed: int = 42,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    fallback_threshold: float = ONSET_THRESHOLD,
    discontinuity_masks: Optional[List[Optional[np.ndarray]]] = None,
    return_meta: bool = False,
) -> Union[Dict[str, float], Dict[str, Tuple[float, Dict]]]:
    """Compute one surrogate threshold per modality (per-modality pooled null).

    Unlike :func:`compute_session_pooled_threshold` (which pools *all* dyads
    across modalities into a single global null), this pools *within* each
    modality so slow/smooth signals (e.g. EDA, low WCC amplitude) and
    fast/spiky signals (e.g. ECG, high WCC amplitude) each get a
    modality-appropriate threshold. This is the canonical v1 onset-threshold
    default: cross-modal comparability is preserved (every dyad of a given
    modality shares one threshold) while the threshold itself is calibrated to
    that modality's null distribution — solving the cross-modality non-uniformity
    problem that a single global pool cannot.

    Parameters
    ----------
    dyad_signals : list of (sig_a, sig_b) tuples
        All dyad signal pairs, in the same order as ``modalities``.
    modalities : list of str
        Modality label for each dyad (same length/order as ``dyad_signals``),
        e.g. ``["EDA", "EDA", "ECG", "ECG"]``. A ``None`` entry is grouped
        under the sentinel key ``"None"``.
    hz, wcc_window_size, surrogate_n, percentile, seed, surrogate_method,
    backend, wclr_max_lag_samples, fallback_threshold
        Forwarded to :func:`compute_session_pooled_threshold` per modality.
    discontinuity_masks : list of optional masks, optional
        Aligned with ``dyad_signals``; only the masks whose modality is pooled
        are forwarded to that modality's pooled-null call.

    Returns
    -------
    Dict[str, float] or Dict[str, Tuple[float, Dict]]
        If ``return_meta`` is False (default): mapping ``modality -> threshold``
        (backward-compatible). If True: mapping ``modality -> (threshold, meta)``
        where ``meta`` carries ``fallback_used`` and ``reason`` so downstream
        reporting can distinguish a surrogate-derived threshold from a fallback
        instead of re-inferring it by float-equality to 0.5.
        If a modality has too few dyads to build a stable null (degenerate
        pooled distribution), that modality's threshold falls back to
        ``fallback_threshold`` (default 0.5) and a warning is logged
        (fail-loud). A modality with zero dyads is not emitted.

    Notes
    -----
    This reuses the exact same surrogate-threshold null machinery as the per-dyad
    and session-pooled paths (``compute_surrogate_threshold`` on IAAFT surrogates),
    but groups by modality instead of by session/condition. It is the symmetry
    partner of :func:`compute_condition_pooled_thresholds` (group-by-condition).

    **Periodicity / strong-autocorrelation guard (BUG-3).** When a modality's
    derived threshold exceeds ``SURROGATE_THRESHOLD_MAX`` (0.9), the modality
    falls back to ``ONSET_THRESHOLD`` (0.5) and a fail-loud warning is emitted
    (``meta["reason"] == "periodicity_ceiling"``) rather than silently using a
    contaminated cut-off.
    """
    if len(modalities) != len(dyad_signals):
        raise ValueError(
            "modalities must have the same length as dyad_signals "
            f"(got {len(modalities)} modalities for {len(dyad_signals)} dyads)."
        )

    # Group dyad indices by modality. None -> "None" sentinel so a batch that
    # never records modality still pools consistently (matches the no-modality
    # BatchComputationPipeline path).
    by_modality: Dict[str, List[int]] = {}
    for i, mod in enumerate(modalities):
        mod_key = str(mod) if mod is not None else "None"
        by_modality.setdefault(mod_key, []).append(i)

    results: Dict[str, float] = {}
    results_meta: Dict[str, Tuple[float, Dict]] = {}
    for mod_key, idxs in by_modality.items():
        mod_signals = [dyad_signals[i] for i in idxs]
        mod_masks = (
            [discontinuity_masks[i] for i in idxs]
            if discontinuity_masks is not None
            else None
        )
        threshold, meta = compute_session_pooled_threshold(
            mod_signals,
            hz=hz,
            wcc_window_size=wcc_window_size,
            surrogate_n=surrogate_n,
            percentile=percentile,
            seed=seed,
            surrogate_method=surrogate_method,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
            fallback_threshold=fallback_threshold,
            discontinuity_masks=mod_masks,
        )
        if meta.get("fallback_used", False):
            logger.warning(
                "compute_session_pooled_thresholds_by_modality: modality %r "
                "fell back to fixed threshold %.3f (%s). n_dyads_used=%s of %s.",
                mod_key, threshold, meta.get("reason", "degenerate null"),
                meta.get("n_dyads_used"), meta.get("n_dyads_input"),
            )
        meta["mode"] = "session_pooled_by_modality"
        meta["modality"] = mod_key
        results[mod_key] = float(threshold)
        results_meta[mod_key] = (float(threshold), meta)
    return results_meta if return_meta else results


def compute_condition_pooled_thresholds(
    condition_signals: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = 95.0,
    seed: int = 42,
    surrogate_method: str = "iaaft",
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    fallback_threshold: float = ONSET_THRESHOLD,
) -> Dict[str, Tuple[float, Dict]]:
    """Compute one pooled surrogate threshold per experimental condition.

    .. deprecated::
        This function does not forward ``discontinuity_masks`` to
        :func:`compute_session_pooled_threshold`, creating an
        observed-vs-null asymmetry when seams are present. Use
        :func:`compute_session_pooled_thresholds_by_modality` instead,
        which accepts and forwards masks.

    Parameters
    ----------
    condition_signals : dict
        Mapping ``condition_label -> list of (sig_a, sig_b) tuples``.
    hz, wcc_window_size, surrogate_n, percentile, seed, surrogate_method,
    backend, wclr_max_lag_samples, fallback_threshold
        Passed to :func:`compute_session_pooled_threshold` for each condition.

    Returns
    -------
    Dict[str, Tuple[float, Dict]]
        Mapping condition -> (threshold, meta).
    """
    results: Dict[str, Tuple[float, Dict]] = {}
    for cond, signals in condition_signals.items():
        threshold, meta = compute_session_pooled_threshold(
            signals,
            hz=hz,
            wcc_window_size=wcc_window_size,
            surrogate_n=surrogate_n,
            percentile=percentile,
            seed=seed,
            surrogate_method=surrogate_method,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
            fallback_threshold=fallback_threshold,
        )
        meta["mode"] = "condition_pooled"
        meta["condition"] = cond
        results[cond] = (threshold, meta)
    return results

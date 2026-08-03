"""
Design-level synchrony controls.

This module contains lightweight, dataset-agnostic controls for separating
three claims that are often conflated in synchrony analyses:

1. synchrony-existence: do two aligned signals show WCC features larger than
   independent autocorrelated signals would produce?
2. dyad-specificity: are real partners stronger than cross-dyad pseudo-pairs?
3. time-alignment: are real partners stronger than within-dyad time-shifted
   alignments?

These controls are descriptive audit components.  They do not prove causality;
they are designed to make shared-stimulus and co-presence alternatives visible.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .dynamic_features import (
    sliding_window_wcc,
    sliding_window_wcc_masked,
    wcc_surrogate_test,
)
from .feature_definitions import ONSET_THRESHOLD, extract_features

SignalPair = Tuple[np.ndarray, np.ndarray]

DEFAULT_AUDIT_FEATURES: Tuple[str, ...] = (
    "mean_synchrony",
    "peak_amplitude",
    "fraction_above_threshold",
    "dwell_time",
    "switching_rate",
)


def _finite_pair(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    strict_length: bool = True,
    on_length_mismatch: str = "warn",
) -> SignalPair:
    """Return same-length finite arrays for a signal pair.

    Parameters
    ----------
    strict_length : bool, default True
        If True (default), unequal input lengths are not silently accepted.
        Behaviour is controlled by ``on_length_mismatch``.
    on_length_mismatch : {"warn", "raise", "truncate"}
        - ``"warn"`` (default): emit ``UserWarning``, then truncate to min length.
        - ``"raise"``: raise ``ValueError`` (recommended for confirmatory audits).
        - ``"truncate"``: legacy silent truncate (discouraged).

    Notes
    -----
    P0-3 fix (2026-07-22): previous versions always truncated to
    ``min(len(a), len(b))`` with no warning, so design-control statistics
    could be computed on a hidden sub-interval when one partner's series
    was shorter.  Joint finite masking still preserves relative alignment
    of kept samples.
    """
    import warnings

    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)
    if a.size != b.size:
        msg = (
            f"_finite_pair: unequal lengths len(a)={a.size}, len(b)={b.size}. "
            f"Truncating to min={min(a.size, b.size)} samples from the start — "
            f"ensure this matches your intended analysis window."
        )
        mode = on_length_mismatch if strict_length else "truncate"
        if mode == "raise":
            raise ValueError(msg)
        if mode == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        # truncate (warn already emitted, or legacy silent)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def extract_pair_features(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    hz: float,
    window_size: int,
    threshold: float = ONSET_THRESHOLD,
    feature_names: Sequence[str] = DEFAULT_AUDIT_FEATURES,
    window_type: str = "rect",
    discontinuity_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute WCC and selected SyncPipe features for one signal pair."""
    a, b = _finite_pair(sig_a, sig_b)
    if a.size < window_size or b.size < window_size:
        return {name: float("nan") for name in feature_names}
    wcc = sliding_window_wcc(a, b, window_size=window_size, hz=hz, window_type=window_type)
    if discontinuity_mask is not None:
        from .dynamic_features import _apply_discontinuity_mask
        wcc = _apply_discontinuity_mask(wcc, discontinuity_mask, window_size)
    feats = extract_features(
        wcc,
        hz=hz,
        wcc_window_sec=window_size / hz if hz > 0 else float(window_size),
        threshold=threshold,
        gap_policy="segment" if discontinuity_mask is not None else None,
    )
    return {name: float(getattr(feats, name, np.nan)) for name in feature_names}


def synchrony_existence_audit(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    hz: float,
    window_size: int,
    surrogate_n: int = 100,
    seed: int = 42,
    window_type: str = "rect",
    discontinuity_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run signal-level IAAFT synchrony-existence audit for one pair.

    Interpretation: a significant result means the observed WCC distributional
    features exceed what independently IAAFT-randomised signals can produce.
    It is necessary-but-not-sufficient evidence for interpersonal coupling.
    Shared-stimulus and co-presence alternatives require design controls.

    Parameters
    ----------
    discontinuity_mask : np.ndarray of bool or None
        Per-sample boundary mask (signal-resolution). When provided, the
        observed WCC and the recomputed surrogate WCC both NaN out windows
        straddling a seam, so the audit does not credit coupling that is an
        artefact of segment concatenation.
    """
    a, b = _finite_pair(sig_a, sig_b)
    if a.size < window_size or b.size < window_size:
        return {
            "audit": "synchrony_existence",
            "null_model": "signal_level_iaaft",
            "status": "failed",
            "reason": "signal_too_short",
            "n_samples": int(min(a.size, b.size)),
        }
    wcc = sliding_window_wcc_masked(
        a, b, window_size=window_size, hz=hz, window_type=window_type,
        discontinuity_mask=discontinuity_mask,
    )
    result = wcc_surrogate_test(
        wcc,
        hz=hz,
        surrogate_n=surrogate_n,
        seed=seed,
        raw_signals=(a, b),
        wcc_window_size=window_size,
        wcc_window_sec=window_size / hz if hz > 0 else float(window_size),
        window_type=window_type,
        discontinuity_mask=discontinuity_mask,
    )
    return {
        "audit": "synchrony_existence",
        "null_model": "signal_level_iaaft",
        "status": "ok",
        "n_samples": int(a.size),
        "n_wcc": int(np.isfinite(wcc).sum()),
        "n_surrogates": int(result.get("n_surrogates", surrogate_n)),
        "per_feature_significant": result.get("per_feature_significant", {}),
        "p_values": {
            k[2:]: float(v)
            for k, v in result.items()
            if k.startswith("p_") and np.isscalar(v)
        },
        "observed": {
            k[4:]: float(v)
            for k, v in result.items()
            if k.startswith("obs_") and np.isscalar(v) and np.isfinite(v)
        },
        "interpretation": (
            "Necessary-but-not-sufficient evidence: signal-level IAAFT tests "
            "whether aligned WCC features exceed independent autocorrelated "
            "signals, but it does not rule out shared stimulus or co-presence."
        ),
    }


def _paired_signflip_p_upper(
    deltas: np.ndarray, *, seed: int = 42, max_draws: int = 20000) -> float:
    """One-sided paired sign-flip p-value for mean(delta) > 0."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    if d.size <= 12:
        masks = np.arange(2 ** d.size, dtype=np.uint64)
        null = []
        for m in masks:
            signs = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(d.size)])
            null.append(float(np.mean(signs * d)))
        null_arr = np.array(null)
    else:
        signs = rng.choice([-1.0, 1.0], size=(max_draws, d.size))
        null_arr = np.mean(signs * d, axis=1)
    return float((np.sum(null_arr >= obs) + 1) / (null_arr.size + 1))


def design_control_audit(
    signal_pairs: Mapping[str, SignalPair],
    *,
    hz: float,
    window_size: int,
    threshold: Union[float, Mapping[str, float]] = ONSET_THRESHOLD,
    feature_names: Sequence[str] = DEFAULT_AUDIT_FEATURES,
    n_pseudo_per_dyad: int = 10,
    shift_lags_sec: Sequence[float] = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0),
    seed: int = 42,
    window_type: str = "rect",
    discontinuity_masks: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run pseudo-pair and time-shift design controls for a cohort.

    Parameters
    ----------
    signal_pairs : mapping
        ``dyad_id -> (person_a_signal, person_b_signal)``.  At least two dyads
        are required for pseudo-pair controls; one dyad is sufficient for
        time-shift controls.
    n_pseudo_per_dyad : int
        Number of pseudo-pair draws per real dyad.  Default 10 (publication
        grade); raise further for very small cohorts where the pseudo-null
        distribution needs more draws to stabilise.

    Returns
    -------
    dict
        JSON-serialisable audit summary with per-feature real, pseudo-pair,
        and time-shift comparisons.
    """
    rng = np.random.default_rng(seed)
    ids = list(signal_pairs.keys())
    if discontinuity_masks is not None:
        missing_masks = sorted(set(ids) - set(discontinuity_masks))
        if missing_masks:
            raise ValueError(
                "discontinuity_masks must use the same keys as signal_pairs; "
                f"missing masks for {missing_masks[:10]}"
            )

    def _threshold_for(pair_id: str) -> float:
        if isinstance(threshold, Mapping):
            if pair_id not in threshold:
                raise ValueError(f"No threshold supplied for design pair {pair_id!r}")
            value = float(threshold[pair_id])
        else:
            value = float(threshold)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite threshold for design pair {pair_id!r}")
        return value

    def _mask_for(pair_id: str):
        return discontinuity_masks.get(pair_id) if discontinuity_masks is not None else None

    def _align_pseudo_pair(sig_a, sig_b, mask_a, mask_b):
        """Jointly align a cross-dyad pseudo-pair and build a combined mask.

        A pseudo-pair combines dyad X's person A with dyad Y's person B. The
        previous implementation passed dyad X's discontinuity mask alongside
        dyad Y's B signal — a mask that describes a DIFFERENT recording and
        therefore gates the wrong samples. It also let ``_finite_pair``
        truncate silently, so the pseudo arm ran on a shorter effective length
        than the real arm (a length confound in the null).

        Here we (1) crop both signals to a common length, (2) keep only the
        jointly-finite sample indices, and (3) build a combined
        discontinuity mask = mask_a AND mask_b sampled at those SAME kept
        indices, so a pseudo window is valid only where BOTH source signals
        are internal to a segment. Returns (a, b, combined_mask, n_kept).
        """
        a = np.asarray(sig_a, dtype=float)
        b = np.asarray(sig_b, dtype=float)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]

        def _crop(m):
            if m is None:
                return None
            m = np.asarray(m, dtype=bool)
            return m[:n] if m.size >= n else None

        ma = _crop(mask_a)
        mb = _crop(mask_b)

        finite = np.isfinite(a) & np.isfinite(b)
        a_f = a[finite]
        b_f = b[finite]
        combined = None
        if ma is not None or mb is not None:
            ca = ma[finite] if ma is not None else np.ones(int(finite.sum()), dtype=bool)
            cb = mb[finite] if mb is not None else np.ones(int(finite.sum()), dtype=bool)
            combined = ca & cb
        return a_f, b_f, combined, int(finite.sum())

    real: Dict[str, Dict[str, float]] = {}
    for dyad_id in ids:
        a, b = signal_pairs[dyad_id]
        real[dyad_id] = extract_pair_features(
            a, b, hz=hz, window_size=window_size,
            threshold=_threshold_for(dyad_id), feature_names=feature_names,
            window_type=window_type, discontinuity_mask=_mask_for(dyad_id),
        )

    pseudo_values: Dict[str, Dict[str, list]] = {
        dyad_id: {f: [] for f in feature_names} for dyad_id in ids
    }
    pseudo_lengths: list = []  # effective (post-alignment) length per pseudo-pair
    if len(ids) >= 2:
        for dyad_id in ids:
            partners = [p for p in ids if p != dyad_id]
            replace = n_pseudo_per_dyad > len(partners)
            chosen = rng.choice(partners, size=n_pseudo_per_dyad, replace=replace)
            a, _ = signal_pairs[dyad_id]
            for partner_id in chosen:
                partner_id = str(partner_id)
                _, b_partner = signal_pairs[partner_id]
                a_al, b_al, mask_al, n_kept = _align_pseudo_pair(
                    a, b_partner, _mask_for(dyad_id), _mask_for(partner_id)
                )
                pseudo_lengths.append(n_kept)
                feats = extract_pair_features(
                    a_al, b_al, hz=hz, window_size=window_size,
                    threshold=_threshold_for(dyad_id), feature_names=feature_names,
                    window_type=window_type, discontinuity_mask=mask_al,
                )
                for f in feature_names:
                    if np.isfinite(feats.get(f, np.nan)):
                        pseudo_values[dyad_id][f].append(feats[f])

    shift_values: Dict[str, Dict[str, list]] = {
        dyad_id: {f: [] for f in feature_names} for dyad_id in ids
    }
    for dyad_id in ids:
        # Crop from the raw (pre-finite) arrays so the discontinuity mask stays
        # index-aligned with the signals, matching how the real/pseudo arms pass
        # the full mask into extract_pair_features. Previously the time-shift arm
        # passed discontinuity_mask=None, so on segmented/gapped data real WCC
        # NaN-outs seam-straddling windows while the shift arm did not — biasing
        # real_minus_time_shift. See design-control mask-consistency fix.
        a_full = np.asarray(signal_pairs[dyad_id][0], dtype=float)
        b_full = np.asarray(signal_pairs[dyad_id][1], dtype=float)
        n = min(a_full.size, b_full.size)
        a_full = a_full[:n]
        b_full = b_full[:n]
        mask_full = _mask_for(dyad_id)
        if mask_full is not None:
            mask_full = np.asarray(mask_full, dtype=bool)
            mask_full = mask_full[:n] if mask_full.size >= n else None
        for lag_sec in shift_lags_sec:
            k = int(round(lag_sec * hz))
            if k == 0 or abs(k) >= n - window_size:
                continue
            if k > 0:
                a_use = a_full[k:]
                b_use = b_full[: n - k]
                # a_use[i] -> a[i+k], b_use[i] -> b[i]; a window is valid only
                # if BOTH shifted signals are internal to a segment there.
                shift_mask = (
                    (mask_full[k:] & mask_full[: n - k]) if mask_full is not None else None
                )
            else:
                a_use = a_full[: n + k]
                b_use = b_full[-k:]
                shift_mask = (
                    (mask_full[: n + k] & mask_full[-k:]) if mask_full is not None else None
                )
            feats = extract_pair_features(
                a_use, b_use, hz=hz, window_size=window_size,
                threshold=_threshold_for(dyad_id), feature_names=feature_names,
                window_type=window_type, discontinuity_mask=shift_mask,
            )
            for f in feature_names:
                if np.isfinite(feats.get(f, np.nan)):
                    shift_values[dyad_id][f].append(feats[f])

    feature_summary: Dict[str, Dict[str, Any]] = {}
    for f in feature_names:
        real_arr = np.array([real[d].get(f, np.nan) for d in ids], dtype=float)

        pseudo_median = np.array([
            np.nanmedian(pseudo_values[d][f]) if pseudo_values[d][f] else np.nan
            for d in ids
        ], dtype=float)
        shift_median = np.array([
            np.nanmedian(shift_values[d][f]) if shift_values[d][f] else np.nan
            for d in ids
        ], dtype=float)

        pseudo_delta = real_arr - pseudo_median
        shift_delta = real_arr - shift_median
        feature_summary[f] = {
            "real_median": float(np.nanmedian(real_arr)),
            "pseudo_pair_median": float(np.nanmedian(pseudo_median)) if np.isfinite(pseudo_median).any() else float("nan"),
            "time_shift_median": float(np.nanmedian(shift_median)) if np.isfinite(shift_median).any() else float("nan"),
            "real_minus_pseudo_mean": float(np.nanmean(pseudo_delta)) if np.isfinite(pseudo_delta).any() else float("nan"),
            "real_minus_time_shift_mean": float(np.nanmean(shift_delta)) if np.isfinite(shift_delta).any() else float("nan"),
            "p_real_gt_pseudo": _paired_signflip_p_upper(pseudo_delta, seed=seed),
            "p_real_gt_time_shift": _paired_signflip_p_upper(shift_delta, seed=seed + 1),
            "n_real": int(np.isfinite(real_arr).sum()),
            "n_pseudo_dyads": int(np.isfinite(pseudo_median).sum()),
            "n_time_shift_dyads": int(np.isfinite(shift_median).sum()),
        }

    return {
        "audit": "design_controls",
        "features": list(feature_names),
        "n_dyads": len(ids),
        "threshold_scope": "per_pair_mapping" if isinstance(threshold, Mapping) else "fixed",
        "discontinuity_masks_applied": bool(discontinuity_masks),
        "pseudo_pair": {
            "enabled": len(ids) >= 2,
            "n_pseudo_per_dyad": int(n_pseudo_per_dyad),
            # Length-transparency for reviewers: pseudo-pairs are jointly
            # aligned (crop to common length + joint finite + combined
            # mask_a AND mask_b). Report the effective-length distribution so
            # a length confound in the null is visible, not hidden.
            "aligned_length_min": (
                int(np.min(pseudo_lengths)) if pseudo_lengths else 0
            ),
            "aligned_length_median": (
                float(np.median(pseudo_lengths)) if pseudo_lengths else float("nan")
            ),
            "aligned_length_max": (
                int(np.max(pseudo_lengths)) if pseudo_lengths else 0
            ),
            "interpretation": (
                "If real pairs exceed pseudo-pairs, evidence is more dyad-specific. "
                "If real ≈ pseudo, shared context/stimulus/co-presence remains plausible."
            ),
        },
        "time_shift": {
            "enabled": True,
            "shift_lags_sec": [float(x) for x in shift_lags_sec],
            "interpretation": (
                "If real pairs exceed time-shift controls, evidence depends on "
                "precise temporal alignment. If time-shift remains high, slow drifts "
                "or shared block structure remain plausible."
            ),
        },
        "feature_summary": feature_summary,
    }

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

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dynamic_features import sliding_window_wcc, wcc_surrogate_test
from .feature_definitions import ONSET_THRESHOLD, extract_features

SignalPair = Tuple[np.ndarray, np.ndarray]

DEFAULT_AUDIT_FEATURES: Tuple[str, ...] = (
    "mean_synchrony",
    "peak_amplitude",
    "fraction_above_threshold",
    "dwell_time",
    "switching_rate",
)


def _finite_pair(sig_a: np.ndarray, sig_b: np.ndarray) -> SignalPair:
    """Return same-length finite arrays for a signal pair."""
    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)
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
) -> Dict[str, float]:
    """Compute WCC and selected SyncPipe features for one signal pair."""
    a, b = _finite_pair(sig_a, sig_b)
    if a.size < window_size or b.size < window_size:
        return {name: float("nan") for name in feature_names}
    wcc = sliding_window_wcc(a, b, window_size=window_size, hz=hz)
    feats = extract_features(
        wcc,
        hz=hz,
        wcc_window_sec=window_size / hz if hz > 0 else float(window_size),
        threshold=threshold,
    )
    return {name: float(getattr(feats, name, np.nan)) for name in feature_names}


def synchrony_existence_audit(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    hz: float,
    window_size: int,
    surrogate_n: int = 99,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run signal-level IAAFT synchrony-existence audit for one pair.

    Interpretation: a significant result means the observed WCC distributional
    features exceed what independently IAAFT-randomised signals can produce.
    It is necessary-but-not-sufficient evidence for interpersonal coupling.
    Shared-stimulus and co-presence alternatives require design controls.
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
    wcc = sliding_window_wcc(a, b, window_size=window_size, hz=hz)
    result = wcc_surrogate_test(
        wcc,
        hz=hz,
        surrogate_n=surrogate_n,
        seed=seed,
        raw_signals=(a, b),
        wcc_window_size=window_size,
        wcc_window_sec=window_size / hz if hz > 0 else float(window_size),
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
    threshold: float = ONSET_THRESHOLD,
    feature_names: Sequence[str] = DEFAULT_AUDIT_FEATURES,
    n_pseudo_per_dyad: int = 3,
    shift_lags_sec: Sequence[float] = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0),
    seed: int = 42,
) -> Dict[str, Any]:
    """Run pseudo-pair and time-shift design controls for a cohort.

    Parameters
    ----------
    signal_pairs : mapping
        ``dyad_id -> (person_a_signal, person_b_signal)``.  At least two dyads
        are required for pseudo-pair controls; one dyad is sufficient for
        time-shift controls.

    Returns
    -------
    dict
        JSON-serialisable audit summary with per-feature real, pseudo-pair,
        and time-shift comparisons.
    """
    rng = np.random.default_rng(seed)
    ids = list(signal_pairs.keys())

    real: Dict[str, Dict[str, float]] = {}
    for dyad_id in ids:
        a, b = signal_pairs[dyad_id]
        real[dyad_id] = extract_pair_features(
            a, b, hz=hz, window_size=window_size,
            threshold=threshold, feature_names=feature_names,
        )

    pseudo_values: Dict[str, Dict[str, list]] = {
        dyad_id: {f: [] for f in feature_names} for dyad_id in ids
    }
    if len(ids) >= 2:
        for dyad_id in ids:
            partners = [p for p in ids if p != dyad_id]
            replace = n_pseudo_per_dyad > len(partners)
            chosen = rng.choice(partners, size=n_pseudo_per_dyad, replace=replace)
            a, _ = signal_pairs[dyad_id]
            for partner_id in chosen:
                _, b_partner = signal_pairs[str(partner_id)]
                feats = extract_pair_features(
                    a, b_partner, hz=hz, window_size=window_size,
                    threshold=threshold, feature_names=feature_names,
                )
                for f in feature_names:
                    if np.isfinite(feats.get(f, np.nan)):
                        pseudo_values[dyad_id][f].append(feats[f])

    shift_values: Dict[str, Dict[str, list]] = {
        dyad_id: {f: [] for f in feature_names} for dyad_id in ids
    }
    for dyad_id in ids:
        a, b = _finite_pair(*signal_pairs[dyad_id])
        n = min(a.size, b.size)
        for lag_sec in shift_lags_sec:
            k = int(round(lag_sec * hz))
            if k == 0 or abs(k) >= n - window_size:
                continue
            if k > 0:
                a_use = a[k:]
                b_use = b[: n - k]
            else:
                a_use = a[: n + k]
                b_use = b[-k:]
            feats = extract_pair_features(
                a_use, b_use, hz=hz, window_size=window_size,
                threshold=threshold, feature_names=feature_names,
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
        "pseudo_pair": {
            "enabled": len(ids) >= 2,
            "n_pseudo_per_dyad": int(n_pseudo_per_dyad),
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

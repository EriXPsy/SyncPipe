"""
Across-stimulus shuffle surrogate (Chang 2024 SCAN-style).

Core idea:
    Two people watch the same sequence of video clips. Shared stimulation
    (ISC) can produce spurious synchrony that cross-pair shuffle cannot
    distinguish — because cross-pair preserves the stimulus identity.

    Across-stim shuffle breaks the stimulus-driven coupling by randomly
    permuting the ORDER of stimulus segments within each person, then
    re-aligning the permuted segments between the two people.  This
    preserves each person's within-segment autocorrelation structure
    while destroying the shared-stimulus temporal alignment.

    The null hypothesis is:
        "The observed synchrony is driven by shared stimulus timing,
         not by interpersonal coupling."

Design:
    - Input: a list of (segment_label, P1_signal, P2_signal) tuples,
      where each tuple represents one stimulus segment (e.g., one video).
    - For each surrogate draw: independently permute the segment order
      for P1 and P2, then compute WCC on the concatenated surrogate
      signals.
    - Compare real synchrony features against the surrogate distribution.

Reference:
    Chang, C. H. C., Nastase, S. A., Zadbood, A., & Hasson, U. (2024).
    How a speaker herds the audience. Social Cognitive and Affective
    Neuroscience, 19(1), nsae059.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def across_stim_shuffle(
    segments: List[Tuple[str, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one across-stimulus shuffle surrogate pair.

    Parameters
    ----------
    segments : list of (label, p1_signal, p2_signal)
        Each tuple is one stimulus segment.  All segments should have the
        same P1/P2 alignment within each tuple (i.e., segment k for P1 was
        recorded during the same stimulus as segment k for P2).

    rng : np.random.Generator
        Random number generator for reproducible permutations.

    Returns
    -------
    p1_surr, p2_surr : np.ndarray
        Concatenated surrogate signals for P1 and P2, same total length
        as the original concatenation.

    Notes
    -----
    The shuffle independently permutes segment indices for P1 and P2,
    then concatenates the permuted segments.  This means that segment k
    for P1 (during stimulus S_k) may be paired with segment j for P2
    (during stimulus S_j, j ≠ k), destroying the shared-stimulus temporal
    alignment while preserving each person's within-segment dynamics.
    """
    n = len(segments)
    if n < 2:
        raise ValueError("Across-stim shuffle requires at least 2 segments.")

    # Extract signals per person
    p1_segs = [s[1] for s in segments]
    p2_segs = [s[2] for s in segments]

    # Independently permute segment order
    p1_perm = rng.permutation(n)
    p2_perm = rng.permutation(n)

    p1_surr = np.concatenate([p1_segs[i] for i in p1_perm])
    p2_surr = np.concatenate([p2_segs[i] for i in p2_perm])
    return p1_surr, p2_surr


def across_stim_shuffle_test(
    segments: List[Tuple[str, np.ndarray, np.ndarray]],
    wcc_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_func: Callable[[np.ndarray], Dict[str, float]],
    n_surr: int = 499,
    seed: int = 2026,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Full across-stimulus shuffle test.

    Parameters
    ----------
    segments : list of (label, p1_signal, p2_signal)
        Stimulus segments with aligned P1/P2 signals.
    wcc_func : callable
        Function (p1, p2) -> wcc_array.  E.g., sliding_window_wcc.
    feature_func : callable
        Function (wcc_array) -> dict of feature_name -> value.
    n_surr : int
        Number of surrogate draws.
    seed : int
        Random seed for reproducibility.
    feature_names : list of str, optional
        Features to extract.  If None, infers from real feature output.

    Returns
    -------
    dict mapping feature_name -> {
        "real": float,
        "surrogate_mean": float,
        "surrogate_sd": float,
        "surrogate_median": float,
        "surrogate_ci95": (float, float),
        "p_value": float,  # two-sided, fraction of surrogates more extreme
        "n_surr": int,
    }
    """
    rng = np.random.default_rng(seed)

    # Real features
    p1_real = np.concatenate([s[1] for s in segments])
    p2_real = np.concatenate([s[2] for s in segments])
    wcc_real = wcc_func(p1_real, p2_real)
    real_feats = feature_func(wcc_real)

    if feature_names is None:
        feature_names = list(real_feats.keys())

    # Surrogate distribution
    surr_feats: Dict[str, list] = {k: [] for k in feature_names}
    for _ in range(n_surr):
        p1_s, p2_s = across_stim_shuffle(segments, rng)
        wcc_s = wcc_func(p1_s, p2_s)
        feats_s = feature_func(wcc_s)
        for k in feature_names:
            v = feats_s.get(k, np.nan)
            if v is not None and np.isfinite(v):
                surr_feats[k].append(float(v))

    # Summary per feature
    results = {}
    for k in feature_names:
        real_v = real_feats.get(k, np.nan)
        surr = np.array(surr_feats[k], dtype=float)
        surr_finite = surr[np.isfinite(surr)]

        if len(surr_finite) < 5 or not np.isfinite(real_v):
            results[k] = {
                "real": real_v,
                "surrogate_mean": np.nan,
                "surrogate_sd": np.nan,
                "surrogate_median": np.nan,
                "surrogate_ci95": (np.nan, np.nan),
                "p_value": np.nan,
                "n_surr": len(surr_finite),
            }
            continue

        # Two-sided p: Phipson-Smyth unbiased estimate (k+1)/(n+1)
        k = min(np.sum(surr_finite >= real_v), np.sum(surr_finite <= real_v))
        n = len(surr_finite)
        p_two = 2.0 * (k + 1) / (n + 1)
        p_two = min(p_two, 1.0)

        ci_lo = np.percentile(surr_finite, 2.5)
        ci_hi = np.percentile(surr_finite, 97.5)

        results[k] = {
            "real": float(real_v) if np.isfinite(real_v) else np.nan,
            "surrogate_mean": float(np.mean(surr_finite)),
            "surrogate_sd": float(np.std(surr_finite)),
            "surrogate_median": float(np.median(surr_finite)),
            "surrogate_ci95": (float(ci_lo), float(ci_hi)),
            "p_value": p_two,
            "n_surr": len(surr_finite),
        }
    return results

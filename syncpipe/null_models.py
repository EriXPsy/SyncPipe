from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import warnings
import numpy as np

from .surrogate import iaaft_surrogate, block_permutation_surrogate, state_transition_shuffle_surrogate
from .preparation import resolve_signal_geometry
from .feature_definitions import compute_surrogate_threshold, ONSET_THRESHOLD, SURROGATE_THRESHOLD_PERCENTILE, SWITCHING_HYSTERESIS_DELTA
from .wcc import sliding_window_wcc, _apply_discontinuity_mask

logger = logging.getLogger(__name__)

# Memory guard (gstack OOM #6): surrogate generation materializes a
# (surrogate_n, n_timepoints) float64 matrix via np.vstack. Under the
# core.DynamicAnalyzer default surrogate_n=5000 this is bounded (~tens of MiB),
# but a high surrogate_n or very long signals can silently OOM. Warn loudly
# (fail loud) *before* allocating instead of crashing the process. This does
# NOT change behaviour for normal inputs and does NOT alter the production
# default surrogate_n=5000.
SURROGATE_MEM_GUARD_BYTES = 512 * 1024 * 1024  # 512 MiB


# ---------------------------------------------------------------------------
# Sliding-window WCC (Weighted Cross-Correlation)
# ---------------------------------------------------------------------------


from .wcc import (
    _make_window_kernel, _sliding_window_wcc_cumsum, _sliding_window_wcc_stride,
    sliding_window_wcc, _apply_discontinuity_mask, sliding_window_wcc_masked,
)



# ---------------------------------------------------------------------------
# Surrogate testing — tiered null models (L0 / L1)
# ---------------------------------------------------------------------------
# Mathematical invariance tiers (see docs/METHOD_LOG.md):
#   L0 (permutation-invariant moments of the WCC value distribution):
#     mean_synchrony, peak_amplitude, synchrony_entropy,
#     bimodality_coefficient — ALL computed from the flat distribution of
#     WCC values with no reference to temporal order, hence mathematically
#     zeroth-order regardless of which interpretive domain (Intensity vs.
#     Structure) they are assigned to elsewhere.
#     -> Correct null: SIGNAL-LEVEL IAAFT (shuffle raw signals, recompute WCC)
#   L1 (local temporal / run-length structure): dwell_time, switching_rate
#     -> Correct null: WCC-LEVEL IAAFT (shuffle WCC, preserves L0 moments)
#
# THE FLAW ITSELF: using a WCC-level IAAFT null to test L0 features is
# mathematically close to void — IAAFT is constructed to converge toward
# preserving the input's own amplitude distribution, so the null mean/max
# end up almost identical to the observed mean/max essentially by
# construction, regardless of whether real coupling exists. This gives
# the test no meaningful power, even though the resulting p-value need not
# land at exactly 1.0 every time (IAAFT's convergence is not bit-exact).

_NULL_MODEL_L0: frozenset = frozenset((
    "mean_synchrony", "peak_amplitude",
    "synchrony_entropy", "bimodality_coefficient",
))
_NULL_MODEL_L1: frozenset = frozenset(("dwell_time", "switching_rate"))


def _prepare_iaaft_segments(
    sig_A: np.ndarray,
    sig_B: np.ndarray,
    *,
    window_size: int,
    discontinuity_mask: Optional[np.ndarray] = None,
    min_segment_samples: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]:
    """Resolve finite contiguous segments eligible for signal-level IAAFT.

    NaN/Inf positions and explicit discontinuities are hard boundaries. Short
    runs are excluded rather than joined or imputed. The default floor ensures
    every retained segment has at least 20 WCC positions and at least 50 raw
    samples, matching the historical signal-level eligibility floor.
    """
    a = np.asarray(sig_A, dtype=float)
    b = np.asarray(sig_B, dtype=float)
    if int(window_size) != window_size or window_size < 2:
        raise ValueError("window_size must be an integer >= 2")
    minimum = (
        max(50, int(window_size) + 19)
        if min_segment_samples is None else int(min_segment_samples)
    )
    if minimum < max(4, int(window_size)):
        raise ValueError("min_segment_samples must be >= max(4, window_size)")

    geometry = resolve_signal_geometry(a, b, discontinuity_mask)
    all_runs = geometry.segments
    runs = list(geometry.segments_at_least(minimum))
    eligible = np.zeros(a.size, dtype=bool)
    for start, end in runs:
        eligible[start:end] = True

    diagnostics = {
        "mode": "segment_wise_iaaft" if len(all_runs) > 1 or not geometry.analysis_mask.all() else "whole_series_iaaft",
        "min_segment_samples": minimum,
        "n_segments_total": len(all_runs),
        "n_segments_used": len(runs),
        "n_segments_excluded_short": len(all_runs) - len(runs),
        "segment_lengths_used": [end - start for start, end in runs],
        "n_samples_total": int(a.size),
        "n_samples_jointly_finite": int(geometry.joint_finite_mask.sum()),
        "n_samples_eligible": int(eligible.sum()),
        "eligible_fraction": float(eligible.mean()) if eligible.size else 0.0,
    }
    return runs, eligible, diagnostics


def _segmentwise_wcc(
    sig_A: np.ndarray,
    sig_B: np.ndarray,
    runs: Sequence[Tuple[int, int]],
    *,
    window_size: int,
    hz: float,
    window_type: str,
) -> np.ndarray:
    """Compute WCC only inside declared contiguous segments on the full axis."""
    a = np.asarray(sig_A, dtype=float)
    b = np.asarray(sig_B, dtype=float)
    out = np.full(max(0, a.size - window_size + 1), np.nan, dtype=float)
    for start, end in runs:
        segment = sliding_window_wcc(
            a[start:end], b[start:end], window_size=window_size,
            hz=hz, window_type=window_type,
        )
        out[start:start + segment.size] = segment
    return out


def _signal_level_surrogate_test(
    sig_A: np.ndarray,
    sig_B: np.ndarray,
    wcc: np.ndarray,
    hz: float,
    surrogate_n: int = 499,
    alpha: float = 0.05,
    seed: int = 42,
    wcc_window_size: Optional[int] = None,
    window_type: str = "rect",
    discontinuity_mask: Optional[np.ndarray] = None,
    min_segment_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Signal-level IAAFT null for L0 features.

    Null model: IAAFT-shuffle raw signals A and B independently,
    recompute WCC, extract L0 features.  This destroys ALL coupling
    (including L0 moments), providing a valid test of existence.

    Parameters
    ----------
    sig_A, sig_B : np.ndarray
        Raw input signals (before WCC computation).
    wcc : np.ndarray
        Observed WCC series (used for length validation only).
    hz : float
        Sampling rate of WCC.
    surrogate_n : int
        Number of surrogates.
    seed : int
        RNG seed.
    wcc_window_size : int, optional
        Window size used for the ORIGINAL observed WCC computation.
        Strongly recommended to pass explicitly: if omitted, this falls
        back to a heuristic guess (``len(wcc)//10``) that may not match
        the window size actually used upstream, which would introduce a
        smoothing mismatch between the observed WCC and the recomputed
        surrogate WCC series — a confound, not a convenience default.
        A warning is logged whenever the fallback fires.

    Returns
    -------
    dict with keys:
      - p_mean_synchrony, p_peak_amplitude, p_bimodality_coefficient
      - null_mean_synchrony, null_peak_amplitude, null_bimodality_coefficient
      - obs_mean_synchrony, obs_peak_amplitude, obs_bimodality_coefficient
      - n_surrogates, notes

    Each of the three null arrays is masked and counted INDEPENDENTLY —
    one feature's surrogate failing (e.g. bimodality_coefficient
    degenerating on a particular surrogate draw) does not contaminate or
    misalign the denominator/slicing used for the other two features.
    """
    from .feature_definitions import (
        compute_bimodality_coefficient,
        compute_peak_amplitude,
        compute_synchrony_entropy,
        smoothed_wcc,
    )

    # Active guard, not decorative: this function tests exactly the audited
    # L0 set below. Audit M3 (2026-09-13): synchrony_entropy was declared in
    # _NULL_MODEL_L0 but never tested anywhere in the chain — coverage gap
    # between the Axis-D governance table and the implementation. It is now
    # audited here: IAAFT preserves the marginal amplitude distribution of
    # each input signal EXACTLY (rank-adjusted output), so the surrogate WCC
    # value distribution — and hence its entropy — is a valid null draw for
    # the entropy of the observed WCC. fraction_above_threshold and
    # peak_abs_amplitude remain declared-L0-but-not-audited (see the SSoT
    # note in feature_definitions.MATHEMATICAL_TIER); they are threshold-
    # dependent / sign-convention descriptors used descriptively only.
    _tested = frozenset((
        "mean_synchrony", "peak_amplitude",
        "bimodality_coefficient", "synchrony_entropy",
    ))
    assert _tested <= _NULL_MODEL_L0, (
        f"_signal_level_surrogate_test tests {_tested}, which is not a "
        f"subset of _NULL_MODEL_L0 ({_NULL_MODEL_L0}). Update one or the "
        f"other — do not let this drift silently."
    )

    logger = logging.getLogger(__name__)
    wcc = np.asarray(wcc, dtype=float)
    valid_mask = np.isfinite(wcc)
    wcc_valid = wcc[valid_mask]
    n_valid = len(wcc_valid)

    sig_A = np.asarray(sig_A, dtype=float)
    sig_B = np.asarray(sig_B, dtype=float)
    if len(sig_A) < 50 or len(sig_B) < 50:
        logger.warning("Raw signals too short for signal-level null")
        return _empty_result("Raw signals too short")

    if wcc_window_size is None:
        wcc_window_size = max(2, n_valid // 10)
        logger.warning(
            "_signal_level_surrogate_test: wcc_window_size not provided; "
            "falling back to a heuristic guess (len(wcc)//10 = %d). This "
            "may not match the window size used to compute the observed "
            "WCC, which would introduce an obs/null smoothing mismatch. "
            "Pass wcc_window_size explicitly to avoid this.",
            wcc_window_size,
        )

    runs, _, segment_info = _prepare_iaaft_segments(
        sig_A, sig_B,
        window_size=wcc_window_size,
        discontinuity_mask=discontinuity_mask,
        min_segment_samples=min_segment_samples,
    )
    eligible_wcc = _segmentwise_wcc(
        sig_A, sig_B, runs,
        window_size=wcc_window_size, hz=hz, window_type=window_type,
    )
    n_eligible_wcc = int(np.isfinite(eligible_wcc).sum())
    segment_info["n_wcc_points_eligible"] = n_eligible_wcc
    if n_eligible_wcc < 20:
        empty = _empty_result(
            f"Segment-wise WCC too short ({n_eligible_wcc}<20 eligible samples)"
        )
        empty["segmentation"] = segment_info
        return empty
    if n_valid < 20:
        empty = _empty_result(f"Observed WCC too short ({n_valid}<20 samples)")
        empty["segmentation"] = segment_info
        return empty

    # P0-1 fix (2026-07-22): L0 peak MUST match SSoT peak_amplitude
    # (3-point boxcar smoothed argmax), NOT raw np.max.  Using raw max
    # made existence-audit p-values test a different statistic than the
    # feature table / L2 / design-control path report under the same name.
    obs_mean = float(np.mean(wcc_valid))
    obs_peak, _ = compute_peak_amplitude(smoothed_wcc(wcc))  # full series; NaNs handled inside
    if not np.isfinite(obs_peak):
        # Fallback if smoothing path is fully non-finite (should be rare given n_valid gate)
        obs_peak = float(np.max(wcc_valid)) if wcc_valid.size else float("nan")
    obs_bc = compute_bimodality_coefficient(wcc_valid)
    obs_entropy = compute_synchrony_entropy(wcc_valid)

    rng = np.random.default_rng(seed)

    # NaN-initialized (NOT np.zeros): a surrogate that fails partway
    # through (too few finite WCC points) must leave a NaN behind, not a
    # spurious 0.0 that would silently masquerade as a valid near-zero
    # null draw and bias both the rejection count and the count of
    # "valid" surrogates used as the Phipson-Smyth denominator.
    null_mean = np.full(surrogate_n, np.nan)
    null_peak = np.full(surrogate_n, np.nan)
    null_bc = np.full(surrogate_n, np.nan)
    null_entropy = np.full(surrogate_n, np.nan)

    for i in range(surrogate_n):
        # Generate A/B IAAFT independently INSIDE each eligible contiguous
        # segment. NaN gaps and session seams remain fixed on the original time
        # axis; no surrogate can borrow spectrum or samples across a boundary.
        A_s = np.full(sig_A.size, np.nan, dtype=float)
        B_s = np.full(sig_B.size, np.nan, dtype=float)
        for start, end in runs:
            A_s[start:end] = iaaft_surrogate(sig_A[start:end], rng=rng)
            B_s[start:end] = iaaft_surrogate(sig_B[start:end], rng=rng)
        wcc_s = _segmentwise_wcc(
            A_s, B_s, runs,
            window_size=wcc_window_size, hz=hz, window_type=window_type,
        )
        wcc_s_valid = wcc_s[np.isfinite(wcc_s)]
        if len(wcc_s_valid) < 10:
            continue  # null_mean[i]/null_peak[i]/null_bc[i] remain NaN
        null_mean[i] = np.mean(wcc_s_valid)
        # Same peak definition as observed (SSoT smoothed); apply to full
        # surrogate WCC (with NaN seams) so smoothing neighborhood matches obs.
        _npk, _ = compute_peak_amplitude(smoothed_wcc(wcc_s))
        null_peak[i] = _npk if np.isfinite(_npk) else float(np.max(wcc_s_valid))
        null_bc[i] = compute_bimodality_coefficient(wcc_s_valid)
        null_entropy[i] = compute_synchrony_entropy(wcc_s_valid)

    # Each feature gets its OWN finite mask, count, and slice — a
    # degenerate bimodality_coefficient draw must not borrow
    # null_mean's denominator or alignment.
    def _phipson_smyth_p(
        null_arr: np.ndarray, obs_val: float
    ) -> Tuple[float, np.ndarray, int, float]:
        finite_null = null_arr[np.isfinite(null_arr)]
        n = finite_null.size
        if n < int(surrogate_n * 0.8):
            logger.warning(f"Only {n}/{surrogate_n} valid surrogates for this feature")
        if n == 0 or not np.isfinite(obs_val):
            return 1.0, finite_null, 0, float("nan")
        # Two-tailed Phipson-Smyth (BUG-4 fix): unify L0 with the L1
        # _wcc_level_surrogate_test, which already uses this conservative
        # two-tailed form.  An upper-tail was methodologically arguable for
        # an existence test, but an inconsistent tail policy across the L0/L1
        # family is the actual defect; two-tail is the conservative,
        # consistent choice and matches tests/validation/test_per_feature_significance.py.
        p_ge = (np.sum(finite_null >= obs_val) + 1) / (n + 1)
        p_le = (np.sum(finite_null <= obs_val) + 1) / (n + 1)
        tail_probability = float(min(p_ge, p_le))
        p = float(min(1.0, 2.0 * tail_probability))
        return p, finite_null, n, tail_probability

    p_mean, null_mean_valid, n_mean, q_mean = _phipson_smyth_p(null_mean, obs_mean)
    p_peak, null_peak_valid, n_peak, q_peak = _phipson_smyth_p(null_peak, obs_peak)
    p_bc, null_bc_valid, n_bc, q_bc = _phipson_smyth_p(null_bc, obs_bc)
    p_ent, null_ent_valid, n_ent, q_ent = _phipson_smyth_p(null_entropy, obs_entropy)

    def _mc_precision(n: int, tail_probability: float) -> Dict[str, float]:
        if n <= 0 or not np.isfinite(tail_probability):
            return {
                "n_valid_surrogates": int(n),
                "min_attainable_two_sided_p": float("nan"),
                "approx_monte_carlo_se": float("nan"),
            }
        # The reported p is two-sided (= 2 * smaller tail), hence both the
        # minimum attainable p and the MCSE carry a factor of two.
        return {
            "n_valid_surrogates": int(n),
            "min_attainable_two_sided_p": float(min(1.0, 2.0 / (n + 1))),
            "approx_monte_carlo_se": float(
                2.0 * np.sqrt(tail_probability * (1.0 - tail_probability) / n)
            ),
        }

    precision = {
        "mean_synchrony": _mc_precision(n_mean, q_mean),
        "peak_amplitude": _mc_precision(n_peak, q_peak),
        "bimodality_coefficient": _mc_precision(n_bc, q_bc),
        "synchrony_entropy": _mc_precision(n_ent, q_ent),
    }

    # Per-feature significance — callers (e.g. InferencePipeline.run_full_cascade)
    # need per-feature pass rates to track frozen primary endpoints
    # rather than an opaque OR across the family.
    per_feature_significant = {
        "mean_synchrony": bool(np.isfinite(p_mean) and p_mean < alpha),
        "peak_amplitude": bool(np.isfinite(p_peak) and p_peak < alpha),
        "bimodality_coefficient": bool(np.isfinite(p_bc) and p_bc < alpha),
        "synchrony_entropy": bool(np.isfinite(p_ent) and p_ent < alpha),
    }

    return {
        "p_mean_synchrony": p_mean,
        "p_peak_amplitude": p_peak,
        "p_bimodality_coefficient": p_bc,
        "p_synchrony_entropy": p_ent,
        "null_mean_synchrony": null_mean_valid,
        "null_peak_amplitude": null_peak_valid,
        "null_bimodality_coefficient": null_bc_valid,
        "null_synchrony_entropy": null_ent_valid,
        "obs_mean_synchrony": obs_mean,
        "obs_peak_amplitude": obs_peak,
        "obs_bimodality_coefficient": obs_bc,
        "obs_synchrony_entropy": obs_entropy,
        "n_surrogates": surrogate_n,
        "n_valid_mean_synchrony": n_mean,
        "n_valid_peak_amplitude": n_peak,
        "n_valid_bimodality_coefficient": n_bc,
        "n_valid_synchrony_entropy": n_ent,
        "null_model": "signal_level_iaaft",
        "per_feature_significant": per_feature_significant,
        "alpha": alpha,
        "surrogate_precision": precision,
        "segmentation": segment_info,
        "notes": "",
    }


def _wcc_level_surrogate_test(
    wcc: np.ndarray,
    hz: float = 1.0,
    surrogate_n: int = 499,
    alpha: float = 0.05,
    seed: int = 42,
    features: Optional[Sequence[str]] = None,
    wcc_window_sec: Optional[float] = None,
    min_wcc_points: int = 30,
    null_model: str = "iaaft",
    block_size: Optional[int] = None,
    threshold: float = ONSET_THRESHOLD,
    gap_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """WCC-level null for L1 features (dwell_time, switching_rate).

    Three null models are supported:

    * ``null_model="iaaft"``: IAAFT-shuffle the WCC series. Preserves L0
      moments and approximately preserves the power spectrum.
    * ``null_model="block_permutation"``: Divide WCC into blocks and
      permute. Preserves local autocorrelation within blocks.
    * ``null_model="state_shuffle"``: Binarize the WCC into elevated/baseline
      segments and shuffle the order of these segments. Preserves the exact
      dwell-time distribution but destroys temporal structure (L1 structure
      null).

    Parameters
    ----------
    wcc : np.ndarray
        Observed WCC time series.
    features : sequence of str, optional
        Which L1 features to extract from surrogates.
    wcc_window_sec : float, optional
        Duration of the WCC sliding window in seconds.
    min_wcc_points : int
        Minimum number of finite WCC points required.
    null_model : {"iaaft", "block_permutation", "state_shuffle"}
        L1 null model.
    block_size : int or None
        Block size for block permutation.
    threshold : float
        Threshold for 'state_shuffle'.
    gap_policy : {"segment", "merge_valid"} or None
        Audit M5 (2026-09-13): forwarded to ``extract_features`` for BOTH
        the observed and the surrogate draws. The legacy behaviour
        NaN-compressed the WCC before recomputing dwell_time /
        switching_rate, which is equivalent to forcing ``merge_valid`` —
        while the scientific canonical path (pipeline_bridge) computes the
        feature table with ``gap_policy="segment"`` on discontinuity-masked
        data. The same dyad could therefore report different dwell_time
        values inside the L1 test and in the manuscript feature table.
        Passing the caller's policy aligns the two; ``None`` preserves the
        extract_features default (merge_valid) for backward compatibility
        with frozen artifacts.

    Raises
    ------
    ValueError
        If ``features`` contains anything outside ``_NULL_MODEL_L1`` or if
        ``null_model`` is unsupported.
    """
    from .feature_definitions import extract_features
    from .surrogate import block_permutation_surrogate

    if features is not None:
        _requested = frozenset(features)
        _bad = _requested - _NULL_MODEL_L1
        if _bad:
            raise ValueError(
                f"_wcc_level_surrogate_test received feature(s) {sorted(_bad)} "
                f"that are not in _NULL_MODEL_L1 ({sorted(_NULL_MODEL_L1)}). "
                f"A WCC-level IAAFT null is mathematically invalid for L0 "
                f"features (it trivially preserves their value) — use "
                f"_signal_level_surrogate_test for those instead."
            )

    logger = logging.getLogger(__name__)
    wcc = np.asarray(wcc, dtype=float)
    valid_mask = np.isfinite(wcc)
    wcc_valid = wcc[valid_mask]
    n_valid = len(wcc_valid)

    if n_valid < min_wcc_points:
        # gstack Finding 6: the early-return path must mirror the NORMAL-path
        # result shape (L1 keys: p_dwell_time / p_switching_rate + their
        # null_/obs_ companions). Otherwise downstream consumers that index those
        # keys (or build DataFrames from the L1 key set) raise KeyError on a short
        # trace — crashing a whole batch if even one dyad falls below
        # min_wcc_points. We deliberately do NOT call the shared _empty_result()
        # here: that helper emits L0-shaped keys (p_mean_synchrony /
        # p_peak_amplitude / p_bimodality_coefficient) which is correct only for
        # the signal-level (L0) test that also depends on it.
        _eff_features = ("dwell_time", "switching_rate") if features is None else tuple(features)
        return {
            **{f"p_{f}": 1.0 for f in _eff_features},
            **{f"null_{f}": np.array([]) for f in _eff_features},
            **{f"obs_{f}": np.nan for f in _eff_features},
            "n_surrogates": 0,
            "null_model": "none",
            "per_feature_significant": {f: False for f in _eff_features},
            "alpha": alpha,
            "applicable": False,
            "notes": f"WCC too short ({n_valid} < {min_wcc_points} samples)",
        }

    if features is None:
        features = ("dwell_time", "switching_rate")

    # Resolve wcc_window_sec: required by extract_features for DTW
    wcc_window_sec_heuristic = wcc_window_sec is None
    if wcc_window_sec_heuristic:
        wcc_window_sec = n_valid / (hz * 10.0)
        logger.warning(
            f"_wcc_level_surrogate_test: wcc_window_sec not provided, "
            f"using heuristic {wcc_window_sec:.1f}s — may introduce "
            f"window-size mismatch confound"
        )

    if null_model not in ("iaaft", "block_permutation", "state_shuffle"):
        raise ValueError(f"null_model must be 'iaaft', 'block_permutation' or 'state_shuffle', got {null_model!r}")
    if null_model == "state_shuffle":
        # BUG-4 degeneracy warning (2026-09-08): state_shuffle re-orders
        # whole elevated/baseline segments, so dwell_time and switching_rate
        # — the two L1 statistics this test reports — are INVARIANT under
        # it by construction (observed == null, p == 1.0 on every input).
        # Keep it available only for order-sensitive custom diagnostics.
        warnings.warn(
            "null_model='state_shuffle' preserves the multiset of "
            "elevated/baseline segments, so dwell_time and switching_rate "
            "are invariant under it and their p-values degenerate to 1.0. "
            "For L1 hypothesis testing use null_model='iaaft' (the "
            "protocol null, V1_PROTOCOL §10).",
            UserWarning,
            stacklevel=2,
        )

    obs_feats = extract_features(
        wcc, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold,
        gap_policy=gap_policy,
    )
    rng = np.random.default_rng(seed)

    # Collect null feature values
    null_values: Dict[str, list] = {f: [] for f in features}

    for i in range(surrogate_n):
        if null_model == "block_permutation":
            wcc_s = block_permutation_surrogate(wcc_valid, rng=rng, block_size=block_size)
        elif null_model == "state_shuffle":
            from .surrogate import state_transition_shuffle_surrogate
            wcc_s = state_transition_shuffle_surrogate(
                wcc_valid, threshold=threshold, rng=rng,
                hysteresis_delta=SWITCHING_HYSTERESIS_DELTA)
        else:
            wcc_s = iaaft_surrogate(wcc_valid, rng=rng)
        feats_s = extract_features(
            wcc_s, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold,
            gap_policy=gap_policy,
        )
        for f in features:
            v = getattr(feats_s, f, np.nan)
            if np.isfinite(v):
                null_values[f].append(v)

    # Compute p-values (correct TWO-TAILED permutation p; Phipson & Smyth, 2010)
    result = {
        "null_model": f"wcc_level_{null_model}",
        "n_surrogates": surrogate_n,
        # Audit m11 (2026-09-14): surface the actual window duration used
        # (and whether it was the heuristic fallback) so a mismatch with
        # the caller's intended window is visible in the output artifact,
        # not only in the log.
        "wcc_window_sec": float(wcc_window_sec),
        "wcc_window_sec_heuristic": bool(wcc_window_sec_heuristic),
    }
    feature_p_values = []
    for f in features:
        obs_v = getattr(obs_feats, f, np.nan)
        null_arr = np.array(null_values[f])
        if len(null_arr) < 10 or not np.isfinite(obs_v):
            p = 1.0
            result[f"p_{f}"] = p
            result[f"null_{f}"] = np.array([])
        else:
            n = len(null_arr)
            p_ge = (np.sum(null_arr >= obs_v) + 1) / (n + 1)
            p_le = (np.sum(null_arr <= obs_v) + 1) / (n + 1)
            p = float(min(1.0, 2.0 * min(p_ge, p_le)))
            result[f"p_{f}"] = p
            result[f"null_{f}"] = null_arr
        result[f"obs_{f}"] = float(obs_v) if np.isfinite(obs_v) else np.nan
        feature_p_values.append(p)

    # Per-feature significance for downstream per-endpoint tracking.
    per_feature_significant = {}
    for f in features:
        per_feature_significant[f] = bool(
            np.isfinite(result.get(f"p_{f}", 1.0))
            and result[f"p_{f}"] < alpha
        )

    result["per_feature_significant"] = per_feature_significant
    result["alpha"] = alpha
    result["applicable"] = True
    result["notes"] = ""
    return result


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return a failed surrogate test result."""
    return {
        "p_mean_synchrony": 1.0,
        "p_peak_amplitude": 1.0,
        "p_bimodality_coefficient": 1.0,
        "p_synchrony_entropy": 1.0,
        "null_mean_synchrony": np.array([]),
        "null_peak_amplitude": np.array([]),
        "null_bimodality_coefficient": np.array([]),
        "null_synchrony_entropy": np.array([]),
        "obs_mean_synchrony": np.nan,
        "obs_peak_amplitude": np.nan,
        "obs_bimodality_coefficient": np.nan,
        "obs_synchrony_entropy": np.nan,
        "n_surrogates": 0,
        "n_valid_mean_synchrony": 0,
        "n_valid_peak_amplitude": 0,
        "n_valid_bimodality_coefficient": 0,
        "n_valid_synchrony_entropy": 0,
        "null_model": "none",
        "per_feature_significant": {},
        "alpha": np.nan,
        "notes": reason,
    }


def wcc_surrogate_test(
    wcc: np.ndarray,
    hz: float = 1.0,
    surrogate_n: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "iaaft",
    raw_signals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    wcc_window_size: Optional[int] = None,
    wcc_window_sec: Optional[float] = None,
    window_type: str = "rect",
    min_wcc_points: int = 30,
    null_model: str = "iaaft",
    block_size: Optional[int] = None,
    threshold: float = ONSET_THRESHOLD,
    discontinuity_mask: Optional[np.ndarray] = None,
    min_segment_samples: Optional[int] = None,
    gap_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Test significance of WCC features using surrogate data.

    Parameters
    ----------
    wcc : np.ndarray
        Observed WCC time series.
    hz : float
        Sampling rate of WCC (Hz).
    surrogate_n : int
        Number of surrogate iterations.
    alpha : float
        Significance threshold.
    seed : int
        Random seed.
    method : str
        Surrogate method (currently only "iaaft").
    null_model : {"iaaft", "block_permutation", "state_shuffle"}
        L1 WCC-level null model (used only when ``raw_signals`` is None).
        Default "iaaft".
    block_size : int or None
        Block size for block-permutation L1 null. If None, derived from WCC
        length.
    raw_signals : tuple of (np.ndarray, np.ndarray), optional
        If provided (sig_A, sig_B), uses SIGNAL-LEVEL IAAFT null
        (correct for L0 features: mean_synchrony, peak_amplitude).
        If None, uses WCC-LEVEL IAAFT null (correct for L1 features)
        but EMITS A WARNING if testing L0 features.
    wcc_window_size : int, optional
        Window size in *samples* used for WCC recomputation in
        signal-level null. Required for correct surrogate WCC.
    wcc_window_sec : float, optional
        Window duration in *seconds* used for feature extraction
        (DTW step parameterisation). Derivable from wcc_window_size
        as ``wcc_window_size / hz`` if omitted. Required for L1 null
        calls to ``extract_features()``.
    min_wcc_points : int
        Minimum number of finite WCC points required. Default 30.
        Only applies to WCC-level null (L1).
    threshold : float
        Threshold for 'state_shuffle'.

    Returns
    -------
    result : dict
        Dictionary with p-values and null distributions.
    """
    logger = logging.getLogger(__name__)

    # Derive wcc_window_sec from wcc_window_size if not provided
    if wcc_window_sec is None and wcc_window_size is not None and hz > 0:
        wcc_window_sec = wcc_window_size / hz

    if raw_signals is not None:
        # SIGNAL-LEVEL null (correct for L0 features)
        return _signal_level_surrogate_test(
            sig_A=raw_signals[0],
            sig_B=raw_signals[1],
            wcc=wcc,
            hz=hz,
            surrogate_n=surrogate_n,
            alpha=alpha,
            seed=seed,
            wcc_window_size=wcc_window_size,
            window_type=window_type,
            discontinuity_mask=discontinuity_mask,
            min_segment_samples=min_segment_samples,
        )
    else:
        # WCC-LEVEL null (correct for L1 features)
        logger.debug(
            "wcc_surrogate_test called without raw_signals — "
            "using WCC-level null for L1 features "
            "(dwell_time, switching_rate)."
        )
        return _wcc_level_surrogate_test(
            wcc=wcc,
            hz=hz,
            surrogate_n=surrogate_n,
            alpha=alpha,
            seed=seed,
            features=("dwell_time", "switching_rate"),
            wcc_window_sec=wcc_window_sec,
            min_wcc_points=min_wcc_points,
            null_model=null_model,
            block_size=block_size,
            threshold=threshold,
            gap_policy=gap_policy,
        )


# ---------------------------------------------------------------------------
# Surrogate-derived threshold computation (from raw signals)
# ---------------------------------------------------------------------------

def compute_surrogate_threshold_from_signals(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    hz: float,
    wcc_window_size: int,
    surrogate_n: int = 200,
    percentile: float = SURROGATE_THRESHOLD_PERCENTILE,
    seed: int = 42,
    discontinuity_mask: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """Compute a per-dyad surrogate-derived onset threshold from raw signals.

    Generates ``surrogate_n`` IAAFT surrogates of ``sig_a`` and ``sig_b``,
    computes WCC for each surrogate pair, pools all finite WCC values,
    and returns the ``percentile``-th quantile.  The result is the WCC
    level this dyad would reach by chance at the chosen false-positive
    rate -- a zero-hypothesis-grounded cut-off rather than an arbitrary
    r-metric anchor (Lykken & Venables 1971; Ben-Shakhar 1985).

    This function encapsulates the full pipeline:
    raw signals → IAAFT surrogates → surrogate WCC → percentile threshold.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Raw physiological signals (finite, same length).
    hz : float
        Sampling rate.
    wcc_window_size : int
        WCC window length in samples.
    surrogate_n : int
        Number of IAAFT replicates (default 200).
    percentile : float
        Quantile for the threshold (default 95).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Tuple[float, bool]
        ``(threshold, is_surrogate_derived)``.  ``threshold`` falls back to
        ``ONSET_THRESHOLD`` (0.5) when the underlying surrogate distribution
        is degenerate (too few finite values) or contaminated by periodicity
        / strong autocorrelation (see :func:`feature_definitions.
        compute_surrogate_threshold`); both fallback paths emit a warning,
        and ``is_surrogate_derived`` is ``False`` exactly when a fallback
        fired.
    """
    rng = np.random.default_rng(seed)
    surrogate_wccs: List[np.ndarray] = []

    # Guard: surrogate computation requires finite raw signals
    sig_a = np.asarray(sig_a, dtype=float)
    sig_b = np.asarray(sig_b, dtype=float)
    if not (np.all(np.isfinite(sig_a)) and np.all(np.isfinite(sig_b))):
        logger.warning(
            "compute_surrogate_threshold_from_signals: non-finite raw "
            "signals detected — cannot build IAAFT surrogates. Falling back "
            "to fixed ONSET_THRESHOLD=%s (is_surrogate_derived=False).",
            ONSET_THRESHOLD,
        )
        return ONSET_THRESHOLD, False

    for _ in range(surrogate_n):
        a_surr = iaaft_surrogate(sig_a, rng)
        b_surr = iaaft_surrogate(sig_b, rng)
        wcc_s = sliding_window_wcc(a_surr, b_surr, wcc_window_size, hz)
        # Exclude the same boundary windows as the observed WCC so the
        # surrogate-derived onset threshold is computed over valid windows.
        wcc_s = _apply_discontinuity_mask(wcc_s, discontinuity_mask, wcc_window_size)
        surrogate_wccs.append(wcc_s)

    # Memory guard (gstack OOM #6): warn before allocating the
    # (surrogate_n, n_timepoints) surrogate matrix if it would exceed the
    # guard budget. Purely a warning — computation is unchanged for normal
    # inputs.
    if surrogate_wccs:
        _n_tp = surrogate_wccs[0].shape[0]
        _est = len(surrogate_wccs) * _n_tp * surrogate_wccs[0].dtype.itemsize
        if _est > SURROGATE_MEM_GUARD_BYTES:
            logger.warning(
                "compute_surrogate_threshold_from_signals: surrogate matrix "
                "would allocate ~%.1f MiB (surrogate_n=%d x %d timepoints). "
                "This may OOM; consider lowering surrogate_n or truncating signals.",
                _est / (1024 * 1024), len(surrogate_wccs), _n_tp,
            )
    surrogate_matrix = np.vstack(surrogate_wccs)  # (surrogate_n, n_timepoints)
    return compute_surrogate_threshold(surrogate_matrix, percentile=percentile)


# ---------------------------------------------------------------------------

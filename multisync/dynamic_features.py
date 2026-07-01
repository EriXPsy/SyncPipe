"""
Dynamic feature extraction — operationalizing synchrony dynamics.

Theoretical Framework
--------------------
SyncPipe conceptualizes interpersonal synchrony through a
morphology-aware dimensional model (INTENSITY / STRUCTURE / TIMING;
see ``docs/DIMENSIONAL_MODEL.md``).  Synchrony Epochs can take
multiple forms — single-peak, oscillatory, sustained, asymmetric decay
and features are classified into three tiers (CORE /
CONDITIONAL / REFERENCE) reflecting cross-morphology robustness.

Feature math lives in :mod:`multisync.feature_definitions` (SSoT).
This module is responsible for WCC computation, surrogate generation,
and thin orchestration wrappers that delegate to the SSoT.

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
import logging
import warnings

from .surrogate import iaaft_surrogate, ft_surrogate, prtf_surrogate  # noqa: F401  # re-export

# Feature math lives in feature_definitions (SSoT); re-export DynamicFeatures
# for backward-compatible imports (multisync.DynamicFeatures, core.py, etc.)
from .feature_definitions import (
    DynamicFeatures,
    extract_features as _ssot_extract_features,
    compute_surrogate_threshold,
    ONSET_THRESHOLD,
    SURROGATE_THRESHOLD_PERCENTILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding-window WCC (Weighted Cross-Correlation)
# ---------------------------------------------------------------------------

def sliding_window_wcc(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    lag_samples: int = 0,
    step_samples: int = 0,
    min_valid_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute sliding-window cross-correlation (WCC) between x and y.

    For each window position, computes Pearson correlation within the window.
    Uses cumsum-based O(n) memory implementation when there are no NaN values
    and step_samples <= 1; falls back to stride_tricks (O(n*w) memory) when NaN values are present.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length).
    window_size : int
        Window size in samples.
    hz : float
        Sampling rate (for time axis, not used in computation directly).
    lag_samples : int
        Lag y by this many samples before correlating.
    step_samples : int
        Step size in samples between consecutive windows.
        Default 0 means every sample (step=1, no skipping).
        When > 1, only computes WCC at positions 0, step_samples, 2*step_samples, ...
        This matches the behavior of converter scripts that used while-loop with step.
    min_valid_ratio : float
        Minimum fraction of valid (non-NaN) pairs within a window for the
        correlation to be computed (vs. returning NaN). Default 0.5 (50%).
        Only applies when NaN values are present (stride_tricks path).

    Returns
    -------
    wcc : 1-D array
        Cross-correlation at each window position.
        Length depends on step_samples:
        - step_samples <= 1: len(x) - window_size + 1
        - step_samples > 1: ceil((len(x) - window_size + 1) / step_samples)
    """
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x and y must have same length: {n} vs {len(y)}")
    if window_size > n:
        return np.array([], dtype=float)

    # Apply lag
    if lag_samples > 0:
        y_lagged = np.full(n, np.nan)
        y_lagged[lag_samples:] = y[:-lag_samples]
    elif lag_samples < 0:
        y_lagged = np.full(n, np.nan)
        y_lagged[:lag_samples] = y[-lag_samples:]
    else:
        y_lagged = y

    # Full-resolution WCC first (no step)
    has_nan = bool(np.isnan(x).any() or np.isnan(y_lagged).any())

    if has_nan:
        mem_estimate = (n - window_size + 1) * window_size * 8 * 4
        if mem_estimate > 1e9:
            logger.warning(
                f"sliding_window_wcc: large memory estimate ({mem_estimate/1e9:.1f} GB) "
                f"due to NaN values forcing stride_tricks fallback. "
                f"Consider filling NaN before calling this function."
            )
        wcc_full = _sliding_window_wcc_stride(x, y_lagged, window_size, min_valid_ratio)
    else:
        wcc_full = _sliding_window_wcc_cumsum(x, y_lagged, window_size)

    # Apply step if requested
    if step_samples > 1:
        return wcc_full[::step_samples]
    return wcc_full


def _sliding_window_wcc_cumsum(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Cumsum-based sliding-window Pearson correlation.
    Assumes no NaN values in x or y.
    Memory: O(n) instead of O(n*w).

    Note
    ----
    Input signals are **pre-demeaned** (using their global means) before
    cumsum to avoid catastrophic cancellation when the signal magnitude
    is much larger than its variance.

    Because we use the **global** mean (not the per-window mean) for
    pre-demeaning, ``mean_x`` and ``mean_y`` computed below within each
    window are NOT exactly zero — therefore the correction term
    ``mean_x * mean_y`` in the covariance formula MUST be kept.  Removing
    it (e.g. ``cov = sum_xy / w``) silently introduces a bias that
    grows with window-to-global-mean discrepancy.
    """
    n = len(x)
    w = float(window_size)

    # ------------------------------------------------------------------
    # Pre-demean using global means to avoid catastrophic cancellation
    # in cumsum.  Window-level means are still recovered correctly
    # because we use sum_xy/w - mean_x*mean_y below.
    # ------------------------------------------------------------------
    mean_x_global = float(np.mean(x))
    mean_y_global = float(np.mean(y))
    x_demeaned = x - mean_x_global
    y_demeaned = y - mean_y_global

    # Cumulative sums of demeaned signals
    cumsum_x = np.cumsum(x_demeaned)
    cumsum_y = np.cumsum(y_demeaned)
    cumsum_xy = np.cumsum(x_demeaned * y_demeaned)
    cumsum_x2 = np.cumsum(x_demeaned ** 2)
    cumsum_y2 = np.cumsum(y_demeaned ** 2)

    # Prepend 0 so that range sums [i, i+w) = cumsum[i+w] - cumsum[i]
    cumsum_x = np.concatenate([[0.0], cumsum_x])
    cumsum_y = np.concatenate([[0.0], cumsum_y])
    cumsum_xy = np.concatenate([[0.0], cumsum_xy])
    cumsum_x2 = np.concatenate([[0.0], cumsum_x2])
    cumsum_y2 = np.concatenate([[0.0], cumsum_y2])

    # Window indices
    i = np.arange(n - window_size + 1)
    i_end = i + window_size

    sum_x = cumsum_x[i_end] - cumsum_x[i]
    sum_y = cumsum_y[i_end] - cumsum_y[i]
    sum_xy = cumsum_xy[i_end] - cumsum_xy[i]
    sum_x2 = cumsum_x2[i_end] - cumsum_x2[i]
    sum_y2 = cumsum_y2[i_end] - cumsum_y2[i]

    # Window-level means (on demeaned signal — close to 0 but not exact)
    mean_x = sum_x / w
    mean_y = sum_y / w

    # ------------------------------------------------------------------
    # FIX: correct Pearson covariance.
    # Previous (buggy) version:  cov = sum_xy / w          (missing -mean_x*mean_y)
    # Correct Pearson formula: cov = sum_xy/w - mean_x*mean_y
    # ------------------------------------------------------------------
    cov = sum_xy / w - mean_x * mean_y
    var_x = sum_x2 / w - mean_x ** 2
    var_y = sum_y2 / w - mean_y ** 2

    # Numerical safety: clamp tiny negatives caused by floating point
    var_x = np.maximum(var_x, 0.0)
    var_y = np.maximum(var_y, 0.0)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    denom = std_x * std_y

    wcc = np.full_like(sum_x, np.nan)
    valid = denom > 1e-10
    wcc[valid] = cov[valid] / denom[valid]
    return np.clip(wcc, -1.0, 1.0)


def _sliding_window_wcc_stride(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    min_valid_ratio: float = 0.5,
) -> np.ndarray:
    """
    Stride-tricks WCC with pairwise deletion for NaN handling.

    NaN strategy:
      1. Compute valid_ratio = fraction of PAIRWISE-valid points per window.
      2. Windows with valid_ratio < min_valid_ratio → NaN (quality gate).
      3. Windows passing the gate use ONLY the pairwise-valid points to
         compute Pearson r (pairwise deletion), so partial-NaN windows
         still yield a valid WCC value rather than propagating NaN.

    This is more robust than listwise deletion (entire window NaN if ANY
    point is NaN) while still enforcing a minimum-data quality threshold.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    x_windows = sliding_window_view(x, window_size)
    y_windows = sliding_window_view(y, window_size)
    n_windows = x_windows.shape[0]

    # Pairwise valid mask: True where BOTH x and y are finite
    pair_valid = ~(np.isnan(x_windows) | np.isnan(y_windows))  # (n_windows, window_size)
    n_pair_valid = pair_valid.sum(axis=1)
    valid_ratio = n_pair_valid / window_size
    passes_gate = valid_ratio >= min_valid_ratio

    wcc = np.full(n_windows, np.nan)

    if not np.any(passes_gate):
        return wcc

    # For windows passing the gate, compute WCC using pairwise-valid points.
    # Use masked arrays for vectorized pairwise deletion.
    xv = np.ma.array(x_windows[passes_gate], mask=~pair_valid[passes_gate])
    yv = np.ma.array(y_windows[passes_gate], mask=~pair_valid[passes_gate])

    # Pairwise means (ignoring masked values)
    x_means = xv.mean(axis=1, keepdims=True)
    y_means = yv.mean(axis=1, keepdims=True)
    x_centered = xv - x_means
    y_centered = yv - y_means

    # Pairwise std and covariance
    x_var = (x_centered ** 2).sum(axis=1)
    y_var = (y_centered ** 2).sum(axis=1)
    cov_xy = (x_centered * y_centered).sum(axis=1)

    denom = np.sqrt(x_var * y_var)
    # Avoid division by zero; np.ma handles this but we want explicit control
    denom_safe = np.where(denom > 1e-10, denom, 1.0)
    wcc_valid = cov_xy / denom_safe
    # Where denom was zero (constant window), set to NaN
    wcc_valid = np.where(denom > 1e-10, wcc_valid, np.nan)

    # Convert masked array result to plain ndarray, fill masked with NaN
    wcc_valid = np.ma.filled(wcc_valid, np.nan)
    wcc_valid = np.clip(wcc_valid, -1.0, 1.0)
    wcc[passes_gate] = wcc_valid

    return wcc


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


def _signal_level_surrogate_test(
    sig_A: np.ndarray,
    sig_B: np.ndarray,
    wcc: np.ndarray,
    hz: float,
    surrogate_n: int = 499,
    alpha: float = 0.05,
    seed: int = 42,
    wcc_window_size: Optional[int] = None,
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
    from .feature_definitions import compute_bimodality_coefficient

    # Active guard, not decorative: this function ONLY ever tests
    # {mean_synchrony, peak_amplitude, bimodality_coefficient}. Assert
    # that set is consistent with _NULL_MODEL_L0 so the constant cannot
    # silently drift out of sync with what this function actually does.
    _tested = frozenset(("mean_synchrony", "peak_amplitude", "bimodality_coefficient"))
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

    if n_valid < 20:
        return _empty_result(f"WCC too short ({n_valid}<20 samples)")

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

    obs_mean = float(np.mean(wcc_valid))
    obs_peak = float(np.max(wcc_valid))
    obs_bc = compute_bimodality_coefficient(wcc_valid)

    rng = np.random.default_rng(seed)

    # NaN-initialized (NOT np.zeros): a surrogate that fails partway
    # through (too few finite WCC points) must leave a NaN behind, not a
    # spurious 0.0 that would silently masquerade as a valid near-zero
    # null draw and bias both the rejection count and the count of
    # "valid" surrogates used as the Phipson-Smyth denominator.
    null_mean = np.full(surrogate_n, np.nan)
    null_peak = np.full(surrogate_n, np.nan)
    null_bc = np.full(surrogate_n, np.nan)

    for i in range(surrogate_n):
        # Signal-level IAAFT: shuffle raw signals, then recompute WCC
        A_s = iaaft_surrogate(sig_A, rng=rng)
        B_s = iaaft_surrogate(sig_B, rng=rng)
        wcc_s = sliding_window_wcc(A_s, B_s, window_size=wcc_window_size, hz=hz)
        wcc_s_valid = wcc_s[np.isfinite(wcc_s)]
        if len(wcc_s_valid) < 10:
            continue  # null_mean[i]/null_peak[i]/null_bc[i] remain NaN
        null_mean[i] = np.mean(wcc_s_valid)
        null_peak[i] = np.max(wcc_s_valid)
        null_bc[i] = compute_bimodality_coefficient(wcc_s_valid)

    # Each feature gets its OWN finite mask, count, and slice — a
    # degenerate bimodality_coefficient draw must not borrow
    # null_mean's denominator or alignment.
    def _phipson_smyth_p(null_arr: np.ndarray, obs_val: float) -> Tuple[float, np.ndarray, int]:
        finite_null = null_arr[np.isfinite(null_arr)]
        n = finite_null.size
        if n < int(surrogate_n * 0.8):
            logger.warning(f"Only {n}/{surrogate_n} valid surrogates for this feature")
        if n == 0 or not np.isfinite(obs_val):
            return 1.0, finite_null, 0
        p = (np.sum(finite_null >= obs_val) + 1) / (n + 1)
        return float(p), finite_null, n

    p_mean, null_mean_valid, n_mean = _phipson_smyth_p(null_mean, obs_mean)
    p_peak, null_peak_valid, n_peak = _phipson_smyth_p(null_peak, obs_peak)
    p_bc, null_bc_valid, n_bc = _phipson_smyth_p(null_bc, obs_bc)

    # Per-feature significance — callers (e.g. InferencePipeline.run_full_cascade)
    # need per-feature pass rates to track pre-registered primary endpoints
    # rather than an opaque OR across the family.
    per_feature_significant = {
        "mean_synchrony": bool(np.isfinite(p_mean) and p_mean < alpha),
        "peak_amplitude": bool(np.isfinite(p_peak) and p_peak < alpha),
        "bimodality_coefficient": bool(np.isfinite(p_bc) and p_bc < alpha),
    }

    return {
        "p_mean_synchrony": p_mean,
        "p_peak_amplitude": p_peak,
        "p_bimodality_coefficient": p_bc,
        "null_mean_synchrony": null_mean_valid,
        "null_peak_amplitude": null_peak_valid,
        "null_bimodality_coefficient": null_bc_valid,
        "obs_mean_synchrony": obs_mean,
        "obs_peak_amplitude": obs_peak,
        "obs_bimodality_coefficient": obs_bc,
        "n_surrogates": surrogate_n,
        "n_valid_mean_synchrony": n_mean,
        "n_valid_peak_amplitude": n_peak,
        "n_valid_bimodality_coefficient": n_bc,
        "null_model": "signal_level_iaaft",
        "per_feature_significant": per_feature_significant,
        "alpha": alpha,
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
) -> Dict[str, Any]:
    """WCC-level null for L1 features (dwell_time, switching_rate).

    Two null models are supported:

    * ``null_model="iaaft"`` (default): IAAFT-shuffle the WCC series. This
      preserves L0 moments (mean, peak, distribution shape) and destroys
      run-length / autoregressive structure.
    * ``null_model="block_permutation"``: Divide the WCC into blocks and
      permute the blocks. This preserves local autocorrelation within each
      block while destroying longer-run structure. It is a more conservative
      null when the WCC has strong local autocorrelation and is the recommended
      robustness check for L1 inference.

    Parameters
    ----------
    wcc : np.ndarray
        Observed WCC time series.
    features : sequence of str, optional
        Which L1 features to extract from surrogates.
        Default: ("dwell_time", "switching_rate"). Every requested feature
        MUST be a member of ``_NULL_MODEL_L1``.
    wcc_window_sec : float, optional
        Duration of the WCC sliding window in seconds.
    min_wcc_points : int
        Minimum number of finite WCC points required. Default 30.
    null_model : {"iaaft", "block_permutation"}
        L1 null model.
    block_size : int or None
        Block size for block permutation. If None, ``max(2, int(sqrt(n)))``.

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
        return _empty_result(
            f"WCC too short ({n_valid} < {min_wcc_points} samples)"
        )

    if features is None:
        features = ("dwell_time", "switching_rate")

    # Resolve wcc_window_sec: required by extract_features for DTW
    if wcc_window_sec is None:
        wcc_window_sec = n_valid / (hz * 10.0)
        logger.warning(
            f"_wcc_level_surrogate_test: wcc_window_sec not provided, "
            f"using heuristic {wcc_window_sec:.1f}s — may introduce "
            f"window-size mismatch confound"
        )

    if null_model not in ("iaaft", "block_permutation"):
        raise ValueError(f"null_model must be 'iaaft' or 'block_permutation', got {null_model!r}")

    obs_feats = extract_features(wcc_valid, hz=hz, wcc_window_sec=wcc_window_sec)
    rng = np.random.default_rng(seed)

    # Collect null feature values
    null_values: Dict[str, list] = {f: [] for f in features}

    for i in range(surrogate_n):
        if null_model == "block_permutation":
            wcc_s = block_permutation_surrogate(wcc_valid, rng=rng, block_size=block_size)
        else:
            wcc_s = iaaft_surrogate(wcc_valid, rng=rng)
        feats_s = extract_features(wcc_s, hz=hz, wcc_window_sec=wcc_window_sec)
        for f in features:
            v = getattr(feats_s, f, np.nan)
            if np.isfinite(v):
                null_values[f].append(v)

    # Compute p-values (correct TWO-TAILED permutation p; Phipson & Smyth, 2010)
    result = {"null_model": f"wcc_level_{null_model}", "n_surrogates": surrogate_n}
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
    result["notes"] = ""
    return result


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return a failed surrogate test result."""
    return {
        "p_mean_synchrony": 1.0,
        "p_peak_amplitude": 1.0,
        "p_bimodality_coefficient": 1.0,
        "null_mean_synchrony": np.array([]),
        "null_peak_amplitude": np.array([]),
        "null_bimodality_coefficient": np.array([]),
        "obs_mean_synchrony": np.nan,
        "obs_peak_amplitude": np.nan,
        "obs_bimodality_coefficient": np.nan,
        "n_surrogates": 0,
        "n_valid_mean_synchrony": 0,
        "n_valid_peak_amplitude": 0,
        "n_valid_bimodality_coefficient": 0,
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
    min_wcc_points: int = 30,
    null_model: str = "iaaft",
    block_size: Optional[int] = None,
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
    null_model : {"iaaft", "block_permutation"}
        L1 WCC-level null model (used only when ``raw_signals`` is None).
        Default "iaaft". Use "block_permutation" as a conservative robustness
        check that preserves local autocorrelation.
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
        )
    else:
        # WCC-LEVEL null (correct for L1 features)
        logger.debug(
            "wcc_surrogate_test called without raw_signals — "
            "using WCC-level IAAFT null for L1 features "
            "(dwell_time, switching_rate). "
            "This is the correct call pattern for L1 null inference."
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
        ``ONSET_THRESHOLD`` (0.5) if fewer than 10 finite surrogate WCC
        values are available (degenerate case); ``is_surrogate_derived`` is
        ``False`` exactly when this fallback fired.
    """
    rng = np.random.default_rng(seed)
    surrogate_wccs: List[np.ndarray] = []

    # Guard: surrogate computation requires finite raw signals
    sig_a = np.asarray(sig_a, dtype=float)
    sig_b = np.asarray(sig_b, dtype=float)
    if not (np.all(np.isfinite(sig_a)) and np.all(np.isfinite(sig_b))):
        return ONSET_THRESHOLD, False

    for _ in range(surrogate_n):
        a_surr = iaaft_surrogate(sig_a, rng)
        b_surr = iaaft_surrogate(sig_b, rng)
        wcc_s = sliding_window_wcc(a_surr, b_surr, wcc_window_size, hz)
        surrogate_wccs.append(wcc_s)

    surrogate_matrix = np.vstack(surrogate_wccs)  # (surrogate_n, n_timepoints)
    return compute_surrogate_threshold(surrogate_matrix, percentile=percentile)


# ---------------------------------------------------------------------------
# Onset threshold — locked in feature_definitions (DECISION-01)
# ---------------------------------------------------------------------------

def extract_dynamic_features(
    wcc: np.ndarray,
    hz: float = 1.0,
    onset_threshold: Optional[float] = None,
    onset_k: float = 2.0,
    max_nan_ratio: float = 0.2,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    aggregation: str = "mean",
    return_raw_profiles: bool = False,
    wcc_window_sec: float = 1.0,
) -> Any:
    """Extract features from a WCC series via the SSoT.

    Thin wrapper delegating to :func:`multisync.feature_definitions.extract_features`.
    No feature computation is implemented here.

    Parameters
    ----------
    wcc : 1-D array
        Windowed cross-correlation time series.
    hz : float
        Sampling rate of the WCC series (Hz).
    onset_threshold : float or None
        WCC threshold (defaults to locked 0.5). Data-driven thresholds removed.
    onset_k, height, distance, prominence, aggregation
        Deprecated. Emit DeprecationWarning; do NOT affect the result.
    max_nan_ratio : float
        Guard: return all-NaN features if NaN fraction exceeds this.
    return_raw_profiles : bool
        Compatibility: when True returns (features, []).
    wcc_window_sec : float
        WCC window length for sustained-crossing scaling (DECISION-02).

    Returns
    -------
    DynamicFeatures | tuple[DynamicFeatures, list]
    """
    # ------------------------------------------------------------------
    # Deprecation surface for legacy peak-centric kwargs (DECISION-04/08)
    # ------------------------------------------------------------------
    if onset_k != 2.0:
        warnings.warn(
            "`onset_k` is deprecated and ignored: data-driven onset "
            "thresholds were removed in v1.0.0 (DECISION-04). "
            "Pass `onset_threshold` explicitly if you need a non-default "
            "threshold for sensitivity analysis.",
            DeprecationWarning,
            stacklevel=2,
        )
    if height is not None or distance is not None or prominence is not None:
        warnings.warn(
            "`height`, `distance`, and `prominence` are deprecated and "
            "ignored: feature math is no longer peak-detection-based "
            "in v1.0.0 (DECISION-08).  See "
            "multisync.feature_definitions for the locked-in definitions.",
            DeprecationWarning,
            stacklevel=2,
        )
    if aggregation != "mean":
        warnings.warn(
            "`aggregation` is deprecated and ignored: each WCC series now "
            "maps to a single DynamicFeatures via the SSoT; multi-peak "
            "aggregation no longer occurs at this layer.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Hard NaN-ratio guard
    wcc_arr = np.asarray(wcc, dtype=float)
    valid = ~np.isnan(wcc_arr)
    if wcc_arr.size == 0:
        nan_ratio = 1.0
    else:
        nan_ratio = 1.0 - float(valid.mean())

    if nan_ratio > max_nan_ratio or int(valid.sum()) < 5:
        _nan_features = DynamicFeatures.from_dict({
            "onset_latency": float("nan"),
            "rise_time": float("nan"),
            "peak_amplitude": float("nan"),
            "recovery_time": float("nan"),
            "dwell_time": float("nan"),
            "switching_rate": float("nan"),
            "mean_synchrony": float("nan"),
            "synchrony_entropy": float("nan"),
        })
        if return_raw_profiles:
            return _nan_features, []
        return _nan_features

    # ------------------------------------------------------------------
    # Resolve threshold (DECISION-01) and delegate to SSoT.
    # ------------------------------------------------------------------
    threshold = ONSET_THRESHOLD if onset_threshold is None else float(onset_threshold)

    features = _ssot_extract_features(
        wcc_arr,
        hz=hz,
        wcc_window_sec=wcc_window_sec,
        threshold=threshold,
    )

    if return_raw_profiles:
        return features, []
    return features


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

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue

                    key = f"{name_a}_{col_a}__{name_b}_{col_b}"

                    # --- Resolve threshold ---
                    if use_surrogate_threshold:
                        thr, is_surr = compute_surrogate_threshold_from_signals(
                            x, y,
                            hz=hz,
                            wcc_window_size=window_size,
                            surrogate_n=surrogate_n,
                            seed=surrogate_seed,
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

                    wcc = sliding_window_wcc(x, y, window_size, hz)
                    feat = extract_dynamic_features(
                        wcc, hz, thr, onset_k,
                        wcc_window_sec=wcc_window_sec,
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

    Returns
    -------
    Tuple[Dict, Dict]
        ``(segmented_features, threshold_meta)``.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names
    t_vec = dataset.time_vector()

    # --- Pre-compute per-dyad thresholds from full-length signals ---
    dyad_thresholds: Dict[str, float] = {}
    threshold_meta: Dict[str, Dict[str, Any]] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue
                    if use_surrogate_threshold:
                        thr, is_surr = compute_surrogate_threshold_from_signals(
                            x, y,
                            hz=hz,
                            wcc_window_size=window_size,
                            surrogate_n=surrogate_n,
                            seed=surrogate_seed,
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
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue

                        x_seg = x[mask]
                        y_seg = y[mask]

                        valid_ratio = (
                            ~np.isnan(x_seg) & ~np.isnan(y_seg)
                        ).sum() / len(x_seg)
                        if valid_ratio < (1.0 - max_nan_ratio):
                            continue

                        wcc = sliding_window_wcc(
                            x_seg, y_seg, window_size, hz
                        )
                        if len(wcc) < 5:
                            continue

                        key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                        thr = dyad_thresholds.get(key, ONSET_THRESHOLD)
                        feat = extract_dynamic_features(
                            wcc, hz, thr, onset_k, max_nan_ratio,
                            wcc_window_sec=wcc_window_sec,
                        )
                        seg_results[key] = feat

        results[label] = seg_results

    return results, threshold_meta

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

Feature math lives in :mod:`syncpipe.feature_definitions` (SSoT).
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
from scipy.signal.windows import get_window
import logging
import warnings

from .surrogate import iaaft_surrogate, ft_surrogate, prtf_surrogate  # noqa: F401  # re-export

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

def _make_window_kernel(window_type: str, window_size: int) -> np.ndarray:
    """Build a taper kernel for the sliding-window WCC.

    The kernel is normalised so its sum equals ``window_size`` (i.e. the
    mean weight is 1), keeping the weighted variance/covariance on the same
    absolute scale as the default rectangular window.  Supported names are
    any ``scipy.signal.windows`` name (e.g. ``'hann'``, ``'hamming'``,
    ``'triang'``, ``'gaussian'``); ``'rect'``/``'boxcar'`` returns uniform
    weights (the legacy behaviour).

    A tapered (e.g. Hann) window reduces the abrupt edge effects of a
    rectangular window on the WCC time series — a standard practice in
    psychophysiology where window boundaries otherwise introduce spurious
    jumps in coupling estimates.
    """
    if window_size <= 1:
        return np.ones(window_size)
    wt = (window_type or "rect").lower()
    if wt in ("rect", "boxcar", "rectangular"):
        kern = np.ones(window_size)
    else:
        try:
            if wt == "gaussian":
                kern = get_window(("gaussian", max(1.0, window_size / 6.0)), window_size)
            elif wt in ("hann", "hanning"):
                kern = get_window("hann", window_size)
            else:
                kern = get_window(wt, window_size)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unsupported window_type={window_type!r}. Use 'rect', 'hann', "
                f"'hamming', 'triang', 'gaussian', or any scipy.signal.windows name. "
                f"({exc})"
            )
    total = float(kern.sum())
    if total > 0:
        kern = kern * (window_size / total)
    return kern


def sliding_window_wcc(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    lag_samples: int = 0,
    step_samples: int = 0,
    min_valid_ratio: float = 0.5,
    window_type: str = "rect",
) -> np.ndarray:
    """
    Compute sliding-window cross-correlation (WCC) between x and y.

    For each window position, computes Pearson correlation within the window.
    Uses cumsum-based O(n) memory implementation when there are no NaN values
    and step_samples <= 1; falls back to stride-tricks (O(n*w) memory) when NaN values are present.

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
    window_type : str
        Window taper applied to each window before computing Pearson r.
        ``'rect'`` (default) = uniform/boxcar window (legacy behaviour) and
        the only path that uses the fast O(n) cumsum backend.
        ``'hann'`` / ``'hamming'`` / ``'triang'`` / ``'gaussian'`` taper the
        window edges, reducing abrupt edge effects on the WCC time series.
        The kernel is normalised to mean weight 1, so results stay on the
        same scale as ``'rect'``.

        Note: a non-rect taper is numerically correct (the dispatcher routes
        it to the stride backend, which aligns the kernel per window start),
        but it forgoes the O(n) cumsum fast path and uses O(n*w) memory.
        Prefer ``'rect'`` for large signals unless tapering is
        methodologically required. Do NOT call the private
        ``_sliding_window_wcc_cumsum`` directly with a taper — it raises
        ``ValueError`` (its globally-tiled weight is phase-shifted for
        non-rect windows and would silently return a wrong WCC).

    Returns
    -------
    wcc : 1-D array
        Cross-correlation at each window position.
        Length depends on step_samples:
        - step_samples <= 1: len(x) - window_size + 1
        - step_samples > 1: ceil((len(x) - window_size + 1) / step_samples)

    Window alignment (LEADING, not centered)
    ----------------------------------------
    This is a **leading-window** WCC: ``wcc[i]`` is the correlation of the
    window spanning samples ``[i, i + window_size - 1]`` (i.e. the window
    *starts* at ``i`` and looks forward).  It is **not** a centered window
    around ``i``.  Consequence for downstream morphology: an episode peak at
    WCC index ``i`` reflects coupling that began at sample ``i * step`` and
    ends at ``(i + window_size - 1) * step``.  Treat the WCC time axis as
    leading-aligned when relating peaks/onsets back to the raw-signal clock.
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

    # The cumsum backend applies the taper kernel as a *globally tiled* weight
    # (weight[i] = kern[i % window_size]).  For a sliding window of stride 1
    # this is only correct when the kernel is uniform ('rect') — otherwise the
    # kernel is phase-shifted by (i % window_size) for off-boundary windows,
    # yielding a WRONG WCC for tapered windows.  Tapered windows must use the
    # stride backend, which aligns the kernel to each window start.  The cumsum
    # backend is retained only for the fast rect + no-NaN case.  (BUG-1 fix:
    # this guarantees one backend per (window_type, has_nan) combo, so a single
    # NaN no longer silently changes the WCC of unaffected windows.)
    use_stride = has_nan or window_type != "rect"
    if use_stride:
        mem_estimate = (n - window_size + 1) * window_size * 8 * 4
        if mem_estimate > 1e9:
            logger.warning(
                f"sliding_window_wcc: large memory estimate ({mem_estimate/1e9:.1f} GB) "
                f"due to NaN values or a tapered window forcing stride_tricks "
                f"fallback. Consider filling NaN / using 'rect' before calling "
                f"this function."
            )
        wcc_full = _sliding_window_wcc_stride(
            x, y_lagged, window_size, min_valid_ratio, window_type
        )
    else:
        wcc_full = _sliding_window_wcc_cumsum(x, y_lagged, window_size, window_type)

    # Apply step if requested
    if step_samples > 1:
        return wcc_full[::step_samples]
    return wcc_full


def _apply_discontinuity_mask(
    wcc: np.ndarray,
    discontinuity_mask: Optional[np.ndarray],
    window_size: Optional[int] = None,
) -> np.ndarray:
    """Invalidate WCC samples that span a signal-level discontinuity.

    Real concatenated recordings (e.g. Lerique rest1/rest_postblock/trials
    seams) contain segment boundaries where the cross-correlation is
    meaningless.  ``discontinuity_mask`` marks, per *signal* sample, whether
    that sample is internal to a single segment (True) or sits on a boundary
    (False).  A WCC window is invalid iff it *contains* any boundary sample,
    so we set those WCC positions to NaN — downstream feature extraction and
    surrogate nulls already skip non-finite WCC, so the boundary is excluded
    uniformly.

    Parameters
    ----------
    wcc : np.ndarray
        WCC time series.
    discontinuity_mask : np.ndarray or None
        If None, ``wcc`` is returned unchanged.
        If length == len(wcc) + window_size - 1 (signal-resolution): a window
        ``i`` is invalid iff any of ``mask[i:i+window_size]`` is False.
        If length == len(wcc) (WCC-resolution): already window-level validity
        flags (True = valid); invalid positions become NaN.
    window_size : int or None
        Required when ``discontinuity_mask`` is signal-resolution.

    Returns
    -------
    np.ndarray
        WCC with discontinuity-spanning windows set to NaN.
    """
    if discontinuity_mask is None:
        return wcc
    wcc = np.asarray(wcc, dtype=float)
    dm = np.asarray(discontinuity_mask)
    if dm.dtype != bool:
        dm = dm.astype(bool)

    if len(dm) == len(wcc):
        # Already WCC-resolution validity flags.
        invalid = ~dm
    elif window_size is not None and len(dm) == len(wcc) + window_size - 1:
        # Signal-resolution: a window is invalid if it contains any False.
        dm_int = dm.astype(np.int8)
        csum = np.concatenate([[0], np.cumsum(dm_int)])
        win_sum = csum[window_size:] - csum[:-window_size]  # length == len(wcc)
        invalid = win_sum < window_size
    else:
        logger.warning(
            "_apply_discontinuity_mask: mask length %d incompatible with "
            "wcc length %d (window_size=%s); skipping mask.",
            len(dm), len(wcc), window_size,
        )
        return wcc

    wcc = wcc.copy()
    wcc[invalid] = np.nan
    return wcc


def sliding_window_wcc_masked(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    lag_samples: int = 0,
    step_samples: int = 0,
    min_valid_ratio: float = 0.5,
    window_type: str = "rect",
    discontinuity_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """sliding_window_wcc with discontinuity-boundary gating.

    Identical to :func:`sliding_window_wcc`, but windows whose span contains a
    segment discontinuity (per ``discontinuity_mask``) are set to NaN so
    downstream feature extraction and surrogate nulls skip them.  See
    :func:`_apply_discontinuity_mask` for mask semantics.
    """
    wcc = sliding_window_wcc(
        x, y, window_size, hz=hz, lag_samples=lag_samples,
        step_samples=step_samples, min_valid_ratio=min_valid_ratio,
        window_type=window_type,
    )
    return _apply_discontinuity_mask(wcc, discontinuity_mask, window_size)


def _sliding_window_wcc_cumsum(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    window_type: str = "rect",
) -> np.ndarray:
    """
    Cumsum-based sliding-window Pearson correlation.
    Assumes no NaN values in x or y.
    Memory: O(n) instead of O(n*w).

    A taper kernel (``window_type``) is applied as a *weighted* Pearson
    correlation within each window: the cumsums accumulate ``kernel * signal``
    so edge points contribute less.  For ``window_type='rect'`` the weights
    are uniform and this reduces exactly to the legacy formula.

    Warning
    -------
    This backend is ONLY valid for ``window_type='rect'``.  For a non-rect
    taper the globally-tiled weight ``kern[i % window_size]`` is phase-shifted
    for off-boundary windows, yielding a numerically INCORRECT WCC (verified:
    ~0.3 absolute error on a [-1, 1] scale for a 'hann' window).  Tapered
    windows must use :func:`sliding_window_wcc`, which routes them to the
    stride backend (correct per-window kernel alignment).  This function
    raises ``ValueError`` for any ``window_type != 'rect'`` to fail loud
    rather than returning corrupted values.  (BUG-1 / Finding 5.)

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

    # Fail loud (Karpathy Rule 12): this backend applies the taper as a
    # *globally tiled* weight (weight[i] = kern[i % window_size]), which is
    # only correct for the uniform 'rect' kernel.  A non-rect taper is
    # phase-shifted for off-boundary windows and yields a WRONG WCC.  Tapered
    # windows must go through sliding_window_wcc (routes to the stride
    # backend, which aligns the kernel per window start).  Refuse loudly
    # rather than silently returning corrupted numbers.  (See BUG-1 / Finding 5.)
    if window_type != "rect":
        raise ValueError(
            f"_sliding_window_wcc_cumsum only supports window_type='rect'; "
            f"got {window_type!r}. Tapered windows must use sliding_window_wcc "
            f"(routes to the stride backend, which aligns the kernel per "
            f"window start). The cumsum backend's globally-tiled weight is "
            f"phase-shifted for non-rect windows and produces a numerically "
            f"INCORRECT WCC."
        )

    # ------------------------------------------------------------------
    # Pre-demean using global means to avoid catastrophic cancellation
    # in cumsum.  Window-level means are still recovered correctly
    # because we use sum_xy/w - mean_x*mean_y below.
    # ------------------------------------------------------------------
    mean_x_global = float(np.mean(x))
    mean_y_global = float(np.mean(y))
    x_demeaned = x - mean_x_global
    y_demeaned = y - mean_y_global

    kern = _make_window_kernel(window_type, window_size)  # length window_size, sum == window_size

    # The kernel is applied *within* each sliding window, so the full-signal
    # weight repeats the kernel every ``window_size`` samples:
    #   weight[i] = kern[i % window_size].
    # Cumulating ``weight * signal`` then yields, for window [p, p+w),
    # Σ_j kern[j] * signal[p+j] — the correct taper.  (For 'rect' this
    # reduces to the legacy uniform-weight cumsum.)
    Wfull = np.tile(kern, (n // window_size) + 1)[:n]

    # Weighted cumulative sums of demeaned signals (full-length weight * signal)
    cumsum_k = np.cumsum(Wfull)
    cumsum_kx = np.cumsum(Wfull * x_demeaned)
    cumsum_ky = np.cumsum(Wfull * y_demeaned)
    cumsum_kxy = np.cumsum(Wfull * x_demeaned * y_demeaned)
    cumsum_kx2 = np.cumsum(Wfull * x_demeaned ** 2)
    cumsum_ky2 = np.cumsum(Wfull * y_demeaned ** 2)

    # Prepend 0 so that range sums [i, i+w) = cumsum[i+w] - cumsum[i]
    cumsum_k = np.concatenate([[0.0], cumsum_k])
    cumsum_kx = np.concatenate([[0.0], cumsum_kx])
    cumsum_ky = np.concatenate([[0.0], cumsum_ky])
    cumsum_kxy = np.concatenate([[0.0], cumsum_kxy])
    cumsum_kx2 = np.concatenate([[0.0], cumsum_kx2])
    cumsum_ky2 = np.concatenate([[0.0], cumsum_ky2])

    # Window indices
    i = np.arange(n - window_size + 1)
    i_end = i + window_size

    Wt = cumsum_k[i_end] - cumsum_k[i]
    sum_x = cumsum_kx[i_end] - cumsum_kx[i]
    sum_y = cumsum_ky[i_end] - cumsum_ky[i]
    sum_xy = cumsum_kxy[i_end] - cumsum_kxy[i]
    sum_x2 = cumsum_kx2[i_end] - cumsum_kx2[i]
    sum_y2 = cumsum_ky2[i_end] - cumsum_ky2[i]

    # Window-level weighted means (on demeaned signal — close to 0 but not exact)
    mean_x = sum_x / Wt
    mean_y = sum_y / Wt

    # ------------------------------------------------------------------
    # FIX: correct Pearson covariance.
    # Previous (buggy) version:  cov = sum_xy / w          (missing -mean_x*mean_y)
    # Correct Pearson formula: cov = sum_xy/w - mean_x*mean_y
    # ------------------------------------------------------------------
    cov = sum_xy / Wt - mean_x * mean_y
    var_x = sum_x2 / Wt - mean_x ** 2
    var_y = sum_y2 / Wt - mean_y ** 2

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
    window_type: str = "rect",
) -> np.ndarray:
    """
    Stride-tricks WCC with pairwise deletion for NaN handling.

    NaN strategy:
      1. Compute valid_ratio = fraction of PAIRWISE-valid points per window.
      2. Windows with valid_ratio < min_valid_ratio → NaN (quality gate).
      3. Windows passing the gate use ONLY the pairwise-valid points to
         compute Pearson r (pairwise deletion), so partial-NaN windows
         still yield a valid WCC value rather than propagating NaN.

    A taper kernel (``window_type``) is applied as a *weighted* Pearson
    correlation (effective weight = kernel * pairwise-valid mask).

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

    # Taper kernel, broadcast over windows: effective weight = kernel * valid.
    kern = _make_window_kernel(window_type, window_size)[None, :]
    w_eff = kern * pair_valid  # (n_windows, window_size); 0 where NaN

    # Global demean means (over finite values) keep the weighted covariance
    # formula consistent with the cumsum path.
    mx = float(np.nanmean(x))
    my = float(np.nanmean(y))
    xg = x - mx
    yg = y - my
    xw = sliding_window_view(xg, window_size)
    yw = sliding_window_view(yg, window_size)

    Wt = w_eff.sum(axis=1)
    sx = (w_eff * xw).sum(axis=1)
    sy = (w_eff * yw).sum(axis=1)
    sxy = (w_eff * xw * yw).sum(axis=1)
    sx2 = (w_eff * xw ** 2).sum(axis=1)
    sy2 = (w_eff * yw ** 2).sum(axis=1)

    # Wt can be 0 for all-NaN windows; np.where does NOT short-circuit the
    # division, so mask the divide inside errstate to avoid RuntimeWarnings
    # (the Wt==0 positions are overwritten with 0.0 / later masked out).
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = np.where(Wt > 0, sx / Wt, 0.0)
        mean_y = np.where(Wt > 0, sy / Wt, 0.0)
        cov = sxy / Wt - mean_x * mean_y
        var_x = sx2 / Wt - mean_x ** 2
        var_y = sy2 / Wt - mean_y ** 2
    var_x = np.maximum(var_x, 0.0)
    var_y = np.maximum(var_y, 0.0)
    denom = np.sqrt(var_x * var_y)

    wcc_valid = np.full(n_windows, np.nan)
    good = passes_gate & (denom > 1e-10)
    wcc_valid[good] = np.clip(cov[good] / denom[good], -1.0, 1.0)
    wcc = wcc_valid

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
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("segment-wise IAAFT requires equal-length 1-D signals")
    if int(window_size) != window_size or window_size < 2:
        raise ValueError("window_size must be an integer >= 2")
    minimum = (
        max(50, int(window_size) + 19)
        if min_segment_samples is None else int(min_segment_samples)
    )
    if minimum < max(4, int(window_size)):
        raise ValueError(
            "min_segment_samples must be >= max(4, window_size)"
        )

    valid = np.isfinite(a) & np.isfinite(b)
    if discontinuity_mask is not None:
        dm = np.asarray(discontinuity_mask)
        if dm.ndim != 1 or dm.size != a.size:
            raise ValueError(
                "discontinuity_mask must be one-dimensional and match signal length"
            )
        valid &= dm.astype(bool)

    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    all_runs = [(int(s), int(e)) for s, e in zip(starts, ends)]
    runs = [(s, e) for s, e in all_runs if e - s >= minimum]
    eligible = np.zeros(a.size, dtype=bool)
    for start, end in runs:
        eligible[start:end] = True

    diagnostics = {
        "mode": "segment_wise_iaaft" if len(all_runs) > 1 or not valid.all() else "whole_series_iaaft",
        "min_segment_samples": minimum,
        "n_segments_total": len(all_runs),
        "n_segments_used": len(runs),
        "n_segments_excluded_short": len(all_runs) - len(runs),
        "segment_lengths_used": [end - start for start, end in runs],
        "n_samples_total": int(a.size),
        "n_samples_jointly_finite": int(np.sum(np.isfinite(a) & np.isfinite(b))),
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
        smoothed_wcc,
    )

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
        # Two-tailed Phipson-Smyth (BUG-4 fix): unify L0 with the L1
        # _wcc_level_surrogate_test, which already uses this conservative
        # two-tailed form.  An upper-tail was methodologically arguable for
        # an existence test, but an inconsistent tail policy across the L0/L1
        # family is the actual defect; two-tail is the conservative,
        # consistent choice and matches tests/validation/test_per_feature_significance.py.
        p_ge = (np.sum(finite_null >= obs_val) + 1) / (n + 1)
        p_le = (np.sum(finite_null <= obs_val) + 1) / (n + 1)
        p = float(min(1.0, 2.0 * min(p_ge, p_le)))
        return p, finite_null, n

    p_mean, null_mean_valid, n_mean = _phipson_smyth_p(null_mean, obs_mean)
    p_peak, null_peak_valid, n_peak = _phipson_smyth_p(null_peak, obs_peak)
    p_bc, null_bc_valid, n_bc = _phipson_smyth_p(null_bc, obs_bc)

    # Per-feature significance — callers (e.g. InferencePipeline.run_full_cascade)
    # need per-feature pass rates to track frozen primary endpoints
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
    if wcc_window_sec is None:
        wcc_window_sec = n_valid / (hz * 10.0)
        logger.warning(
            f"_wcc_level_surrogate_test: wcc_window_sec not provided, "
            f"using heuristic {wcc_window_sec:.1f}s — may introduce "
            f"window-size mismatch confound"
        )

    if null_model not in ("iaaft", "block_permutation", "state_shuffle"):
        raise ValueError(f"null_model must be 'iaaft', 'block_permutation' or 'state_shuffle', got {null_model!r}")

    obs_feats = extract_features(wcc_valid, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold)
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
        feats_s = extract_features(wcc_s, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold)
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
    result["applicable"] = True
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
    window_type: str = "rect",
    min_wcc_points: int = 30,
    null_model: str = "iaaft",
    block_size: Optional[int] = None,
    threshold: float = ONSET_THRESHOLD,
    discontinuity_mask: Optional[np.ndarray] = None,
    min_segment_samples: Optional[int] = None,
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
    gap_policy: Optional[str] = None,
) -> Any:
    """Extract features from a WCC series via the SSoT.

    Thin wrapper delegating to :func:`syncpipe.feature_definitions.extract_features`.
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
            "syncpipe.feature_definitions for the locked-in definitions.",
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
    valid = np.isfinite(wcc_arr)
    if wcc_arr.size == 0:
        nan_ratio = 1.0
    else:
        nan_ratio = 1.0 - float(valid.mean())

    if nan_ratio > max_nan_ratio or int(valid.sum()) < 5:
        logger.warning(
            "extract_dynamic_features: NaN ratio %.3f exceeds "
            "max_nan_ratio %.3f (or only %d valid points < 5) — returning "
            "all-NaN DynamicFeatures for this WCC series.",
            nan_ratio, max_nan_ratio, int(valid.sum()),
        )
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
        # Report the (high) discontinuity fraction even on the all-NaN path.
        _nan_features.nan_fraction = nan_ratio
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
        gap_policy=gap_policy,
    )
    # Discontinuity-masked WCC fraction: mandatory diagnostic for
    # dwell_time / switching_rate confound assessment (Claude review #1 v1b).
    features.nan_fraction = nan_ratio

    if return_raw_profiles:
        return features, []
    return features


def pairing_policy(dataset, cross_modal: bool = False) -> str:
    """Return the effective dyad-pairing policy for the result manifest."""
    if cross_modal:
        return "cross_modal"
    feat_cols = dataset.feature_columns
    if any(len(cols) >= 2 for cols in feat_cols.values()):
        return "same_modality"
    names = dataset.modality_names
    if len(names) == 2 and all(len(feat_cols[name]) == 1 for name in names):
        return "two_file_dyad_fallback"
    return "same_modality_no_pair"


def iter_dyad_pairs(dataset, cross_modal: bool = False):
    """Yield (src_key, name_a, name_b, col_a, col_b, x, y) for each valid pair.

    SINGLE SOURCE OF TRUTH for dyad pairing — every pairing site
    (``DynamicAnalyzer._iter_pairs``, ``extract_features_all_pairs``,
    ``extract_features_segmented``) must call this so the WCC cache, the
    descriptor features, and the segmented features stay key-consistent
    (one dyad = one key). This closes the "divergent pairing paths" gap.

    DEFAULT (cross_modal=False): SAME-MODALITY dyad pairing. For each modality
    with >=2 feature columns, pair its feature columns against each other
    (e.g. ``person_a`` vs ``person_b``). Mirrors the scientific canonical
    record (pipeline_bridge: one record = person_a vs person_b of a single
    modality) and is the v1 default for the CLI ``analyze`` / ``DynamicAnalyzer``
    descriptor path.

    Two-file fallback: if no within-modality pair is found AND the dataset has
    exactly two modalities each with exactly one feature column, pair them. This
    covers the common "two CSVs, two people, one signal" CLI layout — still a
    same-modality dyad, just represented as two modalities.

    OPT-IN (cross_modal=True): legacy cross-modality pairing — every feature
    column of one modality against every feature column of another modality.
    Retained only for exploratory cross-modal description.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names

    if cross_modal:
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue
                        yield (
                            f"{name_a}_{col_a}__{name_b}_{col_b}",
                            name_a, name_b, col_a, col_b, x, y,
                        )
        return

    # SAME-MODALITY default: pair feature columns WITHIN each modality.
    yielded = 0
    for name in names:
        cols = feat_cols[name]
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                x = dataset.get_aligned_array(name, col_a)
                y = dataset.get_aligned_array(name, col_b)
                if x is None or y is None:
                    continue
                yielded += 1
                yield (
                    f"{name}_{col_a}__{name}_{col_b}",
                    name, name, col_a, col_b, x, y,
                )

    # Two-file fallback for the common 2-single-column-CSV layout.
    if yielded == 0 and len(names) == 2:
        ca, cb = feat_cols[names[0]], feat_cols[names[1]]
        if len(ca) == 1 and len(cb) == 1:
            x = dataset.get_aligned_array(names[0], ca[0])
            y = dataset.get_aligned_array(names[1], cb[0])
            if x is not None and y is not None:
                yield (
                    f"{names[0]}_{ca[0]}__{names[1]}_{cb[0]}",
                    names[0], names[1], ca[0], cb[0], x, y,
                )


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

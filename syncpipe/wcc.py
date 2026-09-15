from __future__ import annotations

from typing import Optional
import logging
import numpy as np
from scipy.signal.windows import get_window

logger = logging.getLogger(__name__)

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

    # Apply lag.  Sign convention (audit m13, 2026-09-14 — stated because
    # the wclr backend uses the OPPOSITE-looking construction and the two
    # must never be conflated): lag_samples > 0 shifts y LEFT in time
    # (y_lagged[i] = y[i + lag]), i.e. x[i] is correlated with the FUTURE
    # of y — equivalently y LEADS x by `lag` samples.  lag_samples < 0
    # correlates x[i] with y's PAST (y leads are reversed; x leads y).
    # Boundary samples with no overlapping partner become NaN, never wrap.
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

    Numerical tolerance (audit m1, 2026-09-14): prefix-sum differencing
    accumulates O(n * eps) relative error, so on very long signals
    (> ~1e7 samples) WCC values can drift by ~1e-9..1e-8 in absolute
    terms.  The final ``np.clip(wcc, -1, 1)`` keeps outputs in range and
    the ``denom > 1e-10`` guard masks degenerate windows; if you need
    bit-exact long-run behaviour, segment the signal (the discontinuity
    mask does this already) or use the stride backend.

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
    # NaN hygiene (BUG-2, 2026-09-08): invalid positions must be zeroed
    # BEFORE windowing.  ``0 * NaN = NaN`` would otherwise poison every
    # window sum that overlaps a single NaN, degrading the whole trace to
    # NaN and silently defeating the pairwise-deletion contract above.
    # Zeroing is algebraically exact here: those positions carry w_eff = 0.
    mx = float(np.nanmean(x))
    my = float(np.nanmean(y))
    xg = np.where(np.isfinite(x), x - mx, 0.0)
    yg = np.where(np.isfinite(y), y - my, 0.0)
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

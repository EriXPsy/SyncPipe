"""
Temporal Association Analysis — CCF with rigorous surrogate testing.

**Two modes: signal coupling and cross-modal temporal association.**

.. warning::
   This module detects **temporal precedence** (which signal leads computationally),
   NOT causal cascades. Results may be confounded by shared external stimulus.
   For rigorous causal inference, may use Convergent Cross-Mapping (CCM) as a
   supplementary analysis.

Two analysis modes are supported:

1. ``signal_coupling`` (default): cross-person, cross-modal CCF
   (e.g, person A's EDA vs person B's RESP).

2. ``synchrony_cascade``: meta-synchrony temporal association.  Computes WCC
   curves within-dyad for each modality, then CCF between WCC curves across modalities.
   Surrogate testing applies IAAFT on the *WCC curves themselves*.

Key design decisions:
- CCF computed via scipy.signal.correlate (FFT-based, O(n log n)).
- Edge-effect mitigation via Hanning (tapered cosine) window.
- Surrogate testing via IAAFT (Iterative Amplitude Adjusted Fourier
  Transform) signals.
- p-value: Phipson & Smyth (2010) unbiased estimator.
- Multiple comparisons: Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import correlate as sp_correlate, detrend as sp_detrend

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CCAResult:
    """Result of a single cross-correlation analysis between two modalities."""
    modality_a: str
    modality_b: str
    feature_a: str = ""
    feature_b: str = ""
    lags_sec: np.ndarray = field(default_factory=lambda: np.array([]))
    ccf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lag_sec: float = 0.0
    peak_ccf: float = 0.0
    direction: str = ""
    # Significance
    is_significant: bool = False
    p_value: float = 1.0           # ACTIVE p-value (after BH correction)
    p_value_raw: float = 1.0        # Raw p-value from surrogate testing (before BH)
    p_value_corrected: float = 1.0  # BH FDR corrected p-value (alias of p_value)
    surrogate_n: int = 0
    null_peak_ccf: Optional[np.ndarray] = None  # surrogates' peak CCFs
    # Diagnostics (runtime warnings)
    diagnostics: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Convert arrays to lists for JSON serialization
        lags_sec_list = self.lags_sec.tolist() if len(self.lags_sec) > 0 else []
        ccf_list = self.ccf_values.tolist() if len(self.ccf_values) > 0 else []
        null_peaks = self.null_peak_ccf.tolist() if self.null_peak_ccf is not None and len(self.null_peak_ccf) > 0 else None
        return {
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "lags_sec": lags_sec_list,
            "ccf_values": ccf_list,
            "peak_lag_sec": float(self.peak_lag_sec),
            "peak_ccf": float(self.peak_ccf),
            "direction": self.direction,
            "is_significant": self.is_significant,
            "p_value": float(self.p_value),
            "p_value_raw": float(self.p_value_raw),
            "p_value_corrected": float(self.p_value_corrected),
            "surrogate_n": self.surrogate_n,
            "null_peak_ccf": null_peaks,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CCAResult":
        """Deserialize from a dict (inverse of to_dict)."""
        lags = np.array(d.get("lags_sec", []), dtype=float)
        ccf = np.array(d.get("ccf_values", []), dtype=float)
        null_peaks = np.array(d["null_peak_ccf"], dtype=float) if d.get("null_peak_ccf") is not None else None
        return cls(
            modality_a=d.get("modality_a", ""),
            modality_b=d.get("modality_b", ""),
            feature_a=d.get("feature_a", ""),
            feature_b=d.get("feature_b", ""),
            lags_sec=lags,
            ccf_values=ccf,
            peak_lag_sec=float(d.get("peak_lag_sec", 0.0)),
            peak_ccf=float(d.get("peak_ccf", 0.0)),
            direction=d.get("direction", ""),
            is_significant=bool(d.get("is_significant", False)),
            p_value=float(d.get("p_value", 1.0)),
            p_value_raw=float(d.get("p_value_raw", d.get("p_value", 1.0))),
            p_value_corrected=float(d.get("p_value_corrected", d.get("p_value", 1.0))),
            surrogate_n=int(d.get("surrogate_n", 0)),
            null_peak_ccf=null_peaks,
            diagnostics=d.get("diagnostics", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CCAResult":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


@dataclass
class AssociationEdge:
    """A directed edge in the association graph (Viewer JSON ready)."""
    source: str
    target: str
    lag_sec: float
    ccf_value: float
    p_value: float
    is_significant: bool
    polarity: str = "positive"  # "positive" (excitatory) or "negative" (inhibitory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "lag_sec": float(self.lag_sec),
            "ccf_value": float(self.ccf_value),
            "p_value": float(self.p_value),
            "is_significant": self.is_significant,
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AssociationEdge":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(
            source=d.get("source", ""),
            target=d.get("target", ""),
            lag_sec=float(d.get("lag_sec", 0.0)),
            ccf_value=float(d.get("ccf_value", 0.0)),
            p_value=float(d.get("p_value", 1.0)),
            is_significant=bool(d.get("is_significant", False)),
            polarity=d.get("polarity", "positive"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AssociationEdge":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Surrogate generation — IAAFT (Schreiber & Schmitz, 2000)
# ---------------------------------------------------------------------------

def _iaaft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 200,
    tol: float = 1e-6,
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate an IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogate.

    IAAFT preserves **both** the power spectrum (autocorrelation structure)
    AND the amplitude distribution of the original time series.

    Algorithm (Schreiber & Schmitz, 2000):
    1. Sort original → target amplitude distribution
    2. Initial IAAFT surrogate (phase randomization)
    3. Iteratively: rank-order to match amplitudes, then enforce spectrum
    4. Converge when spectral error < tolerance

    Parameters
    ----------
    x : 1-D array
        Original time series.
    rng : numpy Generator
        Random number generator.
    max_iter : int
        Maximum iterations. Default 200.
    tol : float
        Convergence tolerance. Default 1e-6.
    clip_range : (float, float) or None
        If provided, clip the surrogate to this (min, max) range **after**
        each IAAFT iteration.  This prevents bounded signals (e.g. WCC ∈ [-1,1])
        from drifting outside their physical bounds due to FFT round-off.
        For WCC surrogates, use ``clip_range=(-1.0, 1.0)``.

    Returns
    -------
    surrogate : 1-D array, same length as x.

    References
    ----------
    Schreiber, T., & Schmitz, A. (2000). Surrogate time series.
    *Physica D*, 142(3-4), 346–382.
    """
    from .validation.pgt1_intensity import iaaft_surrogate as _iaaft_canonical

    surr = _iaaft_canonical(x, rng=rng, max_iter=max_iter, tol=tol)
    if clip_range is not None:
        surr = np.clip(surr, clip_range[0], clip_range[1])
    return surr

_prtf_surrogate = _iaaft_surrogate


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _hanning_window(n: int) -> np.ndarray:
    """Periodic Hanning window (scipy.signal.windows.hann equivalent)."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


# ---------------------------------------------------------------------------
# WCC curve computation (vectorized, for synchrony temporal association)
# ---------------------------------------------------------------------------

def _compute_wcc_curve(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step: int,
    hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute WCC over time using vectorized sliding windows.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length, pre-processed to remove NaN).
    window_size : int
        Window size in samples.
    step : int
        Step between consecutive windows (samples).
    hz : float
        Sampling rate (used for time axis).

    Returns
    -------
    times_sec : 1-D array
        Center times of each window (seconds).
    wcc : 1-D array
        WCC values (one per window, range ≈ [-1, 1]).
    """
    n = len(x)
    if n < window_size or step < 1:
        return np.array([]), np.array([])

    from numpy.lib.stride_tricks import sliding_window_view as sw_view

    # Number of complete windows
    n_windows = (n - window_size) // step + 1
    if n_windows <= 0:
        return np.array([]), np.array([])

    # Extract all windows: sw_view(arr, w) → shape (n - w + 1, w)
    x_all_win = sw_view(x, window_size)
    y_all_win = sw_view(y, window_size)

    # Select every `step`-th window
    indices = np.arange(n_windows) * step
    # Guard against out-of-bounds
    mask = indices < len(x_all_win)
    indices = indices[mask]
    n_windows = len(indices)
    if n_windows == 0:
        return np.array([]), np.array([])

    x_win = x_all_win[indices]
    y_win = y_all_win[indices]

    # Center each window (translation-invariant correlation)
    x_mean = x_win.mean(axis=1, keepdims=True)
    y_mean = y_win.mean(axis=1, keepdims=True)
    x_centered = x_win - x_mean
    y_centered = y_win - y_mean

    # Pearson correlation for all windows simultaneously
    sum_xy = (x_centered * y_centered).sum(axis=1)
    sum_x2 = (x_centered ** 2).sum(axis=1)
    sum_y2 = (y_centered ** 2).sum(axis=1)

    denom = np.sqrt(sum_x2 * sum_y2)
    wcc = np.where(denom > 1e-10, sum_xy / denom, 0.0)

    # Window center times (seconds)
    center_samples = indices + window_size // 2
    times_sec = center_samples / hz

    return times_sec, wcc


# ---------------------------------------------------------------------------
# CCF computation (vectorized)
# ---------------------------------------------------------------------------

def compute_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_sec: float,
    hz: float,
    apply_window: bool = True,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation function via numpy correlate.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length, same sampling rate).
    max_lag_sec : float
        Maximum lag to compute, in seconds.
    hz : float
        Sampling rate in Hz.
    apply_window : bool
        If True, apply Hanning window before CCF to reduce edge effects.
    detrend : bool
        If True, apply ``scipy.signal.detrend(type="linear")`` to remove
        slow drift.  Set to ``False`` for WCC curves (which are already
        bounded in [-1, 1]) since their low-frequency drift represents
        meaningful synchrony dynamics.  Default True.

    Returns
    -------
    lags_sec : 1-D array
        Lag values in seconds.
    ccf : 1-D array
        Normalized CCF values (Pearson-like, range ≈ [-1, 1]).

    Notes
    -----
    **Nonlinear drift caveat**: This function uses `scipy.signal.detrend(type="linear")`
    to remove linear slow drift. However, if the signal contains **nonlinear drift**
    (e.g., exponential, polynomial, or U-shaped slow components), linear detrending
    will **fail to remove it completely**, potentially creating **U-shaped pseudo-oscillations**
    in the CCF. For signals with visible nonlinear baseline wander, consider:

    - Using `scipy.signal.detrend(type="constant")` (demean only) + manual nonlinear removal
    - Applying a high-pass filter before CCF
    - Visual inspection of detrended signals
    """
    n = len(x)
    assert len(y) == n, f"x and y must have same length: {n} vs {len(y)}"

    max_lag_samples = int(np.floor(max_lag_sec * hz))

    if n < 2 * max_lag_samples + 1:
        raise ValueError(
            f"Series length ({n}) too short for max_lag ({max_lag_samples} samples). "
            f"Need at least {2 * max_lag_samples + 1}."
        )

    # Detrend: remove linear trend (slow drift) then demean.
    # Using scipy.signal.detrend instead of simple x - x.mean() because
    # biological signals (e.g, EDA) often have global slow drift (gradual
    # sweating across a 5-min session).  Shared linear trends dominate CCF
    # and push peak_idx to max lag boundaries, creating false causality.
    # detrend(type="linear") fits and subtracts the least-squares line,
    # removing slow drift while preserving local fluctuations.
    if detrend:
        x_detrended = sp_detrend(x)
        y_detrended = sp_detrend(y)
        # After detrend, mean is ~0, but re-demean for numerical safety
        x_demean = x_detrended - x_detrended.mean()
        y_demean = y_detrended - y_detrended.mean()
    else:
        x_demean = x - np.nanmean(x)
        y_demean = y - np.nanmean(y)

    # Apply Hanning window for edge-effect mitigation
    if apply_window:
        win = _hanning_window(n)
        x_w = x_demean * win
        y_w = y_demean * win
    else:
        x_w = x_demean
        y_w = y_demean

    # Normalized cross-correlation via FFT convolution — O(n log n).
    # scipy.signal.correlate with method="fft" uses the FFT convolution theorem,
    # which reduces complexity from O(n²) (direct convolution) to O(n log n).
    # This is critical when surrogate_n is large (e.g., 500 surrogates × n²
    # operations would stall on signals with n > 3000 samples).
    nccf = sp_correlate(x_w, y_w, mode="full", method="fft")

    # Normalize to Pearson correlation using UNWINDOWED denominator.
    # The standard Pearson normalization uses the variance of the raw
    # demeaned signal, not the window-weighted variance.  This ensures
    # CCF values stay in [-1, 1] — the window only tapers edge
    # contributions to the numerator (cross-product sum), not the scale.
    norm_x = np.sqrt(np.sum(x_demean ** 2))
    norm_y = np.sqrt(np.sum(y_demean ** 2))
    if norm_x > 0 and norm_y > 0:
        nccf = nccf / (norm_x * norm_y)

    # Extract the lag range: negative lags (x leads) to positive lags (y leads)
    center = n - 1
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    valid = center + lags
    ccf = nccf[valid]

    lags_sec = lags / hz

    return lags_sec, ccf


# ---------------------------------------------------------------------------
# Synchrony temporal association analysis — meta-synchrony (coupling of couplings)
# ---------------------------------------------------------------------------

def _temporal_association_sync(
    dataset: "SynchronyDataset",  # noqa: F821 forward ref
    max_lag_sec: float,
    surrogate_n: int,
    alpha: float,
    seed: int,
    wcc_window_sec: float,
    wcc_step_sec: float,
    hz: float,
    person_cols: Optional[Dict[str, Tuple[str, str]]],
    clip_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[CCAResult], List[AssociationEdge], Dict[str, Dict[str, Any]]]:
    """True meta-synchrony temporal association: CCF between WCC curves of different modalities.

    This answers the question: "Does modality A's synchrony precede
    modality B's synchrony?" — the core of Gordon (2024)'s Flexible
    Synchrony temporal association concept.

    Pipeline:
    1. For each modality, compute WCC curve within-dyad
       (e.g, WCC(EDA_A, EDA_B) → EDA_sync(t)).
    2. For each pair of modalities, compute CCF between their WCC curves.
    3. Surrogate testing: IAAFT on the *WCC curve of modality A* directly.
       This preserves both WCC_A's autocorrelation structure AND its
       amplitude distribution (Wiener-Khinchin theorem) while destroying
       its phase-locking relationship with WCC_B.

    Notes
    -----
    **Why IAAFT on WCC curves, not raw signals?**
    The null hypothesis is: "WCC_A and WCC_B share no temporal
    phase-locking, beyond what is expected from WCC_A's own autocorrelation
    structure."

    IAAFT preserves |FFT(x)| AND amplitude distribution → preserves
    autocorrelation (Wiener-Khinchin theorem) AND WCC bounds.  Applying
    IAAFT to WCC_A keeps its frequency content and amplitude distribution
    intact while randomizing its temporal alignment with WCC_B.  This
    yields a null distribution grounded in the actual statistical
    properties of the synchrony time series.

    IAAFT is strictly superior to FT for WCC surrogates because:
    - WCC is bounded in [-1, 1] (non-Gaussian)
    - FT preserves spectrum but not amplitude distribution
    - FT surrogates may have incorrect bounds → anti-conservative tests
    - IAAFT iteratively matches both spectrum and distribution
    """
    feat_cols = dataset.feature_columns
    modality_names = dataset.modality_names
    rng = np.random.default_rng(seed)

    wcc_window_samples = int(np.floor(wcc_window_sec * hz))
    wcc_step_samples = int(np.floor(wcc_step_sec * hz))

    # ---- Step 1: Compute WCC curves for each modality ----
    # WCC curve = within-dyad synchrony over time for one modality.
    # We need person A and person B's signals for the same modality.
    wcc_curves: Dict[str, np.ndarray] = {}
    wcc_times: Dict[str, np.ndarray] = {}

    for mod_name in modality_names:
        # Resolve person A/B column names
        if person_cols is not None and mod_name in person_cols:
            col_a, col_b = person_cols[mod_name]
        else:
            cols = feat_cols.get(mod_name, [])
            if len(cols) < 2:
                logger.info(
                    "Skipping %s for synchrony temporal association: need >= 2 columns "
                    "(person A and person B), got %d", mod_name, len(cols),
                )
                continue
            col_a, col_b = cols[0], cols[1]
            logger.info(
                "Using implicit column selection for %s: col[0]=%s, col[1]=%s. "
                "Consider passing person_cols={'%s': ('%s', '%s')} for "
                "explicit control.",
                mod_name, col_a, col_b, mod_name, col_a, col_b,
            )

        x = dataset.get_aligned_array(mod_name, col_a)
        y = dataset.get_aligned_array(mod_name, col_b)
        if x is None or y is None:
            continue

        # Pre-process: trim NaN edges, fill internal NaN with linear interp
        either_nan = np.isnan(x) | np.isnan(y)
        if either_nan.sum() == len(x):
            continue
        first_valid = int(np.argmax(~either_nan))
        last_valid = len(either_nan) - 1 - int(np.argmax(~either_nan[::-1]))
        x_trim = x[first_valid : last_valid + 1].copy()
        y_trim = y[first_valid : last_valid + 1].copy()

        if len(x_trim) < wcc_window_samples:
            logger.info(
                "Skipping %s: segment too short (%d < %d window)",
                mod_name, len(x_trim), wcc_window_samples,
            )
            continue

        # Guard: WCC curve must have enough points for meaningful IAAFT.
        # FFT on N points provides ~N/2 independent phase bins; IAAFT with
        # too few points produces highly overlapping surrogates and a null
        # distribution with no statistical power. 50 points is a conservative
        # minimum for IEEE/Nature Methods-level rigor.
        MIN_WCC_POINTS = 50
        wcc_step_samples = int(np.floor(wcc_step_sec * hz))
        session_duration = len(x_trim) / hz
        expected_n = max(0, (len(x_trim) - wcc_window_samples) // wcc_step_samples + 1)

        if expected_n < MIN_WCC_POINTS:
            raise ValueError(
                f"synchrony_cascade requires >={MIN_WCC_POINTS} WCC time points "
                f"per modality for reliable IAAFT surrogates. '{mod_name}' will "
                f"produce only ~{expected_n} points with current settings "
                f"(wcc_window={wcc_window_sec}s, wcc_step={wcc_step_sec}s). "
                f"Try: wcc_step_sec <= {session_duration / MIN_WCC_POINTS:.1f}s, "
                f"or collect longer sessions "
                f"(>= {MIN_WCC_POINTS * wcc_step_sec + wcc_window_sec:.0f}s)."
            )

        # Linear interpolation for internal NaN.
        # Mean imputation creates step discontinuities that produce
        # high-frequency spectral artifacts, contaminating downstream
        # PRTF surrogate testing.  Linear interpolation preserves the
        # local trend and is the standard for time-series data.
        x_clean = (
            pd.Series(x_trim).interpolate(method="linear").ffill().bfill().values
        )
        y_clean = (
            pd.Series(y_trim).interpolate(method="linear").ffill().bfill().values
        )

        # Compute WCC curve (vectorized)
        times, wcc = _compute_wcc_curve(
            x_clean, y_clean, wcc_window_samples, wcc_step_samples, hz,
        )
        if len(wcc) == 0:
            continue

        wcc_curves[mod_name] = wcc
        wcc_times[mod_name] = times

    valid_mods = list(wcc_curves.keys())
    if len(valid_mods) < 2:
        logger.info(
            "Need >= 2 modalities with WCC curves for synchrony temporal association, "
            "got %d: %s", len(valid_mods), valid_mods,
        )
        return [], [], {}

    # ---- Step 2: CCF between WCC curves ----
    # The WCC curve's effective sampling rate is 1 / wcc_step_sec Hz.
    wcc_hz = 1.0 / wcc_step_sec

    cca_results: List[CCAResult] = []
    edges: List[AssociationEdge] = []
    raw_pvals: List[float] = []

    for i, mod_a in enumerate(valid_mods):
        for mod_b in valid_mods[i + 1:]:
            wcc_a = wcc_curves[mod_a]
            wcc_b = wcc_curves[mod_b]

            # Trim to common length for CCF
            min_len = min(len(wcc_a), len(wcc_b))

            # Guard: paired WCC curves need enough points for meaningful CCF.
            # Softer threshold than the per-modality 50-point guard because
            # CCF needs fewer points than PRTF to be numerically stable.
            MIN_CCF_POINTS = 20
            if min_len < MIN_CCF_POINTS:
                logger.info(
                    "Skipping %s vs %s: paired WCC curves too short "
                    "(%d < %d)",
                    mod_a, mod_b, min_len, MIN_CCF_POINTS,
                )
                continue

            wcc_a = wcc_a[:min_len]
            wcc_b = wcc_b[:min_len]
            if min_len < MIN_CCF_POINTS:
                logger.info(
                    "Skipping %s vs %s: paired WCC curves too short "
                    "(%d < %d)",
                    mod_a, mod_b, min_len, MIN_CCF_POINTS,
                )
                continue
            wcc_a = wcc_a[:min_len]
            wcc_b = wcc_b[:min_len]

            # Cap max_lag to feasible range for WCC curve length
            n_seg = min_len
            feasible_lag_sec = (n_seg - 1) / 2.0 / wcc_hz
            effective_max_lag = min(max_lag_sec, feasible_lag_sec)

            # Guard: need >= 3x lag for reliable CCF
            min_required = 3 * int(np.floor(effective_max_lag * wcc_hz)) + 1
            if n_seg < min_required:
                logger.info(
                    "Skipping %s vs %s: WCC curve length %d < %d required",
                    mod_a, mod_b, n_seg, min_required,
                )
                continue

            # Compute observed CCF between WCC curves
            # detrend=False: WCC curves are in [-1, 1] and their low-frequency
            # drift (e.g, synchrony building up over time) is the signal of
            # interest.  Linear detrending would remove this meaningful pattern.
            try:
                lags_sec, ccf_vals = compute_ccf(
                    wcc_a, wcc_b, effective_max_lag, wcc_hz,
                    apply_window=False, detrend=False,
                )
            except ValueError:
                continue

            # Find peak
            abs_ccf = np.abs(ccf_vals)
            peak_idx = int(np.argmax(abs_ccf))
            peak_lag = float(lags_sec[peak_idx])
            peak_val = float(ccf_vals[peak_idx])

            # Determine direction
            if peak_lag < 0:
                direction = f"{mod_a}_sync→{mod_b}_sync"
                source, target = mod_a, mod_b
                peak_lag = abs(peak_lag)
            elif peak_lag > 0:
                direction = f"{mod_b}_sync→{mod_a}_sync"
                source, target = mod_b, mod_a
                peak_lag = abs(peak_lag)
            else:
                direction = "synchronous"
                source, target = mod_a, mod_b

            # ---- Step 3: Surrogate testing ----
            # IAAFT on WCC_A directly (not raw signals).
            # Null hypothesis: WCC_A's autocorrelation structure is preserved,
            # but its phase-locking with WCC_B is destroyed.
            null_peaks = np.empty(surrogate_n)
            for s in range(surrogate_n):
                wcc_a_surr = _iaaft_surrogate(
                    wcc_a, rng, clip_range=clip_range,
                )
                try:
                    _, ccf_s = compute_ccf(
                        wcc_a_surr, wcc_b,
                        effective_max_lag, wcc_hz,
                        apply_window=False, detrend=False,
                    )
                    null_peaks[s] = float(np.max(np.abs(ccf_s)))
                except ValueError:
                    null_peaks[s] = 0.0

            # Phipson & Smyth (2010) unbiased p-value
            p_val = float(
                (1.0 + np.sum(null_peaks >= abs(peak_val)))
                / (surrogate_n + 1.0)
            )
            raw_pvals.append(p_val)

            result = CCAResult(
                modality_a=f"{mod_a}_sync",
                modality_b=f"{mod_b}_sync",
                feature_a="WCC_curve",
                feature_b="WCC_curve",
                lags_sec=lags_sec,
                ccf_values=ccf_vals,
                peak_lag_sec=peak_lag,
                peak_ccf=peak_val,
                direction=direction,
                is_significant=p_val < alpha,
                p_value=p_val,
                p_value_raw=p_val,
                surrogate_n=surrogate_n,
                null_peak_ccf=null_peaks,
            )
            cca_results.append(result)

            if p_val < alpha and direction != "synchronous":
                edges.append(AssociationEdge(
                    source=source,
                    target=target,
                    lag_sec=peak_lag,
                    ccf_value=peak_val,
                    p_value=p_val,
                    is_significant=True,
                    polarity="positive" if peak_val >= 0 else "negative",
                ))

    # ---- BH FDR correction ----
    if len(raw_pvals) > 0:
        corrected = _bh_fdr_correct(raw_pvals, q=alpha)
        edges = []
        for idx, result in enumerate(cca_results):
            result.p_value_corrected = corrected[idx]
            result.is_significant = corrected[idx] < alpha
            result.p_value = corrected[idx]

            if result.is_significant and result.direction != "synchronous":
                if "→" in result.direction:
                    src, tgt = result.direction.split("→")
                else:
                    src = result.modality_a
                    tgt = result.modality_b
                edges.append(AssociationEdge(
                    source=src,
                    target=tgt,
                    lag_sec=result.peak_lag_sec,
                    ccf_value=result.peak_ccf,
                    p_value=result.p_value_corrected,
                    is_significant=result.is_significant,
                    polarity="positive" if result.peak_ccf >= 0 else "negative",
                ))

    metrics = compute_association_metrics(edges, alpha)
    return cca_results, edges, metrics


# ---------------------------------------------------------------------------
# Full temporal association analysis with surrogate testing
# ---------------------------------------------------------------------------

def temporal_association_analysis(
    dataset: "SynchronyDataset",  # noqa: F821  forward ref
    max_lag_sec: float = 30.0,
    surrogate_n: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    apply_window: bool = True,
    mode: str = "signal_coupling",
    wcc_window_sec: float = 30.0,
    wcc_step_sec: float = 15.0,
    person_cols: Optional[Dict[str, Tuple[str, str]]] = None,
    clip_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[CCAResult], List[AssociationEdge], Dict[str, Dict[str, Any]]]:
    """
    Compute temporal association analysis — two modes (signal coupling or cross-modal temporal association).

    **Mode 1: "signal_coupling" (default)**
      Computes CCF between raw signals across persons and modalities
      (e.g, person A's EDA vs person B's RESP).  This is cross-person,
      cross-modal *signal* coupling.  The original behavior.

    **Mode 2: "synchrony_cascade" (true meta-synchrony)**
      Step 1: compute WCC curves within-dyad for each modality
             (e.g, WCC(EDA_A, EDA_B) → EDA_sync(t))
      Step 2: compute CCF between WCC curves of different modalities
             (e.g, CCF(EDA_sync, RESP_sync) → does EDA synchrony
             precede RESP synchrony?)
      This is the "synchrony temporal association" conceptualized by Gordon (2024).

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned and normalized.
    max_lag_sec : float
        Maximum cross-correlation lag in seconds.
    surrogate_n : int
        Number of IAAFT surrogates for significance testing.
    alpha : float
        Significance threshold.  ``is_significant`` is True when
        p < alpha.
    seed : int
        Random seed for reproducibility.
    apply_window : bool
        Apply Hanning window before CCF (signal_coupling mode only).
    mode : str
        ``"signal_coupling"`` (default) or ``"synchrony_cascade"``.
    wcc_window_sec : float
        Window size (seconds) for WCC curve computation.
        Only used when ``mode="synchrony_cascade"``.  Default 30.
    wcc_step_sec : float
        Step size (seconds) between consecutive WCC windows.
        Only used when ``mode="synchrony_cascade"``.  Default 15
        (50% overlap).
    person_cols : dict or None
        Explicit mapping of modality → (person_A_col, person_B_col) for
        synchrony_cascade mode.  Example::

            person_cols = {
                "EDA": ("EDA_A", "EDA_B"),
                "RESP": ("RESP_A", "RESP_B"),
            }

        When ``None`` (default), the first two columns of each modality
        are used (``cols[0]`` and ``cols[1]``), with a warning logged.
        Passing explicit column names is **strongly recommended** to
        avoid silent misidentification when datasets contain extra
    clip_range : (float, float) or None
        If provided, clip IAAFT surrogates to this (min, max) range.
        For WCC-based surrogates, use ``clip_range=(-1.0, 1.0)``
        to prevent FFT round-off from pushing values outside the
        physical bounds of the WCC index.
        columns (e.g, ``["EDA_diff", "EDA_A", "EDA_B"]``).

    Returns
    -------
    cca_results : list of CCAResult
        Detailed results per modality pair (with BH-corrected p-values).
    edges : list of AssociationEdge
        Viewer-ready directed association edges (only significant ones after FDR).
    metrics : dict
        Lightweight graph metrics (in_degree, out_degree, driver_score,
        is_hub, is_follower) per modality.  Computed without networkx.

    Notes
    -----
    "Signal coupling" (cross-person CCF) and "synchrony cascade" (inter-modal
    WCC temporal sequence) are distinct scientific questions. This function
    addresses the latter in ``synchrony_cascade`` mode.
    """
    if not dataset._aligned:
        raise ValueError("Dataset must be aligned before temporal association analysis.")

    hz = dataset.target_hz

    # --- Dispatch based on mode ---
    if mode == "synchrony_cascade":
        return _temporal_association_sync(
            dataset=dataset,
            max_lag_sec=max_lag_sec,
            surrogate_n=surrogate_n,
            alpha=alpha,
            seed=seed,
            wcc_window_sec=wcc_window_sec,
            wcc_step_sec=wcc_step_sec,
            hz=hz,
            person_cols=person_cols,
            clip_range=clip_range,
        )

    # --- Original signal_coupling logic below ---
    feat_cols = dataset.feature_columns
    modality_names = dataset.modality_names
    rng = np.random.default_rng(seed)

    cca_results: List[CCAResult] = []
    edges: List[AssociationEdge] = []
    raw_pvals: List[float] = []  # collect raw p-values for BH correction

    for i, name_a in enumerate(modality_names):
        for name_b in modality_names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue

                    # Trim leading and trailing NaN only — never slice internal gaps.
                    either_nan = np.isnan(x) | np.isnan(y)
                    if either_nan.sum() == len(x):
                        logger.debug(
                            "Skipping %s/%s x %s/%s: entirely NaN",
                            name_a, col_a, name_b, col_b,
                        )
                        continue  # entirely NaN
                    # Find first/last valid index to trim edges
                    first_valid = int(np.argmax(~either_nan))
                    last_valid = len(either_nan) - 1 - int(np.argmax(~either_nan[::-1]))
                    x_trim = x[first_valid : last_valid + 1].copy()
                    y_trim = y[first_valid : last_valid + 1].copy()
                    if len(x_trim) < 20:
                        logger.debug(
                            "Skipping %s/%s x %s/%s: segment too short (%d < 20)",
                            name_a, col_a, name_b, col_b, len(x_trim),
                        )
                        continue
                    # Fill internal NaN with local mean instead of 0.
                    # Zero-filling creates false synchrony when both signals
                    # have NaN at the same positions (0 vs 0 = "perfect sync").
                    # Local mean is a more conservative imputation.
                    nan_ratio = np.isnan(x_trim).sum() / len(x_trim)
                    if nan_ratio > 0.1:
                        logger.warning(
                            "High NaN ratio (%.1f%%) in %s/%s x %s/%s, "
                            "results may be unreliable",
                            nan_ratio * 100, name_a, col_a, name_b, col_b,
                        )
                    # Linear interpolation for internal NaN (consistent with
                    # synchrony_cascade mode).  Mean imputation creates
                    # step discontinuities → high-frequency spectral artifacts.
                    x_clean = (
                        pd.Series(x_trim).interpolate(method="linear")
                        .ffill().bfill().values
                    )
                    y_clean = (
                        pd.Series(y_trim).interpolate(method="linear")
                        .ffill().bfill().values
                    )

                    # --- Nonlinear drift detection (strict heuristic) ---
                    # Only trigger when linear detrending FAILED to remove
                    # the dominant trend: if residual variance >95% of original,
                    # the drift is essentially nonlinear and CCF may contain
                    # U-shaped pseudo-oscillations.
                    # Threshold is strict (95%) to avoid Alert Fatigue:
                    # most EDA/fNIRS signals have mild nonlinear components
                    # that do NOT invalidate CCF results.
                    diagnostics = []  # local diagnostics for this pair
                    for sig, label in [(x_clean, f"{name_a}/{col_a}"), (y_clean, f"{name_b}/{col_b}")]:
                        sig_valid = sig[~np.isnan(sig)]
                        if len(sig_valid) < 30:
                            continue
                        var_orig = np.var(sig_valid)
                        var_res = np.var(sp_detrend(sig_valid))
                        # Trigger only if residual variance > 95% of original
                        # (linear detrending removed <5% of variance)
                        if var_orig > 0 and var_res > 0.95 * var_orig:
                            diagnostics.append({
                                "type": "warning",
                                "message": (
                                    f"Severe nonlinear drift detected in {label}. "
                                    f"Linear detrending removed <5% of signal variance. "
                                    f"CCF results may contain spurious oscillations. "
                                    f"Recommendation: apply high-pass filter (e.g., 0.01Hz) "
                                    f"in preprocessing (MNE/NeuroKit) before importing."
                                )
                            })

                    # Cap max_lag_sec to the mathematically feasible range for
                    # this particular segment.  CCF needs n >= 2*max_lag + 1,
                    # so max_lag <= (n-1)//2.  Silently cap instead of crashing.
                    n_seg = len(x_clean)
                    feasible_lag_samples = (n_seg - 1) // 2
                    feasible_lag_sec = feasible_lag_samples / hz
                    effective_max_lag = min(max_lag_sec, feasible_lag_sec)
                    if effective_max_lag < max_lag_sec:
                        logger.info(
                            "Capped max_lag from %.1fs to %.1fs for "
                            "%s/%s x %s/%s (segment length %d)",
                            max_lag_sec, effective_max_lag,
                            name_a, col_a, name_b, col_b, n_seg,
                        )

                    # Degrees-of-freedom guard: CCF + Surrogate testing on very
                    # short segments produces noise peaks, not real synchrony.
                    # Require n >= 3 * effective_max_lag_samples + 1 so that
                    # there are at least 3 full lag-windows of data.  If not,
                    # skip this pair entirely rather than emit noisy results.
                    min_required = 3 * int(np.floor(effective_max_lag * hz)) + 1
                    if n_seg < min_required:
                        logger.warning(
                            "Skipping %s/%s x %s/%s: segment (%d samples) is "
                            "too short for reliable CCF+Surrogate analysis "
                            "(need >= %d = 3×max_lag_samples+1). "
                            "Results would be statistical noise.",
                            name_a, col_a, name_b, col_b, n_seg, min_required,
                        )
                        continue

                    # Compute observed CCF
                    lags_sec, ccf_vals = compute_ccf(
                        x_clean, y_clean, effective_max_lag, hz, apply_window
                    )

                    # Find peak
                    abs_ccf = np.abs(ccf_vals)
                    peak_idx = int(np.argmax(abs_ccf))
                    peak_lag = float(lags_sec[peak_idx])
                    peak_val = float(ccf_vals[peak_idx])

                    # Determine direction
                    if peak_lag < 0:
                        direction = f"{name_a}→{name_b}"
                        source, target = name_a, name_b
                        peak_lag = abs(peak_lag)
                    elif peak_lag > 0:
                        direction = f"{name_b}→{name_a}"
                        source, target = name_b, name_a
                        peak_lag = abs(peak_lag)
                    else:
                        direction = "synchronous"
                        source, target = name_a, name_b

                    # --- Surrogate testing (IAAFT) ---
                    null_peaks = np.empty(surrogate_n)
                    for s in range(surrogate_n):
                        x_surr = _prtf_surrogate(x_clean, rng)
                        y_surr = _prtf_surrogate(y_clean, rng)
                        _, ccf_s = compute_ccf(
                            x_surr, y_surr, effective_max_lag, hz, apply_window
                        )
                        null_peaks[s] = np.max(np.abs(ccf_s))

                    # p-value: Phipson & Smyth (2010) unbiased permutation
                    # p-value estimator.  The observed statistic is treated as
                    # one draw from the null distribution, so the minimum
                    # possible p is 1 / (surrogate_n + 1) rather than 0.
                    #
                    # Formula: p = (1 + #{surrogates >= |observed|}) / (N + 1)
                    #
                    # Why not np.mean(null > obs)?
                    #   - With N=500, a perfectly extreme signal gives p=0.0,
                    #     which is statistically non-standard and rejected by
                    #     most journals.  The +1 correction ensures p ∈ (0, 1].
                    # Why >= instead of >?
                    #   - Strict > is anti-conservative on discrete null
                    #     distributions; >= gives correct coverage (Phipson &
                    #     Smyth, Stat Appl Genet Mol Biol, 2010).
                    surrogate_n_actual = len(null_peaks)
                    p_val = float(
                        (1.0 + np.sum(null_peaks >= abs(peak_val)))
                        / (surrogate_n_actual + 1.0)
                    )
                    is_sig = p_val < alpha
                    raw_pvals.append(p_val)

                    result = CCAResult(
                        modality_a=name_a,
                        modality_b=name_b,
                        feature_a=col_a,
                        feature_b=col_b,
                        lags_sec=lags_sec,
                        ccf_values=ccf_vals,
                        peak_lag_sec=peak_lag,
                        peak_ccf=peak_val,
                        direction=direction,
                        is_significant=is_sig,
                        p_value=p_val,
                        p_value_raw=p_val,  # raw p-value from surrogate testing
                        surrogate_n=surrogate_n,
                        null_peak_ccf=null_peaks,
                        diagnostics=diagnostics,
                    )
                    cca_results.append(result)

                    if is_sig and direction != "synchronous":
                        edges.append(AssociationEdge(
                            source=source,
                            target=target,
                            lag_sec=peak_lag,
                            ccf_value=peak_val,
                            p_value=p_val,
                            is_significant=True,
                            polarity="positive" if peak_val >= 0 else "negative",
                        ))

    # --- Post-hoc: Benjamini-Hochberg FDR correction ---
    # Raw p-values from pairwise surrogate tests are subject to the
    # multiple-comparisons problem.  With m modality pairs, the family-wise
    # error rate at alpha=0.05 approaches 1 - (0.95)^m (e.g, 40% for
    # m=10, ~90% for m=45).  Apply BH correction to control FDR.
    if len(raw_pvals) > 0:
        corrected = _bh_fdr_correct(raw_pvals, q=alpha)
        # Update cca_results with corrected p-values
        # Also rebuild edges based on corrected significance
        edges = []
        for idx, result in enumerate(cca_results):
            result.p_value_corrected = corrected[idx]
            result.is_significant = corrected[idx] < alpha
            # Update p_value to corrected (for downstream consumption)
            result.p_value = corrected[idx]

            # Rebuild edges with corrected significance
            if result.is_significant and result.direction != "synchronous":
                # Determine source/target from direction string
                if "→" in result.direction:
                    src, tgt = result.direction.split("→")
                else:
                    src = result.modality_a
                    tgt = result.modality_b
                edges.append(AssociationEdge(
                    source=src,
                    target=tgt,
                    lag_sec=result.peak_lag_sec,
                    ccf_value=result.peak_ccf,
                    p_value=result.p_value_corrected,
                    is_significant=result.is_significant,
                    polarity="positive" if result.peak_ccf >= 0 else "negative",
                ))

    # --- Compute lightweight graph metrics (no networkx) ---
    metrics = compute_association_metrics(edges, alpha)

    return cca_results, edges, metrics


# ---------------------------------------------------------------------------
# Multiple Comparisons Correction — Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def _bh_fdr_correct(p_values: List[float], q: float = 0.05) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Returns adjusted p-values (q-values).  A test is significant if
    adjusted_p <= q.

    Parameters
    ----------
    p_values : list of float
        Raw p-values (uncorrected).
    q : float
        Target FDR level (default 0.05).

    Returns
    -------
    adjusted : list of float
        Adjusted p-values, same order as input.
        adjusted[i] <= q  →  reject H_0 (significant).

    Notes
    -----
    **PRDS assumption**: BH procedure assumes **Positive Regression Dependency
    on a Subset (PRDS)** for strict FDR control. If p-values are strongly
    **negatively correlated** (e.g., one significant result makes others less
    likely to be significant), the actual FDR may **exceed q**.

    In SyncPipe, CCF p-values from **overlapping time windows** may violate
    PRDS. Consider:
    
    - Using **independent tests only** (e.g., separate dyads, separate epochs)
    - Applying **permutation-based FDR** (more robust to dependence)
    - Reporting **raw p-values + correction method** for transparency

    See: Issue #4 "FDR的关联依赖性（PRDS假设）隐患" in project notes.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed = [(i, p) for i, p in enumerate(p_values)]
    indexed_sorted = sorted(indexed, key=lambda x: x[1])

    # Compute adjusted p-values (BH step-up)
    # adjusted_p_(j) = min_{k >= j} (m * p_(k) / k)
    # Then enforce monotonicity: adj_p_(j) <= adj_p_(j+1)
    raw_adj = [0.0] * m
    for j in range(1, m + 1):
        p_j = indexed_sorted[j - 1][1]
        raw_adj[j - 1] = min(1.0, p_j * m / j)

    # Enforce non-decreasing order (adjusted p-values can only increase)
    for j in range(m - 2, -1, -1):
        raw_adj[j] = min(raw_adj[j], raw_adj[j + 1])

    # Map back to original order
    adjusted = [0.0] * m
    for j in range(m):
        orig_idx = indexed_sorted[j][0]
        adjusted[orig_idx] = raw_adj[j]

    return adjusted


# ---------------------------------------------------------------------------
# Lightweight Graph Metrics (no networkx dependency)
# ---------------------------------------------------------------------------

def compute_association_metrics(
    edges: List[AssociationEdge],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute lightweight graph metrics from association edges.

    Uses only collections.Counter — no networkx needed.
    Computes:
    - in_degree: how many edges point TO this modality (driven by others)
    - out_degree: how many edges originate FROM this modality (drives others)
    - driver_score: out_degree - in_degree
    - temporal_precedence_score: alias for driver_score (signal-level only)

    CRITICAL: These are **signal-level temporal descriptions**, NOT
    psychological role labels.  A "driver" in the association graph means
    "this modality's signal tends to precede others in time" — it does
    NOT mean the person is psychologically "leading" the interaction.
    Causal interpretation requires experimental design (e.g., perturbation).

    Parameters
    ----------
    edges : list of AssociationEdge
        Directed edges (only significant ones).
    alpha : float
        Significance threshold (for labeling; already filtered by caller).

    Returns
    -------
    metrics : dict
        {modality_name: {"in_degree": int, "out_degree": int,
                         "driver_score": int, "is_temporal_precedent": bool}}
    """
    from collections import Counter

    out_degrees = Counter(e.source for e in edges if e.is_significant)
    in_degrees = Counter(e.target for e in edges if e.is_significant)

    all_modalities = set(out_degrees.keys()) | set(in_degrees.keys())
    metrics: Dict[str, Dict[str, Any]] = {}

    for mod in all_modalities:
        out_d = out_degrees.get(mod, 0)
        in_d = in_degrees.get(mod, 0)
        driver = out_d - in_d
        metrics[mod] = {
            "in_degree": in_d,
            "out_degree": out_d,
            "driver_score": driver,
            "temporal_precedence_score": driver,
            "is_hub": out_d >= 2,              # backward-compat alias
            "is_follower": in_d >= 2,          # backward-compat alias
            "is_temporal_precedent": out_d >= 2,
        }

    return metrics







# ---------------------------------------------------------------------------
# Time-Varying Association — Rolling Window CCF for Dynamic Temporal Precedence Analysis
# ---------------------------------------------------------------------------

def rolling_association(
    dataset: "SynchronyDataset",
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    max_lag_sec: float = 10.0,
    apply_window: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute time-varying CCF in overlapping windows.

    The global CCF from ``temporal_association_analysis`` compresses the entire session
    into a single lead-lag estimate.  In real social interactions,
    leadership is **dynamic** — person A may drive synchrony in the first
    5 minutes while B takes over later.  ``rolling_association`` reveals these
    turn-taking dynamics by computing CCF per window and tracking how
    peak lag and direction change over time.

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned (normalized is recommended but not required).
    window_sec : float
        Duration of each analysis window in seconds.  Default 60.
        Should be long enough for reliable CCF: ``window_sec > 3 * max_lag_sec``.
    step_sec : float
        Step between consecutive windows.  Default 30 (50% overlap).
        Smaller steps = finer temporal resolution, more computation.
    max_lag_sec : float
        Maximum CCF lag per window.  Default 10.
        **Must be < window_sec / 3** for statistical reliability.
    apply_window : bool
        Apply Hanning window within each CCF (default True).

    Returns
    -------
    windows : list of dict
        Each dict represents one time window:
        {
            "window_start": float,
            "window_end": float,
            "window_center": float,
            "pairs": [{ ... }],
            "driver_score": { modality: int },
        }

    Notes
    -----
    No surrogate testing is performed per-window (too expensive for
    rolling analysis).  **WARNING**: the direction assignments in
    ``pairs`` are **NOT statistically validated** for any
    individual window.  Pure noise will also produce a peak lag
    and a direction label.

    **Correct usage**: run ``temporal_association_analysis`` on the full session
    first; only apply ``rolling_association`` to *already-significant*
    pairs to decompose their temporal dynamics.

    The output dicts contain a ``"significance_tested": False``
    field and a ``"warning"`` field to remind consumers of this
    limitation.
    """
    if not dataset._aligned:
        raise ValueError("Dataset must be aligned before rolling association.")

    if max_lag_sec >= window_sec / 3:
        raise ValueError(
            f"max_lag_sec ({max_lag_sec}) must be < window_sec / 3 "
            f"({window_sec / 3:.1f}).  CCF needs >=3x max_lag of data "
            f"per window for reliable estimation."
        )

    hz = dataset.target_hz
    if hz <= 0:
        raise ValueError(f"Invalid sampling rate: {hz} Hz")

    win_samples = int(np.floor(window_sec * hz))
    step_samples = int(np.floor(step_sec * hz))
    min_samples = 2 * int(np.floor(max_lag_sec * hz)) + 1

    if win_samples < min_samples:
        raise ValueError(
            f"Window ({win_samples} samples) too short for max_lag={max_lag_sec}s. "
            f"Need at least {min_samples} samples."
        )

    feat_cols = dataset.feature_columns
    modality_names = dataset.modality_names

    min_len = min(
        len(dataset.modalities[name])
        for name in modality_names
        if dataset.modalities[name] is not None
    )

    results: List[Dict[str, Any]] = []

    for start in range(0, min_len - win_samples + 1, step_samples):
        end = start + win_samples
        t_start = float(start / hz)
        t_end = float(end / hz)
        t_center = float((start + end) / 2 / hz)

        pairs: List[Dict[str, Any]] = []
        driver_delta: Dict[str, int] = {}

        for i, name_a in enumerate(modality_names):
            for name_b in modality_names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue

                        x_win = x[start:end].copy()
                        y_win = y[start:end].copy()

                        either_nan = np.isnan(x_win) | np.isnan(y_win)
                        if either_nan.sum() > 0.2 * len(x_win):
                            continue

                        # Linear interpolation (consistent across both modes).
                        # Mean imputation creates step discontinuities that
                        # produce high-frequency spectral artifacts.
                        x_clean = (
                            pd.Series(x_win).interpolate(method="linear")
                            .ffill().bfill().values
                        )
                        y_clean = (
                            pd.Series(y_win).interpolate(method="linear")
                            .ffill().bfill().values
                        )

                        try:
                            lags, ccf = compute_ccf(
                                x_clean, y_clean, max_lag_sec, hz, apply_window
                            )
                        except ValueError:
                            continue

                        peak_idx = int(np.argmax(np.abs(ccf)))
                        peak_lag = float(lags[peak_idx])
                        peak_val = float(ccf[peak_idx])

                        if peak_lag < 0:
                            direction = f"{name_a}→{name_b}"
                            driver_delta[name_a] = driver_delta.get(name_a, 0) + 1
                        elif peak_lag > 0:
                            direction = f"{name_b}→{name_a}"
                            driver_delta[name_b] = driver_delta.get(name_b, 0) + 1
                        else:
                            direction = "synchronous"

                        pairs.append({
                            "modality_a": name_a,
                            "modality_b": name_b,
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "peak_lag_sec": abs(peak_lag),
                            "peak_ccf": peak_val,
                            "direction": direction,
                        })

        results.append({
            "window_start": t_start,
            "window_end": t_end,
            "window_center": t_center,
            "pairs": pairs,
            "driver_score": driver_delta,
            "significance_tested": False,
            "warning": (
                "No surrogate testing performed per window. Direction "
                "assignments are NOT statistically validated. "
                "Run temporal_association_analysis() on significant pairs first, then "
                "use rolling_association() to decompose their temporal dynamics."
            ),
        })

    return results


# Re-export for convenience
from .dataset import SynchronyDataset  # noqa: E402, F401

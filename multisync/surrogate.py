"""
Lightweight surrogate-generation module providing FT and IAAFT surrogates.

Shared public API used by the main analysis pipeline (dynamic_features.py,
core.py) so surrogate generation lives in one place rather than the
validation sub-package.
"""

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FT surrogate (Fourier-phase randomization)
# ---------------------------------------------------------------------------

def ft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one FT surrogate (Fourier-phase randomization) of ``x``.

    This is the standard phase-randomized Fourier-transform surrogate
    (Theiler et al. 1992): take the FFT, replace the phases with uniform
    random phases while keeping the magnitudes, then invert.  It preserves
    the power spectrum (and hence the linear autocorrelation function) of
    ``x``.  The amplitude distribution is **not** preserved; under phase
    randomization the output approaches Gaussianity by the Central Limit
    Theorem (Schreiber & Schmitz 2000).

    Parameters
    ----------
    x : np.ndarray
        Input time series (1-D, finite values required).
    rng : np.random.Generator
        Random number generator for reproducible phase draws.

    Returns
    -------
    np.ndarray
        FT surrogate of ``x``, same length as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("ft_surrogate requires finite input (no NaN).")
    n = x.size
    X = np.fft.rfft(x)
    magnitudes = np.abs(X)

    # Random phases for non-DC, non-Nyquist bins
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=magnitudes.size)
    random_phases[0] = 0.0                     # DC must be real
    if n % 2 == 0:
        random_phases[-1] = 0.0              # Nyquist must be real (even n)

    X_surr = magnitudes * np.exp(1j * random_phases)
    surr = np.fft.irfft(X_surr, n=n)
    return surr


# Backward-compatible alias
prtf_surrogate = ft_surrogate


# ---------------------------------------------------------------------------
# IAAFT surrogate (Iterative Amplitude-Adjusted Fourier Transform)
# ---------------------------------------------------------------------------

def block_permutation_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    block_size: Optional[int] = None,
) -> np.ndarray:
    """Generate a block-permutation surrogate of ``x``.

    The series is divided into contiguous blocks of length ``block_size`` and
    the blocks are randomly permuted. This preserves local autocorrelation
    within each block while destroying longer-run temporal structure.

    Parameters
    ----------
    x : np.ndarray
        Input time series (1-D, finite values required).
    rng : np.random.Generator
        Random number generator.
    block_size : int or None
        Block length in samples. If None, set to ``max(2, int(sqrt(n)))``.

    Returns
    -------
    np.ndarray
        Block-permutation surrogate of ``x``, same length as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("block_permutation_surrogate requires finite input (no NaN).")
    n = x.size
    if n < 4:
        return x.copy()

    if block_size is None:
        block_size = max(2, int(np.sqrt(n)))
    block_size = max(2, min(block_size, n))

    n_blocks = int(np.ceil(n / block_size))
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    shuffled = np.concatenate([blocks[i] for i in order])
    return shuffled[:n]


def state_transition_shuffle_surrogate(
    wcc: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    hysteresis_delta: float = 0.0,
) -> np.ndarray:
    """Generate a state-transition shuffle surrogate.

    This null model binarizes the WCC trace into elevated (above threshold)
    and baseline segments, then shuffles the ORDER of these segments.
    It preserves the exact distribution of dwell times and baseline times,
    but destroys any long-range temporal organization or event-locked structure.

    Parameters
    ----------
    wcc : np.ndarray
        Observed WCC series.
    threshold : float
        Binarization threshold.
    rng : np.random.Generator
        Random number generator.
    hysteresis_delta : float
        Hysteresis for binarization (Schmitt trigger).

    Returns
    -------
    np.ndarray
        Shuffled WCC series.
    """
    from .feature_definitions import _binarize_with_hysteresis

    n = len(wcc)
    above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)

    # Degenerate case: no transitions — entire trace is baseline or elevated.
    # Just permute the time points as a fallback.
    if not above.any() or above.all():
        return rng.permutation(wcc)

    # Run-length encode with guarded indexing
    padded = np.concatenate(([not above[0]], above, [not above[-1]]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Guard: clamp ends to [0, n] and ensure starts/ends are matched
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    if len(starts) == 0 or len(ends) == 0:
        return rng.permutation(wcc)
    # Match starts and ends (diff may produce mismatched counts at boundaries)
    n_pairs = min(len(starts), len(ends))
    starts = starts[:n_pairs]
    ends = ends[:n_pairs]
    ends = np.clip(ends, 0, n)

    # Extract segments with safe slicing
    segments = []
    if not above[0] and starts[0] > 0:
        segments.append(wcc[:starts[0]])

    for i in range(len(starts)):
        si = int(starts[i])
        ei = int(ends[i])
        if ei > si:
            segments.append(wcc[si:ei])
        if i < len(starts) - 1:
            ni = int(starts[i + 1])
            if ei < ni:
                segments.append(wcc[ei:ni])
        elif ei < n:
            segments.append(wcc[ei:])

    if not segments:
        return rng.permutation(wcc)

    # Separate into elevated and baseline pools, skip empty segments
    elevated_pool = [seg for seg in segments
                     if len(seg) > 0 and np.nanmean(seg) >= threshold - hysteresis_delta]
    baseline_pool = [seg for seg in segments
                     if len(seg) > 0 and np.nanmean(seg) < threshold - hysteresis_delta]

    if not elevated_pool or not baseline_pool:
        return rng.permutation(wcc)

    rng.shuffle(elevated_pool)
    rng.shuffle(baseline_pool)

    # Re-assemble with alternating elevated-baseline order
    shuffled_segments = []
    is_elevated = above[0]
    e_idx = 0
    b_idx = 0
    for _ in range(len(segments)):
        if is_elevated:
            if e_idx < len(elevated_pool):
                shuffled_segments.append(elevated_pool[e_idx])
                e_idx += 1
        else:
            if b_idx < len(baseline_pool):
                shuffled_segments.append(baseline_pool[b_idx])
                b_idx += 1
        is_elevated = not is_elevated

    res = np.concatenate(shuffled_segments)
    return res[:n]


def iaaft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Generate one IAAFT surrogate of ``x``.

    The IAAFT algorithm alternates between matching the empirical amplitude
    distribution and matching the Fourier magnitudes.  A finite surrogate
    cannot, in general, preserve both constraints exactly.  SyncPipe returns
    the final **rank-adjusted** sequence: the empirical amplitude distribution
    is preserved exactly (up to floating-point ordering/ties), while the power
    spectrum / linear autocorrelation is matched approximately.  This is the
    appropriate default for SyncPipe's signal-level and WCC-level nulls, where
    the null should not change the marginal value distribution.

    Parameters
    ----------
    x : np.ndarray
        Input time series (1-D, finite values required).
    rng : np.random.Generator
        Random number generator.
    max_iter : int
        Maximum number of iterative adjustment cycles.
    tol : float
        Convergence tolerance on iterative signal change.

    Returns
    -------
    np.ndarray
        IAAFT surrogate of ``x`` with the same empirical amplitude
        distribution as ``x`` and an approximately matched power spectrum.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("iaaft_surrogate requires finite input (no NaN).")
    n = x.size
    if n < 4:
        return x.copy()

    # Step 1: target sorted values (empirical amplitude distribution)
    x_sorted = np.sort(x)

    # Step 2: initial FT surrogate (phase randomization)
    X = np.fft.rfft(x)
    magnitudes = np.abs(X)
    phases = np.angle(X)

    random_phases = rng.uniform(-np.pi, np.pi, size=magnitudes.size)
    random_phases[0] = phases[0]       # preserve DC
    if n % 2 == 0:
        random_phases[-1] = phases[-1]  # preserve Nyquist (even n)

    X_init = magnitudes * np.exp(1j * random_phases)
    x_surr = np.fft.irfft(X_init, n=n)

    # Step 3: iterative amplitude-spectrum matching
    for _ in range(max_iter):
        # (a) Match amplitude distribution via rank ordering.
        rank_order = np.argsort(np.argsort(x_surr))
        x_adjusted = x_sorted[rank_order]

        # (b) FFT of rank-ordered surrogate
        X_adj = np.fft.rfft(x_adjusted)

        # (c) Replace magnitudes with original power spectrum
        X_new = magnitudes * np.exp(1j * np.angle(X_adj))

        # (d) IFFT back to time domain
        x_new = np.fft.irfft(X_new, n=n)

        # (e) Convergence check on iterative signal change
        signal_change = float(np.sum((x_new - x_surr) ** 2))
        x_surr = x_new
        if signal_change < tol * n:
            break

    # Final rank adjustment: exact empirical amplitude distribution, approximate
    # spectrum. Returning x_surr here would instead privilege the exact spectrum
    # and allow the marginal distribution to drift.
    final_rank_order = np.argsort(np.argsort(x_surr))
    return x_sorted[final_rank_order]

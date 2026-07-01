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


def iaaft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Generate one IAAFT surrogate of ``x``.

    Preserves **both** the power spectrum **and** the amplitude distribution
    of ``x``.  This is the primary (more conservative, field-standard)
    null model; see ``ft_surrogate`` for the phase-randomization
    robustness comparator.

    Parameters
    ----------
    x : np.ndarray
        Input time series (1-D, finite values required).
    rng : np.random.Generator
        Random number generator.
    max_iter : int
        Maximum number of iterative adjustment cycles.
    tol : float
        Convergence tolerance on spectral error (sum-of-squares).

    Returns
    -------
    np.ndarray
        IAAFT surrogate of ``x``.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("iaaft_surrogate requires finite input (no NaN).")
    n = x.size
    if n < 4:
        return x.copy()

    # Step 1: target sorted values (amplitude distribution)
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
    prev_amp = None
    for _ in range(max_iter):
        # (a) Match amplitude distribution via rank ordering
        rank_order = np.argsort(np.argsort(x_surr))
        x_adjusted = x_sorted[rank_order]

        # (b) FFT of rank-ordered surrogate
        X_adj = np.fft.rfft(x_adjusted)

        # (c) Replace magnitudes with original power spectrum
        X_new = magnitudes * np.exp(1j * np.angle(X_adj))

        # (d) IFFT
        x_new = np.fft.irfft(X_new, n=n)

        # (e) Convergence check on SIGNAL change
        if prev_amp is not None:
            signal_change = float(np.sum((x_new - x_surr) ** 2))
            if signal_change < tol * n:
                return x_new
        prev_amp = x_new.copy()
        x_surr = x_new

    return x_surr

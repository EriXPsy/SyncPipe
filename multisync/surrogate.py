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

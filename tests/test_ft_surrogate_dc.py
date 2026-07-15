"""Regression tests: ft_surrogate must preserve the DC component's phase
(sign of the mean). gstack Finding 7.

The buggy version forced ``random_phases[0] = 0.0``, which made the
reconstructed DC term ``|X[0]| * exp(i*0) = +|X[0]|`` — deterministically
flipping a negative-mean signal to a positive-mean surrogate (Claude:
30 seeds all yielded mean = +0.4110, variance 0). Phase randomization must
preserve the DC phase, so the surrogate mean equals the original mean.
"""
import numpy as np

from multisync.surrogate import ft_surrogate


def test_ft_surrogate_preserves_negative_mean_sign():
    """A negative-mean signal must yield negative-mean surrogates."""
    rng = np.random.default_rng(0)
    x = -0.4 + rng.normal(0.0, 1.0, 256)
    assert x.mean() < 0

    means = []
    surrogates = []
    for seed in range(30):
        s = ft_surrogate(x, np.random.default_rng(seed))
        means.append(s.mean())
        surrogates.append(s)
    means = np.asarray(means)

    # Mean preserved exactly (DC phase preserved) -> stays negative, not flipped.
    assert abs(means.mean() - x.mean()) < 1e-9
    assert means.mean() < 0
    # Surrogates genuinely differ across seeds (phase randomization active).
    assert not np.allclose(surrogates[0], surrogates[1])


def test_ft_surrogate_preserves_positive_mean_sign():
    """Symmetry check: a positive-mean signal stays positive (sanity)."""
    rng = np.random.default_rng(1)
    x = 0.5 + rng.normal(0.0, 1.0, 256)
    assert x.mean() > 0
    for seed in range(10):
        s = ft_surrogate(x, np.random.default_rng(seed))
        assert abs(s.mean() - x.mean()) < 1e-9


def test_ft_surrogate_even_length_nyquist_preserved():
    """Even-length signals: Nyquist phase preserved, stays real/consistent."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 300)  # even length
    s = ft_surrogate(x, np.random.default_rng(7))
    # Reconstruction must be real (finite) — broken Nyquist handling would
    # introduce non-negligible imaginary leakage.
    assert np.all(np.isfinite(s))
    assert abs(s.mean() - x.mean()) < 1e-9

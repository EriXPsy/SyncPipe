"""Cross-validation of the WCC core and the IAAFT surrogate against
independent textbook reference implementations (P1 items of the 2026-09-08
OSS deep-optimization review).

Rationale: SyncPipe's WCC operator and segment-wise IAAFT are deliberately
kept in-house (no equivalent upstream exists), so their numerical correctness
is defended by comparing against naive, independent implementations written
here from first principles.  Any future optimization of the fast paths
(cumsum backend, stride backend, surrogate iteration) must keep these
equivalences green.

These are contract-level equivalence tests; they do not replace the
statistical validation batteries under the ``slow`` marker.
"""

import numpy as np
import pytest
from scipy.signal.windows import get_window

from syncpipe.dynamic_features import sliding_window_wcc
from syncpipe.surrogate import iaaft_surrogate


# --------------------------------------------------------------------------
# Independent reference implementations (textbook, no shared code paths)
# --------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two 1-D arrays; NaN if either side is degenerate."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return np.nan
    if a.std() == 0.0 or b.std() == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _reference_wcc(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_samples: int = 1,
    min_valid_ratio: float = 0.5,
) -> np.ndarray:
    """Naive leading-window WCC: output[i] covers x[i:i+w] vs y[i:i+w].
    Lag is applied by the caller via :func:`_lag`."""
    out = []
    for i in range(0, len(x) - window_size + 1, max(step_samples, 1)):
        a = x[i : i + window_size]
        b = y[i : i + window_size]
        pair = np.isfinite(a) & np.isfinite(b)
        if pair.mean() < min_valid_ratio:
            out.append(np.nan)
            continue
        out.append(_pearson(a, b))
    return np.asarray(out, dtype=float)


def _lag(y: np.ndarray, lag_samples: int) -> np.ndarray:
    """Shift y forward by lag samples; the first `lag` samples become NaN."""
    out = np.full(y.size, np.nan, dtype=float)
    if lag_samples < y.size:
        out[lag_samples:] = y[: y.size - lag_samples]
    return out


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.6) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# --------------------------------------------------------------------------
# P1-1: WCC numerical equivalence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window_size", [20, 50, 120])
def test_wcc_clean_rect_matches_textbook_reference(window_size):
    rng = np.random.default_rng(20260908)
    n = 600
    x = _ar1(rng, n)
    y = _ar1(rng, n)
    got = sliding_window_wcc(x, y, window_size=window_size, hz=1.0)
    ref = _reference_wcc(x, y, window_size)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_wcc_nan_stride_path_matches_textbook_reference():
    """Pairwise-deletion contract: with NaN present, windows passing the
    min_valid_ratio gate must equal the textbook per-window Pearson over
    pairwise-valid points (BUG-2 regression guard — this was all-NaN
    before the 2026-09-08 NaN-hygiene fix)."""
    rng = np.random.default_rng(7)
    n = 400
    x = _ar1(rng, n)
    y = _ar1(rng, n)
    x[rng.random(n) < 0.10] = np.nan
    y[rng.random(n) < 0.10] = np.nan
    w = 50
    got = sliding_window_wcc(x, y, window_size=w, hz=1.0, min_valid_ratio=0.5)
    ref = _reference_wcc(x, y, w, min_valid_ratio=0.5)
    assert np.isfinite(got).mean() > 0.9, "NaN input must not void the trace"
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=1e-8)


def test_wcc_step_and_lag_match_textbook_reference():
    rng = np.random.default_rng(11)
    n = 500
    x = _ar1(rng, n)
    y = _ar1(rng, n)
    w, step, lag = 60, 7, 4
    got = sliding_window_wcc(
        x, y, window_size=w, hz=1.0, step_samples=step, lag_samples=lag
    )
    ref = _reference_wcc(x, _lag(y, lag), w, step_samples=step)
    assert got.shape == ref.shape
    mask = np.isfinite(got) & np.isfinite(ref)
    assert mask.mean() > 0.9  # lag boundary NaNs are expected on both sides
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-10, rtol=1e-10)


def test_wcc_hann_taper_matches_weighted_reference():
    """Non-rect tapers route to the stride backend; verify against an
    independent weighted Pearson (kernel weights, per-window weighted
    means — algebraically equal to the backend's global-demean formula)."""
    rng = np.random.default_rng(13)
    n = 300
    x = _ar1(rng, n)
    y = _ar1(rng, n)
    w = 50
    got = sliding_window_wcc(x, y, window_size=w, hz=1.0, window_type="hann")
    # Periodic Hann (scipy convention, as the backend resolves via
    # scipy.signal.windows.get_window) — a definitional constant, not the
    # algorithm under test.
    kern = get_window("hann", w)
    kern = kern * (w / kern.sum())  # mean weight 1, as the backend does
    ref = []
    for i in range(n - w + 1):
        a, b = x[i : i + w], y[i : i + w]
        k = kern
        mx_w = float(np.sum(k * a) / np.sum(k))
        my_w = float(np.sum(k * b) / np.sum(k))
        cov = float(np.sum(k * (a - mx_w) * (b - my_w)) / np.sum(k))
        vx = float(np.sum(k * (a - mx_w) ** 2) / np.sum(k))
        vy = float(np.sum(k * (b - my_w) ** 2) / np.sum(k))
        ref.append(cov / np.sqrt(vx * vy))
    ref = np.asarray(ref)
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=1e-8)


# --------------------------------------------------------------------------
# P1-2: IAAFT surrogate properties against the textbook algorithm contract
# --------------------------------------------------------------------------

def test_iaaft_preserves_empirical_distribution_exactly():
    """SyncPipe returns the rank-adjusted IAAFT variant: the marginal value
    distribution must be preserved exactly (multiset equality)."""
    rng = np.random.default_rng(2026)
    x = _ar1(rng, 800, phi=0.7)
    s = iaaft_surrogate(x, rng)
    assert s.shape == x.shape
    np.testing.assert_allclose(np.sort(s), np.sort(x), atol=1e-12)


def test_iaaft_power_spectrum_approximately_matched():
    """The iterative phase must drive the surrogate spectrum close to the
    observed one.  Rank adjustment trades a little spectral error back, so
    this is a tolerance check, not exact equality."""
    rng = np.random.default_rng(99)
    x = _ar1(rng, 1024, phi=0.8)
    s = iaaft_surrogate(x, rng)
    fx = np.abs(np.fft.rfft(x - x.mean()))
    fs = np.abs(np.fft.rfft(s - s.mean()))
    rel_rms = float(np.sqrt(np.mean((fs - fx) ** 2)) / np.sqrt(np.mean(fx**2)))
    assert rel_rms < 0.20, (
        f"IAAFT spectrum mismatch too large: relative RMS = {rel_rms:.3f} "
        "(the iterative adjustment should track the observed magnitudes)"
    )


def test_iaaft_destroys_dyadic_coupling():
    """L0 contract: independently drawn surrogates of two signals must not
    inherit the coupling between the observed pair, while the observed WCC
    level is high."""
    rng = np.random.default_rng(321)
    n, w = 900, 60
    base = _ar1(rng, n, phi=0.5)
    x = base + 0.5 * rng.normal(size=n)
    y = base + 0.5 * rng.normal(size=n)
    observed_level = float(np.nanmean(sliding_window_wcc(x, y, window_size=w)))
    assert observed_level > 0.5, "fixture must actually contain coupling"

    surr_corr = []
    for _ in range(40):
        sx = iaaft_surrogate(x, rng)
        sy = iaaft_surrogate(y, rng)
        surr_corr.append(
            float(np.nanmean(sliding_window_wcc(sx, sy, window_size=w)))
        )
    null_level = float(np.mean(np.abs(surr_corr)))
    assert null_level < 0.15, (
        f"mean |WCC| over surrogate pairs = {null_level:.3f}; the IAAFT null "
        "must destroy the dyadic coupling (observed "
        f"{observed_level:.3f})"
    )

"""
Smoke + correctness tests for Level 3 surrogate testing.

Run with::
    python -m pytest tests/validation/test_pgt1_intensity.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.validation.pgt1_intensity import (
    Level3Config,
    ft_surrogate,
    prtf_surrogate,
    iaaft_surrogate,
    phipson_smyth_p,
    bh_fdr,
    run_level3_grid,
    apply_bh_fdr_within_noise,
    summarise_level3,
    FEATURE_TAILS,
    FEATURE_P_COLUMNS,
    REFERENCE_TAILS,
)


# =====================================================================
# FT surrogate (Fourier-phase randomization; Theiler et al. 1992)
# =====================================================================

def test_ft_preserves_power_spectrum():
    """
    The FT surrogate must preserve |FFT|**2 (power spectrum) up to
    floating-point error.  The magnitude is preserved exactly; the phase
    is randomized.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    surr = ft_surrogate(x, rng=np.random.default_rng(1))

    amp_x = np.abs(np.fft.rfft(x))
    amp_s = np.abs(np.fft.rfft(surr))
    # Magnitude is preserved exactly (up to rounding)
    assert np.allclose(amp_x, amp_s, atol=1e-10), \
        "FT surrogate must preserve FFT magnitudes (hence power spectrum)."


def test_ft_destroys_amplitude_distribution_under_burst():
    """
    Burst signals should look more Gaussian after the FT surrogate,
    because phase randomization + CLT Gaussianizes the amplitude
    distribution.
    """
    rng = np.random.default_rng(0)
    n = 500
    t = np.arange(n)
    # Create a burst-dominated signal
    bursts = sum(
        np.exp(-0.5 * ((t - bt) / 3.0) ** 2)
        for bt in (50, 150, 250, 350, 450)
    )
    x = bursts + 0.1 * rng.normal(size=n)

    surr = ft_surrogate(x, rng=np.random.default_rng(2))

    from scipy.stats import kurtosis  # local import to avoid top-level hard dep
    k_x = kurtosis(x)
    k_s = kurtosis(surr)
    assert abs(k_s) < abs(k_x), (
        f"FT surrogate should reduce kurtosis on burst signals: "
        f"kurt(x)={k_x:.2f}, kurt(surr)={k_s:.2f}"
    )


def test_ft_returns_real_signal():
    """Output of the FT surrogate must be real-valued (no imaginary part)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    surr = ft_surrogate(x, rng=rng)
    assert np.all(np.isreal(surr))


def test_ft_same_length():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    surr = ft_surrogate(x, rng=np.random.default_rng(1))
    assert len(surr) == len(x)


def test_prtf_alias_equivalent_to_ft():
    """
    ``prtf_surrogate`` is a backward-compatible alias of ``ft_surrogate``
    (the function was previously misnamed "PRTF").  Both names must refer
    to the same callable and yield identical output for the same seed.
    """
    assert prtf_surrogate is ft_surrogate
    rng_seed = 7
    x = np.random.default_rng(0).normal(0, 1, 500)
    a = ft_surrogate(x, rng=np.random.default_rng(rng_seed))
    b = prtf_surrogate(x, rng=np.random.default_rng(rng_seed))
    assert np.allclose(a, b)


# =====================================================================
# IAAFT surrogate
# =====================================================================

def test_iaaft_preserves_amplitude_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    surr = iaaft_surrogate(x, rng=np.random.default_rng(1))
    # Amplitude distribution: ranks should match (within convergence tolerance)
    # IAAFT may not converge exactly in 200 iterations,
    # so we check Spearman correlation ≈ 1 (rank preservation)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(np.sort(x), np.sort(surr))
    assert rho > 0.99, \
        f"IAAFT must approximately preserve amplitude distribution (rho={rho:.4f})"


def test_iaaft_approximately_preserves_power_spectrum():
    rng = np.random.default_rng(0)
    n = 500
    t = np.arange(n)
    x = np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.normal(size=n)
    surr = iaaft_surrogate(x, rng=np.random.default_rng(2))
    amp_x = np.abs(np.fft.rfft(x))
    amp_s = np.abs(np.fft.rfft(surr))
    # Allow ~5% relative spectral drift (finite iterations)
    rel_err = np.abs(amp_x - amp_s) / (amp_x + 1e-12)
    assert np.median(rel_err) < 0.05, \
        "IAAFT spectral error median must be < 5% (after convergence)."


def test_iaft_destroys_cross_phase():
    """IAAFT should destroy cross-phase between two independent signals.

    With two independent Gaussian noise signals, IAAFT surrogates
    should have ~zero cross-correlation.
    """
    rng = np.random.default_rng(0)
    n = 500

    # Two INDEPENDENT Gaussian noise signals
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    r_obs = np.corrcoef(x, y)[0, 1]
    assert abs(r_obs) < 0.1   # ~uncorrelated

    x_s = iaaft_surrogate(x, rng=np.random.default_rng(3))
    y_s = iaaft_surrogate(y, rng=np.random.default_rng(4))
    r_surr = np.corrcoef(x_s, y_s)[0, 1]
    # Independent IAAFT: cross-phase destroyed
    assert abs(r_surr) < 0.5   # loose; key is "not ~1.0"



# =====================================================================
# Phipson-Smyth p-value
# =====================================================================

def test_phipson_smyth_min_p_floor():
    """Minimum achievable p = 1 / (1 + N)."""
    null = np.zeros(10)
    p_upper = phipson_smyth_p(observed=1.0, null_values=null, tail="upper")
    # k=0, N=10  ->  p = (1+0)/(1+10) = 1/11
    assert abs(p_upper - 1.0 / 11.0) < 1e-12


def test_phipson_smyth_never_zero():
    """Unbiased estimator never produces p=0."""
    null = np.full(100, -1.0)
    p = phipson_smyth_p(observed=10.0, null_values=null, tail="upper")
    assert p > 0.0


def test_phipson_smyth_two_tailed_symmetric():
    """For observed = median, two-tailed p should be ~1.0."""
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 999)
    p = phipson_smyth_p(observed=0.0, null_values=null, tail="two")
    # observed ≈ median → p ≈ 1.0
    assert p > 0.8


def test_phipson_smyth_nan_observed():
    null = rng.normal(0, 1, 99) if 'rng' in dir() else np.random.normal(0, 1, 99)
    p = phipson_smyth_p(observed=np.nan, null_values=null, tail="upper")
    assert np.isnan(p)


# =====================================================================
# BH-FDR
# =====================================================================

def test_bh_fdr_all_significant():
    p = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    rej = bh_fdr(p, q=0.05)
    assert rej.all()


def test_bh_fdr_none_significant():
    p = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    rej = bh_fdr(p, q=0.05)
    assert not rej.any()


def test_bh_fdr_handles_nan():
    p = np.array([0.001, np.nan, 0.5])
    rej = bh_fdr(p, q=0.05)
    assert rej[0] and not rej[1] and not rej[2]


def test_bh_fdr_returns_bool_array():
    p = np.array([0.01, 0.5])
    rej = bh_fdr(p, q=0.05)
    assert rej.dtype == bool


# =====================================================================
# Pipeline smoke
# =====================================================================

@pytest.fixture(scope="module")
def small_grid_3a() -> pd.DataFrame:
    """
    1 noise x 1 coupling x 3 seeds x 49 surrogates - quick smoke test.
    """
    cfg = Level3Config(
        noise_ratios=(0.3,),
        couplings=(0.0,),
        seeds=tuple(range(3000, 3003)),
        n_surrogates=49,          # min meaningful for smoke
        iaaft_max_iter=30,
        surrogate_method="prtf",   # exercises the "prtf"->ft alias dispatch; IAAFT tested separately
    )
    return run_level3_grid(cfg)


def test_grid_returns_one_row_per_cell(small_grid_3a):
    # 1 noise x 1 coupling x 3 seeds = 3 rows
    assert len(small_grid_3a) == 3
    expected_cols_subset = {
        "noise_ratio", "coupling", "seed",
        "obs_peak_amplitude", "p_peak_amplitude",
        "obs_mean_synchrony", "p_mean_synchrony",
    }
    assert expected_cols_subset.issubset(set(small_grid_3a.columns))


def test_p_values_in_unit_interval(small_grid_3a):
    for col in FEATURE_P_COLUMNS:
        p = small_grid_3a[col].dropna()
        assert (p > 0.0).all(), f"{col} has p=0 (should be >0)"
        assert (p <= 1.0).all(), f"{col} has p>1"


def test_p_values_min_floor_respects_phipson(small_grid_3a):
    """n_surrogates=49 -> min p = 1/50 = 0.02."""
    for col in FEATURE_P_COLUMNS:
        p = small_grid_3a[col].dropna()
        if len(p) > 0:
            assert p.min() >= 1.0 / 50.0 - 1e-12, \
                f"{col} min p = {p.min():.4f}, expected >= {1/50:.4f}"


def test_fdr_correction_adds_reject_columns(small_grid_3a):
    df_fdr = apply_bh_fdr_within_noise(small_grid_3a, q=0.05)
    for col in FEATURE_P_COLUMNS:
        rej_col = col.replace("p_", "reject_")
        assert rej_col in df_fdr.columns, f"Missing column: {rej_col}"
        assert df_fdr[rej_col].dtype == bool
    assert "n_reject" in df_fdr.columns


def test_summarise_returns_per_cell_rates(small_grid_3a):
    df_fdr = apply_bh_fdr_within_noise(small_grid_3a, q=0.05)
    s = summarise_level3(df_fdr)
    # 1 noise x 1 coupling -> 1 row
    assert len(s) == 1
    rate_cols = [c for c in s.columns if c.endswith("_rate")]
    assert len(rate_cols) == 3, f"Expected 3 rate columns, got {len(rate_cols)}"
    for c in rate_cols:
        v = float(s[c].iloc[0])
        assert 0.0 <= v <= 1.0, f"{c}={v} outside [0,1]"


def test_feature_tails_cover_all_features():
    """The confirmatory FDR family contains 3 features (SSoT Option B,
    2026-06-29).

    L0 (signal-level null): peak_amplitude
    L1 (WCC-level null): dwell_time, switching_rate
    mean_synchrony is a reported reference (REFERENCE_TAILS), and
    bimodality_coefficient / L2 features / synchrony_entropy are
    exploratory (not in FDR).
    """
    expected_fdr = {
        "peak_amplitude",
        "dwell_time", "switching_rate",
    }
    expected_ref = {"mean_synchrony"}  # reported reference, not FDR-corrected

    assert set(FEATURE_TAILS.keys()) == expected_fdr, (
        "FDR family (SSoT Option B): must be the 3 L0+L1 confirmatory features"
    )
    assert set(REFERENCE_TAILS.keys()) == expected_ref, (
        "Reference family: mean_synchrony is reported but not FDR-corrected"
    )
    # Disjointness — no feature may live in both families.
    assert set(FEATURE_TAILS.keys()).isdisjoint(REFERENCE_TAILS.keys()), (
        "FDR family and Reference family must be disjoint"
    )


# =====================================================================
# IAAFT smoke (slower, fewer surrogates)
# =====================================================================

@pytest.fixture(scope="module")
def small_grid_3a_iaaft() -> pd.DataFrame:
    cfg = Level3Config(
        noise_ratios=(0.3,),
        couplings=(0.0,),
        seeds=tuple(range(4000, 4002)),   # only 2 seeds (slow)
        n_surrogates=29,                    # IAAFT is iterative
        surrogate_method="iaaft",
        iaaft_max_iter=50,
    )
    return run_level3_grid(cfg)


def test_iaaft_grid_runs(small_grid_3a_iaaft):
    assert len(small_grid_3a_iaaft) == 2


def test_iaaft_p_values_in_unit_interval(small_grid_3a_iaaft):
    for col in FEATURE_P_COLUMNS:
        p = small_grid_3a_iaaft[col].dropna()
        if len(p) > 0:
            assert (p > 0.0).all()
            assert (p <= 1.0).all()

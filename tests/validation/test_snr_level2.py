"""Smoke tests for Level 2 SNR robustness validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.validation.snr import (
    Level2Config,
    run_level2_grid,
    summarise_level2,
    robustness_curves,
)


@pytest.fixture(scope="module")
def small_grid_df() -> pd.DataFrame:
    """3 noise x 2 coupling x 4 seeds = 24 cells (fast)."""
    cfg = Level2Config(
        noise_ratios=(0.1, 0.5, 1.0),
        couplings=(0.0, 0.7),
        seeds=tuple(range(2000, 2004)),
    )
    return run_level2_grid(cfg)


def test_grid_returns_one_row_per_cell(small_grid_df):
    assert len(small_grid_df) == 3 * 2 * 4
    expected_cols = {
        "noise_ratio", "coupling", "seed",
        "onset_threshold",
        "peak_amplitude", "mean_synchrony",
        "onset_latency", "onset_defined",
        "recovery_time", "recovery_defined",
        "rise_time", "synchrony_entropy",
    }
    assert expected_cols.issubset(set(small_grid_df.columns))


def test_peak_amplitude_decreases_with_noise(small_grid_df):
    """At fixed high coupling, raising noise_ratio must drag down peak."""
    sub = small_grid_df[small_grid_df["coupling"] == 0.7]
    means = sub.groupby("noise_ratio")["peak_amplitude"].mean()
    # Strict monotonicity is too strong (Monte Carlo wiggles); demand
    # cleanest > noisiest with a margin.
    assert means.loc[0.1] - means.loc[1.0] > 0.05, (
        f"Expected peak_amplitude to drop with rising noise; got {means.to_dict()}"
    )


def test_mean_synchrony_near_zero_at_coupling_zero(small_grid_df):
    sub = small_grid_df[small_grid_df["coupling"] == 0.0]
    assert sub["mean_synchrony"].abs().mean() < 0.15


def test_summarise_two_way_groupby(small_grid_df):
    s = summarise_level2(small_grid_df)
    assert len(s) == 3 * 2  # 3 noise x 2 coupling
    expected_cols = {
        "noise_ratio", "coupling",
        "peak_amplitude_mean", "peak_amplitude_sd",
        "onset_n_valid_fraction", "recovery_n_valid_fraction",
        "n_seeds", "onset_threshold",
    }
    assert expected_cols.issubset(s.columns)


def test_summarise_guards_against_mixed_thresholds(small_grid_df):
    df1 = small_grid_df.copy()
    df2 = small_grid_df.copy()
    df2["onset_threshold"] = 0.4
    mixed = pd.concat([df1, df2], ignore_index=True)
    with pytest.raises(ValueError, match="single onset_threshold"):
        summarise_level2(mixed)


def test_robustness_curves_shape(small_grid_df):
    pivot = robustness_curves(small_grid_df, "peak_amplitude")
    assert pivot.shape == (3, 2)  # 3 noise levels x 2 couplings
    assert set(pivot.index) == {0.1, 0.5, 1.0}
    assert set(pivot.columns) == {0.0, 0.7}


def test_robustness_curves_unknown_feature_raises(small_grid_df):
    with pytest.raises(KeyError):
        robustness_curves(small_grid_df, "not_a_real_feature")


def test_n_valid_fraction_bounded(small_grid_df):
    """Definedness fractions must lie in [0, 1]."""
    s = summarise_level2(small_grid_df)
    for col in ("onset_n_valid_fraction", "recovery_n_valid_fraction"):
        assert s[col].min() >= 0.0
        assert s[col].max() <= 1.0

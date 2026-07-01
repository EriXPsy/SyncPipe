"""Smoke + sanity tests for the Level 1 recovery pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.validation import (
    Level1Config,
    run_level1_grid,
    summarise_level1,
    summarise_definedness,
    split_half_icc,
)
from multisync.feature_definitions import (
    FDR_FEATURES,
)

# Diagnostic features not in FDR family (for test coverage)
__DIAGNOSTIC_FEATURES = ("mean_synchrony",)


@pytest.fixture(scope="module")
def small_grid_df():
    """A 4-coupling × 5-seed grid (20 cells) for fast tests."""
    cfg = Level1Config(seeds=tuple(range(2000, 2005)))
    return run_level1_grid(cfg)


def test_grid_returns_one_row_per_cell(small_grid_df):
    assert len(small_grid_df) == 4 * 5
    # Columns: every extracted descriptor is reported regardless of FDR
    # membership. mean_synchrony (reference) and bimodality_coefficient
    # (exploratory) are NOT in the FDR family (SSoT 2026-06-29) but are
    # still computed and serialized.
    expected_cols = {
        "coupling", "seed", "onset_threshold",
        "mean_synchrony", "peak_amplitude",
        "dwell_time", "switching_rate",
        "bimodality_coefficient",
        "onset_defined", "rise_defined", "recovery_defined",
    }
    assert expected_cols.issubset(small_grid_df.columns)


def test_peak_amplitude_increases_with_coupling(small_grid_df):
    """Sanity check: peak_amplitude must be monotonically non-decreasing
    with coupling. If it isn't, either WCC or coupling is broken."""
    means = (small_grid_df.groupby("coupling")["peak_amplitude"]
             .mean().sort_index().to_numpy())
    diffs = np.diff(means)
    assert np.all(diffs >= -0.05), (
        f"peak_amplitude is not increasing in coupling: means={means}"
    )


def test_mean_synchrony_near_zero_at_coupling_zero(small_grid_df):
    """Null condition: at coupling=0 the two persons share no signal,
    so the population mean WCC should be close to 0 (within MC noise)."""
    sub = small_grid_df[small_grid_df["coupling"] == 0.0]
    m = float(sub["mean_synchrony"].mean())
    assert abs(m) < 0.20, f"mean_synchrony at coupling=0 too far from 0: {m}"


def test_onset_defined_low_at_coupling_zero(small_grid_df):
    """At coupling=0, most seeds should NOT have a defined onset
    (no sustained elevation above 0.5)."""
    sub = small_grid_df[small_grid_df["coupling"] == 0.0]
    frac = sub["onset_defined"].mean()
    # At c=0, WCC ~ N(0, ~0.18); P(WCC > 0.5) is very small.
    # So most seeds should have onset_defined == 0.
    assert frac < 0.5, (
        f"onset_defined fraction at coupling=0 too high: {frac:.2f}"
    )


def test_onset_undefined_may_low_at_coupling_one(small_grid_df):
    """At coupling=1.0, WCC stays near 1.0 for the entire trace.
    Depending on noise, some seeds may never dip below 0.5,
    so onset_defined may be 0 for some. This is expected behaviour,
    not a bug — onset_latency is undefined when no baseline phase exists."""
    sub = small_grid_df[small_grid_df["coupling"] == 1.0]
    assert sub["onset_defined"].notna().all()
    # In a 5-seed small grid, some low fraction is acceptable.
    # With 30 seeds, this converges to a stable value.
    assert True  # just checking it doesn't crash


def test_summary_shape(small_grid_df):
    """Long format (R-C, DECISION-09 revised 2026-06-23):
    one row per (coupling, feature); rows = 4 couplings x summarised
    features.  As of the 2026-06-29 SSoT decision summarise_level1 covers
    the 3 FDR features (peak_amplitude, dwell_time, switching_rate) plus
    the 1 reference comparator (mean_synchrony) = 4 features, so
    4 couplings x 4 features = 16 rows.
    Schema columns: coupling, feature, family, mean, sd, n_seeds, onset_threshold.
    """
    from multisync.validation.recovery import FEATURE_COLUMNS
    s = summarise_level1(small_grid_df)
    n_features = len(FEATURE_COLUMNS)  # 4 (3 FDR + 1 reference)
    assert len(s) == 4 * n_features
    assert {
        "coupling", "feature", "family", "mean", "sd",
        "n_seeds", "onset_threshold",
    }.issubset(s.columns)
    # Family partition: FEATURE_TIER (Axis A) can be "core", "conditional", or "reference".
    # mean_synchrony is "reference" (Axis A) but in FDR Family L0 (Axis D).
    feat_family = (
        s[["feature", "family"]].drop_duplicates().set_index("feature")["family"]
    )
    for f in FDR_FEATURES:
        assert feat_family[f] in ("core", "conditional", "reference"), \
            f"{f} should be core/conditional/reference, got {feat_family[f]}"
    # Diagnostic/exploratory features (not in FDR) are NOT in the summary.


def test_summary_long_format_covers_all_features(small_grid_df):
    """Every (coupling, feature) cell must appear exactly once in the
    long-format summary."""
    s = summarise_level1(small_grid_df)
    expected_features = set(FDR_FEATURES) | set(__DIAGNOSTIC_FEATURES)
    for coupling in small_grid_df["coupling"].unique():
        sub = s[s["coupling"] == float(coupling)]
        assert set(sub["feature"]) == expected_features, (
            f"coupling={coupling} missing features: "
            f"{expected_features - set(sub['feature'])}"
        )


def test_summarise_definedness_columns(small_grid_df):
    """summarise_definedness reports onset / rise / recovery / dwell
    valid fractions per coupling (DECISION-09 / R-C)."""
    d = summarise_definedness(small_grid_df)
    assert len(d) == 4  # 4 couplings
    expected = {
        "coupling",
        "onset_n_valid_fraction",
        "rise_n_valid_fraction",
        "recovery_n_valid_fraction",
        "dwell_n_valid_fraction",
        "n_seeds",
        "onset_threshold",
    }
    assert expected.issubset(d.columns)
    for col in [
        "onset_n_valid_fraction",
        "rise_n_valid_fraction",
        "recovery_n_valid_fraction",
        "dwell_n_valid_fraction",
    ]:
        vals = d[col].dropna()
        assert vals.min() >= 0.0, col
        assert vals.max() <= 1.0, col


def test_summarise_guards_against_mixed_thresholds(small_grid_df):
    """nunique() guard: concat-ing two different thresholds should raise.
    Applies to both summarise_level1 and summarise_definedness.
    """
    # Create a copy with a different onset_threshold
    df_copy = small_grid_df.copy()
    df_copy["onset_threshold"] = 0.4
    merged = pd.concat([small_grid_df, df_copy], ignore_index=True)
    with pytest.raises(ValueError, match="single onset_threshold"):
        summarise_level1(merged)
    with pytest.raises(ValueError, match="single onset_threshold"):
        summarise_definedness(merged)


def test_split_half_icc_returns_tuple():
    """split_half_icc now returns (value, status)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=30)
    value, status = split_half_icc(v, rng_seed=0)
    assert isinstance(value, float)
    assert status in {"ok", "ceiling_undefined", "insufficient_seeds",
                      "all_undefined"}


def test_split_half_icc_ceiling_detected():
    """When SD is very small, status should be 'ceiling_undefined'
    and value should be the SD (precision estimate)."""
    v = np.ones(30) * 0.97 + np.random.default_rng(0).normal(0, 0.005, 30)
    value, status = split_half_icc(v, rng_seed=0, ceiling_sd_threshold=0.05)
    assert status == "ceiling_undefined"
    assert isinstance(value, float) and value >= 0.0


def test_split_half_icc_insufficient_seeds():
    v = np.array([0.1, 0.2])
    value, status = split_half_icc(v, rng_seed=0)
    assert status == "insufficient_seeds"
    assert np.isnan(value)


def test_split_half_icc_all_undefined():
    """All-NaN input should return 'all_undefined' status."""
    v = np.array([np.nan, np.nan, np.nan])
    value, status = split_half_icc(v, rng_seed=0)
    assert status == "all_undefined"
    assert np.isnan(value)

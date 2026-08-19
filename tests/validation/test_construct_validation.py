"""Validation contracts for peak duration bias and rival explanations."""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.validation.discriminant import (
    SCENARIO_METADATA,
    generate_discriminant_pair,
    run_discriminant_benchmark,
)
from syncpipe.validation.peak_duration import (
    duration_dependability_curve,
    moving_block_peak_stability,
    simulate_peak_duration_bias,
)

pytestmark = [pytest.mark.slow]


def test_block_stability_is_deterministic_and_honestly_labelled():
    x = 0.2 + 0.3 * np.sin(np.linspace(0, 20, 240))
    a = moving_block_peak_stability(x, block_length=30, n_resamples=40, seed=3)
    b = moving_block_peak_stability(x, block_length=30, n_resamples=40, seed=3)
    assert np.array_equal(a["resampled_values"], b["resampled_values"])
    assert a["interval_label"] == "block_resampling_stability_not_population_CI"
    assert a["stability_interval"][0] <= a["stability_interval"][1]


def test_block_stability_rejects_discontinuous_trace():
    x = np.linspace(0, 1, 100)
    x[50] = np.nan
    with pytest.raises(ValueError, match="finite contiguous"):
        moving_block_peak_stability(x, block_length=20, n_resamples=10)


def test_duration_dependability_uses_contiguous_complete_blocks():
    traces = {
        "d1": np.r_[np.full(60, 0.2), np.full(60, 0.8)],
        "d2": np.r_[np.full(60, 0.3), np.full(60, 0.7)],
        "d3": np.r_[np.full(60, 0.4), np.full(60, 0.6)],
    }
    blocks, summary = duration_dependability_curve(traces, durations=(30, 60))
    assert set(blocks["duration_points"]) == {30, 60}
    assert len(blocks[blocks["duration_points"] == 60]) == 6
    row = summary.set_index("duration_points").loc[60]
    assert row["n_dyads_with_repeats"] == 3
    assert row["median_abs_first_last_diff"] > 0


def test_independent_null_peak_increases_with_duration():
    _, summary = simulate_peak_duration_bias(
        durations=(120, 300, 600), n_replicates=80,
        window_size=30, phi=0.9, seed=1,
    )
    means = summary.sort_values("duration_samples")["mean_null_peak"].to_numpy()
    assert np.all(np.diff(means) > 0), means


@pytest.mark.parametrize("scenario", tuple(SCENARIO_METADATA))
def test_discriminant_generators_are_finite(scenario):
    a, b = generate_discriminant_pair(scenario, n_samples=300, seed=7)
    assert a.shape == b.shape == (300,)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.std(a) == pytest.approx(1.0)
    assert np.std(b) == pytest.approx(1.0)


def test_shared_input_is_distinct_from_independent_control():
    independent = []
    shared = []
    for seed in range(12):
        a, b = generate_discriminant_pair("independent_ar1", n_samples=400, seed=seed)
        independent.append(np.corrcoef(a, b)[0, 1])
        a, b = generate_discriminant_pair("shared_stimulus", n_samples=400, seed=seed)
        shared.append(np.corrcoef(a, b)[0, 1])
    assert np.median(shared) > np.median(independent) + 0.4


def test_discriminant_benchmark_reports_l0_and_design_controls():
    values, summary, controls = run_discriminant_benchmark(
        scenarios=("independent_ar1", "shared_stimulus", "reciprocal_var"),
        n_replicates=6,
        n_samples=300,
        window_size=20,
        surrogate_n=19,
        shift_lags_sec=(-60.0, 60.0),
        seed=2,
    )
    assert len(values) == 18
    assert set(summary["scenario"]) == {
        "independent_ar1", "shared_stimulus", "reciprocal_var"
    }
    assert "construct_false_positive_rate" in summary
    assert controls.set_index("scenario").loc[
        "reciprocal_var", "real_minus_pseudo_mean"
    ] > 0

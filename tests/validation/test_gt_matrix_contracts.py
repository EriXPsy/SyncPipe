"""Phase-1 ground-truth matrix contracts.

These tests check generator truth and status semantics, not scientific validity.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncpipe.validation.gt_matrix import (
    GT_SCENARIOS,
    make_gt_scenario,
    validate_gt_contract,
)


@pytest.mark.parametrize("name", tuple(GT_SCENARIOS))
def test_matrix_scenarios_have_structured_truth_and_finite_shape(name):
    scenario = make_gt_scenario(name, n_samples=240, seed=7)
    assert scenario.x_A.shape == scenario.x_B.shape == (240,)
    assert scenario.name == name
    assert scenario.family
    assert scenario.truth
    assert scenario.intended_claim.startswith("synthetic")
    finite = np.isfinite(scenario.x_A) & np.isfinite(scenario.x_B)
    if name != "missingness":
        assert finite.all()


def test_matrix_covers_null_coupling_rivals_missingness_and_episodes():
    assert set(GT_SCENARIOS) == {
        "null", "zero_lag_linear", "lead_lag", "shared_driver_slow_drift",
        "missingness", "nonstationary_episode", "independent_autocorr_null",
        "reciprocal_var", "shared_driver_no_direct_coupling",
    }
    assert make_gt_scenario("null", n_samples=240).truth["coupling"] == 0.0
    assert make_gt_scenario("lead_lag", n_samples=240).truth["direction"] == "A_leads_B"
    assert make_gt_scenario("missingness", n_samples=240).truth["missing_fraction"] == 0.15


def test_contract_separates_type_i_power_definedness_and_claimability():
    null_result = validate_gt_contract(make_gt_scenario("null", n_samples=240))
    positive_result = validate_gt_contract(make_gt_scenario("zero_lag_linear", n_samples=240))
    missing_result = validate_gt_contract(make_gt_scenario("missingness", n_samples=240))
    assert null_result.type_i_calibration == "required"
    assert null_result.power == "not_applicable"
    assert positive_result.power == "required"
    assert positive_result.type_i_calibration == "not_applicable"
    assert missing_result.definedness == "not_tested"
    assert missing_result.numerical_correctness == "not_tested"
    assert validate_gt_contract(make_gt_scenario("null"), observed_defined=True).definedness == "pass"
    assert {null_result.claimability, positive_result.claimability} == {"not_claimable"}


def test_contract_does_not_claim_all_undefined_data():
    scenario = make_gt_scenario("missingness", n_samples=240)
    result = validate_gt_contract(scenario, observed_defined=False)
    assert result.definedness == "undefined"
    assert result.claimability == "not_claimable"


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown GT scenario"):
        make_gt_scenario("real_data", n_samples=240)


def test_generator_routing_truth_and_seed_match_discriminant():
    from syncpipe.validation.discriminant import generate_discriminant_pair

    routes = {
        "null": "independent_ar1",
        "independent_autocorr_null": "independent_ar1",
        "reciprocal_var": "reciprocal_var",
        "shared_driver_slow_drift": "common_drift",
        "shared_driver_no_direct_coupling": "shared_stimulus",
    }
    for name, route in routes.items():
        scenario = make_gt_scenario(name, seed=7)
        expected = generate_discriminant_pair(route, n_samples=240, seed=7)
        np.testing.assert_array_equal(scenario.x_A, expected[0])
        np.testing.assert_array_equal(scenario.x_B, expected[1])
        assert scenario.truth["direct_coupling"] == (name == "reciprocal_var")
        assert scenario.truth["independence_null"] == (route == "independent_ar1")
        if name == "reciprocal_var":
            assert "ar_phi" not in scenario.truth
            assert scenario.truth["own_coefficient"] == 0.62
            assert scenario.truth["cross_coefficient"] == 0.22
        else:
            assert scenario.truth["ar_phi"] == 0.9
    a = make_gt_scenario("null", seed=7)
    b = make_gt_scenario("independent_autocorr_null", seed=7)
    np.testing.assert_array_equal(a.x_A, b.x_A)
    np.testing.assert_array_equal(a.x_B, b.x_B)


def test_reciprocal_truth_matches_independent_recurrence_reference():
    scenario = make_gt_scenario("reciprocal_var", seed=7)
    rng = np.random.default_rng(7)
    n = 240
    # The discriminant generator consumes two AR noise streams first.
    for _ in range(2):
        rng.normal()
        rng.normal(size=n)
    a, b = np.empty(n), np.empty(n)
    a[0], b[0] = rng.normal(size=2)
    eps_a, eps_b = rng.normal(size=n), rng.normal(size=n)
    own = scenario.truth["own_coefficient"]
    cross = scenario.truth["cross_coefficient"]
    for i in range(1, n):
        a[i] = own * a[i - 1] + cross * b[i - 1] + eps_a[i]
        b[i] = own * b[i - 1] + cross * a[i - 1] + eps_b[i]
    np.testing.assert_allclose(scenario.x_A, (a - a.mean()) / a.std())
    np.testing.assert_allclose(scenario.x_B, (b - b.mean()) / b.std())


def test_runner_all_cells_numeric_reference_and_seed_consistency(monkeypatch):
    import syncpipe.validation.gt_matrix as matrix
    from pandas.testing import assert_frame_equal

    calls = []

    def audit(a, b, **kwargs):
        calls.append(kwargs)
        return {"status": "ok", "p_values": {"peak_amplitude": 0.01}}

    monkeypatch.setattr(matrix, "synchrony_existence_audit", audit)
    rows, summary = matrix.run_gt_matrix(n_replicates=2, surrogate_n=20, seed=7)
    expected_cells = {(name, 240, 30) for name in GT_SCENARIOS}
    expected_cells |= {(name, n, w) for name in ("null", "reciprocal_var")
                       for n, w in ((240, 60), (480, 30))}
    assert set(zip(summary.scenario, summary.n_samples, summary.window_size)) == expected_cells
    assert len(rows) == len(calls) == 26
    assert (rows.numerical_correctness == "pass").all()
    assert (summary.n_effective == 2).all()
    assert (summary.rate == 1).all()
    assert (summary.claimability == "not_claimable").all()
    assert (summary.statistical_calibration == "not_established").all()
    assert set(rows.loc[rows.scenario.isin(["null", "independent_autocorr_null"]), "rate_kind"]) == {"statistical_fpr"}
    assert set(rows.loc[rows.scenario.str.startswith("shared_driver"), "rate_kind"]) == {"wrong_interpersonal_interpretation_risk"}
    for index, name in enumerate(GT_SCENARIOS):
        selected = rows[rows.scenario == name]
        assert set(selected.seed) == {7 + index * 100_000, 8 + index * 100_000}
        assert (selected.audit_seed == selected.seed + 10_000_000).all()
    for row, call in zip(rows.itertuples(), calls):
        assert call["seed"] == row.audit_seed
        assert call["window_size"] == row.window_size
        assert call["surrogate_n"] == 20
    repeat, repeat_summary = matrix.run_gt_matrix(n_replicates=2, surrogate_n=20, seed=7)
    assert_frame_equal(rows, repeat)
    assert_frame_equal(summary, repeat_summary)
    for name in GT_SCENARIOS:
        first, second = make_gt_scenario(name, seed=7), make_gt_scenario(name, seed=7)
        np.testing.assert_array_equal(first.x_A, second.x_A)
        np.testing.assert_array_equal(first.x_B, second.x_B)


def test_runner_undefined_denominator_and_numeric_failure(monkeypatch):
    import syncpipe.validation.gt_matrix as matrix

    monkeypatch.setattr(matrix, "synchrony_existence_audit", lambda *args, **kwargs: {
        "status": "unmeasurable", "reason": "test_undefined", "p_values": {"peak_amplitude": np.nan}})
    original = matrix.sliding_window_wcc
    monkeypatch.setattr(matrix, "sliding_window_wcc", lambda *args, **kwargs: original(*args, **kwargs) + 0.1)
    rows, summary = matrix.run_gt_matrix(n_replicates=1, surrogate_n=20)
    assert (rows.numerical_correctness == "fail").all()
    assert not rows.effective.any()
    assert rows.p_peak_amplitude.isna().all()
    assert rows.rejected.isna().all()
    assert (summary.n_effective == 0).all()
    assert (summary.n_unmeasurable == 1).all()
    assert summary[["rate", "wilson95_lower", "wilson95_upper"]].isna().all().all()
    assert (summary.statistical_calibration == "unmeasurable").all()


def test_wilson_reference_zero_denominator_and_invalid_counts():
    from syncpipe.validation.gt_matrix import wilson_interval

    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(0, 3) == pytest.approx((0, 0.5614970317550454))
    assert wilson_interval(1, 3) == pytest.approx((0.06149194472039621, 0.7923403991979523))
    assert wilson_interval(3, 3) == pytest.approx((0.4385029682449546, 1))
    for successes, trials in ((-1, 3), (4, 3), (0, -1)):
        with pytest.raises(ValueError):
            wilson_interval(successes, trials)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syncpipe.feature_definitions import REFERENCE_FEATURE
from syncpipe.inference_pipeline import InferencePipeline, _apply_global_modality_fdr
from syncpipe.null_models import _wcc_level_surrogate_test
from syncpipe.validation.l2_between_condition import between_condition_fdr


ROOT = Path(__file__).resolve().parents[2]


def test_v1_positioning_document_declares_narrow_scope():
    text = (ROOT / "docs" / "V1_POSITIONING_AND_ROADMAP.md").read_text(encoding="utf-8")
    assert "窄而深" in text
    assert "WCC 是 estimator，不是 construct" in text
    assert "L0/L1 不得事后筛选 L2" in text


def test_risk_register_covers_priority_levels():
    text = (ROOT / "docs" / "RISK_REGISTER.md").read_text(encoding="utf-8")
    for level in ("P0", "P1", "P2"):
        assert f"### {level}-" in text


def test_unknown_confirmatory_feature_cannot_escape_fdr_family():
    class Result:
        feature = "unregistered_feature"
        claimable = True
        p_raw = 0.01
        p_fdr = 0.01
        significant_05 = True
        observed_diff = null_mean = null_sd = perm_effect_size = 0.0
        difference_q25 = difference_q75 = median_ci_low = median_ci_high = 0.0
        median_ci_confidence = 0.95
        median_ci_bounded = False
        median_ci_method = permutation_method = "test"
        n_null_draws = min_attainable_p = approx_monte_carlo_se = 0.0
        n_dyads = defined_a = defined_b = 0
        p_definedness = 1.0
        definedness_status = "complete"

    with pytest.raises(ValueError, match="no registered FDR family"):
        _apply_global_modality_fdr(
            {"ECG": {"per_feature": [Result()]}}, alpha=0.05
        )


def _paired_table(modalities=("ECG",)):
    return pd.DataFrame([
        {"dyad_id": f"d{i}", "condition": condition, "modality": modality,
         "peak_amplitude": i / 100 + shift * (1 + i / 10)}
        for modality in modalities for i in range(10)
        for condition, shift in (("A", 0.7), ("B", 0.1))
    ])


@pytest.mark.parametrize("unknowns", [("custom",), ("custom", "another")])
@pytest.mark.parametrize("scope", ["global", "within_modality"])
def test_exploratory_features_run_without_confirmatory_singletons(unknowns, scope):
    df = _paired_table(("ECG", "EDA"))
    features = ["peak_amplitude", *unknowns, REFERENCE_FEATURE[0]]
    for feature in features[1:]:
        df[feature] = df["peak_amplitude"]
    results = InferencePipeline(df).test_l2_by_modality(
        feature_cols=features, contrast=("A", "B"), fdr_scope=scope,
    )
    for payload in results.values():
        assert payload["n_significant"] == 1
        for result in payload["per_feature"]:
            if result.feature == "peak_amplitude":
                assert np.isfinite(result.p_fdr)
            else:
                assert not result.claimable
                assert not result.significant_05
                assert np.isnan(result.p_fdr)
                assert result.p_raw < 0.05
        if scope == "global":
            assert payload["fdr_family_size"] == {"L0": 2}
        else:
            assert payload["fdr_family_size"] == 2 + len(unknowns)


@pytest.mark.parametrize("policy", ["flag", "gate"])
def test_all_undefined_is_not_claimable_even_in_flag_mode(policy):
    df = _paired_table()
    df["peak_amplitude"] = np.nan
    payload = between_condition_fdr(
        df, dyad_col="dyad_id", condition_values=("A", "B"),
        undefined_policy=policy,
    )
    result = payload["per_feature"][0]
    assert result.n_dyads == 0
    assert result.permutation_method == "not_run"
    assert not result.claimable
    assert not result.significant_05
    assert payload["n_significant"] == 0


@pytest.mark.parametrize("case", ["unpaired", "single_condition"])
def test_nonestimable_design_fails_without_publishing_l2(case):
    df = _paired_table()
    if case == "unpaired":
        df.loc[df.condition == "B", "dyad_id"] += "_other"
    else:
        df = df[df.condition == "A"]
    pipe = InferencePipeline(df)
    with pytest.raises(ValueError, match="Only 0 dyads|Condition 'B' not found"):
        pipe.run_group_condition_inference(contrast=("A", "B"))
    assert pipe._group_inference_results is None
    assert pipe._l2_results is None


@pytest.mark.parametrize("status", ["supported", "not_supported", "untestable"])
def test_l0_l1_status_never_selects_l2_dyads(status):
    df = _paired_table()
    pipe = InferencePipeline(df)
    baseline = pipe.run_group_condition_inference(contrast=("A", "B"))["ECG"]
    audits = {f"d{i}": {"status": status, "per_feature_significant": {
        "peak_amplitude": status == "supported"}} for i in range(10)}
    pipe._l0_results = audits
    pipe._l1_results = audits
    pipe._synchrony_existence_results = audits
    actual = pipe.run_group_condition_inference(contrast=("A", "B"))["ECG"]
    assert actual["n_dyads"] == 10
    assert actual["per_feature"][0].n_dyads == 10
    pd.testing.assert_frame_equal(actual["summary_df"], baseline["summary_df"])
    pd.testing.assert_frame_equal(pipe.df, df)


def test_l1_untestable_status_is_distinct_from_non_significance():
    short_trace = np.zeros(10)
    result = _wcc_level_surrogate_test(short_trace, hz=1.0, surrogate_n=20, seed=1)
    assert result["applicable"] is False
    assert result["notes"]
    assert result["p_dwell_time"] == 1.0
    assert result["n_surrogates"] == 0


def test_l1_summary_reports_untestable_traces_separately():
    pipe = InferencePipeline(_paired_table())
    pipe._l1_results = {
        "short": {"applicable": False, "notes": "too short"},
        "valid": {
            "applicable": True,
            "per_feature_significant": {"switching_rate": False},
        },
    }
    text = pipe.summarize()
    assert "Not applicable: 1 trace(s) excluded from denominator" in text


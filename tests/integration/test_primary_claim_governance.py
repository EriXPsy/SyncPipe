"""Regression tests for primary-modality L2 claim governance."""
from types import SimpleNamespace

import pytest

from syncpipe.evidence import build_evidence_chain


def _group(significant):
    return {"per_feature": [SimpleNamespace(
        feature="peak_amplitude", claimable=True,
        significant_05=significant, condition_a="rest", condition_b="task",
    )]}


@pytest.mark.parametrize("reverse", [False, True])
def test_comparator_cannot_promote_primary_l2_claim(reverse):
    items = [("EDA", _group(False)), ("ECG", _group(True))]
    graph = build_evidence_chain(
        endpoint="peak_amplitude",
        existence_gate={"primary_pass": False, "primary_modalities": ["EDA"]},
        design=None, across_stimulus=None,
        group=dict(reversed(items) if reverse else items), alpha=.05,
    )
    assert not graph.decision.claimable_condition_difference
    assert graph.stage("L2").status.value == "not_supported"


def test_missing_primary_modality_cannot_borrow_comparator_result():
    graph = build_evidence_chain(
        endpoint="peak_amplitude",
        existence_gate={"primary_pass": False, "primary_modalities": ["EDA"]},
        design=None, across_stimulus=None, group={"ECG": _group(True)}, alpha=.05,
    )
    assert not graph.decision.claimable_condition_difference
    assert graph.stage("L2").status.value == "inconclusive"


def test_primary_l2_claim_does_not_require_l0_selection():
    graph = build_evidence_chain(
        endpoint="peak_amplitude",
        existence_gate={"primary_pass": False, "primary_modalities": ["EDA"]},
        design=None, across_stimulus=None, group={"EDA": _group(True)}, alpha=.05,
    )
    assert graph.decision.claimable_condition_difference
    assert graph.stage("L2").statistics["supporting_contrasts"] == [
        {"modality": "EDA", "condition_a": "rest", "condition_b": "task"}
    ]


def test_unregistered_primary_modalities_are_inconclusive():
    graph = build_evidence_chain(
        endpoint="peak_amplitude", existence_gate={"primary_pass": False},
        design=None, across_stimulus=None, group={"EDA": _group(True)}, alpha=.05,
    )
    assert not graph.decision.claimable_condition_difference
    assert graph.stage("L2").status.value == "inconclusive"

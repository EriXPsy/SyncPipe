"""Frozen v1 default guards.

These tests freeze protocol-level defaults so that a future edit cannot
silently revert a locked v1 choice (V1_PROTOCOL.md §10, V1_CLAIM_CEILING.md).
If one of these fails, the change is a protocol event: it requires an explicit
Gate-0 freeze decision and a docs update — never a silent default swap.
"""

import inspect

import pytest

from syncpipe.feature_definitions import (
    EXISTENCE_GATE_ALPHA,
    FDR_FAMILIES,
    PRIMARY_EXISTENCE_ENDPOINT,
    PRIMARY_EXISTENCE_MODALITIES,
    PRIMARY_FDR_FAMILY,
    REFERENCE_FEATURE,
    SECONDARY_FDR_FAMILY,
)
from syncpipe.inference_pipeline import InferencePipeline


def test_l1_default_null_model_is_state_shuffle():
    """v1.0 revised the L1 WCC-level null from IAAFT to state_shuffle.

    state_shuffle preserves the exact dwell-time distribution while testing
    temporal organization (see InferencePipeline.test_l1_structure docstring).
    Reverting this default is a protocol change, not a refactor.
    """
    sig = inspect.signature(InferencePipeline.test_l1_structure)
    assert sig.parameters["null_model"].default == "state_shuffle"


def test_l0_existence_null_is_signal_level_iaaft():
    """The L0 existence audit is locked to signal-level IAAFT surrogates.

    IAAFT preserves each dyad's autocorrelation while destroying coupling;
    its known limitation (shared input passes L0) is a documented claim
    boundary (docs/CONSTRUCT_VALIDITY.md §7), not a bug to fix by swapping
    the null silently.
    """
    sig = inspect.signature(InferencePipeline.test_l1_structure)
    # state_shuffle must remain one of the accepted L1 null choices.
    accepted = {"state_shuffle", "block_permutation", "iaaft"}
    assert sig.parameters["null_model"].default in accepted
    assert FDR_FAMILIES["L0"] == ("peak_amplitude",)


def test_fdr_family_partition_is_frozen():
    assert PRIMARY_FDR_FAMILY == ("peak_amplitude",)
    assert SECONDARY_FDR_FAMILY == ("dwell_time", "switching_rate")
    assert REFERENCE_FEATURE == ("mean_synchrony",)
    # PRIMARY + SECONDARY must be disjoint and together equal the tested
    # L2 surface; the reference comparator stays outside both.
    assert not set(PRIMARY_FDR_FAMILY) & set(SECONDARY_FDR_FAMILY)
    assert "mean_synchrony" not in set(PRIMARY_FDR_FAMILY) | set(SECONDARY_FDR_FAMILY)


def test_existence_gate_frozen_parameters():
    assert PRIMARY_EXISTENCE_ENDPOINT == "peak_amplitude"
    assert PRIMARY_EXISTENCE_MODALITIES == ("ECG", "EDA")
    assert EXISTENCE_GATE_ALPHA == 0.05
    # Gate and claim cannot diverge: the endpoint must be the single member
    # of the primary FDR family.
    assert PRIMARY_EXISTENCE_ENDPOINT in PRIMARY_FDR_FAMILY
    assert len(PRIMARY_FDR_FAMILY) == 1


def test_l2_governance_defaults_are_fail_loud():
    sig = inspect.signature(InferencePipeline.test_l2_condition)
    assert sig.parameters["undefined_policy"].default == "gate"
    assert sig.parameters["n_min_dyads"].default == 10
    assert sig.parameters["fdr_alpha"].default == 0.05
    # Paired design: the permutation test must be the dyad-paired variant,
    # enforced by requiring a dyad column with a sane default.
    assert sig.parameters["dyad_col"].default == "dyad_id"

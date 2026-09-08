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


def test_l1_default_null_model_is_iaaft():
    """V1_PROTOCOL §10 locks the L1 WCC-level null to IAAFT on the WCC
    trace, and the SSoT null-design table (dynamic_features.py) maps the
    L1 features (dwell_time, switching_rate) to WCC-level IAAFT.

    BUG-4 regression guard: a v1.0 revision switched the pipeline default
    to 'state_shuffle' on the rationale that it "preserves the dwell-time
    distribution" — but state_shuffle re-orders whole elevated/baseline
    segments, making BOTH L1 statistics invariant by construction
    (p == 1.0 on every input).  The default was restored to the protocol
    null on 2026-09-08.
    """
    sig = inspect.signature(InferencePipeline.test_l1_structure)
    assert sig.parameters["null_model"].default == "iaaft"
    # state_shuffle remains selectable (with a degeneracy warning) for
    # order-sensitive diagnostics, but must never be the default again.
    accepted = {"state_shuffle", "block_permutation", "iaaft"}
    assert sig.parameters["null_model"].default in accepted
    assert FDR_FAMILIES["L1"] == ("dwell_time", "switching_rate")


def test_l0_existence_null_is_signal_level_iaaft():
    """The L0 existence audit is locked to signal-level IAAFT surrogates.

    IAAFT preserves each dyad's autocorrelation while destroying coupling;
    its known limitation (shared input passes L0) is a documented claim
    boundary (docs/CONSTRUCT_VALIDITY.md §7), not a bug to fix by swapping
    the null silently.
    """
    assert FDR_FAMILIES["L0"] == ("peak_amplitude",)
    # BUG-3 regression guard: the existence audit must derive per-pair
    # seeds (label + master seed), never reuse one master stream across
    # dyads — a shared stream correlated surrogate draws across dyads
    # (r ≈ +0.33) and inflated the group-null spread ~1.46x.
    from syncpipe.inference_pipeline import _pair_seed

    s1 = _pair_seed(42, "dyad_001__ECG")
    s2 = _pair_seed(42, "dyad_002__ECG")
    assert s1 != s2
    assert _pair_seed(42, "dyad_001__ECG") == s1  # deterministic
    assert _pair_seed(7, "dyad_001__ECG") != s1   # master seed matters


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

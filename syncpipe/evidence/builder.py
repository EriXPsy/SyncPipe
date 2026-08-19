"""Build the E0-E5 evidence graph from pipeline stage outputs."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..__about__ import EVIDENCE_SCHEMA_VERSION
from .models import (
    ClaimDecision, EvidenceChain, EvidenceProfile, EvidenceStageResult,
    EvidenceStatus,
)


def _design_p(design: Optional[Dict[str, Any]], endpoint: str, key: str) -> float:
    if not isinstance(design, dict):
        return float("nan")
    return float(
        design.get("feature_summary", {}).get(endpoint, {}).get(key, np.nan)
    )


def _across_p(across: Optional[Dict[str, Any]], endpoint: str) -> float:
    if not isinstance(across, dict):
        return float("nan")
    return float(across.get("results", {}).get(endpoint, {}).get("p_value", np.nan))


def _holm_two(p_a: float, p_b: float) -> Tuple[float, float]:
    """Holm adjustment for the two generic design-control hypotheses."""
    values = np.asarray([p_a, p_b], dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return tuple(float(x) for x in values)
    order = np.argsort(values)
    adjusted = np.empty(2, dtype=float)
    adjusted[order[0]] = min(1.0, 2.0 * values[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]], values[order[1]])
    return float(adjusted[0]), float(adjusted[1])


def _design_resolution(design: Optional[Dict[str, Any]], endpoint: str) -> float:
    if not isinstance(design, dict):
        return float("nan")
    n = design.get("feature_summary", {}).get(endpoint, {}).get("n_real")
    if n is None or int(n) < 1:
        return float("nan")
    n = int(n)
    return float(1.0 / (2 ** n)) if n <= 12 else float(1.0 / 20001.0)


def _group_endpoint_status(
    group: Optional[Dict[str, Any]], endpoint: str
) -> Tuple[EvidenceStatus, str]:
    if not isinstance(group, dict):
        return EvidenceStatus.INCONCLUSIVE, "group inference not run"
    found = False
    claimable = False
    for payload in group.values():
        if not isinstance(payload, dict):
            continue
        for result in payload.get("per_feature", ()):
            if getattr(result, "feature", None) != endpoint:
                continue
            found = True
            if bool(getattr(result, "claimable", False)):
                claimable = True
                if bool(getattr(result, "significant_05", False)):
                    return EvidenceStatus.SUPPORTED, "claimable endpoint contrast passed FDR"
    if claimable:
        return EvidenceStatus.NOT_SUPPORTED, "claimable endpoint contrast did not pass FDR"
    if found:
        return EvidenceStatus.INCONCLUSIVE, "endpoint result was not claimable"
    return EvidenceStatus.INVALID, "declared endpoint absent from group inference"


def build_evidence_chain(
    *,
    endpoint: str,
    existence_gate: Dict[str, Any],
    design: Optional[Dict[str, Any]],
    across_stimulus: Optional[Dict[str, Any]],
    group: Optional[Dict[str, Any]],
    alpha: float,
) -> EvidenceChain:
    """Create typed stages and propagate the strongest defensible claim."""
    e0_supported = bool(existence_gate.get("primary_pass", False))
    primary_results = [
        item for item in existence_gate.get("per_modality", {}).values()
        if isinstance(item, dict) and item.get("is_primary")
    ]
    e0_testable = any(
        np.isfinite(item.get("min_attainable_two_sided_p", np.nan))
        and item["min_attainable_two_sided_p"] < alpha
        for item in primary_results
    )
    if e0_supported:
        e0_status = EvidenceStatus.SUPPORTED
        e0_reason = "second-order group surrogate gate passed"
    elif not primary_results or not e0_testable:
        e0_status = EvidenceStatus.INCONCLUSIVE
        e0_reason = "primary existence test absent or lacks p-value resolution"
    else:
        e0_status = EvidenceStatus.NOT_SUPPORTED
        e0_reason = "testable second-order group surrogate gate did not pass"
    e0 = EvidenceStageResult(
        stage_id="E0", name="co_fluctuation_existence",
        question="Does the endpoint exceed independent autocorrelated signals?",
        status=e0_status,
        permitted_claim=(
            "synchrony-like co-fluctuation above the declared independent-signal null"
            if e0_supported else "descriptive co-fluctuation only"
        ),
        rival_addressed="independent autocorrelated dynamics",
        unresolved_rivals=("shared input", "partner mismatch", "slow drift", "co-presence"),
        reason=e0_reason,
        statistics={
            "primary_pass": e0_supported, "alpha": float(alpha),
            "testable_at_alpha": e0_testable,
        },
    )

    e1 = EvidenceStageResult(
        stage_id="E1", name="temporal_structure",
        question="Does the WCC trace contain structure beyond its local-state null?",
        status=EvidenceStatus.INCONCLUSIVE,
        permitted_claim="no E1 claim; the canonical chain does not yet aggregate L1",
        rival_addressed="unordered/local WCC states",
        unresolved_rivals=("temporal structure not evaluated at group level",),
        reason="group-level L1 stage not implemented in the canonical chain",
    )

    p_pseudo_raw = _design_p(design, endpoint, "p_real_gt_pseudo")
    p_shift_raw = _design_p(design, endpoint, "p_real_gt_time_shift")
    p_pseudo, p_shift = _holm_two(p_pseudo_raw, p_shift_raw)
    design_resolution = _design_resolution(design, endpoint)
    design_testable = not np.isfinite(design_resolution) or design_resolution < alpha
    if design is None:
        e2_status = EvidenceStatus.INCONCLUSIVE
        e2_reason = "pseudo-pair control not run"
    elif not design_testable:
        e2_status = EvidenceStatus.INCONCLUSIVE
        e2_reason = f"minimum attainable p={design_resolution:.6g} is not below alpha"
    elif np.isfinite(p_pseudo):
        e2_status = EvidenceStatus.SUPPORTED if p_pseudo < alpha else EvidenceStatus.NOT_SUPPORTED
        e2_reason = f"Holm-adjusted p_real_gt_pseudo={p_pseudo:.6g}"
    else:
        e2_status = EvidenceStatus.INVALID
        e2_reason = "pseudo-pair p-value unavailable"
    e2 = EvidenceStageResult(
        stage_id="E2", name="partner_specificity",
        question="Do real partners exceed mismatched partners?",
        status=e2_status,
        permitted_claim=("partner-specific association" if e2_status is EvidenceStatus.SUPPORTED else "no partner-specificity claim"),
        rival_addressed="arbitrary partner pairing",
        unresolved_rivals=("dyad-specific common cause", "shared input", "co-presence"),
        reason=e2_reason,
        statistics={
            "p_raw": p_pseudo_raw, "p_holm": p_pseudo,
            "min_attainable_p": design_resolution,
        },
    )

    if design is None:
        e3_status = EvidenceStatus.INCONCLUSIVE
        e3_reason = "time-shift control not run"
    elif not design_testable:
        e3_status = EvidenceStatus.INCONCLUSIVE
        e3_reason = f"minimum attainable p={design_resolution:.6g} is not below alpha"
    elif np.isfinite(p_shift):
        e3_status = EvidenceStatus.SUPPORTED if p_shift < alpha else EvidenceStatus.NOT_SUPPORTED
        e3_reason = f"Holm-adjusted p_real_gt_time_shift={p_shift:.6g}"
    else:
        e3_status = EvidenceStatus.INVALID
        e3_reason = "time-shift p-value unavailable"
    e3 = EvidenceStageResult(
        stage_id="E3", name="alignment_specificity",
        question="Does the endpoint depend on original temporal alignment?",
        status=e3_status,
        permitted_claim=("alignment-specific association" if e3_status is EvidenceStatus.SUPPORTED else "no alignment-specificity claim"),
        rival_addressed="slow drift and coarse block timing",
        unresolved_rivals=("event-locked shared input", "simultaneous artifact", "co-presence"),
        reason=e3_reason,
        statistics={
            "p_raw": p_shift_raw, "p_holm": p_shift,
            "min_attainable_p": design_resolution,
        },
    )

    p_across = _across_p(across_stimulus, endpoint)
    across_n = (
        across_stimulus.get("results", {}).get(endpoint, {}).get("n_surr")
        if isinstance(across_stimulus, dict) else None
    )
    across_resolution = (
        float(min(1.0, 2.0 / (int(across_n) + 1)))
        if across_n is not None and int(across_n) > 0 else float("nan")
    )
    if across_stimulus is None:
        e4_status = EvidenceStatus.INCONCLUSIVE
        e4_reason = "shared-stimulus control not run"
    elif np.isfinite(across_resolution) and across_resolution >= alpha:
        e4_status = EvidenceStatus.INCONCLUSIVE
        e4_reason = f"minimum attainable two-sided p={across_resolution:.6g} is not below alpha"
    elif np.isfinite(p_across):
        e4_status = EvidenceStatus.SUPPORTED if p_across < alpha else EvidenceStatus.NOT_SUPPORTED
        e4_reason = f"p_across_stimulus={p_across:.6g}"
    else:
        e4_status = EvidenceStatus.INVALID
        e4_reason = "shared-stimulus p-value unavailable"
    e4 = EvidenceStageResult(
        stage_id="E4", name="shared_input_specificity",
        question="Does the endpoint exceed the measured shared-stimulus null?",
        status=e4_status,
        permitted_claim=("not fully explained by measured shared stimulus" if e4_status is EvidenceStatus.SUPPORTED else "no shared-input-specificity claim"),
        rival_addressed="measured common stimulus schedule",
        unresolved_rivals=("unmeasured common input", "co-presence", "simultaneous artifact"),
        reason=e4_reason,
        statistics={
            "p_across_stimulus": p_across,
            "min_attainable_two_sided_p": across_resolution,
        },
    )

    e5 = EvidenceStageResult(
        stage_id="E5", name="interaction_contingency",
        question="Does coordination require reciprocal interaction rather than co-presence?",
        status=EvidenceStatus.INCONCLUSIVE,
        permitted_claim="no interaction-contingency claim",
        rival_addressed="non-interactive co-presence and parallel responding",
        unresolved_rivals=("co-presence", "parallel response", "unmeasured common cause"),
        reason="live/replay, yoked, or equivalent reciprocity-breaking contrast not supplied",
    )

    l2_status, l2_reason = _group_endpoint_status(group, endpoint)
    l2_supported = l2_status is EvidenceStatus.SUPPORTED
    l2 = EvidenceStageResult(
        stage_id="L2", name="condition_difference",
        question="Does the declared endpoint differ across the pre-specified conditions?",
        status=l2_status,
        permitted_claim=("condition difference in the declared endpoint" if l2_supported else "no confirmatory condition-difference claim"),
        rival_addressed="within-dyad condition-label exchangeability",
        unresolved_rivals=("construct interpretation depends on E0-E5",),
        reason=l2_reason,
    )

    stages = (e0, e1, e2, e3, e4, e5, l2)
    profile = EvidenceProfile(
        supported=tuple(s.stage_id for s in stages if s.status is EvidenceStatus.SUPPORTED),
        not_supported=tuple(s.stage_id for s in stages if s.status is EvidenceStatus.NOT_SUPPORTED),
        inconclusive=tuple(s.stage_id for s in stages if s.status is EvidenceStatus.INCONCLUSIVE),
        not_applicable=tuple(s.stage_id for s in stages if s.status is EvidenceStatus.NOT_APPLICABLE),
        invalid=tuple(s.stage_id for s in stages if s.status is EvidenceStatus.INVALID),
    )
    progression = [e0, e2, e3, e4, e5]
    highest = "none"
    claim = "descriptive co-fluctuation only"
    ceiling_blocked = False
    for stage in progression:
        if not ceiling_blocked and stage.status is EvidenceStatus.SUPPORTED:
            highest = stage.stage_id
            claim = stage.permitted_claim
        else:
            ceiling_blocked = True
    blocked = [s.stage_id for s in progression if s.status is not EvidenceStatus.SUPPORTED]
    unresolved = [
        rival for stage in stages
        if stage.status is not EvidenceStatus.SUPPORTED
        for rival in stage.unresolved_rivals
    ]
    # E5 is never currently supported; always preserve the hard ceiling.
    forbidden = (
        "causal interpersonal coupling",
        "interaction contingency without a reciprocity-breaking control",
        "lead-lag direction",
        "universal synchrony biomarker",
    )
    decision = ClaimDecision(
        highest_supported_stage=highest,
        permitted_claim=claim,
        condition_claim=l2.permitted_claim,
        claimable_condition_difference=l2_supported,
        blocked_by=tuple(blocked),
        unresolved_rivals=tuple(dict.fromkeys(unresolved)),
        forbidden_claims=forbidden,
    )
    return EvidenceChain(
        version=EVIDENCE_SCHEMA_VERSION,
        endpoint=endpoint,
        stages=stages,
        profile=profile,
        decision=decision,
    )


__all__ = ["build_evidence_chain"]

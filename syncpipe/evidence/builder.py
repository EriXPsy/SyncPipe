"""Build the E0-E5 evidence graph from pipeline stage outputs."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..__about__ import EVIDENCE_SCHEMA_VERSION
from .models import ClaimDecision, EvidenceChain, EvidenceStageResult, EvidenceStatus


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


def _group_endpoint_supported(group: Optional[Dict[str, Any]], endpoint: str) -> bool:
    if not isinstance(group, dict):
        return False
    for payload in group.values():
        if not isinstance(payload, dict):
            continue
        for result in payload.get("per_feature", ()):
            if (
                getattr(result, "feature", None) == endpoint
                and bool(getattr(result, "significant_05", False))
                and bool(getattr(result, "claimable", False))
            ):
                return True
    return False


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
    e0 = EvidenceStageResult(
        stage_id="E0", name="co_fluctuation_existence",
        question="Does the endpoint exceed independent autocorrelated signals?",
        status=EvidenceStatus.SUPPORTED if e0_supported else EvidenceStatus.NOT_SUPPORTED,
        permitted_claim=(
            "synchrony-like co-fluctuation above the declared independent-signal null"
            if e0_supported else "descriptive co-fluctuation only"
        ),
        rival_addressed="independent autocorrelated dynamics",
        unresolved_rivals=("shared input", "partner mismatch", "slow drift", "co-presence"),
        reason="second-order group surrogate gate",
        statistics={"primary_pass": e0_supported, "alpha": float(alpha)},
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

    p_pseudo = _design_p(design, endpoint, "p_real_gt_pseudo")
    if design is None:
        e2_status = EvidenceStatus.INCONCLUSIVE
        e2_reason = "pseudo-pair control not run"
    elif np.isfinite(p_pseudo):
        e2_status = EvidenceStatus.SUPPORTED if p_pseudo < alpha else EvidenceStatus.NOT_SUPPORTED
        e2_reason = f"p_real_gt_pseudo={p_pseudo:.6g}"
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
        statistics={"p_real_gt_pseudo": p_pseudo},
    )

    p_shift = _design_p(design, endpoint, "p_real_gt_time_shift")
    if design is None:
        e3_status = EvidenceStatus.INCONCLUSIVE
        e3_reason = "time-shift control not run"
    elif np.isfinite(p_shift):
        e3_status = EvidenceStatus.SUPPORTED if p_shift < alpha else EvidenceStatus.NOT_SUPPORTED
        e3_reason = f"p_real_gt_time_shift={p_shift:.6g}"
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
        statistics={"p_real_gt_time_shift": p_shift},
    )

    p_across = _across_p(across_stimulus, endpoint)
    if across_stimulus is None:
        e4_status = EvidenceStatus.INCONCLUSIVE
        e4_reason = "shared-stimulus control not run"
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
        statistics={"p_across_stimulus": p_across},
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

    l2_supported = _group_endpoint_supported(group, endpoint)
    l2 = EvidenceStageResult(
        stage_id="L2", name="condition_difference",
        question="Does the declared endpoint differ across the pre-specified conditions?",
        status=EvidenceStatus.SUPPORTED if l2_supported else EvidenceStatus.NOT_SUPPORTED,
        permitted_claim=("condition difference in the declared endpoint" if l2_supported else "no confirmatory condition-difference claim"),
        rival_addressed="within-dyad condition-label exchangeability",
        unresolved_rivals=("construct interpretation depends on E0-E5",),
        reason="dyad-paired permutation with claimability and multiplicity gates",
    )

    stages = (e0, e1, e2, e3, e4, e5, l2)
    progression = [e0, e2, e3, e4, e5]
    highest = "none"
    claim = "descriptive co-fluctuation only"
    blocked = []
    unresolved = []
    for stage in progression:
        if stage.status is EvidenceStatus.SUPPORTED:
            highest = stage.stage_id
            claim = stage.permitted_claim
        else:
            blocked.append(stage.stage_id)
            unresolved.extend(stage.unresolved_rivals)
            # Claims are cumulative: do not jump over an unsupported lower stage.
            break
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
        blocked_by=tuple(blocked),
        unresolved_rivals=tuple(dict.fromkeys(unresolved)),
        forbidden_claims=forbidden,
    )
    return EvidenceChain(
        version=EVIDENCE_SCHEMA_VERSION,
        endpoint=endpoint,
        stages=stages,
        decision=decision,
    )


__all__ = ["build_evidence_chain"]

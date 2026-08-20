"""Internal result objects used to build the plain-language report."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"
    INVALID = "invalid"


@dataclass(frozen=True)
class EvidenceStageResult:
    """Result of one question, such as partner or timing specificity."""

    stage_id: str
    name: str
    question: str
    status: EvidenceStatus
    permitted_claim: str
    rival_addressed: str
    unresolved_rivals: Tuple[str, ...] = ()
    reason: str = ""
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "question": self.question,
            "status": self.status.value,
            "permitted_claim": self.permitted_claim,
            "rival_addressed": self.rival_addressed,
            "unresolved_rivals": list(self.unresolved_rivals),
            "reason": self.reason,
            "statistics": dict(self.statistics),
        }


@dataclass(frozen=True)
class EvidenceProfile:
    """Lists which checks passed, failed, lacked information, or were invalid."""

    supported: Tuple[str, ...]
    not_supported: Tuple[str, ...]
    inconclusive: Tuple[str, ...]
    not_applicable: Tuple[str, ...]
    invalid: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": list(self.supported),
            "not_supported": list(self.not_supported),
            "inconclusive": list(self.inconclusive),
            "not_applicable": list(self.not_applicable),
            "invalid": list(self.invalid),
        }


@dataclass(frozen=True)
class ClaimDecision:
    """Short conclusion plus explanations that remain possible."""

    highest_supported_stage: str
    permitted_claim: str
    condition_claim: str
    claimable_condition_difference: bool
    blocked_by: Tuple[str, ...]
    unresolved_rivals: Tuple[str, ...]
    forbidden_claims: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "highest_supported_stage": self.highest_supported_stage,
            "permitted_claim": self.permitted_claim,
            "condition_claim": self.condition_claim,
            "claimable_condition_difference": self.claimable_condition_difference,
            "blocked_by": list(self.blocked_by),
            "unresolved_rivals": list(self.unresolved_rivals),
            "forbidden_claims": list(self.forbidden_claims),
        }


@dataclass(frozen=True)
class EvidenceChain:
    """All checks and the conclusion used by REPORT.md and JSON output."""

    version: str
    endpoint: str
    stages: Tuple[EvidenceStageResult, ...]
    profile: EvidenceProfile
    decision: ClaimDecision

    def stage(self, stage_id: str) -> EvidenceStageResult:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(stage_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "endpoint": self.endpoint,
            "stages": [item.to_dict() for item in self.stages],
            "profile": self.profile.to_dict(),
            "decision": self.decision.to_dict(),
        }


__all__ = [
    "EvidenceStatus", "EvidenceStageResult", "EvidenceProfile",
    "ClaimDecision", "EvidenceChain",
]

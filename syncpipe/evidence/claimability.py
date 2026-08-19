"""Derive machine-readable claimability from evidence and L2 outputs."""
from __future__ import annotations
from typing import Any, Dict, List
from ..export.runtime import safe_float as _safe_float

def derive_claimability(chain: Dict[str, Any]) -> Dict[str, Any]:
    """Extract per-feature claimability from the L2 result of the chain.

    ``group_condition_inference`` is always the modality-keyed shape
    ``{modality: l2_dict}`` (1c), so each entry is walked and the modality is
    recorded on every row — a claim about ECG and a claim about EDA are distinct
    hypotheses and must not be collapsed into one anonymous feature list.
    """
    group = chain.get("group_condition_inference") or {}
    per_feature: List[Dict[str, Any]] = []

    def _collect(modality: str, l2: Any) -> None:
        if not isinstance(l2, dict):
            return
        pf = l2.get("per_feature")
        if not pf:
            return
        elig = l2.get("eligibility_status")
        for r in pf:
            feat = getattr(r, "feature", None)
            if feat is None:
                continue
            per_feature.append({
                "modality": modality,
                "feature": feat,
                "p_fdr": _safe_float(getattr(r, "p_fdr", None)),
                "significant_05": bool(getattr(r, "significant_05", False)),
                "claimable": getattr(r, "claimable", None),
                "definedness_status": getattr(r, "definedness_status", None),
                "eligibility_status": (elig.get(feat) if isinstance(elig, dict) else elig),
                "n_dyads": _safe_float(getattr(r, "n_dyads", None)),
                "observed_diff": _safe_float(getattr(r, "observed_diff", None)),
                "difference_q25": _safe_float(getattr(r, "difference_q25", None)),
                "difference_q75": _safe_float(getattr(r, "difference_q75", None)),
                "median_ci_low": _safe_float(getattr(r, "median_ci_low", None)),
                "median_ci_high": _safe_float(getattr(r, "median_ci_high", None)),
                "median_ci_bounded": bool(getattr(r, "median_ci_bounded", False)),
                "permutation_method": getattr(r, "permutation_method", None),
                "n_null_draws": int(getattr(r, "n_null_draws", 0)),
                "min_attainable_p": _safe_float(getattr(r, "min_attainable_p", None)),
                "approx_monte_carlo_se": _safe_float(
                    getattr(r, "approx_monte_carlo_se", None)
                ),
            })

    for mod in sorted(group.keys(), key=str):
        _collect(str(mod), group[mod])

    evidence_graph = chain.get("evidence_graph", {})
    return {
        "stage_status": chain.get("stage_status", {}),
        "claim_ceiling": chain.get("claim_ceiling"),
        "claim_decision": evidence_graph.get("decision", {}),
        "evidence_stages": evidence_graph.get("stages", []),
        "per_feature": per_feature,
    }


__all__ = ["derive_claimability"]

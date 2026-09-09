"""Render the short report most users should read first.

Report layering (P2, 2026-09-09): confirmatory results lead the document
(bottom line, primary endpoint, pre-specified secondary family); exploratory
and reference measures are folded into a clearly-labelled section at the end
so they are visible for transparency but cannot be mistaken for the
confirmatory conclusion.
"""
from __future__ import annotations

from ..feature_definitions import (
    REFERENCE_FEATURE,
    SECONDARY_FDR_FAMILY,
)

# Exploratory descriptors: implemented and reported for transparency, with
# no validated confirmatory null of their own (see docs/FEATURE_TABLE).
EXPLORATORY_FEATURES: tuple[str, ...] = ("bimodality_coefficient",)


def _plain_status(value: str) -> str:
    return {
        "supported": "YES",
        "not_supported": "NO CLEAR SUPPORT",
        "inconclusive": "NOT ENOUGH INFORMATION",
        "not_applicable": "NOT NEEDED",
        "invalid": "COULD NOT TEST",
    }.get(str(value), str(value).upper())


def _plain_reason(reason: str) -> str:
    text = str(reason)
    replacements = {
        "second-order group surrogate gate passed": "the group result exceeded the randomized independent-signal comparison",
        "testable second-order group surrogate gate did not pass": "the randomized comparison ran but the group result did not clearly exceed it",
        "primary existence test absent or lacks p-value resolution": "there were too few usable randomized comparisons to answer this question",
        "pseudo-pair control not run": "mismatched partners were not tested",
        "time-shift control not run": "shifted timing was not tested",
        "shared-stimulus control not run": "shared stimulus was not tested",
        "group inference not run": "conditions were not compared",
        "claimable endpoint contrast passed FDR": "the planned condition comparison was reportable after multiple-test correction",
        "claimable endpoint contrast did not pass FDR": "the planned condition comparison was reportable but not clearly different",
        "endpoint result was not claimable": "the condition result did not meet the reporting requirements",
    }
    if text in replacements:
        return replacements[text]
    if "minimum attainable" in text and "not below alpha" in text:
        return "the available sample or randomizations could not reach the required significance level"
    if text.startswith("Holm-adjusted p_real_gt_pseudo="):
        return "real and mismatched partners were compared with correction for the two design checks"
    if text.startswith("Holm-adjusted p_real_gt_time_shift="):
        return "original and shifted timing were compared with correction for the two design checks"
    if text.startswith("p_across_stimulus="):
        return "the observed result was compared with shuffled stimulus order"
    return text.replace("dyad-paired permutation", "paired condition-label swapping")


def _stage_name(stage_id: str) -> str:
    return {
        "E0": "Different from independent signals?",
        "E1": "Is there tested time structure?",
        "E2": "Specific to the real partners?",
        "E3": "Dependent on the original timing?",
        "E4": "Beyond the measured shared stimulus?",
        "E5": "Dependent on reciprocal interaction?",
        "L2": "Different between the chosen conditions?",
    }.get(stage_id, stage_id)


def build_report_markdown(
    records, cfg, chain, qc, exclusions, environment, paths=None
) -> str:
    """Build a plain-language report; technical details remain in JSON files."""
    graph = chain.get("evidence_graph") or {}
    decision = graph.get("decision") or {}
    stages = graph.get("stages") or []
    endpoint = cfg.resolved_endpoint_spec()
    unresolved = decision.get("unresolved_rivals") or []

    lines = [
        "# SyncPipe analysis report",
        "",
        "## Bottom line",
        "",
        f"**Strongest conclusion supported:** {decision.get('permitted_claim', 'No conclusion available.')}",
        "",
        f"**Condition comparison:** {decision.get('condition_claim', 'Not available.')}",
        "",
    ]
    if unresolved:
        lines += [
            "**Still not ruled out:** " + ", ".join(str(x) for x in unresolved),
            "",
        ]

    lines += [
        "## What was checked",
        "",
        "| Question | Result | Why |",
        "|---|---|---|",
    ]
    for stage in stages:
        lines.append(
            f"| {_stage_name(stage.get('stage_id', ''))} | "
            f"**{_plain_status(stage.get('status', 'unknown'))}** | "
            f"{_plain_reason(stage.get('reason', ''))} |"
        )

    lines += [
        "",
        "## Main comparison",
        "",
        f"- Measure compared: `{endpoint.name}`",
        f"- Conditions: `{cfg.contrast[0]}` vs `{cfg.contrast[1]}`",
        f"- Signal types used for the main conclusion: {', '.join(cfg.resolved_primary_modalities())}",
        "",
    ]
    group = chain.get("group_condition_inference") or {}
    found = False
    for modality, payload in sorted(group.items(), key=lambda x: str(x[0])):
        if not isinstance(payload, dict):
            continue
        for result in payload.get("per_feature", ()):
            if getattr(result, "feature", None) != endpoint.name:
                continue
            found = True
            if getattr(result, "median_ci_bounded", False):
                interval = (
                    f"95% interval {result.median_ci_low:.3g} to "
                    f"{result.median_ci_high:.3g}"
                )
            else:
                interval = "95% interval could not be bounded at this sample size"
            p_fdr = getattr(result, "p_fdr", float("nan"))
            lines.append(
                f"- **{modality}:** median difference {result.observed_diff:.3g}; "
                f"{interval}; adjusted p={p_fdr:.4g}; "
                f"usable pairs={result.n_dyads}; "
                f"reportable={'yes' if result.claimable else 'no'}"
            )
    if not found:
        lines.append("- No usable result was produced for the main measure.")

    # ---- Layered reporting: secondary family, then folded exploratory. ----
    secondary = [f for f in SECONDARY_FDR_FAMILY if f != endpoint.name]
    exploratory = [f for f in (*REFERENCE_FEATURE, *EXPLORATORY_FEATURES)
                   if f != endpoint.name]
    reference_names = tuple(REFERENCE_FEATURE)
    rows_by_feature: dict[str, list] = {}
    for modality, payload in sorted(group.items(), key=lambda x: str(x[0])):
        if not isinstance(payload, dict):
            continue
        for result in payload.get("per_feature", ()):
            feat = getattr(result, "feature", None)
            if feat is not None:
                rows_by_feature.setdefault(feat, []).append(
                    (modality, result)
                )

    if secondary and rows_by_feature:
        lines += [
            "",
            "## Secondary measures",
            "",
            "Pre-specified secondary family (reported in parallel with the "
            "main measure; multiple-testing corrected within this family):",
            "",
        ]
        for feat in secondary:
            for modality, result in rows_by_feature.get(feat, ()):
                lines.append(
                    f"- **{feat}** ({modality}): median difference "
                    f"{result.observed_diff:.3g}; adjusted p={result.p_fdr:.4g}; "
                    f"usable pairs={result.n_dyads}; "
                    f"reportable={'yes' if result.claimable else 'no'}"
                )

    if exploratory and rows_by_feature:
        lines += [
            "",
            "<details>",
            f"<summary>Exploratory and reference measures "
            f"({', '.join(exploratory)}) — folded; not confirmatory</summary>",
            "",
            "These are reported for transparency. They are not part of the "
            "pre-specified confirmatory families above, are not covered by "
            "the same multiplicity control, and cannot support claims on "
            "their own.",
            "",
        ]
        for feat in exploratory:
            label = (
                "reference measure" if feat in reference_names
                else "exploratory measure"
            )
            for modality, result in rows_by_feature.get(feat, ()):
                lines.append(
                    f"- **{feat}** ({modality}, {label}): median difference "
                    f"{result.observed_diff:.3g}; adjusted "
                    f"p={result.p_fdr:.4g}; usable pairs={result.n_dyads}"
                )
        lines += ["", "</details>", ""]

    lines += [
        "",
        "## Data used",
        "",
        f"- Rows listed: {qc.get('total_rows', 0)}",
        f"- Rows analyzed: {qc.get('included', 0)}",
        f"- Rows excluded: {qc.get('excluded', 0)}",
        f"- Sampling rate: {qc.get('hz')} Hz",
        f"- Sliding-window length: {cfg.window_size} samples",
        "",
    ]
    if exclusions:
        lines += ["## Excluded data", ""]
        for item in exclusions:
            lines.append(
                f"- {item.dyad_id} / {item.modality} / {item.condition}: "
                f"{item.detail}"
            )
        lines.append("")

    lines += [
        "## Important limits",
        "",
        "- A positive result does not prove causality, direction, relationship quality, or clinical meaning.",
        "- Shared events, co-presence, movement, or unmeasured common causes may still explain the result unless directly tested above.",
        "- Read `evidence_graph.json` for the full machine-readable checks and `qc_report.json` for data-quality details.",
        "",
        "## Reproducibility details",
        "",
        f"- SyncPipe {environment.get('syncpipe_version')} · seed {environment.get('seed')}",
        f"- Git revision `{environment.get('git_hash')}`",
    ]
    if paths:
        lines += ["", "## Output files", ""]
        for name in sorted(paths, key=str):
            lines.append(f"- `{name}`")
    return "\n".join(lines)


__all__ = ["build_report_markdown"]

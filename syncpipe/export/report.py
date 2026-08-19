"""Human-readable canonical report rendering."""
from __future__ import annotations
from typing import Any, Dict, Optional

def build_report_markdown(
    records, cfg, chain, qc, exclusions, environment, paths=None
) -> str:
    """Human-readable markdown summary of a canonical run."""
    endpoint = cfg.resolved_endpoint_spec()
    lines = [
        "# SyncPipe v1 — Canonical Analysis Report",
        "",
        f"- SyncPipe version: **{environment.get('syncpipe_version')}**",
        f"- Git hash: `{environment.get('git_hash')}`",
        f"- Seed: {environment.get('seed')}",
        f"- hz: {qc.get('hz')} | window_size: {cfg.window_size} | "
        f"window_type: {cfg.window_type}",
        f"- Contrast: {cfg.contrast}",
        f"- Declared primary endpoint: {endpoint.name} | "
        f"primary modalities: {cfg.primary_modalities}",
        f"- Endpoint estimand: {endpoint.estimand}",
        f"- Endpoint null: {endpoint.null.name} ({endpoint.null.tail})",
        f"- FDR scope: {cfg.fdr_scope} | undefined_policy: {cfg.undefined_policy} | "
        f"observation_policy: {cfg.observation_policy} | eligibility_policy: {cfg.eligibility_policy}",
        f"- Rows in manifest: {qc.get('total_rows')} | included: {qc.get('included')} | "
        f"excluded: {qc.get('excluded')}",
        "",
        "## Pipeline summary",
        "",
        chain.get("summary", ""),
        "",
        "## Claim ceiling",
        "",
        chain.get("claim_ceiling", ""),
        "",
    ]
    if exclusions:
        lines += ["## Exclusions", ""]
        for e in exclusions:
            lines.append(
                f"- dyad={e.dyad_id} modality={e.modality} "
                f"condition={e.condition}: [{e.stage}/{e.code}] {e.detail}"
            )
        lines.append("")
    lines += ["## Output files", ""]
    if paths:
        # Derived from the actual write log, so a newly dumped artifact can never
        # go unreported. REPORT.md itself is not in `paths` yet at this point,
        # which is correct: this is that file.
        for name in sorted(paths, key=str):
            lines.append(f"- `{name}`")
    else:
        lines.append(
            "(Output inventory unavailable: report rendered without a write log.)"
        )
    return "\n".join(lines)

__all__ = ["build_report_markdown"]

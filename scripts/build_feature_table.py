#!/usr/bin/env python
"""
build_feature_table.py — Single Source of Truth → authoritative feature table.

Reads the canonical metadata dicts from ``multisync.feature_definitions``
(FEATURE_TIER, MATHEMATICAL_TIER, FDR_FAMILIES, REFERENCE_FEATURE,
INTENSITY/STRUCTURE/TEMPORAL_FEATURES) and a single curated annotation block
below, then emits:

    docs/FEATURE_TABLE.csv      (machine-readable)
    docs/FEATURE_TABLE.md       (human-readable, for README / paper)

The table CANNOT drift from the code: every column except the prose
"interpretation"/"paradigm"/"primary" annotations is derived live from the
module. The annotations are validated against the code at build time —
if a feature is added/removed in feature_definitions.py without updating
this file, the build FAILS loudly.

Run:  python scripts/build_feature_table.py
Test: tests/test_feature_table_consistency.py asserts MD/CSV match the code.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow direct execution as `python scripts/build_feature_table.py` from the
# multisync-core directory without requiring editable installation first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisync.feature_definitions import (
    FEATURE_TIER,
    MATHEMATICAL_TIER,
    FDR_FAMILIES,
    FDR_FEATURES,
    REFERENCE_FEATURE,
    INTENSITY_FEATURES,
    STRUCTURE_FEATURES,
    TEMPORAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Null model per mathematical tier (Axis D) — the ONLY driver of null choice.
# ---------------------------------------------------------------------------
NULL_MODEL = {
    "L0": "Signal-level IAAFT (shuffle raw signals, recompute WCC)",
    "L1": "WCC-level IAAFT (shuffle WCC; preserves L0 moments)",
    "L2": "Exploratory in v1; validated timing/morphology existence null deferred to v2",
}

# ---------------------------------------------------------------------------
# Curated prose annotations. These are the ONLY hand-written cells.
# Keys MUST exactly match feature_definitions.FEATURE_TIER (checked below).
#   primary   : is this a PRIMARY confirmatory feature for v1.0?
#   paradigm  : where the feature is meaningful / where it breaks
#   interpret : one-line scientific reading
# ---------------------------------------------------------------------------
ANNOTATIONS = {
    "mean_synchrony": dict(
        primary="Reference only (reported, NOT in FDR)",
        paradigm="All paradigms; most robust, least specific",
        interpret="Average moment-to-moment coupling magnitude.",
    ),
    "peak_amplitude": dict(
        primary="PRIMARY (intensity)",
        paradigm="All paradigms; cross-paradigm robust",
        interpret="Strongest sustained coupling reached during interaction.",
    ),
    "fraction_above_threshold": dict(
        primary="Exploratory-secondary (occupancy; NOT in FDR)",
        paradigm="All paradigms with threshold justification; report threshold metadata",
        interpret="Fraction of finite WCC samples above the synchrony threshold (coverage).",
    ),
    "dwell_time": dict(
        primary="PRIMARY (structure)",
        paradigm="Continuous & event paradigms; needs sufficient trace length",
        interpret="Mean duration of high-synchrony episodes (persistence).",
    ),
    "switching_rate": dict(
        primary="PRIMARY (structure)",
        paradigm="Continuous & event paradigms; sensitive to window size",
        interpret="How often synchrony crosses in/out of high-coupling state.",
    ),
    "bimodality_coefficient": dict(
        primary="Exploratory (distributional; not in the FDR family)",
        paradigm="All paradigms; distribution shape, not temporal order",
        interpret="Degree to which synchrony is bistable (high vs low) rather than graded.",
    ),
    "synchrony_entropy": dict(
        primary="Exploratory (distributional; NOT in FDR)",
        paradigm="All paradigms; distribution shape, not temporal order",
        interpret="Dispersion/unpredictability of the synchrony distribution.",
    ),
    "inter_peak_cv": dict(
        primary="Exploratory-secondary (temporal regularity; NOT in FDR)",
        paradigm="Long, oscillatory traces with >= 3 prominent peaks; report definedness rate",
        interpret="CV of inter-peak intervals (regular vs irregular synchrony events).",
    ),
    "first_peak_time": dict(
        primary="Exploratory-secondary (event timing; NOT in FDR)",
        paradigm="Any morphology with >= 1 prominent peak; report definedness rate",
        interpret="Time of the first prominent above-threshold synchrony peak.",
    ),
    "onset_latency": dict(
        primary="Exploratory — EVENT-LOCKED paradigms ONLY",
        paradigm="Event/stimulus-locked ONLY; undefined (NaN) in free interaction",
        interpret="Time from event onset to first sustained high-synchrony crossing.",
    ),
    "rise_time": dict(
        primary="Exploratory — EVENT-LOCKED paradigms ONLY",
        paradigm="Event/stimulus-locked ONLY; estimator-shape confound (see Limitations)",
        interpret="Speed of synchrony build-up (WCC-derived; not a physiological waveform).",
    ),
    "recovery_time": dict(
        primary="Exploratory — EVENT-LOCKED paradigms ONLY",
        paradigm="Event/stimulus-locked ONLY; estimator-shape confound (see Limitations)",
        interpret="Time for synchrony to return toward baseline after a peak.",
    ),
}

COLUMNS = [
    "feature",
    "functional_tier",     # Axis A
    "informational_class", # intensity / structure / temporal
    "computed",            # always True in v1.0
    "primary",             # prose
    "in_FDR_family",       # Axis C
    "fdr_family",          # L0 / L1 / —
    "math_tier",           # Axis D
    "null_model",          # derived from math_tier
    "paradigm",            # prose
    "interpretation",      # prose
]


def informational_class(name: str) -> str:
    if name in INTENSITY_FEATURES:
        return "intensity"
    if name in STRUCTURE_FEATURES:
        return "structure"
    if name in TEMPORAL_FEATURES:
        return "temporal"
    return "—"


def fdr_family_of(name: str) -> str:
    for fam, members in FDR_FAMILIES.items():
        if name in members:
            return fam
    return "—"


def build_rows():
    # Consistency guard: annotations must cover exactly the coded features.
    coded = set(FEATURE_TIER)
    annotated = set(ANNOTATIONS)
    if coded != annotated:
        missing = coded - annotated
        extra = annotated - coded
        raise SystemExit(
            "FEATURE TABLE DRIFT DETECTED — update scripts/build_feature_table.py:\n"
            f"  features in code but NOT annotated: {sorted(missing)}\n"
            f"  annotated but NOT in code:          {sorted(extra)}"
        )

    rows = []
    # Stable order: intensity → structure → temporal, reference first within ties
    order = list(INTENSITY_FEATURES) + list(STRUCTURE_FEATURES) + list(TEMPORAL_FEATURES)
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        mt = MATHEMATICAL_TIER[name]
        rows.append({
            "feature": name,
            "functional_tier": FEATURE_TIER[name],
            "informational_class": informational_class(name),
            "computed": "yes",
            "primary": ANNOTATIONS[name]["primary"],
            "in_FDR_family": "yes" if name in FDR_FEATURES else "no",
            "fdr_family": fdr_family_of(name),
            "math_tier": mt,
            "null_model": NULL_MODEL[mt],
            "paradigm": ANNOTATIONS[name]["paradigm"],
            "interpretation": ANNOTATIONS[name]["interpret"],
        })
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_md(rows, path: Path):
    n_fdr = len(FDR_FEATURES)
    n_total = len(FEATURE_TIER)
    lines = []
    lines.append("# SyncPipe — Authoritative Feature Table (Single Source of Truth)\n")
    lines.append(
        "> **Auto-generated** by `scripts/build_feature_table.py` directly from "
        "`feature_definitions.py`. Do not hand-edit. "
        f"Total features computed: **{n_total}**; FDR-family (confirmatory multiplicity set): "
        f"**{n_fdr}** ({', '.join(FDR_FEATURES)}); Reference (reported, not corrected): "
        f"**{', '.join(REFERENCE_FEATURE)}**.\n"
    )
    lines.append(
        "\n**Four orthogonal axes** govern every feature:\n"
        "- **Functional tier (Axis A)** — extraction robustness label: `reference` / `core` / `conditional`.\n"
        "- **Informational class** — Results organisation: `intensity` / `structure` / `temporal`.\n"
        "- **FDR family (Axis C)** — whether the feature is in the confirmatory multiplicity-corrected set.\n"
        "- **Mathematical tier (Axis D)** — *sole* determinant of the null model: `L0` / `L1` / `L2`.\n"
    )
    # Compact table
    hdr = ["Feature", "Func. tier", "Class", "Primary?", "FDR?", "Fam", "Math", "Null model", "Paradigm validity", "Interpretation"]
    lines.append("\n| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['feature']}`",
            r["functional_tier"],
            r["informational_class"],
            r["primary"],
            r["in_FDR_family"],
            r["fdr_family"],
            r["math_tier"],
            r["null_model"],
            r["paradigm"],
            r["interpretation"],
        ]) + " |")
    lines.append("\n## Null-model legend\n")
    for tier, desc in NULL_MODEL.items():
        lines.append(f"- **{tier}** — {desc}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    repo_root = Path(__file__).resolve().parents[1]  # multisync-core/
    docs = repo_root / "docs"
    rows = build_rows()
    write_csv(rows, docs / "FEATURE_TABLE.csv")
    write_md(rows, docs / "FEATURE_TABLE.md")
    print(f"Wrote {docs/'FEATURE_TABLE.csv'} and {docs/'FEATURE_TABLE.md'} "
          f"({len(rows)} features; {len(FDR_FEATURES)} in FDR family).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

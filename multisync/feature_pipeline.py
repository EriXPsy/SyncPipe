"""
Pipeline 1: Feature pipeline.

Purpose: Help users understand what features are available, what they mean,
and how to choose the right ones for their research question.

This is a thin, user-friendly wrapper around ``feature_definitions.py`` (SSoT).
It does NOT compute anything — it only explains and selects.

Note on ``_FEATURE_CATALOG`` vs SSoT
------------------------------------
The catalog exists only to carry richer human-readable metadata (description,
unit, typical range) that lives nowhere else.  Every field that also exists in
the SSoT is NOT restated here:

* ``axis`` is derived mechanically from ``FEATURE_AXIS`` at construction time,
  so it can never drift.
* ``tier`` and ``fdr_member`` are still written next to the prose (they carry
  inline justification comments), but an import-time guard asserts they agree
  with ``FEATURE_TIER`` / ``FDR_FEATURES``, so a drifted copy fails loudly at
  import rather than silently shipping a mislabeled feature.
"""

from typing import Dict, List, Optional

import warnings

from .feature_definitions import (
    FDR_FEATURES,
    FEATURE_AXIS,
    FEATURE_TIER,
    get_fdr_features as _ssot_get_fdr_features,
)


class FeatureInfo:
    """Human-readable information about one feature."""

    def __init__(
        self,
        name: str,
        tier: str,
        fdr_member: bool,
        description: str,
        unit: str,
        typical_range: str,
    ):
        # `axis` is never passed in: it is read from the SSoT so the catalog
        # cannot hold a second, drifting copy of the axis assignment. A feature
        # missing from FEATURE_AXIS is a genuine SSoT gap, so fail loudly.
        if name not in FEATURE_AXIS:
            raise AssertionError(
                f"_FEATURE_CATALOG lists '{name}', which has no axis in the SSoT "
                f"FEATURE_AXIS. Add it to one of INTENSITY/STRUCTURE/"
                f"TEMPORAL_FEATURES in feature_definitions.py."
            )
        self.name = name
        self.tier = tier
        self.axis = FEATURE_AXIS[name]
        self.fdr_member = fdr_member
        self.description = description
        self.unit = unit
        self.typical_range = typical_range

    def summary(self) -> str:
        return (
            f"{self.name} [{self.tier}/{self.axis}]"
            + (" (FDR)" if self.fdr_member else " (diagnostic)")
            + f": {self.description}"
        )


_FEATURE_CATALOG: Dict[str, FeatureInfo] = {
    "mean_synchrony": FeatureInfo(
        name="mean_synchrony",
        tier="reference",
        fdr_member=False,           # reference comparator; NOT in FDR_FEATURES (SSoT 2026-06-29)
        description="Mean WCC value across the epoch — overall coupling strength baseline.",
        unit="Pearson r [-1, 1]",
        typical_range="0.0 to 0.6",
    ),
    "peak_amplitude": FeatureInfo(
        name="peak_amplitude",
        tier="core",
        fdr_member=True,            # in FDR_FAMILIES["L0"]
        description="Maximum WCC value at the dominant peak — peak coupling intensity.",
        unit="Pearson r [-1, 1]",
        typical_range="0.2 to 0.9",
    ),
    "onset_latency": FeatureInfo(
        name="onset_latency",
        tier="conditional",         # matches FEATURE_TIER
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from epoch start to first above-threshold WCC crossing — "
        "how quickly synchrony emerges.",
        unit="seconds",
        typical_range="0 to 30 s",
    ),
    "rise_time": FeatureInfo(
        name="rise_time",
        tier="conditional",         # matches FEATURE_TIER
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from 25% to 75% of peak amplitude — coordination build-up speed.",
        unit="seconds",
        typical_range="1 to 20 s",
    ),
    "recovery_time": FeatureInfo(
        name="recovery_time",
        tier="conditional",         # matches FEATURE_TIER
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from peak to 50% decay — how long coupling persists after peaking.",
        unit="seconds",
        typical_range="1 to 30 s",
    ),
    "fraction_above_threshold": FeatureInfo(
        name="fraction_above_threshold",
        tier="conditional",
        fdr_member=False,
        description="Fraction of finite WCC samples above the synchrony threshold — above-threshold occupancy.",
        unit="proportion [0, 1]",
        typical_range="0.0 to 1.0",
    ),
    "dwell_time": FeatureInfo(
        name="dwell_time",
        tier="core",
        fdr_member=True,            # in FDR_FAMILIES["L1"]
        description="Mean duration of above-threshold intervals — "
        "how long dyads stay in a synchronized state.",
        unit="seconds",
        typical_range="1 to 60 s",
    ),
    "switching_rate": FeatureInfo(
        name="switching_rate",
        tier="core",
        fdr_member=True,            # in FDR_FAMILIES["L1"]
        description="Number of threshold crossings per minute — "
        "frequency of entering/leaving synchronized states.",
        unit="crossings/min",
        typical_range="0 to 10",
    ),
    "synchrony_entropy": FeatureInfo(
        name="synchrony_entropy",
        tier="conditional",
        fdr_member=False,           # excluded: collinear with mean_synchrony (rho=-0.94)
        description="Shannon entropy of the WCC distribution — "
        "diversity of coupling states visited.",
        unit="bits",
        typical_range="1 to 5",
    ),
    "bimodality_coefficient": FeatureInfo(
        name="bimodality_coefficient",
        tier="conditional",
        fdr_member=False,           # exploratory descriptor; not in the FDR family
        description="Sarle's bimodality coefficient (BC) of WCC values — "
        "degree to which coupling follows a dual-state (on/off) pattern.",
        unit="dimensionless [0, 1]",
        typical_range="0.3 to 0.8",
    ),
    "first_peak_time": FeatureInfo(
        name="first_peak_time",
        tier="conditional",
        fdr_member=False,
        description="Time of the first prominent above-threshold WCC peak.",
        unit="seconds",
        typical_range="paradigm-dependent",
    ),
    "inter_peak_cv": FeatureInfo(
        name="inter_peak_cv",
        tier="conditional",
        fdr_member=False,
        description="Coefficient of variation of intervals between prominent WCC peaks.",
        unit="dimensionless",
        typical_range="paradigm-dependent",
    ),
}


# Import-time anti-drift guard: the catalog carries richer human-readable
# metadata that cannot be mechanically derived, but its `tier`/`fdr_member`
# fields MUST agree with the SSoT. Assert that here so a future SSoT edit that
# is not mirrored into the catalog fails loudly at import rather than silently
# shipping a mislabeled feature tier in user-facing help.
for _cat_name, _info in _FEATURE_CATALOG.items():
    if _cat_name not in FEATURE_TIER:
        raise AssertionError(
            f"_FEATURE_CATALOG lists '{_cat_name}', which is not in the SSoT "
            f"FEATURE_TIER. Remove it or add it to feature_definitions.FEATURE_TIER."
        )
    if _info.tier != FEATURE_TIER[_cat_name]:
        raise AssertionError(
            f"_FEATURE_CATALOG['{_cat_name}'].tier={_info.tier!r} disagrees with "
            f"SSoT FEATURE_TIER[{_cat_name!r}]={FEATURE_TIER[_cat_name]!r}. "
            f"Update the catalog to match the SSoT."
        )
    _ssot_fdr = _cat_name in FDR_FEATURES
    if _info.fdr_member != _ssot_fdr:
        raise AssertionError(
            f"_FEATURE_CATALOG['{_cat_name}'].fdr_member={_info.fdr_member} disagrees "
            f"with SSoT membership (name in FDR_FEATURES = {_ssot_fdr}). "
            f"Update the catalog to match the SSoT."
        )
del _cat_name, _info


def list_features(tier: Optional[str] = None, axis: Optional[str] = None) -> List[FeatureInfo]:
    """Return all features, optionally filtered by tier or axis.

    Parameters
    ----------
    tier : str or None
        Filter by functional tier: "core", "conditional", "reference".
        Matching is case-insensitive; an unrecognised value warns and returns [].
    axis : str or None
        Filter by informational axis: "intensity", "structure", "temporal".
        Matching is case-insensitive; an unrecognised value warns and returns [].

    Returns
    -------
    List of FeatureInfo instances.
    """
    result = list(_FEATURE_CATALOG.values())
    if tier is not None:
        t = tier.lower()
        valid_tiers = sorted({f.tier for f in result})
        if t not in valid_tiers:
            warnings.warn(
                f"list_features: unknown tier {tier!r}. Valid tiers: {valid_tiers}. "
                f"Returning empty list."
            )
            return []
        result = [f for f in result if f.tier == t]
    if axis is not None:
        a = axis.lower()
        valid_axes = sorted({f.axis for f in result})
        if a not in valid_axes:
            warnings.warn(
                f"list_features: unknown axis {axis!r}. Valid axes: {valid_axes}. "
                f"Returning empty list."
            )
            return []
        result = [f for f in result if f.axis == a]
    return result


def explain_feature(name: str) -> Optional[FeatureInfo]:
    """Return detailed explanation of a single feature.

    Parameters
    ----------
    name : str
        Feature name, e.g. "peak_amplitude", "dwell_time".

    Returns
    -------
    FeatureInfo or None if not found.
    """
    return _FEATURE_CATALOG.get(name)


def get_fdr_features(full_family_fdr: bool = False) -> List[str]:
    """Return the list of features included in the L2 BH-FDR correction.

    Parameters
    ----------
    full_family_fdr : bool, default False
        Forwarded to the SSoT ``get_fdr_features``.  False returns the frozen
        PRIMARY confirmatory family (``PRIMARY_FDR_FAMILY``, n=1:
        ``peak_amplitude``) — NOT the full ``FDR_FEATURES`` triple. The primary
        claim rests on a single pre-registered endpoint so that gating existence
        and claiming group inference cannot smuggle in an OR-across-a-family;
        ``dwell_time`` / ``switching_rate`` are reported in parallel as SECONDARY,
        BH-corrected within their own family. True returns all 12 implemented
        features for a strictly-more-conservative, reviewer-proof single BH-FDR
        step.
    """
    return _ssot_get_fdr_features(full_family_fdr)


def recommend_features(research_question: str = "general") -> Dict[str, List[str]]:
    """Recommend feature sets based on the research question.

    Parameters
    ----------
    research_question : str
        One of: "general", "intensity", "dynamics", "structure", "full".
        Matching is case-insensitive. An unrecognised value raises ValueError
        with the list of valid options (no silent fallback to "general").

    Returns
    -------
    Dict with keys "primary", "supplementary", "reference", and rationale.
    """
    recommendations = {
        "general": {
            "primary": list(FDR_FEATURES),
            "supplementary": ["fraction_above_threshold", "bimodality_coefficient", "synchrony_entropy", "first_peak_time", "inter_peak_cv"],
            "reference": ["mean_synchrony"],
            "rationale": (
                "General-purpose v1 set: peak_amplitude, dwell_time, and "
                "switching_rate are the primary FDR-family descriptors when "
                "thresholding is group-comparable; distribution/timing descriptors "
                "are exploratory-secondary."
            ),
        },
        "intensity": {
            "primary": ["peak_amplitude"],
            "supplementary": ["fraction_above_threshold"],
            "reference": ["mean_synchrony"],
            "rationale": (
                "Focus on coupling magnitude: peak_amplitude as the primary "
                "intensity workhorse; mean_synchrony remains a reference comparator."
            ),
        },
        "dynamics": {
            "primary": [],
            "supplementary": ["onset_latency", "rise_time", "recovery_time", "first_peak_time", "inter_peak_cv", "peak_amplitude"],
            "reference": ["mean_synchrony"],
            "rationale": (
                "Timing/morphology descriptors are exploratory in v1 (not yet "
                "confirmatory), so there is no primary *timing* endpoint to recommend "
                "yet. If your research question is really about dynamic *process* "
                "structure — how long dyads stay coordinated and how often they switch "
                "states — use the 'structure' preset: dwell_time and switching_rate are "
                "the current FDR-family confirmatory descriptors. Report timing "
                "descriptors only with paradigm restrictions and definedness rates."
            ),
        },
        "structure": {
            "primary": ["dwell_time", "switching_rate"],
            "supplementary": ["fraction_above_threshold", "bimodality_coefficient", "synchrony_entropy"],
            "reference": ["mean_synchrony"],
            "rationale": (
                "Focus on coordination structure: dwell_time and switching_rate "
                "summarize above-threshold state persistence/flexibility when the "
                "threshold is comparable; occupancy and distribution-shape descriptors "
                "are reported as exploratory-secondary."
            ),
        },
        "full": {
            "primary": list(FDR_FEATURES),
            "supplementary": ["fraction_above_threshold", "bimodality_coefficient", "synchrony_entropy", "onset_latency", "rise_time", "recovery_time", "first_peak_time", "inter_peak_cv"],
            "reference": ["mean_synchrony"],
            "rationale": (
                "Complete v1 descriptor map: primary FDR-family descriptors plus "
                "reference and exploratory diagnostics."
            ),
        },
    }
    key = research_question.lower()
    if key not in recommendations:
        valid = sorted(recommendations.keys())
        raise ValueError(
            f"recommend_features: unknown research_question {research_question!r}. "
            f"Valid options are: {valid}."
        )
    return recommendations[key]


def print_feature_table():
    """Print a formatted table of all features for quick reference."""
    header = f"{'Feature':<25} {'Tier':<12} {'Axis':<12} {'FDR':<5} {'Unit':<18}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for f in _FEATURE_CATALOG.values():
        lines.append(
            f"{f.name:<25} {f.tier:<12} {f.axis:<12} "
            f"{'yes' if f.fdr_member else 'no':<5} {f.unit:<18}"
        )
    lines.append(sep)
    return "\n".join(lines)

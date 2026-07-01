"""
Pipeline 1: Feature pipeline.

Purpose: Help users understand what features are available, what they mean,
and how to choose the right ones for their research question.

This is a thin, user-friendly wrapper around ``feature_definitions.py`` (SSoT).
It does NOT compute anything — it only explains and selects.

Note on ``_FEATURE_CATALOG`` vs SSoT
------------------------------------
The catalog's ``tier`` and ``fdr_member`` fields are kept in sync with
``FEATURE_TIER`` and ``FDR_FEATURES`` in the SSoT.  When the SSoT changes,
this file MUST be updated manually — there is no mechanical derivation,
because the catalog carries richer human-readable metadata (HKB
interpretation, typical range, unit) that lives nowhere else.
"""

from typing import Dict, List, Optional

from .feature_definitions import (
    FDR_FEATURES,
    REFERENCE_FEATURE,
    CORE_FEATURES,
    CONDITIONAL_FEATURES,
)


class FeatureInfo:
    """Human-readable information about one feature."""

    def __init__(
        self,
        name: str,
        tier: str,
        axis: str,
        fdr_member: bool,
        description: str,
        hkb_interpretation: str,
        unit: str,
        typical_range: str,
    ):
        self.name = name
        self.tier = tier
        self.axis = axis
        self.fdr_member = fdr_member
        self.description = description
        self.hkb_interpretation = hkb_interpretation
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
        axis="intensity",
        fdr_member=False,           # reference comparator; NOT in FDR_FEATURES (SSoT 2026-06-29)
        description="Mean WCC value across the epoch — overall coupling strength baseline.",
        hkb_interpretation="AR(1) baseline: average proximity to the attractor. "
        "Captures tonic synchrony level independent of phase structure.",
        unit="Pearson r [-1, 1]",
        typical_range="0.0 to 0.6",
    ),
    "peak_amplitude": FeatureInfo(
        name="peak_amplitude",
        tier="core",
        axis="intensity",
        fdr_member=True,            # in FDR_FAMILIES["L0"]
        description="Maximum WCC value at the dominant peak — peak coupling intensity.",
        hkb_interpretation="Maximum depth of the shared attractor well. "
        "Higher values indicate stronger momentary coordination.",
        unit="Pearson r [-1, 1]",
        typical_range="0.2 to 0.9",
    ),
    "onset_latency": FeatureInfo(
        name="onset_latency",
        tier="conditional",         # matches FEATURE_TIER
        axis="temporal",
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from epoch start to first above-threshold WCC crossing — "
        "how quickly synchrony emerges.",
        hkb_interpretation="Time spent waiting near the attractor before convergence. "
        "Shorter latency = faster mutual entrainment.",
        unit="seconds",
        typical_range="0 to 30 s",
    ),
    "rise_time": FeatureInfo(
        name="rise_time",
        tier="conditional",         # matches FEATURE_TIER
        axis="temporal",
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from 25% to 75% of peak amplitude — coordination build-up speed.",
        hkb_interpretation="Convergence rate toward the shared attractor. "
        "Faster rise = stronger coupling pull.",
        unit="seconds",
        typical_range="1 to 20 s",
    ),
    "recovery_time": FeatureInfo(
        name="recovery_time",
        tier="conditional",         # matches FEATURE_TIER
        axis="temporal",
        fdr_member=False,           # L2 exploratory; not in FDR_FEATURES
        description="Time from peak to 50% decay — how long coupling persists after peaking.",
        hkb_interpretation="Escape rate from the shared attractor. "
        "Longer recovery = deeper attractor well (slower decay).",
        unit="seconds",
        typical_range="1 to 30 s",
    ),
    "fraction_above_threshold": FeatureInfo(
        name="fraction_above_threshold",
        tier="conditional",
        axis="structure",
        fdr_member=False,
        description="Fraction of finite WCC samples above the synchrony threshold — above-threshold occupancy.",
        hkb_interpretation="Coverage of the synchronized state, independent of episode ordering. "
        "Use as an exploratory occupancy descriptor, not a primary endpoint.",
        unit="proportion [0, 1]",
        typical_range="0.0 to 1.0",
    ),
    "dwell_time": FeatureInfo(
        name="dwell_time",
        tier="core",
        axis="structure",
        fdr_member=True,            # in FDR_FAMILIES["L1"]
        description="Mean duration of above-threshold intervals — "
        "how long dyads stay in a synchronized state.",
        hkb_interpretation="Residence time in the synchronized attractor. "
        "Longer dwells = more stable coordination modes.",
        unit="seconds",
        typical_range="1 to 60 s",
    ),
    "switching_rate": FeatureInfo(
        name="switching_rate",
        tier="core",
        axis="structure",
        fdr_member=True,            # in FDR_FAMILIES["L1"]
        description="Number of threshold crossings per minute — "
        "frequency of entering/leaving synchronized states.",
        hkb_interpretation="Attractor landscape flexibility. "
        "Higher switching = more frequent mode transitions.",
        unit="crossings/min",
        typical_range="0 to 10",
    ),
    "synchrony_entropy": FeatureInfo(
        name="synchrony_entropy",
        tier="conditional",
        axis="structure",
        fdr_member=False,           # excluded: collinear with mean_synchrony (rho=-0.94)
        description="Shannon entropy of the WCC distribution — "
        "diversity of coupling states visited.",
        hkb_interpretation="Complexity of the coordination landscape. "
        "Higher entropy = richer repertoire of coupling configurations.",
        unit="bits",
        typical_range="1 to 5",
    ),
    "bimodality_coefficient": FeatureInfo(
        name="bimodality_coefficient",
        tier="conditional",
        axis="structure",
        fdr_member=False,           # exploratory descriptor; not in the FDR family
        description="Sarle's bimodality coefficient (BC) of WCC values — "
        "degree to which coupling follows a dual-state (on/off) pattern.",
        hkb_interpretation="Evidence for two distinct attractor states. "
        "BC > 0.555 suggests bimodal coupling (synchronized vs. unsynchronized).",
        unit="dimensionless [0, 1]",
        typical_range="0.3 to 0.8",
    ),
    "first_peak_time": FeatureInfo(
        name="first_peak_time",
        tier="conditional",
        axis="temporal",
        fdr_member=False,
        description="Time of the first prominent above-threshold WCC peak.",
        hkb_interpretation="Exploratory trace-morphology timing descriptor. "
        "Report only with definedness rates and paradigm restrictions.",
        unit="seconds",
        typical_range="paradigm-dependent",
    ),
    "inter_peak_cv": FeatureInfo(
        name="inter_peak_cv",
        tier="conditional",
        axis="temporal",
        fdr_member=False,
        description="Coefficient of variation of intervals between prominent WCC peaks.",
        hkb_interpretation="Exploratory regularity/irregularity descriptor for multi-peak traces. "
        "Requires enough defined peaks and is not a v1 confirmatory endpoint.",
        unit="dimensionless",
        typical_range="paradigm-dependent",
    ),
}


def list_features(tier: Optional[str] = None, axis: Optional[str] = None) -> List[FeatureInfo]:
    """Return all features, optionally filtered by tier or axis.

    Parameters
    ----------
    tier : str or None
        Filter by functional tier: "core", "conditional", "reference".
    axis : str or None
        Filter by informational axis: "intensity", "structure", "temporal".

    Returns
    -------
    List of FeatureInfo instances.
    """
    result = list(_FEATURE_CATALOG.values())
    if tier:
        result = [f for f in result if f.tier == tier]
    if axis:
        result = [f for f in result if f.axis == axis]
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


def get_fdr_features() -> List[str]:
    """Return the list of features included in the FDR family."""
    return list(FDR_FEATURES)


def get_core_features() -> List[str]:
    """Return core v1 descriptor names."""
    return list(CORE_FEATURES)


def get_conditional_features() -> List[str]:
    """Return conditional feature names."""
    return list(CONDITIONAL_FEATURES)


def get_reference_feature() -> str:
    """Return the reference feature name (mean_synchrony)."""
    return REFERENCE_FEATURE[0]


def recommend_features(research_question: str = "general") -> Dict[str, List[str]]:
    """Recommend feature sets based on the research question.

    Parameters
    ----------
    research_question : str
        One of: "general", "intensity", "dynamics", "structure", "full".

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
                "Timing/morphology descriptors are exploratory in v1. Report them "
                "only with paradigm restrictions and definedness rates; they are not "
                "primary confirmatory endpoints."
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
    return recommendations.get(research_question, recommendations["general"])


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

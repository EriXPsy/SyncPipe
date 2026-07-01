"""
multisync/feature_definitions.py
================================

Single Source of Truth (SSoT) for feature mathematics,
operating on a Windowed Cross-Correlation (WCC) time series.

SSoT boundary update (v1 measurement-infrastructure architecture)
-----------------------------------------------------------------
This module is the **mathematical SSoT**: it implements feature definitions,
constants, extraction helpers, and serialization.  External-facing feature
status for README/demo/manuscript Table 1 lives in ``multisync.feature_status``.
That communication table is deliberately simpler: source level, incremental
information, paradigm restrictions, recommended audit/test, evidence status,
and risk.  See ``docs/METHOD_LOG.md``.

Responsibility (Module Contract)
--------------------------------------
This module is responsible for: the mathematical definitions and
computation functions of all implemented WCC-derived features.

This module MUST NOT: compute WCC, generate surrogates, read files,
or produce figures.  Those belong to ``dynamic_features``,
``validation.pgt1_intensity``, ``io``, and ``scripts.plot_*`` respectively.

All other modules MUST import feature math from here rather
than reimplement it.

Three-Axis Classification System
---------------------------------------------
Features are classified along three INDEPENDENT axes. Axis C (FDR
membership) is deliberately NOT a mechanical derivation of Axis A — see
the rationale below.

**Axis A — Functional tier** (extraction robustness, empirically derived):
  Core:        Morphology-independent, cross-paradigm robust
  Conditional: Assumes SCR-like single-peak morphology, or pending
               cross-paradigm/construct-validity confirmation
  Reference:   Baseline comparator, always computed → report-only,
               never FDR-eligible

**Axis B — Informational tier** (what kind of synchrony information):
  Intensity:   Magnitude of moment-to-moment coupling
  Structure:   Temporal organisation — sustained vs intermittent
  Temporal:    Event timing — when episodes occur

**Axis C — FDR membership** (statistical inference family, independently
gated): membership in ``FDR_FEATURES`` is NOT automatically granted by
Core/Conditional tier status. A feature enters the FDR family only when
a dated DECISION_LOG entry documents validation evidence that PRECEDES
the entry's own timestamp (cross-paradigm defined-rate table and/or a
dedicated ground-truth construct-validity test). This separation exists
specifically to prevent "promote first, validate after" sequencing —
see the inline NOTE on ``bimodality_coefficient`` below, which is
currently the one feature whose Conditional-tier classification and
FDR-family inclusion need an explicit, separately-dated DECISION entry
to confirm the evidence-before-decision ordering.

Every feature has one functional tier AND one informational tier;
FDR membership is tracked separately in ``FDR_FEATURES`` (Axis C) and
must not be re-derived by filtering ``FEATURE_TIER``.

**Axis D — Mathematical invariance tier (L0/L1/L2)** (driver axis for null model selection):
  L0 (permutation-invariant):   mean_synchrony, peak_amplitude,
                             synchrony_entropy, bimodality_coefficient
    → Null model: SIGNAL-LEVEL IAAFT (destroy all coupling, including L0 moments)
  L1 (local temporal structure): dwell_time, switching_rate,
                             bimodality_coefficient (structural semantics)
    → Null model: WCC-LEVEL IAAFT (preserve L0 moments, destroy run-length)
  L2 (event-locked morphology): onset_latency, rise_time, recovery_time
    → Null model: CIRCULAR TIME-SHIFT (preserve L0+L1, destroy absolute phase)
  See ``docs/surrogate_threshold_design.md`` for the full rationale.
  Axis D is the sole driver for null model selection in ``dynamic_features.py``.
  Axes A/B/C are external communication labels (functional, informational, FDR).
  A feature's mathematical tier is NOT derived from its functional tier.

Functional tiers:
  REFERENCE  (1 feature, report-only, never FDR-eligible)
    mean_synchrony

  EXPLORATORY OCCUPANCY (implemented, not FDR)
    fraction_above_threshold  Fraction of finite WCC values >= threshold;
                              permutation-invariant coverage descriptor

  CORE       (3 features, cross-morphology, cross-paradigm; FDR family)
    peak_amplitude        [Intensity]
    dwell_time            [Structure — stability pole]
    switching_rate        [Structure — flexibility pole]

  CONDITIONAL (5 features, morphology/paradigm-dependent; FDR family)
    onset_latency         [Temporal]  Requires low→high ignition trajectory
    recovery_time         [Temporal]  Requires identifiable post-peak decay
    rise_time             [Temporal]  Requires clear single-peak morphology
    synchrony_entropy     [Structure] Shannon entropy of WCC distribution;
                          bridges Structure and Temporal dimensions
    bimodality_coefficient [Structure] Bimodality of WCC distribution;
                          detects high/low state separability (exploratory
                          descriptor, not in the FDR family)

**Morphology-Agnostic Timers** (diagnostic, not in FDR family)
    first_peak_time       Time of first prominent peak — all morphologies
    baseline_fraction     Fraction below threshold before first peak
    inter_peak_cv         CV of inter-peak intervals — metastability descriptor
    These are computed but do NOT enter FDR; promotion to Core requires
    further validation (see §4.5 Future Directions).

References
----------
- Bassett, D. S., Wymbs, N. F., Porter, M. A., Mucha, P. J., Carlson, J. M.,
  & Grafton, S. T. (2011). Dynamic reconfiguration of human brain networks
  during learning. *PNAS*, 108(18), 7641-7646.
  Boker, S. M., Xu, M., Rotondo, J. L., & King, K. (2002). "Windowed cross-correlation
  and peak picking for the analysis of variability in the association
  between behavioral time series." Psychological Methods, 7(3), 338–355.
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
  (2nd ed.). Lawrence Erlbaum.
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2007). The electrodermal
  system. In *Handbook of Psychophysiology* (3rd ed.).
- Gordon, I., Tomashin, A., & Mayo, O. (2024). A theory of flexible
  multimodal synchrony. *Psychological Review*, 132(3), 680–718.
- Kelso, J. A. S. (1995). *Dynamic Patterns*. MIT Press.
- Tognoli, E., & Kelso, J. A. S. (2014). The metastable brain.
  *Neuron*, 81(1), 35-48.
- Pfister, R., Schwarz, K. A., Janczyk, M., Dale, R., & Freeman, J. B.
  (2013). Good things peak in pairs: a note on the bimodality coefficient.
  *Frontiers in Psychology*, 4, 700.
  NOTE: the commonly-cited "Ellison 1987" attribution for the BC formula
  and the 0.555 threshold has NOT been independently verified against a
  primary source as of this writing; confirm before citing in the methods
  paper. Sarle's SAS documentation is the most consistently traceable
  origin found so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as _dc_fields, MISSING as _DC_MISSING
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Locked constants (DECISION-01, DECISION-03, DECISION-04, DECISION-05)
# ---------------------------------------------------------------------------

ONSET_THRESHOLD: float = 0.5
"""
**Primary analysis now uses surrogate-derived threshold** (see
:func:`compute_surrogate_threshold`).  This constant is retained as the
fallback for callers that do not supply an empirical threshold, and for
the sensitivity sweep [0.3, 0.7] .
"""

SURROGATE_THRESHOLD_PERCENTILE: float = 95.0
"""percentile used for surrogate-derived threshold (default 95th).

The surrogate null distribution for a dyad is built by computing WCC on
``n_surrogates`` IAAFT-randomised signal pairs.  The threshold is the
``SURROGATE_THRESHOLD_PERCENTILE``-th quantile of all surrogate WCC values,
representing the highest WCC level reachable by chance at the given false-
positive rate.
"""

PEAK_SMOOTHING_WINDOW: int = 3
"""DECISION-04: 3-point boxcar smoothing for peak detection (Boucsein 2012)."""

RISE_LOW_FRAC: float = 0.25
RISE_HIGH_FRAC: float = 0.75
"""DECISION-03: 25%-75% quartile rise time (Boucsein 2012)."""

RECOVERY_FRAC: float = 0.50
"""DECISION-05: half-recovery time (Boucsein 2012; Dawson et al. 2007)."""

SWITCHING_HYSTERESIS_DELTA: float = 0.05
"""Hysteresis band for state binarization.

WCC values within ``[threshold - delta, threshold + delta)`` retain the
previous state (Schmitt trigger logic), eliminating boundary jitter from
oscillatory traces straddling the threshold.  Default 0.05 in WCC units
(r-metric); set to 0.0 to recover the legacy non-hysteresis behaviour.
"""


# re-export of LEAKAGE_DELTA_AUC_THRESHOLD (primary definition in prediction.py)
def __getattr__(name: str):
    if name == "LEAKAGE_DELTA_AUC_THRESHOLD":
        try:
            from .prediction import LEAKAGE_DELTA_AUC_THRESHOLD
        except ImportError as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}: the "
                f"'prediction' submodule this constant is re-exported from "
                f"appears to have been removed from the codebase. If "
                f"prediction.py was intentionally descoped, delete this "
                f"shim entirely rather than leaving a dangling re-export."
            ) from exc
        return LEAKAGE_DELTA_AUC_THRESHOLD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Functional tier classification (primary axis)
# ---------------------------------------------------------------------------

FEATURE_TIER: Dict[str, str] = {
    # Reference — baseline comparator; always computed but NOT in FDR
    "mean_synchrony":          "reference",
    # Core — morphology-independent, cross-paradigm robust; FDR family
    "peak_amplitude":          "core",
    "dwell_time":              "core",
    "switching_rate":          "core",
    # Conditional — morphology/paradigm-dependent; FDR family
    "onset_latency":           "conditional",
    "recovery_time":           "conditional",
    "rise_time":               "conditional",
    "synchrony_entropy":       "conditional",
    "bimodality_coefficient":  "conditional",
    # Occupancy descriptor — implemented in SSoT but external status is
    # exploratory-secondary (see feature_status.py); NOT in FDR_FEATURES.
    "fraction_above_threshold": "conditional",
    # Morphology-agnostic timing descriptors — implemented in SSoT but
    # external status is exploratory-secondary (see feature_status.py);
    # NOT in FDR_FEATURES.  Definedness is paradigm-dependent (require
    # >= 3 prominent peaks / >= 1 prominent peak respectively).
    "inter_peak_cv":           "conditional",
    "first_peak_time":         "conditional",
}
"""Functional tier for every feature (Axis A of the three-axis classification).

Tiers are:
  - "core"        : morphology-independent, cross-paradigm robust
  - "conditional" : assumes SCR-like single-peak morphology
  - "reference"   : baseline comparator, always computed

MATHEMATICAL_TIER (Axis D) is the driver for null-model selection;
FEATURE_TIER (Axis A) is the external communication label.
"""


# ---------------------------------------------------------------------------
# Mathematical invariance tier (Axis D — driver axis for null model)
# ---------------------------------------------------------------------------
# This is the SOLE determinant of which null model to use in surrogate
# testing.  It is NOT derived from FEATURE_TIER (Axis A) — a feature
# can be "conditional" (Axis A) but L0 (Axis D), e.g. BC.
#
# L0 (permutation-invariant):
#   Features whose values are unchanged under permutation of WCC indices.
#   Null model: SIGNAL-LEVEL IAAFT (shuffle raw signals, recompute WCC).
#   Tests "existence of coupling beyond chance".
#
# L1 (local temporal structure):
#   Features that depend on local run-length / autoregressive structure
#   but NOT on absolute time anchors.
#   Null model: WCC-LEVEL IAAFT (shuffle WCC, preserves L0 moments).
#   Tests "incremental temporal structure beyond mean/peak".
#
# L2 (event-locked morphology):
#   Features that depend on absolute phase anchors (t=0 = stimulus onset).
#   Null model: CIRCULAR TIME-SHIFT (preserves L0+L1, destroys phase).
#   Tests "event-locked timing beyond duration/dynamics".
#
# Reference: docs/surrogate_threshold_design.md (nested null architecture).

MATHEMATICAL_TIER: Dict[str, str] = {
    # L0 — permutation-invariant (signal-level null)
    "mean_synchrony":          "L0",
    "peak_amplitude":          "L0",
    "synchrony_entropy":       "L0",
    "bimodality_coefficient":  "L0",
    "fraction_above_threshold": "L0",
    # L1 — local temporal structure (WCC-level null)
    "dwell_time":              "L1",
    "switching_rate":          "L1",
    # L2 — event-locked morphology (circular time-shift null)
    "onset_latency":           "L2",
    "rise_time":               "L2",
    "recovery_time":           "L2",
    # L2 — peak-timing descriptors (depend on the ordering and spacing of
    # threshold-crossing peaks; NOT permutation-invariant)
    "inter_peak_cv":           "L2",
    "first_peak_time":         "L2",
}
"""Mathematical invariance tier (Axis D).

This is the SOLE driver for null-model selection.
External labels (Core/Conditional/Reference) are in FEATURE_TIER (Axis A).
"""


# ---------------------------------------------------------------------------
# FDR families reorganised by mathematical tier (Axis D)
# ---------------------------------------------------------------------------
# Each family shares the SAME null model (signal-level or WCC-level IAAFT).
# Two-stage BH-FDR: first within-family, then across families
# (or: one-stage across all, with families as covariates — TBD).
#
# Family L0: existence test (signal-level null)
# Family L1: structural increment test (WCC-level null)
#
# L2 features are EXPLORATORY (not in FDR). Their peak-timing existence
# null uses a cyclic block-bootstrap (Kunsch 1989; Politis & Romano 1992);
# the existence test is deferred to v2 pending signal-level validation.
# See docs/METHOD_LOG.md and scripts/validate_timing_descriptors.py.

FDR_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "L0": (
        # peak_amplitude is the v1 primary intensity workhorse for the
        # group-condition FDR family.
        #
        # mean_synchrony is the REFERENCE comparator (Axis A): always
        # reported and still tested by the synchrony-existence audit, but
        # NOT entered into the confirmatory group-inference FDR family. It
        # remains an L0 feature mathematically (MATHEMATICAL_TIER +
        # _NULL_MODEL_L0 in dynamic_features.py).
        #
        # bimodality_coefficient is a permutation-invariant L0
        # distribution-shape descriptor used by the synchrony-existence
        # audit, but is exploratory for confirmatory group inference and
        # not in the FDR family.
        "peak_amplitude",
    ),
    "L1": (
        "dwell_time",
        "switching_rate",
    ),
}
"""FDR families grouped by mathematical invariance tier (Axis D).

Family L0: null = signal-level IAAFT (destroy all coupling)
Family L1: null = WCC-level IAAFT   (preserve L0, destroy run-length)
"""

# Backward-compat flat tuple (all FDR-family features, both tiers)
FDR_FEATURES: Tuple[str, ...] = (
    *FDR_FAMILIES["L0"],
    *FDR_FAMILIES["L1"],
    # L2 timing/event features remain EXCLUDED: their peak-timing existence
    # null (cyclic block-bootstrap) is under development and deferred to v2.
    # "onset_latency", "rise_time", "recovery_time",
)
"""Tuple of all FDR-family feature names (Axis C).

This is the flattened union of FDR_FAMILIES["L0"] and FDR_FAMILIES["L1"].
L2 features are EXCLUDED: their peak-timing existence null (cyclic block-
bootstrap, deferred to v2) is not yet validated.
"""

# FDR family (Axis C) — DELIBERATELY maintained as an explicit, independent
# tuple, NOT derived from FEATURE_TIER via comprehension. Core/Conditional
# tier membership (Axis A) is a statement about extraction robustness;
# FDR-family membership (Axis C) is a statement about whether a feature has
# cleared its own, separately-dated validation gate. Collapsing the two
# into one mechanical derivation is exactly the "promote first, validate
# after" failure mode this module's governance is meant to prevent.
# The primary group-condition FDR family is exactly
# {peak_amplitude, dwell_time, switching_rate}; mean_synchrony is a
# reported reference and bimodality_coefficient an exploratory descriptor,
# both kept as L0 features only for the synchrony-existence audit (see
# _NULL_MODEL_L0 in dynamic_features.py), a separate axis from confirmatory
# multiplicity correction.
#
# Defensive consistency check: verify every FDR entry has a FEATURE_TIER.
for _name in FDR_FEATURES:
    if _name not in FEATURE_TIER:
        raise AssertionError(
            f"FDR_FEATURES contains '{_name}', which has no entry in "
            f"FEATURE_TIER. Every FDR-family feature must have an "
            f"explicit functional-tier classification."
        )
    # NOTE: "reference" features CAN be in FDR_FEATURES. Functional tier
    #   "reference" (Axis A, "always computed, reported as baseline") and
    #   mathematical tier "L0" (Axis D, "use signal-level null") are
    #   independent axes.
    if FEATURE_TIER[_name] == "reference":
        import logging
        logging.getLogger(__name__).debug(
            "'%s' is tiered as 'reference' but listed in FDR_FEATURES "
            "(allowed: reference is Axis A, FDR membership is driven "
            "by mathematical tier Axis D).",
            _name,
        )
del _name

REFERENCE_FEATURE: Tuple[str, ...] = (
    "mean_synchrony",
)
"""Reference feature — always computed, reported alongside FDR family
but does NOT enter multiplicity correction.  Singular (mean_synchrony)
as of 2026-06-17."""

# ---------------------------------------------------------------------------
# Informational tier classification (secondary axis — organises Results)
# ---------------------------------------------------------------------------

INTENSITY_FEATURES: Tuple[str, ...] = (
    "mean_synchrony",
    "peak_amplitude",
)
"""Features reporting the magnitude of moment-to-moment coupling."""

STRUCTURE_FEATURES: Tuple[str, ...] = (
    "fraction_above_threshold",
    "dwell_time",
    "switching_rate",
    "synchrony_entropy",
    "bimodality_coefficient",
)
"""Features reporting the temporal organisation of synchrony —
sustained vs intermittent state distribution."""

TEMPORAL_FEATURES: Tuple[str, ...] = (
    "onset_latency",
    "rise_time",
    "recovery_time",
    "inter_peak_cv",
    "first_peak_time",
)
"""Features reporting the timing of synchrony events —
when episodes occur within an interaction."""

CORE_FEATURES: Tuple[str, ...] = tuple(
    name for name, tier in FEATURE_TIER.items() if tier == "core"
)
CONDITIONAL_FEATURES: Tuple[str, ...] = tuple(
    name for name, tier in FEATURE_TIER.items() if tier == "conditional"
)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class DynamicFeatures:
    """Container for FDR-family features + reference + diagnostics + definedness flags."""

    # --- FDR family (L0: 2 + L1: 3 = 5; DECISION-09 revised 2026-06-23)
    onset_latency: float = float("nan")
    rise_time: float = float("nan")
    peak_amplitude: float = float("nan")
    recovery_time: float = float("nan")
    dwell_time: float = float("nan")
    switching_rate: float = float("nan")
    synchrony_entropy: float = float("nan")

    # --- Reference (baseline comparator; report-only) ---
    mean_synchrony: float = float("nan")

    # --- Conditional (promoted 2026-06-20; has default for backward compat
    #     with pre-2026-06-20 artifacts that lack this field) ---
    bimodality_coefficient: float = float("nan")

    # --- Exploratory occupancy descriptor (implemented 2026-06-29;
    #     not in FDR family) ---
    fraction_above_threshold: float = float("nan")

    # --- Exploratory morphology-agnostic timing descriptors (wired
    #     2026-06-29; not in FDR family). May be NaN (undefined) on short
    #     or subthreshold traces; report definedness rates alongside. ---
    inter_peak_cv: float = float("nan")
    first_peak_time: float = float("nan")

    # --- Definedness flags ---
    onset_defined: int = 0
    rise_defined: int = 0
    recovery_defined: int = 0

    # --- Meta ---
    notes: str = ""
    params: Dict[str, float] = field(default_factory=dict)

    FDR_KEYS = FDR_FEATURES  # plain class attribute, NOT a dataclass field
    # Alias for FDR_FEATURES; kept because the test-suite references it.
    CONFIRMATORY_KEYS = FDR_FEATURES  # plain class attribute, NOT a field

    def to_dict(self) -> Dict[str, float]:
        """Export all feature values + definedness flags as a flat dict.

        Includes both FDR-family features (L0+L1) and non-FDR features
        (L2 event-locked + synchrony_entropy), so downstream callers such
        as ``_extract_six_features`` and Level 2/3 summarisers can access
        the full feature set without a second call to ``extract_features``.
        """
        d: Dict[str, float] = {
            k: getattr(self, k)
            for k in (
                # FDR-family (L0: mean_synchrony, peak_amplitude)
                #         + (L1: dwell_time, switching_rate, bimodality_coefficient)
                *self.FDR_KEYS,
                # Reference — always computed, in FDR L0
                "mean_synchrony",
                # L2 event-locked (exploratory; not in FDR)
                "onset_latency",
                "rise_time",
                "recovery_time",
                # L0 diagnostics (permutation-invariant; not in FDR)
                "synchrony_entropy",
                # Distribution-shape descriptor; L0 math tier, exploratory
                # (removed from FDR family 2026-06-29) but still reported.
                "bimodality_coefficient",
                "fraction_above_threshold",
                # Exploratory timing descriptors (not in FDR)
                "inter_peak_cv",
                "first_peak_time",
                # Definedness flags
                "onset_defined",
                "rise_defined",
                "recovery_defined",
            )
        }
        if self.notes:
            d["_notes"] = self.notes  # type: ignore[assignment]
        if self.params:
            d["_params"] = self.params  # type: ignore[assignment]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicFeatures":
        """Deserialize from a dict produced by :meth:`to_dict` (or compatible).

        Tolerant to:
        - Missing reference / definedness keys (defaults applied)
        - Extra unknown keys (silently ignored)

        Strict on:
        - The FDR-family keys MUST be present (subject to the
          backward-compatibility default exception below).  Pre-lock-in
          v2 artifacts lacking ``dwell_time``, ``switching_rate``, or
          ``synchrony_entropy`` will raise ``KeyError`` to force
          migration (see DECISION_LOG.md).
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"DynamicFeatures.from_dict expected dict, got "
                f"{type(data).__name__}"
            )

        known = {
            *cls.FDR_KEYS,
            "mean_synchrony",
            "bimodality_coefficient",
            "fraction_above_threshold",
            "inter_peak_cv",
            "first_peak_time",
            "onset_defined",
            "rise_defined",
            "recovery_defined",
        }
        kwargs: Dict[str, Any] = {k: data[k] for k in known if k in data}

        # FDR keys without a dataclass default are strictly required.
        # Fields with a default (e.g. bimodality_coefficient, added
        # 2026-06-20) are tolerated as missing for backward compatibility
        # with pre-promotion artifacts.
        _fields_with_defaults = {
            f.name for f in _dc_fields(cls)
            if f.default is not _DC_MISSING
        }
        missing = [
            k for k in cls.FDR_KEYS
            if k not in kwargs and k not in _fields_with_defaults
        ]
        if missing:
            raise KeyError(
                f"DynamicFeatures.from_dict: missing required key(s) "
                f"{missing}.  This may indicate a pre-lock-in v2 artifact; "
                f"see docs/DECISION_LOG.md for migration guidance."
            )

        notes = data.get("_notes", "") or ""
        params = data.get("_params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        return cls(notes=notes, params=params, **kwargs)


# ---------------------------------------------------------------------------
# Smoothed peak (DECISION-04)
# ---------------------------------------------------------------------------

def smoothed_wcc(wcc: np.ndarray, window: int = PEAK_SMOOTHING_WINDOW) -> np.ndarray:
    """3-point boxcar smoothing with same-mode boundary (DECISION-04)."""
    kernel = np.ones(window) / window
    return np.convolve(wcc, kernel, mode="same")


def find_dominant_peak(wcc_smoothed: np.ndarray) -> Optional[int]:
    """Return index of the dominant (= global argmax) smoothed peak,
    or ``None`` if all values are NaN."""
    if not np.isfinite(wcc_smoothed).any():
        return None
    return int(np.nanargmax(wcc_smoothed))


# ---------------------------------------------------------------------------
# DECISION-02 · onset_latency helpers
# ---------------------------------------------------------------------------

def _sustained_crossing_index(above: np.ndarray, k: int) -> Optional[int]:
    """First index i such that ``above[i : i+k]`` is all True AND
    there exists j < i with ``above[j]`` False (a baseline phase exists)."""
    n = above.size
    if n < k:
        return None
    if not (~above).any():  # entire trace elevated -> no baseline
        return None

    seen_baseline = False
    run = 0
    for i in range(n):
        if not above[i]:
            seen_baseline = True
            run = 0
            continue
        run += 1
        if seen_baseline and run >= k:
            return i - k + 1
    return None


# ---------------------------------------------------------------------------
# DECISION-02 · onset_latency
# ---------------------------------------------------------------------------
# K = 5% of WCC window, computed in seconds then converted to samples,
# so the sustained-crossing duration is invariant to sampling rate.

def compute_onset_latency(
    wcc: np.ndarray,
    hz: float,
    wcc_window_sec: float,
    threshold: float = ONSET_THRESHOLD,
) -> Tuple[float, int]:
    """
    DECISION-02 · onset_latency.

    Definition
    ----------
    The first transition from a baseline phase (WCC < threshold) to a
    sustained elevated phase (WCC >= threshold for at least K consecutive
    samples).  K is scaled to 5% of the WCC window length (in seconds,
    then converted to samples)::

        k_seconds = max(1.0, 0.05 * wcc_window_sec)
        k = max(2, round(k_seconds * hz))

    Returns
    -------
    (latency_seconds, defined_flag)
        - latency_seconds : float
            Seconds from start to the first sustained crossing.  NaN if
            undefined.
        - defined_flag : int (0/1)
            1 iff the trace exhibits both a baseline and a sustained
            elevated phase.  0 means scientifically undefined (NOT a bug).
    """
    finite = np.isfinite(wcc)
    if not finite.any():
        return float("nan"), 0
    above = (wcc >= threshold) & finite
    k_seconds = max(1.0, 0.05 * wcc_window_sec)
    k = max(2, int(round(k_seconds * hz)))
    idx = _sustained_crossing_index(above, k)
    if idx is None:
        return float("nan"), 0
    return float(idx) / hz, 1


# ---------------------------------------------------------------------------
# DECISION-03 · rise_time
# ---------------------------------------------------------------------------

def compute_rise_time(
    wcc: np.ndarray,
    peak_index: int,
    peak_value: float,
    hz: float,
    baseline: float = ONSET_THRESHOLD,
) -> Tuple[float, int]:
    """
    DECISION-03 · rise_time.

    Quartile rise time on the segment ``wcc[: peak_index + 1]``::

        level_25 = baseline + 0.25 * (peak_value - baseline)
        level_75 = baseline + 0.75 * (peak_value - baseline)
        rise_time = (t_75 - t_25) / hz

    Notes
    -----
    Assumes single-peak morphology. On oscillatory traces (57% of Lerique),
    captures a phase segment rather than genuine synchrony build-up (ρ≈0.14).
    Tier: CONDITIONAL.
    """
    if not np.isfinite(peak_value):
        return float("nan"), 0
    amp = peak_value - baseline
    if amp <= 0:
        return float("nan"), 0

    level_25 = baseline + RISE_LOW_FRAC * amp
    level_75 = baseline + RISE_HIGH_FRAC * amp

    seg = wcc[: peak_index + 1]
    above_25 = np.where((seg >= level_25) & np.isfinite(seg))[0]
    if above_25.size == 0:
        return float("nan"), 0
    t_25 = int(above_25[0])

    seg2 = wcc[t_25 : peak_index + 1]
    above_75 = np.where((seg2 >= level_75) & np.isfinite(seg2))[0]
    if above_75.size == 0:
        return float("nan"), 0
    t_75 = t_25 + int(above_75[0])

    duration_samples = t_75 - t_25
    if duration_samples == 0:
        return float("nan"), 0  # noise-spike guard: see docstring
    return float(duration_samples) / hz, 1


# ---------------------------------------------------------------------------
# DECISION-04 · peak_amplitude
# ---------------------------------------------------------------------------

def compute_peak_amplitude(wcc_smoothed: np.ndarray) -> Tuple[float, Optional[int]]:
    """DECISION-04 · peak_amplitude = max of 3-point smoothed WCC.

    Returns ``(peak_value, peak_index)``.  If all NaN, returns ``(NaN, None)``.
    """
    idx = find_dominant_peak(wcc_smoothed)
    if idx is None:
        return float("nan"), None
    return float(wcc_smoothed[idx]), idx


# ---------------------------------------------------------------------------
# DECISION-05 · recovery_time (half-recovery)
# ---------------------------------------------------------------------------

def compute_recovery_time(
    wcc: np.ndarray,
    peak_index: int,
    peak_value: float,
    hz: float,
    baseline: float = ONSET_THRESHOLD,
) -> Tuple[float, int]:
    """
    DECISION-05 · half-recovery time.

    From the dominant peak, time until WCC drops to::

        half_level = baseline + 0.5 * (peak_value - baseline)

    Notes
    -----
    On oscillatory traces, WCC drops below half_level as part of the
    oscillation cycle, not synchrony decay (ρ≈0.47). Tier: CONDITIONAL.
    """
    if not np.isfinite(peak_value):
        return float("nan"), 0
    amp = peak_value - baseline
    if amp <= 0:
        return float("nan"), 0

    half_level = baseline + RECOVERY_FRAC * amp
    post_peak = wcc[peak_index:]
    below = np.where((post_peak <= half_level) & np.isfinite(post_peak))[0]
    if below.size == 0:
        return float("nan"), 0
    return float(below[0]) / hz, 1


# ---------------------------------------------------------------------------
# DECISION-06 · shared state binarization (Schmitt trigger)
# ---------------------------------------------------------------------------

def _binarize_with_hysteresis(
    wcc: np.ndarray,
    threshold: float,
    hysteresis_delta: float = SWITCHING_HYSTERESIS_DELTA,
) -> np.ndarray:
    """Schmitt-trigger binarization of WCC into elevated / baseline states.

    State becomes ``True`` when WCC >= ``threshold + delta``,
    becomes ``False`` when WCC < ``threshold - delta``,
    and retains the previous state inside the hysteresis band.

    This eliminates spurious boundary crossings on oscillatory traces
    that straddle the threshold — the primary source of noise in
    :func:`compute_switching_rate` (DECISION-06b, revised 2026-06-20).

    Parameters
    ----------
    wcc : np.ndarray
        WCC time series (may contain NaN).
    threshold : float
        Centre of the hysteresis band.
    hysteresis_delta : float
        Half-width of the hysteresis band.  ``0.0`` recovers the legacy
        non-hysteresis binarization (``WCC >= threshold``).

    Returns
    -------
    np.ndarray[bool]
        Boolean state array (``True`` = elevated).  NaN positions are
        ``False``.
    """
    finite = np.isfinite(wcc)
    n = wcc.shape[0]
    states = np.zeros(n, dtype=bool)
    if not finite.any() or n == 0:
        return states

    if hysteresis_delta <= 0:
        return (wcc >= threshold) & finite

    enter = threshold + hysteresis_delta
    exit_ = threshold - hysteresis_delta
    state = False
    for i in range(n):
        if not finite[i]:
            states[i] = state
            continue
        if not state and wcc[i] >= enter:
            state = True
        elif state and wcc[i] < exit_:
            state = False
        states[i] = state
    return states


# ---------------------------------------------------------------------------
# DECISION-06a · dwell_time
# ---------------------------------------------------------------------------

def compute_dwell_time(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    hysteresis_delta: float = SWITCHING_HYSTERESIS_DELTA,
) -> float:
    """DECISION-06a · dwell_time = mean elevated run-length (seconds).

    Binarize WCC via Schmitt-trigger hysteresis (DECISION-06b, revised
    2026-06-20), run-length encode, take the arithmetic mean over
    elevated runs.  Returns NaN if there are zero elevated runs.

    Parameters
    ----------
    hysteresis_delta : float
        Half-width of the hysteresis band.  ``0.0`` recovers legacy
        non-hysteresis binarization.  See
        :data:`SWITCHING_HYSTERESIS_DELTA`.

    Notes
    -----
    Hysteresis eliminates boundary jitter near ``threshold``, producing
    more stable dwell estimates on oscillatory traces.
    """
    finite = np.isfinite(wcc)
    if not finite.any():
        return float("nan")
    above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)
    if not above.any():
        return float("nan")

    padded = np.concatenate(([False], above, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    run_lengths = ends - starts
    if run_lengths.size == 0:
        return float("nan")
    return float(np.mean(run_lengths)) / hz


# ---------------------------------------------------------------------------
# DECISION-06b · switching_rate
# ---------------------------------------------------------------------------

def compute_switching_rate(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    hysteresis_delta: float = SWITCHING_HYSTERESIS_DELTA,
) -> float:
    """DECISION-06b · switching_rate = state transitions per minute.

    Binarize WCC via Schmitt-trigger hysteresis (revised 2026-06-20),
    count both ``False->True`` and ``True->False`` transitions, normalize
    by duration in minutes.

    Parameters
    ----------
    hysteresis_delta : float
        Half-width of the hysteresis band.  ``0.0`` recovers legacy
        non-hysteresis binarization.  See
        :data:`SWITCHING_HYSTERESIS_DELTA`.

    Notes
    -----
    The hysteresis band (default ±0.05) eliminates boundary jitter from
    oscillatory traces straddling threshold — the primary noise source
    in the pre-2026-06-20 implementation.  PGT-2 validation showed
    switching_rate Spearman ρ improved from ~0.22 to substantially
    higher values after this fix.
    """
    finite = np.isfinite(wcc)
    if not finite.any():
        return float("nan")
    above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)
    if above.size < 2:
        return float("nan")
    transitions = int(np.sum(above[1:] != above[:-1]))
    duration_min = above.size / hz / 60.0
    if duration_min == 0:
        return float("nan")
    return float(transitions) / duration_min


# ---------------------------------------------------------------------------
# Reference features (NOT in FDR family — always computed, report-only)
# ---------------------------------------------------------------------------

def compute_mean_synchrony(wcc: np.ndarray) -> float:
    """Reference: arithmetic mean over finite WCC values."""
    finite = wcc[np.isfinite(wcc)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def compute_fraction_above_threshold(
    wcc: np.ndarray,
    threshold: float = ONSET_THRESHOLD,
) -> float:
    """Exploratory occupancy: fraction of finite WCC values >= threshold.

    This is a permutation-invariant L0 descriptor: reordering the WCC trace
    leaves the value unchanged.  It reports *coverage* of above-threshold
    synchrony, not episode duration or switching structure.  It is therefore
    complementary to ``dwell_time`` and ``switching_rate`` but should be
    interpreted cautiously because it is threshold-dependent and often
    redundant with mean/peak synchrony.

    Not in ``FDR_FEATURES`` in v1.
    """
    finite = np.isfinite(wcc)
    n_finite = int(np.sum(finite))
    if n_finite == 0:
        return float("nan")
    return float(np.sum((wcc >= threshold) & finite) / n_finite)


def compute_synchrony_entropy(wcc: np.ndarray, n_bins: int = 20) -> float:
    """Conditional: Shannon entropy of WCC amplitude distribution.

    Enters the FDR family (DECISION-09, revised 2026-06-17).
    Bridges Structure and Temporal information dimensions.

    The histogram range is data-adaptive: ``[finite.min(), finite.max()]``
    rather than the theoretical ``[-1, 1]``.  This ensures all ``n_bins``
    bins are informative — with physiological WCC typically spanning
    [-0.2, 0.9], a fixed [-1, 1] range leaves >50% of bins empty and
    depresses sensitivity.
    """
    finite = wcc[np.isfinite(wcc)]
    if finite.size < 10:
        return float("nan")
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return float("nan")
    counts, _ = np.histogram(finite, bins=n_bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    if p.size < 2:
        return float("nan")
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Bimodality Coefficient (diagnostic; DECISION-17)
# ---------------------------------------------------------------------------

def compute_bimodality_coefficient(wcc: np.ndarray) -> float:
    """Bimodality Coefficient (BC) of the WCC amplitude distribution.

    .. math::
        BC = \\frac{\\gamma^2 + 1}{\\kappa}

    where :math:`\\gamma` is skewness and :math:`\\kappa` is the
    (non-excess) kurtosis.  BC > 0.555 indicates a bimodal distribution
    (Ellison 1987; Pfister et al. 2013).

    For alternating high/low coupling (PGT-2), BC directly measures the
    separability of the two synchrony states.  Unlike Shannon entropy
    (which is trajectory-blind and conflates state count with state
    occupancy), BC is sensitive to whether the WCC distribution exhibits
    two distinct modes.

    Tier: CONDITIONAL (set 2026-06-20). FDR-family membership: REMOVED from
    the confirmatory group-condition FDR family on 2026-06-29 (Option B),
    because its membership was provisional and lacked dated, pre-decision
    cross-paradigm evidence. It is retained as a permutation-invariant L0
    distribution-shape descriptor for the synchrony-existence audit
    (MATHEMATICAL_TIER + _NULL_MODEL_L0 in dynamic_features.py), but is
    exploratory for confirmatory group inference.
    """
    finite = wcc[np.isfinite(wcc)]
    if finite.size < 10:
        return float("nan")
    from scipy.stats import skew, kurtosis
    sk = float(skew(finite))
    kt_excess = float(kurtosis(finite))  # scipy returns excess kurtosis
    kurt = kt_excess + 3.0  # convert to proper kurtosis
    if kurt <= 0:
        return float("nan")
    return (sk ** 2 + 1.0) / kurt


# ---------------------------------------------------------------------------
# DECISION-17 · Morphology-agnostic timing features
# ---------------------------------------------------------------------------


def compute_first_peak_time(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    min_prominence: float = 0.15,
) -> float:
    """Time of the first prominent peak above threshold (seconds).

    Independent of single-peak assumption — meaningful for oscillatory,
    single-peak, and sustained morphologies alike.  A peak is defined as
    a local maximum within :attr:`ONSET_THRESHOLD`-exceeding segments
    whose prominence (height above adjacent troughs) exceeds
    ``min_prominence``.

    Returns NaN if no prominent peak exists (subthreshold traces).
    """
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any() or n < 3:
        return float("nan")
    x = np.where(finite, wcc, -np.inf)
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            left_min = np.min(x[max(0, i - 1)::-1][:50]) if i > 0 else x[i]
            right_min = np.min(x[i + 1: i + 51]) if i + 1 < n else x[i]
            base = max(left_min, right_min)
            if x[i] - base >= min_prominence:
                return float(i) / hz
    return float("nan")


def compute_baseline_fraction(
    wcc: np.ndarray,
    threshold: float = ONSET_THRESHOLD,
    min_prominence: float = 0.15,
) -> float:
    """Fraction of samples below threshold *before* the first prominent peak.

    High values (~1.0) indicate a prolonged baseline period (single-peak
    morphology).  Low values (~0.0) indicate the trace starts above
    threshold (sustained).  Intermediate values indicate intermittent
    early crossings (oscillatory).

    Returns NaN if no prominent peak exists.
    """
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any() or n < 3:
        return float("nan")
    x = np.where(finite, wcc, -np.inf)
    first_peak_idx = -1
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            left_min = np.min(x[max(0, i - 1)::-1][:50]) if i > 0 else x[i]
            right_min = np.min(x[i + 1: i + 51]) if i + 1 < n else x[i]
            base = max(left_min, right_min)
            if x[i] - base >= min_prominence:
                first_peak_idx = i
                break
    if first_peak_idx < 1:
        return float("nan")
    above = (wcc[:first_peak_idx] >= threshold) & finite[:first_peak_idx]
    n_pre = len(above)
    if n_pre == 0:
        return float("nan")
    return float((~above).mean())


def compute_inter_peak_cv(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    min_prominence: float = 0.15,
    min_peaks: int = 3,
) -> float:
    """Coefficient of variation of inter-peak intervals (CV = std / mean).

    Requires >= ``min_peaks`` prominent peaks (default 3, i.e. >= 2 gaps).
    Low CV (~0.0-0.3) indicates REGULAR oscillation; high CV (~0.6+)
    indicates IRREGULAR intermittency. This is the most direct
    morphological descriptor of Kelso-style metastable coordination
    dynamics.

    The minimum of 3 peaks (rather than 2) and the use of the unbiased
    sample standard deviation (``ddof=1``) are deliberate: with exactly
    2 peaks there is only 1 gap, making CV undefined in any meaningful
    sense, and with exactly 2 gaps (3 peaks) the population-standard-
    deviation convention (``ddof=0``) and the sample convention
    (``ddof=1``) disagree by a large, sample-size-dependent factor.
    Requiring 3 peaks plus ``ddof=1`` reduces (but does not eliminate)
    this small-n instability; report definedness rates alongside this
    feature, as with all morphology-agnostic timers.

    Returns NaN if fewer than ``min_peaks`` peaks exist.
    """
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any() or n < 3:
        return float("nan")
    x = np.where(finite, wcc, -np.inf)
    all_peaks = []
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            left_min = np.min(x[max(0, i - 1)::-1][:50]) if i > 0 else x[i]
            right_min = np.min(x[i + 1: i + 51]) if i + 1 < n else x[i]
            base = max(left_min, right_min)
            if x[i] - base >= min_prominence:
                all_peaks.append(i)
    if len(all_peaks) < max(min_peaks, 2):
        return float("nan")
    gaps = np.diff(all_peaks).astype(float) / hz
    mean_gap = gaps.mean()
    if mean_gap <= 0:
        return float("nan")
    ddof = 1 if gaps.size > 1 else 0
    return float(gaps.std(ddof=ddof) / mean_gap)


# ---------------------------------------------------------------------------
# Surrogate-derived threshold (DECISION-01 revised 2026-06-21)
# ---------------------------------------------------------------------------

def compute_surrogate_threshold(
    wcc_surrogates: np.ndarray,
    percentile: float = SURROGATE_THRESHOLD_PERCENTILE,
) -> Tuple[float, bool]:
    """Compute a per-dyad surrogate-derived onset threshold.

    The threshold is the ``percentile``-th quantile of all finite WCC values
    across ``n_surrogates`` IAAFT-randomised WCC series.  Its semantic:
    "the WCC level this dyad would reach by chance (at the given false-positive
    rate)" — a zero-hypothesis-grounded cut-off rather than an arbitrary
    r-metric anchor.

    Methodological lineage: Lykken & Venables (1971), Ben-Shakhar (1985).
    See docs/surrogate_threshold_design.md for full rationale.

    Parameters
    ----------
    wcc_surrogates : np.ndarray, shape (n_surrogates, n_timepoints)
        2-D array of WCC series computed on IAAFT-randomised signal pairs.
        Each row is one surrogate replicate.
    percentile : float, optional
        Quantile to use (default 95).  Set to 90 for a more liberal threshold.

    Returns
    -------
    Tuple[float, bool]
        ``(threshold, is_surrogate_derived)``. ``threshold`` falls back to
        ``ONSET_THRESHOLD`` (0.5) if fewer than 10 finite surrogate values
        are available (degenerate case); ``is_surrogate_derived`` is
        ``False`` exactly when this fallback fired, so callers can flag
        which dyads received a data-driven threshold and which received
        the fixed fallback — this distinction MUST be reported alongside
        any dwell_time / switching_rate / onset_latency computed under
        the surrogate-threshold specification (cf. the explicit
        ``_defined`` flag convention used elsewhere in this module).

    Notes
    -----
    **Session-level pooling** (DECISION-01r): pool ALL timepoints across ALL
    surrogate replicates before computing the quantile.  This gives a single
    threshold per session, preserving cross-condition comparability (Task A
    in surrogate_threshold_design.md).

    For condition-level thresholds (sensitivity analysis), call this function
    separately for each condition's surrogate WCC slice.
    """
    wcc_surrogates = np.asarray(wcc_surrogates, dtype=float)
    if wcc_surrogates.ndim == 1:
        wcc_surrogates = wcc_surrogates.reshape(1, -1)
    finite = wcc_surrogates[np.isfinite(wcc_surrogates)]
    if finite.size < 10:
        return ONSET_THRESHOLD, False  # degenerate fallback, explicitly flagged
    return float(np.percentile(finite, percentile)), True


# ---------------------------------------------------------------------------
# High-level entry: extract all features + diagnostics
# ---------------------------------------------------------------------------

def extract_features(
    wcc: np.ndarray,
    hz: float,
    wcc_window_sec: float,
    threshold: float = ONSET_THRESHOLD,
    paradigm: str = "auto",
) -> DynamicFeatures:
    """Compute features + diagnostics from a WCC series.

    This is THE single entry point for feature extraction across
    SyncPipe.  Both ``multisync.dynamic_features.extract_dynamic_features``
    and ``multisync.validation.recovery._run_single_cell`` MUST delegate
    here.

    Parameters
    ----------
    wcc : 1-D array
        Windowed cross-correlation series (may contain NaN).
    hz : float
        Sampling rate of the WCC series (Hz).
    wcc_window_sec : float
        Length of the WCC window in seconds; used to scale the
        sustained-crossing length K (DECISION-02).
    threshold : float, optional
        Onset / dwell / switching threshold.  Defaults to DECISION-01 (0.5).
        Override only for sensitivity analysis.
    paradigm : str, optional
        "event" — all features computed (event-locked design).
        "continuous" — rise_time and recovery_time are set to NaN (they
        require a single dominant onset→peak→recovery cycle; in continuous
        mode the multiple-episode structure is captured by dwell/switching
        instead).
        "auto" (default) — identical to "event".  Kept for backward
        compatibility; prefer explicitly specifying "event" or
        "continuous" in new code.
        DECISION-16 (2026-06-03).

    Returns
    -------
    DynamicFeatures
    """
    wcc = np.asarray(wcc, dtype=float)

    # Smoothed peak first (DECISION-04) -- anchors rise/recovery indexing
    sm = smoothed_wcc(wcc)
    peak_value, peak_idx = compute_peak_amplitude(sm)

    # Onset is decoupled from peak (DECISION-08)
    onset_lat, onset_def = compute_onset_latency(
        wcc, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold,
    )

    if peak_idx is not None:
        rise_t, rise_def = compute_rise_time(
            wcc, peak_index=peak_idx, peak_value=peak_value,
            hz=hz, baseline=threshold,
        )
        rec_t, rec_def = compute_recovery_time(
            wcc, peak_index=peak_idx, peak_value=peak_value,
            hz=hz, baseline=threshold,
        )
    else:
        rise_t, rise_def, rec_t, rec_def = float("nan"), 0, float("nan"), 0

    dwell = compute_dwell_time(wcc, hz=hz, threshold=threshold)
    switch = compute_switching_rate(wcc, hz=hz, threshold=threshold)

    mean_s = compute_mean_synchrony(wcc)
    frac_above = compute_fraction_above_threshold(wcc, threshold=threshold)
    entropy = compute_synchrony_entropy(wcc)
    bc = compute_bimodality_coefficient(wcc)
    ipc = compute_inter_peak_cv(wcc, hz=hz, threshold=threshold)
    fpt = compute_first_peak_time(wcc, hz=hz, threshold=threshold)

    # Undefined onset/rise/recovery → filled with wcc_window_sec (conservative
    # upper bound). _defined flags encode structural distinction for downstream
    # classifiers.
    if not onset_def:
        onset_lat = float(wcc_window_sec)
    if not rise_def:
        rise_t = float(wcc_window_sec)
    if not rec_def:
        rec_t = float(wcc_window_sec)

    notes: list[str] = []

    # DECISION-16: paradigm-aware feature reporting
    if paradigm == "continuous":
        rise_t = float("nan")
        rec_t = float("nan")
        rise_def = 0
        rec_def = 0
        notes.append("rise/recovery set NaN (continuous paradigm)")

    return DynamicFeatures(
        onset_latency=onset_lat,
        rise_time=rise_t,
        peak_amplitude=peak_value,
        recovery_time=rec_t,
        dwell_time=dwell,
        switching_rate=switch,
        mean_synchrony=mean_s,
        synchrony_entropy=entropy,
        bimodality_coefficient=bc,
        fraction_above_threshold=frac_above,
        inter_peak_cv=ipc,
        first_peak_time=fpt,
        onset_defined=int(onset_def),
        rise_defined=int(rise_def),
        recovery_defined=int(rec_def),
        notes="; ".join(notes),
        params={
            "threshold": float(threshold),
            "hz": float(hz),
            "wcc_window_sec": float(wcc_window_sec),
        },
    )


__all__ = [
    # Constants
    "ONSET_THRESHOLD",
    "SURROGATE_THRESHOLD_PERCENTILE",
    "PEAK_SMOOTHING_WINDOW",
    "RISE_LOW_FRAC",
    "RISE_HIGH_FRAC",
    "RECOVERY_FRAC",
    "SWITCHING_HYSTERESIS_DELTA",
    "compute_surrogate_threshold",
    # Functional tier (primary classification)
    "FEATURE_TIER",
    "FDR_FEATURES",
    "REFERENCE_FEATURE",
    "CORE_FEATURES",
    "CONDITIONAL_FEATURES",
    # Informational tier (secondary classification)
    "INTENSITY_FEATURES",
    "STRUCTURE_FEATURES",
    "TEMPORAL_FEATURES",
    # Container
    "DynamicFeatures",
    # Helpers
    "smoothed_wcc",
    "find_dominant_peak",
    "_binarize_with_hysteresis",
    # Individual feature computations
    "compute_onset_latency",
    "compute_rise_time",
    "compute_peak_amplitude",
    "compute_recovery_time",
    "compute_dwell_time",
    "compute_switching_rate",
    "compute_mean_synchrony",
    "compute_fraction_above_threshold",
    "compute_synchrony_entropy",
    # Bimodality Coefficient (CONDITIONAL, promoted 2026-06-20)
    "compute_bimodality_coefficient",
    # Morphology-agnostic timers (DECISION-17)
    "compute_first_peak_time",
    "compute_baseline_fraction",
    "compute_inter_peak_cv",
    # High-level entry
    "extract_features",
]

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

Four-Axis Classification System
---------------------------------------------
Features are classified along four INDEPENDENT axes. Axis C (FDR
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
  L2 (event-locked / peak-timing morphology): onset_latency, rise_time,
                             recovery_time, first_peak_time, inter_peak_cv
    → v1 status: exploratory descriptors. Their existence null is not validated
      for confirmatory use in v1; see ``docs/METHOD_LOG.md``.
  Axis D guides null-model selection where a validated null exists.
  Axes A/B/C are external communication labels (functional, informational, FDR).
  A feature's mathematical tier is NOT derived from its functional tier.

Functional tiers:
  REFERENCE  (1 feature, report-only, never FDR-eligible)
    mean_synchrony

  EXPLORATORY OCCUPANCY (implemented, not FDR)
    fraction_above_threshold  Fraction of finite WCC values >= threshold;
                              permutation-invariant coverage descriptor

  CORE       (3 implemented primary v1 descriptors; FDR family when the
              thresholding scheme is group-comparable)
    peak_amplitude        [Intensity]
    dwell_time            [Structure — stability pole]
    switching_rate        [Structure — flexibility pole]

  CONDITIONAL / EXPLORATORY (implemented, reported with restrictions;
                             NOT automatically FDR-family members)
    onset_latency         [Temporal]  Event-locked only
    recovery_time         [Temporal]  Event-locked only
    rise_time             [Temporal]  Event-locked only
    synchrony_entropy     [Structure] Distribution diversity diagnostic
    bimodality_coefficient [Structure] Distribution-shape diagnostic
    fraction_above_threshold [Occupancy] Threshold coverage descriptor
    first_peak_time / inter_peak_cv [Timing] Exploratory morphology descriptors

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

import logging
from dataclasses import dataclass, field, fields as _dc_fields, MISSING as _DC_MISSING
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Locked constants (DECISION-01, DECISION-03, DECISION-04, DECISION-05)
# ---------------------------------------------------------------------------

ONSET_THRESHOLD: float = 0.5
"""
**Canonical v1 onset threshold = per-modality pooled IAAFT surrogate
threshold** (see :func:`compute_session_pooled_thresholds_by_modality`).

The primary scientific path (``records_to_inference_inputs`` /
``BatchComputationPipeline`` with ``onset_threshold="session_pooled"``) derives
one surrogate threshold *per modality* by pooling IAAFT surrogates across all
dyads of that modality. This calibrates the episode threshold to each
modality's null distribution — slow/smooth signals (e.g. EDA, low WCC
amplitude) and fast/spiky signals (e.g. ECG, high WCC amplitude) therefore get
different, modality-appropriate thresholds, while every dyad of a modality
still shares one threshold for cross-dyad comparability.

This constant (0.5) is the **fallback / sensitivity value only**: it is used
when a modality's pooled null is degenerate (too few dyads), and as the fixed
baseline in sensitivity sweeps such as [0.3, 0.7] and paper reproductions.
"""

SURROGATE_THRESHOLD_PERCENTILE: float = 95.0
"""percentile used for surrogate-derived threshold (default 95th).

The surrogate null distribution for a dyad is built by computing WCC on
``n_surrogates`` IAAFT-randomised signal pairs.  The threshold is the
``SURROGATE_THRESHOLD_PERCENTILE``-th quantile of all surrogate WCC values,
representing the highest WCC level reachable by chance at the given false-
positive rate.
"""

SURROGATE_THRESHOLD_MAX: float = 0.9
"""Hard sanity ceiling for a surrogate-derived onset threshold (BUG-3).

A null 95th-percentile WCC at or above this level is almost always an
*artifact* of the surrogate method, not a genuine "high sync by chance"
level: IAAFT surrogates preserve the autocorrelation/periodicity of the
ORIGINAL signals, so for periodic or strongly autocorrelated data the
surrogate WCC distribution is shifted upward.  When the derived threshold
exceeds this bound we fail loud and fall back to :data:`ONSET_THRESHOLD`
rather than silently using a contaminated cut-off.
"""

PEAK_SMOOTHING_WINDOW: int = 3
"""DECISION-04: 3-point boxcar smoothing for peak detection (Boucsein 2012)."""

RISE_LOW_FRAC: float = 0.25
RISE_HIGH_FRAC: float = 0.75
"""DECISION-03: 25%-75% quartile rise time (Boucsein 2012)."""

RECOVERY_FRAC: float = 0.50
"""DECISION-05: half-recovery time (Boucsein 2012; Dawson et al. 2007)."""

# ---------------------------------------------------------------------------
# B3 eligibility thresholds (frozen 2026-07-21)
# ---------------------------------------------------------------------------

T_DEF_MIN_WCC_POINTS: int = 3
"""DECISION-04 hard floor: minimum finite WCC sampling points per dyad.

Episode features — onset_latency, rise_time, recovery_time, first_peak_time,
inter_peak_cv (Axis D L2) — require a geometrically distinguishable
onset -> peak -> recovery sequence.  The peak-detection step uses a 3-point
boxcar (:data:`PEAK_SMOOTHING_WINDOW`, DECISION-04), RISE is interpolated on
the 25%-75% span (:data:`RISE_LOW_FRAC`/:data:`RISE_HIGH_FRAC`), and RECOVERY
uses the 50% span (:data:`RECOVERY_FRAC`).  A WCC trajectory with fewer than 3
finite points cannot support onset + peak + recovery as three *separable*
points, so episode-feature extraction is mathematically undefined (returns
NaN + an ineligible flag) rather than silently degenerate.  ``T_def = 3`` is
a hard floor of the feature definition, NOT a tunable default.
"""

N_MIN_DYADS_FDR: int = 10
"""Minimum dyad count for meaningful BH-FDR correction (B3 freeze).

BH-FDR rejection resolution is discrete: with N=10 dyads the smallest
non-zero p-value is ~0.1, so at alpha=0.05 a group cannot reject the null
unless p=0 exactly.  The codebase already treats "< 10" as a small-sample
boundary — :func:`compute_surrogate_threshold` falls back to ONSET_THRESHOLD
when "fewer than 10 finite surrogate values" are available (degenerate-case
branch at the surrogate-threshold derivation).  B4 bake-off shows Lerique
N=31 LOO is 100% stable; the three real datasets have N=176/46/23 — all >10.
Thus ``n_min = 10`` only excludes absurdly small pilots and constrains no
real SyncPipe analysis.  Groups below this floor are WARNING-flagged as
unreliable, never silently accepted.
"""


def check_eligibility(
    n_wcc_points: int,
    n_dyads: int,
) -> Tuple[bool, bool]:
    """Lightweight eligibility gate for SyncPipe analysis (B3 freeze).

    Pure, dependency-free check of the two frozen B3 floors:

    * ``wcc_points_ok`` — True iff ``n_wcc_points >= T_DEF_MIN_WCC_POINTS``.
      A dyad whose WCC trajectory has fewer than 3 finite sampling points
      cannot define episode features (peak/recovery need onset + peak +
      recovery as three separable points); it is *episode-feature ineligible*.
    * ``n_dyads_ok`` — True iff ``n_dyads >= N_MIN_DYADS_FDR``.  A group with
      fewer than 10 dyads yields BH-FDR results of uninterpretable
      resolution at alpha=0.05; such results are WARNING-flagged as
      unreliable.

    Parameters
    ----------
    n_wcc_points : int
        Number of finite WCC sampling points for a single dyad.
    n_dyads : int
        Number of dyads in the dataset / analysis group.

    Returns
    -------
    Tuple[bool, bool]
        ``(wcc_points_ok, n_dyads_ok)``.

    Examples
    --------
    >>> check_eligibility(2, 11)
    (False, True)
    >>> check_eligibility(3, 10)
    (True, True)
    """
    wcc_points_ok = n_wcc_points >= T_DEF_MIN_WCC_POINTS
    n_dyads_ok = n_dyads >= N_MIN_DYADS_FDR
    return (wcc_points_ok, n_dyads_ok)


SWITCHING_HYSTERESIS_DELTA: float = 0.05
"""Hysteresis band for state binarization.

WCC values within ``[threshold - delta, threshold + delta)`` retain the
previous state (Schmitt trigger logic), eliminating boundary jitter from
oscillatory traces straddling the threshold.  Default 0.05 in WCC units
(r-metric); set to 0.0 to recover the legacy non-hysteresis behaviour.
"""

DEFAULT_PROMINENCE_WINDOW_SEC: float = 50.0
"""Default look-back / look-ahead window (seconds) for peak prominence.

Used by morphology-agnostic timing features to locate local minima on
either side of a candidate peak.  Converted to samples via
``max(1, round(DEFAULT_PROMINENCE_WINDOW_SEC * hz))`` so that the
physical window size is independent of the sampling rate.
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
# Private utilities (shared low-level helpers)
# ---------------------------------------------------------------------------

def _find_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find runs of True in a boolean array (shared run-length detector).

    Consolidates the parallel diff-based run detectors that previously lived
    in ``compute_dwell_time`` (this module), ``extract_episodes``
    (morphology), ``state_transition_shuffle_surrogate`` (surrogate), and the
    NaN-gap check (qc).  Given a boolean ``mask``, returns ``(starts, ends)``
    where ``starts[k]`` is the first ``True`` index of the k-th run and
    ``ends[k]`` is the one-past-last ``True`` index, so that
    ``mask[starts[k]:ends[k]]`` is the k-th run and ``ends[k] - starts[k]`` its
    length.  An empty or all-False mask yields two empty arrays.

    Implementation: pad with ``False`` at both ends and detect
    ``False->True`` (diff == +1) and ``True->False`` (diff == -1) transitions
    via ``np.diff``.  The fixed ``[False]`` sentinels are essential — using
    ``[not mask[0]]`` would inject a phantom transition when the trace starts
    or ends in the ``True`` state.  This is robust to runs spanning the array
    boundary and to empty masks.
    """
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], m, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return starts, ends


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
# L2 (event-locked / peak-timing morphology):
#   Features that depend on absolute phase anchors or peak ordering/spacing.
#   v1 status: exploratory.  A validated existence null for these descriptors
#   is deferred to v2; do not treat them as confirmatory endpoints.
#
# Reference: docs/METHOD_LOG.md (v1 audited evidence-chain architecture).

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

# ---------------------------------------------------------------------------
# Frozen PRIMARY / SECONDARY confirmatory FDR families (②)
# ---------------------------------------------------------------------------
# The PRIMARY confirmatory claim rests on ONE frozen endpoint, aligned
# with PRIMARY_EXISTENCE_ENDPOINT (③ existence gate): gating existence and
# claiming group-inference on a single feature avoids the hidden
# multiple-comparison of an OR-across-a-family. dwell_time / switching_rate are
# CONFIRMATORY descriptors (each has its own validated L1 existence null) and
# are reported in PARALLEL as SECONDARY — BH-corrected within their own small
# family — but they do NOT enter the primary claim's denominator.
PRIMARY_FDR_FAMILY: Tuple[str, ...] = (
    *FDR_FAMILIES["L0"],
)
SECONDARY_FDR_FAMILY: Tuple[str, ...] = (
    *FDR_FAMILIES["L1"],
)

# FDR_FEATURES stays the FULL confirmatory family (primary + secondary) for
# descriptive export / guard compatibility. The primary manuscript claim uses
# PRIMARY_FDR_FAMILY; the secondary report uses SECONDARY_FDR_FAMILY.
# Backward-compat flat tuple (all FDR-family features, both tiers)
FDR_FEATURES: Tuple[str, ...] = (
    *PRIMARY_FDR_FAMILY,
    *SECONDARY_FDR_FAMILY,
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


PRIMARY_EXISTENCE_ENDPOINT: str = "peak_amplitude"
"""Pre-registered PRIMARY endpoint for the L0 synchrony-existence gate.

The existence audit tests several signal-level features (mean_synchrony,
peak_amplitude, bimodality_coefficient), but only ONE frozen
endpoint decides whether the existence stage is "supported". Gating on a
single frozen feature — rather than an OR across the whole family —
avoids inflating the false-positive rate of the gate itself (any-of-k
significant is a hidden multiple-comparison). The remaining audited features
are reported alongside but do NOT enter the gate decision.

``peak_amplitude`` is chosen because it is the v1 primary intensity
workhorse and the L0 member of the confirmatory group-inference family
(FDR_FAMILIES["L0"]). Defined as a module constant so the gate never drifts
via a hard-coded string scattered across call sites."""

# Consistency guard: the primary existence endpoint must be a real feature
# and must be the frozen L0 confirmatory member.
if PRIMARY_EXISTENCE_ENDPOINT not in FEATURE_TIER:
    raise AssertionError(
        f"PRIMARY_EXISTENCE_ENDPOINT '{PRIMARY_EXISTENCE_ENDPOINT}' has no "
        f"FEATURE_TIER entry."
    )
if PRIMARY_EXISTENCE_ENDPOINT not in FDR_FAMILIES["L0"]:
    raise AssertionError(
        f"PRIMARY_EXISTENCE_ENDPOINT '{PRIMARY_EXISTENCE_ENDPOINT}' must be a "
        f"member of FDR_FAMILIES['L0'] {FDR_FAMILIES['L0']}."
    )
if PRIMARY_EXISTENCE_ENDPOINT not in PRIMARY_FDR_FAMILY:
    raise AssertionError(
        f"PRIMARY_EXISTENCE_ENDPOINT '{PRIMARY_EXISTENCE_ENDPOINT}' must be the "
        f"sole member of PRIMARY_FDR_FAMILY {PRIMARY_FDR_FAMILY} (the primary "
        f"confirmatory FDR claim must match the primary existence gate)."
    )


# ---------------------------------------------------------------------------
# ③ Existence gate: per-modality primary-modality structure
# ---------------------------------------------------------------------------
# The existence gate is NOT "any dyad significant" (an any-of-k hidden
# multiple-comparison whose false-positive rate explodes with N). It is
# evaluated PER MODALITY against a dyad-majority threshold, and only the
# pre-registered PRIMARY modalities decide the gate. The physiological
# rationale for the default primary set is frozen here so the choice is
# defensible a priori (NOT selected because these modalities "looked best"
# in the data — that would be circular endpoint selection).
#
#   * ECG (→ IBI/HRV) and EDA (→ SCL) are purely autonomic outputs that a
#     participant cannot voluntarily control, so their synchrony is the most
#     objective readout of physiological coupling (Berntson/Cacioppo/Quigley
#     1991; Palumbo et al. 2017; Chatel-Goldman 2014). EDA is the purest
#     (single sympathetic innervation, no parasympathetic antagonism).
#   * RESP is the one physiological channel that is BOTH autonomic and
#     voluntarily controllable (breath-hold / paced breathing), so its
#     synchrony can reflect co-regulated breathing strategy rather than
#     spontaneous coupling. It is a SENSITIVITY/comparator modality: reported
#     but excluded from the gate.
#
# This set is DATASET-SPECIFIC (Lerique ECG/EDA/RESP). Datasets with other
# channel compositions must define their own primary set via config; the
# default below is the Lerique physiological primary set.
PRIMARY_EXISTENCE_MODALITIES: Tuple[str, ...] = ("ECG", "EDA")
"""Pre-registered primary modalities for the existence gate (Lerique).

Each primary modality must independently reach a dyad-majority pass rate on
PRIMARY_EXISTENCE_ENDPOINT for that modality to count as supporting
existence; the gate is satisfied if AT LEAST ONE primary modality does so
(ECG and EDA are two readouts of the same autonomic-synchrony construct, so
one channel's confirmation suffices — requiring both would over-tighten the
gate and reproduce the false-negative problem this design avoids)."""

EXISTENCE_GATE_MIN_PASS_RATE: float = 0.5
"""Dyad-majority threshold for the per-modality existence gate.

A primary modality supports existence only when the fraction of its dyads
significant on PRIMARY_EXISTENCE_ENDPOINT strictly EXCEEDS this value
(> 0.5, i.e. a genuine majority). Chosen because a "synchrony exists in this
modality" claim is indefensible below half the dyads."""


# ---------------------------------------------------------------------------
# Full feature set for reviewer-proof FDR (critique A, 2026-07-07)
# ---------------------------------------------------------------------------
# ALL_FEATURES is the complete set of 12 implemented features (the union of
# every functional tier).  By default the L2 BH-FDR correction uses only the
# frozen PRIMARY confirmatory family (PRIMARY_FDR_FAMILY, n=1).  Passing
# ``full_family_fdr=True`` enters ALL 12 features into a SINGLE BH-FDR step.
#
# This is the most CONSERVATIVE multiplicity correction possible (more tests
# => stricter BH threshold), so it directly answers the "cherry-picking 3/12"
# critique: if the frozen core survives even the all-features-inclusive
# procedure, the result is robust to any family-selection objection.  The
# primary manuscript endpoint remains FDR_FEATURES (frozen); the
# full-family version is a supplementary, strictly-more-stringent check.

ALL_FEATURES: Tuple[str, ...] = tuple(FEATURE_TIER.keys())
"""Every implemented feature (Axis A functional tiers union).

12 features: mean_synchrony (reference) + peak_amplitude, dwell_time,
switching_rate (core) + onset_latency, rise_time, recovery_time,
synchrony_entropy, bimodality_coefficient, fraction_above_threshold,
first_peak_time, inter_peak_cv (conditional/exploratory).  Used by the
``full_family_fdr`` option to enter all features into one BH-FDR step.
"""


def get_fdr_features(full_family_fdr: bool = False) -> List[str]:
    """Return the feature set entered into the L2 between-condition BH-FDR.

    Parameters
    ----------
    full_family_fdr : bool, default False
        False (default) — the frozen PRIMARY confirmatory family
        (``PRIMARY_FDR_FAMILY``, n=1: peak_amplitude). This is the primary
        manuscript endpoint, aligned with ``PRIMARY_EXISTENCE_ENDPOINT``.
        True — all 12 implemented features (``ALL_FEATURES``) enter a single
        BH-FDR step.  Strictly more conservative; used as a supplementary,
        reviewer-proof check that the frozen core survives even the
        most inclusive multiplicity correction.

    Returns
    -------
    list of str
        Feature names to pass as ``feature_cols`` to the L2 test / inference
        pipeline.
    """
    return list(ALL_FEATURES) if full_family_fdr else list(PRIMARY_FDR_FAMILY)


def get_primary_fdr_features() -> List[str]:
    """Pre-registered PRIMARY confirmatory endpoints (single, peak_amplitude).

    These enter the primary BH-FDR; the primary manuscript claim rests on them.
    """
    return list(PRIMARY_FDR_FAMILY)


def get_secondary_fdr_features() -> List[str]:
    """SECONDARY confirmatory descriptors reported in parallel (dwell_time,
    switching_rate). BH-corrected within their own small family, but NOT part of
    the primary claim's denominator.
    """
    return list(SECONDARY_FDR_FAMILY)

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

FEATURE_AXIS: Dict[str, str] = {
    **{name: "intensity" for name in INTENSITY_FEATURES},
    **{name: "structure" for name in STRUCTURE_FEATURES},
    **{name: "temporal" for name in TEMPORAL_FEATURES},
}
"""Informational axis for every feature (Axis B), derived from the three
axis tuples above so there is exactly ONE source of truth for the mapping.

Consumers (e.g. feature_pipeline's user-facing catalog) must read the axis
from here rather than restating it, so an edit to the tuples above can never
drift out of sync with a hand-maintained copy."""

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

    # --- Implemented descriptors.  The v1 primary FDR family is the explicit
    #     FDR_FEATURES tuple: peak_amplitude, dwell_time, switching_rate.
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

    # --- Optional imputed timing values for machine-learning workflows.
    #     The scientific timing fields above remain raw values and are NaN when
    #     undefined.  These companion fields retain the conservative legacy
    #     imputation (undefined -> wcc_window_sec) only for callers that
    #     explicitly need filled duration-like predictors.
    onset_latency_imputed: float = float("nan")
    rise_time_imputed: float = float("nan")
    recovery_time_imputed: float = float("nan")

    # --- Definedness flags ---
    onset_defined: int = 0
    rise_defined: int = 0
    recovery_defined: int = 0

    # --- Data-quality diagnostic (2026-07-14) ---
    # Fraction of the WCC series that is NaN (i.e. discontinuity-masked
    # windows). Reported alongside dwell_time / switching_rate so reviewers
    # can assess whether missingness correlates with experimental condition —
    # a confound for the "quality of synchrony" argument built on those two
    # FDR-family features (Claude review finding #1, mitigation v1b).
    nan_fraction: float = float("nan")

    # --- Meta ---
    notes: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

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
                # Primary FDR-family descriptors
                *self.FDR_KEYS,
                # Reference — always computed, not in the primary FDR family
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
                # Data-quality diagnostic (discontinuity-masked WCC fraction)
                "nan_fraction",
                # Explicit ML-only timing imputations
                "onset_latency_imputed",
                "rise_time_imputed",
                "recovery_time_imputed",
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

        Round-trip invariant:
        - Every public dataclass field exported by ``to_dict()`` is accepted
          here. Missing fields fall back to dataclass defaults so older artifacts
          remain readable, while extra unknown keys are ignored.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"DynamicFeatures.from_dict expected dict, got "
                f"{type(data).__name__}"
            )

        # Keep deserialization aligned with the dataclass rather than with a
        # hand-maintained subset.  The previous subset silently dropped several
        # non-FDR descriptors (onset/rise/recovery/synchrony_entropy), breaking
        # ``DynamicFeatures.from_dict(x.to_dict()).to_dict()`` round-trips.
        known = {f.name for f in _dc_fields(cls)} - {"notes", "params"}
        kwargs: Dict[str, Any] = {k: data[k] for k in known if k in data}

        # Backward compatibility: all current feature fields have dataclass
        # defaults.  If a future required field is added without a default, keep
        # the existing migration guard rather than silently constructing a
        # partially-defined feature object.
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
                f"{missing}. This may indicate an incompatible artifact; "
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
        ``False``; the hysteresis memory is also reset at a NaN so that
        the post-gap sample is evaluated from a clean baseline (a
        discontinuity must not bridge two otherwise-separated runs).

    Notes
    -----
    Prior to 2026-07-08 the ``hysteresis_delta > 0`` branch assigned
    ``states[i] = state`` at NaN positions, i.e. it *bridged* the gap by
    inheriting the previous run state.  That silently let a dwell/switch
    run continue across an injected ``discontinuity_mask`` NaN, defeating
    the whole point of the mask.  NaN positions are now hard ``False`` and
    the hysteresis memory resets, exactly as the docstring promised.
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
            states[i] = False
            state = False  # reset hysteresis memory at the discontinuity
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

def _finite_segments(finite: np.ndarray):
    """Yield (start, end) slices of contiguous finite samples (end exclusive)."""
    starts, ends = _find_runs(np.asarray(finite, dtype=bool))
    return zip(starts.tolist(), ends.tolist())


def compute_dwell_time(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    hysteresis_delta: float = SWITCHING_HYSTERESIS_DELTA,
    gap_policy: str = "merge_valid",
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
    gap_policy : {"segment", "merge_valid"}
        How missing (NaN) samples affect episode geometry (P1-1, 2026-07-22):

        * ``"merge_valid"`` (default) — drop NaNs then run-length on the
          compressed boolean series (2026-07-13 behaviour).  Bridges short
          sensor dropouts *inside* one continuous episode so they do not
          fabricate structure.  This is the behaviour the unit tests and the
          FDR-family extraction rely on; it restores the pre-P1-1 default.
        * ``"segment"`` — split the trace into contiguous *finite* segments and
          compute elevated runs **within each segment only**.  Opt-in for
          concatenated paradigms (e.g. Lerique trials) where discontinuity
          seams must not glue two real-world episodes into one.

    Notes
    -----
    Hysteresis eliminates boundary jitter near ``threshold``, producing
    more stable dwell estimates on oscillatory traces.
    """
    if gap_policy not in ("segment", "merge_valid"):
        raise ValueError(
            f"gap_policy must be 'segment' or 'merge_valid', got {gap_policy!r}"
        )
    finite = np.isfinite(wcc)
    if not finite.any():
        return float("nan")
    above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)

    run_lengths_list = []
    if gap_policy == "merge_valid":
        # 2026-07-13: compress to finite samples (may glue across seams).
        above_valid = above[finite]
        if above_valid.any():
            starts, ends = _find_runs(above_valid)
            run_lengths_list.extend((ends - starts).tolist())
    else:
        # Default: per contiguous finite segment (no cross-seam glue).
        for s, e in _finite_segments(finite):
            seg = above[s:e]
            if not seg.any():
                continue
            starts, ends = _find_runs(seg)
            run_lengths_list.extend((ends - starts).tolist())

    if not run_lengths_list:
        return float("nan")
    return float(np.mean(run_lengths_list)) / hz


# ---------------------------------------------------------------------------
# DECISION-06b · switching_rate
# ---------------------------------------------------------------------------

def compute_switching_rate(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    hysteresis_delta: float = SWITCHING_HYSTERESIS_DELTA,
    gap_policy: str = "merge_valid",
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
    gap_policy : {"segment", "merge_valid"}
        Same semantics as :func:`compute_dwell_time`.  Default
        ``"merge_valid"`` (2026-07-13 behaviour) counts transitions on the
        NaN-compressed series so short dropouts inside one episode do not
        inflate the rate.  ``"segment"`` is opt-in for concatenated paradigms
        where a discontinuity seam must not inject a fake switch.

    Notes
    -----
    The hysteresis band (default ±0.05) eliminates boundary jitter from
    oscillatory traces straddling threshold — the primary noise source
    in the pre-2026-06-20 implementation.  PGT-2 validation showed
    switching_rate Spearman ρ improved from ~0.22 to substantially
    higher values after this fix.
    """
    if gap_policy not in ("segment", "merge_valid"):
        raise ValueError(
            f"gap_policy must be 'segment' or 'merge_valid', got {gap_policy!r}"
        )
    finite = np.isfinite(wcc)
    if not finite.any():
        return float("nan")
    above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)

    transitions = 0
    if gap_policy == "merge_valid":
        above_valid = above[finite]
        if above_valid.size >= 2:
            transitions = int(np.sum(above_valid[1:] != above_valid[:-1]))
    else:
        for s, e in _finite_segments(finite):
            seg = above[s:e]
            if seg.size >= 2:
                transitions += int(np.sum(seg[1:] != seg[:-1]))

    duration_min = float(finite.sum()) / hz / 60.0
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

    Tier: CONDITIONAL. NOT a member of the confirmatory group-condition FDR
    family (FDR_FEATURES = peak_amplitude, dwell_time, switching_rate). An
    earlier revision (DECISION-09, 2026-06-17) proposed FDR membership, but
    that was never carried into the locked v1.0 confirmatory set, so the prior
    "Enters the FDR family" wording here was stale. Retained as a
    permutation-invariant L0 amplitude-distribution descriptor for exploratory
    analysis (see compute_bimodality_coefficient for the analogous removal).
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
    if float(np.nanstd(finite)) < 1e-12:
        return float("nan")
    from scipy.stats import skew, kurtosis
    sk = float(skew(finite))
    kt_excess = float(kurtosis(finite))  # scipy returns excess kurtosis
    kurt = kt_excess + 3.0  # convert to proper kurtosis
    # Only guard against division-by-zero. Negative proper kurtosis
    # (platykurtic) is a legitimate distribution shape and must NOT be
    # blanked to NaN; BC with a negative denominator is interpretable.
    if abs(kurt) < 1e-12:
        return float("nan")
    return (sk ** 2 + 1.0) / kurt


# ---------------------------------------------------------------------------
# DECISION-17 · Morphology-agnostic timing features
# ---------------------------------------------------------------------------


def _find_prominent_peaks(
    x: np.ndarray,
    threshold: float,
    min_prominence: float,
    window_samples: int,
) -> list:
    """Return indices of prominent local maxima in ``x``.

    A peak is a local maximum above ``threshold`` whose prominence
    (height above the higher of its left and right neighbouring troughs,
    searched within ``window_samples`` on each side) exceeds
    ``min_prominence``.

    Parameters
    ----------
    x : np.ndarray
        1-D array (NaN → -inf).
    threshold : float
        Minimum peak height.
    min_prominence : float
        Minimum prominence.
    window_samples : int
        Look-back / look-ahead window in samples.  Should be scaled by
        the sampling rate so that the physical window is hz-independent.
    """
    n = len(x)
    peaks = []
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            left_min = np.min(x[max(0, i - window_samples):i])
            right_min = np.min(x[i + 1: min(n, i + 1 + window_samples)])
            base = max(left_min, right_min)
            if x[i] - base >= min_prominence:
                peaks.append(i)
    return peaks


def compute_first_peak_time(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    min_prominence: float = 0.15,
    prominence_window_sec: float = DEFAULT_PROMINENCE_WINDOW_SEC,
) -> float:
    """Time of the first prominent peak above threshold (seconds).

    Independent of single-peak assumption — meaningful for oscillatory,
    single-peak, and sustained morphologies alike.  A peak is defined as
    a local maximum within :attr:`ONSET_THRESHOLD`-exceeding segments
    whose prominence (height above adjacent troughs) exceeds
    ``min_prominence``.

    The trough-search window scales with hz so that the physical look-back /
    look-ahead duration is independent of the sampling rate.

    Returns NaN if no prominent peak exists (subthreshold traces).
    """
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any() or n < 3:
        return float("nan")
    x = np.where(finite, wcc, -np.inf)
    window_samples = max(1, round(prominence_window_sec * hz))
    peaks = _find_prominent_peaks(x, threshold, min_prominence, window_samples)
    if not peaks:
        return float("nan")
    return float(peaks[0]) / hz


def compute_baseline_fraction(
    wcc: np.ndarray,
    hz: float,
    threshold: float = ONSET_THRESHOLD,
    min_prominence: float = 0.15,
    prominence_window_sec: float = DEFAULT_PROMINENCE_WINDOW_SEC,
) -> float:
    """Fraction of samples below threshold *before* the first prominent peak.

    High values (~1.0) indicate a prolonged baseline period (single-peak
    morphology).  Low values (~0.0) indicate the trace starts above
    threshold (sustained).  Intermediate values indicate intermittent
    early crossings (oscillatory).

    .. note::
       This function is NOT called by ``extract_features()`` (the main
       entry point).  It is available for external scripts that need the
       pre-first-peak baseline fraction as a standalone descriptor.

       ``hz`` became a required parameter when the internal peak search's
       prominence window was fixed to scale with sampling rate (it was
       previously a hardcoded 50-sample window regardless of ``hz``).
       Existing callers must now pass ``hz`` explicitly.

    Returns NaN if no prominent peak exists.
    """
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any() or n < 3:
        return float("nan")
    x = np.where(finite, wcc, -np.inf)
    window_samples = max(1, round(prominence_window_sec * hz))
    peaks = _find_prominent_peaks(x, threshold, min_prominence, window_samples)
    if not peaks:
        return float("nan")
    first_peak_idx = peaks[0]
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
    prominence_window_sec: float = DEFAULT_PROMINENCE_WINDOW_SEC,
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
    window_samples = max(1, round(prominence_window_sec * hz))
    all_peaks = _find_prominent_peaks(x, threshold, min_prominence, window_samples)
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
    See ``docs/METHOD_LOG.md`` for the current v1 threshold stance.

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
        ``ONSET_THRESHOLD`` (0.5) when (a) fewer than 10 finite surrogate
        values are available (degenerate case) or (b) the derived threshold
        exceeds :data:`SURROGATE_THRESHOLD_MAX` (periodicity / strong-
        autocorrelation artifact of IAAFT surrogates; BUG-3).  Both fallback
        paths emit a ``logger.warning`` so the substitution is *never* silent;
        ``is_surrogate_derived`` is ``False`` exactly when a fallback fired,
        so callers can flag
        which dyads received a data-driven threshold and which received
        the fixed fallback — this distinction MUST be reported alongside
        any dwell_time / switching_rate / onset_latency computed under
        the surrogate-threshold specification (cf. the explicit
        ``_defined`` flag convention used elsewhere in this module).

    Notes
    -----
    **Per-modality pooled surrogate threshold is the canonical v1 default.**
    In the pipeline, thresholds are computed by
    ``multisync.session_threshold.compute_session_pooled_thresholds_by_modality``,
    which pools the surrogate null *within* each modality so slow/smooth and
    fast/spiky signals each get a modality-appropriate cut-off (cross-modal
    comparability preserved, threshold calibrated to each modality's null).
    This ``compute_surrogate_threshold`` function is the single-dyad primitive
    that the pooled paths build on.

    **Session-level / all-dyad pooling is an OPTIONAL granularity**, not the
    default: pool ALL timepoints across ALL surrogate replicates and ALL dyads
    before computing the quantile for a single shared threshold (use
    ``multisync.session_threshold``).  Condition-level thresholds (sensitivity
    analysis) can be obtained by calling this function separately for each
    condition's surrogate WCC slice.
    """
    wcc_surrogates = np.asarray(wcc_surrogates, dtype=float)
    if wcc_surrogates.ndim == 1:
        wcc_surrogates = wcc_surrogates.reshape(1, -1)
    finite = wcc_surrogates[np.isfinite(wcc_surrogates)]
    if finite.size < 10:
        # Degenerate case: not enough finite surrogate WCC values to form a
        # reliable null distribution.  Fail LOUD (do NOT silently substitute)
        # and fall back to the locked DECISION-01 threshold.
        logging.getLogger(__name__).warning(
            "compute_surrogate_threshold: only %d finite surrogate WCC "
            "values (< 10) available — falling back to fixed ONSET_THRESHOLD"
            "=%s. Surrogate-derived threshold NOT used for this dyad/session.",
            finite.size, ONSET_THRESHOLD,
        )
        return ONSET_THRESHOLD, False
    derived = float(np.percentile(finite, percentile))
    # Periodicity / strong-autocorrelation guard (BUG-3): an extreme null
    # threshold means the surrogate (null) distribution is contaminated by
    # the preserved autocorrelation structure, NOT a genuine high sync-by-
    # chance level.  Fail LOUD and fall back to the fixed threshold.
    if derived > SURROGATE_THRESHOLD_MAX:
        logging.getLogger(__name__).warning(
            "compute_surrogate_threshold: derived threshold %.3f exceeds "
            "sanity ceiling %.2f — likely a periodicity/autocorrelation "
            "artifact of IAAFT surrogates. Falling back to fixed "
            "ONSET_THRESHOLD=%s for this modality/session.",
            derived, SURROGATE_THRESHOLD_MAX, ONSET_THRESHOLD,
        )
        return ONSET_THRESHOLD, False
    return derived, True


# ---------------------------------------------------------------------------
# High-level entry: extract all features + diagnostics
# ---------------------------------------------------------------------------

def extract_features(
    wcc: np.ndarray,
    hz: float,
    wcc_window_sec: float,
    threshold: float = ONSET_THRESHOLD,
    paradigm: str = "auto",
    gap_policy: Optional[str] = None,
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
    gap_policy : {"segment", "merge_valid"} or None
        Forwarded to dwell_time / switching_rate.  ``None`` (default) leaves
        those functions on their own default (``merge_valid``).  Pass
        ``"segment"`` when the WCC was gated by a discontinuity mask so
        episodes are not glued across seams (P1-R2).

    Returns
    -------
    DynamicFeatures
        Raw event-timing fields are ``NaN`` when scientifically undefined.
        Companion ``*_imputed`` fields carry the conservative legacy fill for
        downstream ML workflows that explicitly need imputed duration-like
        predictors.
    """
    wcc = np.asarray(wcc, dtype=float)
    # P1-4: always report discontinuity / missing fraction from SSoT so
    # callers that bypass extract_dynamic_features still get the diagnostic.
    _finite_frac = float(np.isfinite(wcc).mean()) if wcc.size else 0.0
    _nan_fraction = 1.0 - _finite_frac

    # Empty / fully non-finite WCC: return structured NaNs (do not raise inside
    # scipy peak helpers with "v cannot be empty").
    if wcc.size == 0 or not np.isfinite(wcc).any():
        return DynamicFeatures(
            nan_fraction=float(_nan_fraction),
            notes="empty_or_nonfinite_wcc",
            params={
                "threshold": float(threshold),
                "hz": float(hz),
                "wcc_window_sec": float(wcc_window_sec),
                "gap_policy": gap_policy if gap_policy is not None else "merge_valid",
            },
        )

    # Smoothed peak first (DECISION-04) -- anchors rise/recovery indexing
    sm = smoothed_wcc(wcc)
    peak_value, peak_idx = compute_peak_amplitude(sm)

    # Onset is decoupled from peak (DECISION-08)
    onset_lat, onset_def = compute_onset_latency(
        wcc, hz=hz, wcc_window_sec=wcc_window_sec, threshold=threshold,
    )

    if peak_idx is not None:
        # Intentional split (DECISION-03/05, 2026-07-13): the peak anchor
        # (peak_index, peak_value) is taken from the smoothed series `sm` for
        # robustness, but rise/recovery crossing searches run on the RAW `wcc`.
        # The crossing timing must reflect the real (unsmoothed) signal, and
        # the quartile/half-recovery levels are referenced to `peak_value`
        # (the smoothed amplitude), not to wcc[peak_index]. peak_index is a
        # valid index into both arrays (they are equal-length and aligned),
        # so there is no indexing error. We deliberately do NOT search in
        # `sm`: smoothed_wcc zero-pads at the boundaries, which would fabricate
        # spurious crossings near the trace edges.
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

    _dwell_kwargs = {"hz": hz, "threshold": threshold}
    if gap_policy is not None:
        _dwell_kwargs["gap_policy"] = gap_policy
    dwell = compute_dwell_time(wcc, **_dwell_kwargs)
    switch = compute_switching_rate(wcc, **_dwell_kwargs)

    mean_s = compute_mean_synchrony(wcc)
    frac_above = compute_fraction_above_threshold(wcc, threshold=threshold)
    entropy = compute_synchrony_entropy(wcc)
    bc = compute_bimodality_coefficient(wcc)
    ipc = compute_inter_peak_cv(wcc, hz=hz, threshold=threshold)
    fpt = compute_first_peak_time(wcc, hz=hz, threshold=threshold)

    # Scientific timing fields remain raw: undefined == NaN.  Companion
    # *_imputed fields preserve the old conservative upper-bound fill only for
    # downstream ML workflows that explicitly need complete duration-like
    # predictors.
    onset_lat_raw = onset_lat
    rise_t_raw = rise_t
    rec_t_raw = rec_t
    onset_lat_imp = onset_lat if onset_def else float(wcc_window_sec)
    rise_t_imp = rise_t if rise_def else float(wcc_window_sec)
    rec_t_imp = rec_t if rec_def else float(wcc_window_sec)

    notes: list[str] = []

    # DECISION-16: paradigm-aware feature reporting
    if paradigm == "continuous":
        rise_t_raw = float("nan")
        rec_t_raw = float("nan")
        rise_t_imp = float("nan")
        rec_t_imp = float("nan")
        rise_def = 0
        rec_def = 0
        notes.append("rise/recovery set NaN (continuous paradigm)")

    return DynamicFeatures(
        onset_latency=onset_lat_raw,
        rise_time=rise_t_raw,
        peak_amplitude=peak_value,
        recovery_time=rec_t_raw,
        dwell_time=dwell,
        switching_rate=switch,
        mean_synchrony=mean_s,
        synchrony_entropy=entropy,
        bimodality_coefficient=bc,
        fraction_above_threshold=frac_above,
        inter_peak_cv=ipc,
        first_peak_time=fpt,
        onset_latency_imputed=onset_lat_imp,
        rise_time_imputed=rise_t_imp,
        recovery_time_imputed=rec_t_imp,
        onset_defined=int(onset_def),
        rise_defined=int(rise_def),
        recovery_defined=int(rec_def),
        nan_fraction=float(_nan_fraction),
        notes="; ".join(notes),
        params={
            "threshold": float(threshold),
            "hz": float(hz),
            "wcc_window_sec": float(wcc_window_sec),
            "timing_imputation_rule": "undefined event timing -> wcc_window_sec; continuous event-only timing -> NaN",
            "gap_policy": gap_policy if gap_policy is not None else "merge_valid",  # default bridges short dropouts (DECISION 2026-07-13); segment is opt-in for concatenated paradigms
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
    "T_DEF_MIN_WCC_POINTS",
    "N_MIN_DYADS_FDR",
    "check_eligibility",
    "compute_surrogate_threshold",
    # Functional tier (primary classification)
    "FEATURE_TIER",
    "FDR_FEATURES",
    "REFERENCE_FEATURE",
    "PRIMARY_EXISTENCE_ENDPOINT",
    "PRIMARY_EXISTENCE_MODALITIES",
    "EXISTENCE_GATE_MIN_PASS_RATE",
    "ALL_FEATURES",
    "get_fdr_features",
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

"""
tests/test_feature_definitions.py
=================================

Tests for the v0.x.0 Methodology Lock-In Single Source of Truth.

These tests are intentionally **independent** of any other test in the
project: they import only from ``multisync.feature_definitions`` and
``numpy``, so they can be run in isolation::

    pytest tests/test_feature_definitions.py -v

The assertions encode the locked DECISIONs (see
``docs/METHODOLOGY_LOCK_IN.md``).  If any of these assertions fail,
methodology has been (intentionally or accidentally) reverted -- consult
``docs/DECISION_LOG.md`` before proceeding.
"""

from __future__ import annotations

import numpy as np
import pytest

from multisync.feature_definitions import (
    DynamicFeatures,
    ONSET_THRESHOLD,
    PEAK_SMOOTHING_WINDOW,
    RECOVERY_FRAC,
    RISE_HIGH_FRAC,
    RISE_LOW_FRAC,
    compute_baseline_fraction,
    compute_dwell_time,
    compute_first_peak_time,
    compute_fraction_above_threshold,
    compute_inter_peak_cv,
    compute_onset_latency,
    compute_peak_amplitude,
    compute_recovery_time,
    compute_rise_time,
    compute_switching_rate,
    compute_synchrony_entropy,
    extract_features,
    find_dominant_peak,
    smoothed_wcc,
)


# ---------------------------------------------------------------------------
# Sanity: locked constants
# ---------------------------------------------------------------------------

def test_locked_constants_have_expected_values():
    """DECISION-01 / 03 / 04 / 05 numerical anchors."""
    assert ONSET_THRESHOLD == 0.5
    assert PEAK_SMOOTHING_WINDOW == 3
    assert RISE_LOW_FRAC == 0.25
    assert RISE_HIGH_FRAC == 0.75
    assert RECOVERY_FRAC == 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_wave(n=300, hz=1.0, period_sec=60.0, high=0.8, low=0.0):
    """Alternating high/low blocks; each block lasts period_sec / 2."""
    t = np.arange(n) / hz
    return np.where((t // (period_sec / 2)) % 2 == 0, high, low).astype(float)


# ---------------------------------------------------------------------------
# Smoothing & peak
# ---------------------------------------------------------------------------

def test_smoothed_wcc_preserves_length():
    wcc = np.random.RandomState(0).randn(101)
    sm = smoothed_wcc(wcc)
    assert sm.shape == wcc.shape


def test_smoothed_peak_resists_single_spike():
    """DECISION-04 defence: 3-point smoothing must blunt a single sample spike."""
    wcc = np.zeros(100)
    wcc[50] = 1.0  # single-sample spike
    sm = smoothed_wcc(wcc)
    # 1/3 boxcar average -> peak around 1/3, not 1.0
    assert sm.max() < 0.5
    assert sm.max() > 0.30


def test_find_dominant_peak_returns_argmax():
    wcc = np.array([0.1, 0.3, 0.9, 0.2, 0.5])
    assert find_dominant_peak(wcc) == 2


def test_find_dominant_peak_all_nan_returns_none():
    wcc = np.full(10, np.nan)
    assert find_dominant_peak(wcc) is None


# ---------------------------------------------------------------------------
# DECISION-02 · onset_latency
# ---------------------------------------------------------------------------

def test_onset_undefined_when_trace_all_above_threshold():
    """A trace with no baseline phase has scientifically undefined onset.

    Raw timing semantics: undefined onset is NaN.  The companion
    onset_latency_imputed field preserves the conservative wcc_window_sec fill
    for downstream ML workflows that explicitly need imputation.
    """
    wcc = np.full(300, 0.9)
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats.onset_defined == 0
    assert np.isnan(feats.onset_latency)
    assert feats.onset_latency_imputed == 30.0


def test_onset_undefined_when_trace_all_below_threshold():
    """All-below trace has no onset — raw value is NaN, imputed value is explicit."""
    wcc = np.full(300, 0.2)
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats.onset_defined == 0
    assert np.isnan(feats.onset_latency)
    assert feats.onset_latency_imputed == 30.0


def test_onset_defined_for_clear_step_up():
    """A trace that starts low and steps up well above threshold should
    have a defined onset at roughly the step boundary."""
    wcc = np.concatenate([np.full(60, 0.0), np.full(240, 0.9)])
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats.onset_defined == 1
    # K = max(2, round(0.05 * 1 * 30)) = 2; first sustained crossing -> idx 60
    assert feats.onset_latency == pytest.approx(60.0, abs=2.0)


def test_onset_requires_sustained_crossing_not_single_spike():
    """A single-sample crossing must NOT trigger onset; sustained crossing must."""
    wcc = np.full(300, 0.0)
    wcc[100] = 0.9  # single-sample spike, should NOT trigger
    feats_single = compute_onset_latency(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats_single[1] == 0  # undefined

    wcc2 = wcc.copy()
    wcc2[150:160] = 0.9  # 10-sample sustained crossing, MUST trigger
    feats_sustained = compute_onset_latency(wcc2, hz=1.0, wcc_window_sec=30.0)
    assert feats_sustained[1] == 1
    assert feats_sustained[0] == pytest.approx(150.0, abs=2.0)


# ---------------------------------------------------------------------------
# DECISION-03 · rise_time independence from threshold position
# ---------------------------------------------------------------------------

def test_rise_time_equivalence_on_linear_ramp():
    """DECISION-03 boundary condition (documented invariant):

    On a *linear* ramp, the 25-75 quartile definition and the naive
    'onset -> peak' definition behave nearly identically -- both spread
    by the same proportional amount across threshold choices.

    This is NOT a refutation of DECISION-03; it is a documented limit
    case.  The discriminant-validity advantage of quartile rise time
    appears only on non-linear (sigmoid / curved) ramps, which is the
    realistic regime for WCC traces.  See ``test_rise_time_robustness_on_sigmoid_ramp``
    for the regime where DECISION-03 actually pays off.

    Historical note (DECISION_LOG entry, 2026-05-23):
    A prior version of this test asserted ``spread/mean < 0.40`` on a
    linear ramp and FAILED with measured 0.646.  Hand derivation by the
    project lead (S.C. Chen) showed the assertion was geometry-blind:
    quartile and onset-to-peak coincide on linear functions.  The test
    was reframed to lock in *both* the linear-equivalence fact and the
    non-linear discriminant fact.
    """
    wcc = np.concatenate([
        np.full(40, 0.3),                    # clear baseline
        np.linspace(0.4, 1.0, 100),          # linear ramp
        np.full(160, 1.0),                   # plateau
    ])

    f_low = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.4)
    f_mid = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.5)
    f_hi  = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.7)

    rises = [f_low.rise_time, f_mid.rise_time, f_hi.rise_time]
    assert all(np.isfinite(r) for r in rises)

    # On a linear ramp the spread is large (~0.65) by construction.
    # We assert (i) all three values are finite, and (ii) the spread is
    # bounded above -- the latter to detect catastrophic regressions
    # (e.g. a future SSoT bug returning 0 or thousands of seconds).
    spread = max(rises) - min(rises)
    rel_spread = spread / np.mean(rises)
    assert 0.40 < rel_spread < 1.00, (
        f"linear-ramp rise_time spread out of expected band "
        f"[0.40, 1.00]: got {rel_spread:.3f}, rises={rises}"
    )


def test_rise_time_robustness_on_sigmoid_ramp():
    """DECISION-03 discriminant validity: on a *sigmoid* (non-linear) ramp,
    the 25-75 quartile rise time is substantially less threshold-sensitive
    than a naive 'onset -> peak' would be.

    Construction: an S-shaped transition where most of the slope is
    concentrated in the middle.  The 25%-75% band lives inside the
    high-slope core and is largely invariant to where 'baseline' sits
    along the long, flat tails.  An 'onset -> peak' definition would
    instead chase the threshold deep into the long lower tail and inflate
    rise_time dramatically as threshold drops.
    """
    # 300-sample trace, hz=1.0
    # baseline (60) + sigmoid transition (180) + plateau (60)
    sig_len = 180
    x = np.linspace(-6, 6, sig_len)
    sigmoid = 1.0 / (1.0 + np.exp(-x))           # 0..1 across 180 samples
    sigmoid_segment = 0.30 + 0.65 * sigmoid       # 0.30 .. 0.95

    wcc = np.concatenate([
        np.full(60, 0.30),
        sigmoid_segment,
        np.full(60, 0.95),
    ])

    f_low = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.40)
    f_mid = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.50)
    f_hi  = extract_features(wcc, hz=1.0, wcc_window_sec=30.0, threshold=0.60)

    rises = [f_low.rise_time, f_mid.rise_time, f_hi.rise_time]
    assert all(np.isfinite(r) for r in rises), (
        f"All quartile rise_times should be finite on a clean sigmoid, "
        f"got {rises}"
    )

    spread = max(rises) - min(rises)
    rel_spread = spread / np.mean(rises)

    # DECISION-03 claim: on non-linear ramps, quartile rise_time is
    # tight (< 0.40 relative spread) across the sensitivity band.
    # If this ever fails, either the SSoT regressed or the claim itself
    # needs Reversal (see DECISION_LOG.md).
    assert rel_spread < 0.40, (
        f"DECISION-03 discriminant claim violated: relative spread "
        f"{rel_spread:.3f} on sigmoid (expected < 0.40). "
        f"rises={rises}"
    )


def test_rise_time_nan_when_flat_trace():
    wcc = np.full(200, 0.3)  # never crosses baseline
    peak_value = 0.3
    peak_index = 50
    r, d = compute_rise_time(wcc, peak_index, peak_value, hz=1.0)
    assert np.isnan(r)
    assert d == 0


# ---------------------------------------------------------------------------
# DECISION-04 · peak_amplitude returns smoothed max
# ---------------------------------------------------------------------------

def test_peak_amplitude_returns_smoothed_value_not_raw_max():
    """DECISION-04: peak must come from the 3-point smoothed series."""
    wcc = np.full(100, 0.4)
    wcc[50] = 1.0  # single-sample spike
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    # raw max is 1.0; smoothed max is roughly (0.4 + 1.0 + 0.4) / 3 ~ 0.6
    assert feats.peak_amplitude < 0.8
    assert feats.peak_amplitude > 0.4


# ---------------------------------------------------------------------------
# DECISION-05 · half-recovery, not full
# ---------------------------------------------------------------------------

def test_recovery_uses_half_level_not_baseline():
    """DECISION-05: recovery_time = peak -> half_level, NOT peak -> baseline.
    A trace that decays from 1.0 only to 0.6 (above baseline 0.5) must
    still have a defined recovery_time because half_level = 0.75 < 1.0
    and the trace crosses 0.75 on the way down."""
    wcc = np.concatenate([
        np.full(50, 0.2),
        np.linspace(0.2, 1.0, 50),
        np.linspace(1.0, 0.6, 100),  # decays only to 0.6, never to baseline 0.5
        np.full(100, 0.6),
    ])
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats.recovery_defined == 1
    assert np.isfinite(feats.recovery_time)


def test_recovery_nan_when_never_returns():
    """If trace stays at peak forever after peak, half-recovery undefined."""
    wcc = np.concatenate([
        np.full(50, 0.2),
        np.linspace(0.2, 1.0, 50),
        np.full(200, 1.0),  # plateau, never drops
    ])
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    assert feats.recovery_defined == 0
    assert np.isnan(feats.recovery_time)
    assert feats.recovery_time_imputed == 30.0


# ---------------------------------------------------------------------------
# DECISION-06a · dwell_time
# ---------------------------------------------------------------------------

def test_dwell_time_matches_known_block_length():
    """30s elevated blocks alternating with 30s baseline -> mean dwell = 30s."""
    wcc = _square_wave(n=300, hz=1.0, period_sec=60.0, high=0.8, low=0.0)
    d = compute_dwell_time(wcc, hz=1.0)
    assert d == pytest.approx(30.0, abs=1.0)


def test_dwell_time_nan_when_no_elevated_phase():
    wcc = np.full(300, 0.1)
    assert np.isnan(compute_dwell_time(wcc, hz=1.0))


# ---------------------------------------------------------------------------
# DECISION-06b · switching_rate
# ---------------------------------------------------------------------------

def test_switching_rate_matches_known_transitions():
    """300s with alternating 30s blocks: ~9 internal transitions over 5 min.

    (5 high blocks of 30s each between 5 low blocks => 9 transitions
    inside the trace, plus possibly an opening edge depending on phase.)
    """
    wcc = _square_wave(n=300, hz=1.0, period_sec=60.0, high=0.8, low=0.0)
    rate = compute_switching_rate(wcc, hz=1.0)
    # 9 transitions / 5 min = 1.8/min ; allow some tolerance for edges
    assert rate == pytest.approx(1.8, abs=0.5)


def test_switching_rate_zero_when_constant_trace():
    wcc = np.full(300, 0.9)
    rate = compute_switching_rate(wcc, hz=1.0)
    assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gap-robustness (DECISION, 2026-07-13): missing points must not fabricate
# structure in the two confirmatory episode features.
# ---------------------------------------------------------------------------

def _single_episode_with_gap(n=100, onset=20, offset=80, hi=0.8, lo=0.1,
                              gap_start=48, gap_end=53):
    """One continuous elevated run (onset:offset) interrupted by a NaN gap in
    its middle; baseline (lo) elsewhere. The gap is the ONLY break, so the
    episode is genuinely one run of (offset-onset - gap_len) valid samples."""
    wcc = np.full(n, lo, dtype=float)
    wcc[onset:offset] = hi
    wcc[gap_start:gap_end] = np.nan
    return wcc


def test_dwell_time_artifact_gap_does_not_split_run():
    """A single elevated episode interrupted by an artifact gap must remain
    one run (dwell ~ valid length), not be split into two short runs."""
    wcc = _single_episode_with_gap()
    d = compute_dwell_time(wcc, hz=1.0)
    # 60 elevated samples minus the 5-sample gap = 55 valid samples, not the
    # ~27.5 a hard split into two runs would yield.
    assert d == pytest.approx(55.0, abs=2.0)


def test_switching_rate_artifact_gap_does_not_inflate():
    """An artifact gap inside one episode must not add transitions: the rate
    stays equal to the continuous (no-gap) version, not double it."""
    wcc = _single_episode_with_gap()
    rate_gap = compute_switching_rate(wcc, hz=1.0)
    wcc_cont = np.full(100, 0.1, dtype=float)
    wcc_cont[20:80] = 0.8
    rate_cont = compute_switching_rate(wcc_cont, hz=1.0)
    # With the fix the gap adds 0 transitions; without it the run would split
    # into two and the rate would roughly double (~2.5/min here).
    assert rate_gap == pytest.approx(rate_cont, abs=0.2)
    assert rate_gap < 1.5


def test_switching_rate_duration_uses_valid_time():
    """Two fully-valid elevated blocks + trailing NaN padding: the transition
    count is identical with/without the fix, but the duration denominator is
    valid time only (3 transitions / 1 valid-min = 3/min), not the full length
    (which would give 3 / 2 min = 1.5 / min)."""
    n = 120
    wcc = np.full(n, 0.1, dtype=float)
    wcc[0:21] = 0.8           # elevated block 1 (indices 0..20)
    wcc[30:51] = 0.8          # elevated block 2 (indices 30..50)
    wcc[60:120] = np.nan      # trailing NaN padding (no valid samples after)
    rate = compute_switching_rate(wcc, hz=1.0)
    # 3 transitions (up, down-up, down) over 60 valid seconds = 3 / min.
    assert rate == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# Occupancy descriptor · fraction_above_threshold
# ---------------------------------------------------------------------------

def test_fraction_above_threshold_counts_finite_values_only():
    wcc = np.array([0.6, 0.4, np.nan, 0.8, 0.2])
    assert compute_fraction_above_threshold(wcc, threshold=0.5) == pytest.approx(2 / 4)


def test_fraction_above_threshold_is_order_invariant_and_non_fdr():
    wcc = np.array([0.1, 0.7, 0.2, 0.9, 0.8])
    shuffled = np.array([0.8, 0.1, 0.9, 0.2, 0.7])
    assert compute_fraction_above_threshold(wcc, threshold=0.5) == pytest.approx(
        compute_fraction_above_threshold(shuffled, threshold=0.5)
    )
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=5.0, threshold=0.5)
    assert feats.fraction_above_threshold == pytest.approx(3 / 5)
    assert "fraction_above_threshold" in feats.to_dict()
    assert "fraction_above_threshold" not in DynamicFeatures.CONFIRMATORY_KEYS


def test_timing_descriptors_are_wired_and_non_fdr():
    """inter_peak_cv and first_peak_time are wired into extract_features,
    exported in to_dict, and excluded from the FDR family (2026-06-29)."""
    # An oscillatory trace with several prominent above-threshold peaks.
    t = np.linspace(0, 20 * np.pi, 400)
    wcc = 0.5 + 0.4 * np.sin(t)
    feats = extract_features(wcc, hz=2.0, wcc_window_sec=30.0, threshold=0.5)
    d = feats.to_dict()
    assert "inter_peak_cv" in d
    assert "first_peak_time" in d
    assert np.isfinite(feats.first_peak_time)
    assert np.isfinite(feats.inter_peak_cv)
    assert "inter_peak_cv" not in DynamicFeatures.CONFIRMATORY_KEYS
    assert "first_peak_time" not in DynamicFeatures.CONFIRMATORY_KEYS


def test_timing_descriptors_are_nan_when_undefined():
    """Subthreshold / flat traces yield NaN (undefined) timing descriptors,
    which downstream code must report as definedness gaps rather than 0."""
    wcc = np.full(100, 0.1)
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=10.0, threshold=0.5)
    assert np.isnan(feats.inter_peak_cv)
    assert np.isnan(feats.first_peak_time)


# ---------------------------------------------------------------------------
# DECISION-06 · diagnostics excluded from confirmatory family
# ---------------------------------------------------------------------------

def test_synchrony_entropy_is_diagnostic_only():
    """synchrony_entropy is L0 mathematically (permutation-invariant) but
    NOT in the confirmatory FDR family (exploratory). peak_amplitude is a
    genuine FDR-family member, used here as a positive control."""
    assert "synchrony_entropy" not in DynamicFeatures.CONFIRMATORY_KEYS
    assert "peak_amplitude" in DynamicFeatures.CONFIRMATORY_KEYS


def test_confirmatory_family_has_exactly_three_keys():
    """2026-06-29 revised (SSoT Option B): primary group-condition FDR
    family = 3 features.
    L0: peak_amplitude (signal-level null)
    L1: dwell_time, switching_rate (WCC-level null)
    mean_synchrony (reference) and bimodality_coefficient (exploratory)
    are reported but NOT in the confirmatory FDR family; both remain L0
    features for the separate synchrony-existence audit."""
    assert len(DynamicFeatures.CONFIRMATORY_KEYS) == 3
    assert set(DynamicFeatures.CONFIRMATORY_KEYS) == {
        "peak_amplitude",
        "dwell_time",
        "switching_rate",
    }
    assert "mean_synchrony" not in DynamicFeatures.CONFIRMATORY_KEYS
    assert "bimodality_coefficient" not in DynamicFeatures.CONFIRMATORY_KEYS


# ---------------------------------------------------------------------------
# extract_features end-to-end smoke test
# ---------------------------------------------------------------------------

def test_extract_features_returns_finite_on_clean_synthetic():
    """A well-formed synthetic trace must produce 6 finite confirmatory features."""
    wcc = np.concatenate([
        np.full(30, 0.1),                          # baseline
        np.linspace(0.1, 0.9, 30),                 # rise
        np.full(60, 0.9),                          # plateau
        np.linspace(0.9, 0.3, 30),                 # decay
        np.full(150, 0.3),                         # recovery baseline
    ])
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    for k in DynamicFeatures.CONFIRMATORY_KEYS:
        v = getattr(feats, k)
        assert np.isfinite(v), f"{k} should be finite, got {v}"


def test_extract_features_handles_all_nan_gracefully():
    wcc = np.full(300, np.nan)
    feats = extract_features(wcc, hz=1.0, wcc_window_sec=30.0)
    # Must NOT raise, must return NaNs with definedness flags = 0
    assert feats.onset_defined == 0
    assert feats.rise_defined == 0
    assert feats.recovery_defined == 0
    assert np.isnan(feats.peak_amplitude)


# ---------------------------------------------------------------------------
# Prominence-window hz-invariance (regression test)
#
# compute_first_peak_time / compute_baseline_fraction / compute_inter_peak_cv
# previously shared a peak-prominence lookback/lookahead window hardcoded at
# 50 SAMPLES, not scaled by hz. The same real-world synchrony pattern
# resampled at a higher rate could silently fail to detect any peak at all
# (NaN) purely because the fixed 50-sample window no longer reached the
# neighboring trough. See docs/DECISION_LOG.md.
# ---------------------------------------------------------------------------

def _make_oscillation(hz: float, duration_sec: float = 60.0, n_cycles: float = 3.0) -> np.ndarray:
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    return 0.4 + 0.35 * np.sin(2 * np.pi * n_cycles / duration_sec * t)


def test_timing_descriptors_are_hz_invariant():
    """The same real-world oscillation, resampled at different hz, must
    yield the same first_peak_time (seconds) and the same inter_peak_cv,
    and must never silently degrade to NaN at higher sampling rates."""
    results = []
    for hz in (1.0, 5.0, 20.0):
        wcc = _make_oscillation(hz)
        fpt = compute_first_peak_time(wcc, hz=hz, threshold=0.5, min_prominence=0.15)
        cv = compute_inter_peak_cv(wcc, hz=hz, threshold=0.5, min_prominence=0.15)
        bf = compute_baseline_fraction(wcc, hz=hz, threshold=0.5, min_prominence=0.15)
        assert np.isfinite(fpt), f"first_peak_time went NaN at hz={hz}"
        assert np.isfinite(cv), f"inter_peak_cv went NaN at hz={hz}"
        assert np.isfinite(bf), f"baseline_fraction went NaN at hz={hz}"
        results.append((hz, fpt, cv, bf))

    ref_fpt, ref_cv, ref_bf = results[0][1], results[0][2], results[0][3]
    for hz, fpt, cv, bf in results[1:]:
        assert fpt == pytest.approx(ref_fpt, abs=1.0 / 1.0), (
            f"first_peak_time not hz-invariant: hz=1.0 -> {ref_fpt}, hz={hz} -> {fpt}"
        )
        assert cv == pytest.approx(ref_cv, abs=0.05), (
            f"inter_peak_cv not hz-invariant: hz=1.0 -> {ref_cv}, hz={hz} -> {cv}"
        )
        assert bf == pytest.approx(ref_bf, abs=0.05), (
            f"baseline_fraction not hz-invariant: hz=1.0 -> {ref_bf}, hz={hz} -> {bf}"
        )


def test_compute_baseline_fraction_requires_hz():
    """hz became a required parameter when the prominence window was fixed
    to scale with sampling rate; calling without it must raise, not
    silently fall back to the old sample-count behavior."""
    wcc = _make_oscillation(hz=1.0)
    with pytest.raises(TypeError):
        compute_baseline_fraction(wcc, threshold=0.5)  # type: ignore[call-arg]

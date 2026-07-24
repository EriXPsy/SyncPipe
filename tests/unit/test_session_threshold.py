"""
Unit tests for per-modality pooled onset thresholds.

Covers :func:`multisync.session_threshold.compute_session_pooled_thresholds_by_modality`
and the per-modality wiring of
:func:`multisync.pipeline_bridge.records_to_inference_inputs`.

The canonical v1 onset threshold is one IAAFT surrogate threshold *per modality*
(slow/smooth EDA vs fast/spiky ECG get different thresholds). Fixed 0.5 is only
the fallback / sensitivity value. These tests prove:

1. one threshold key per distinct modality (grouping),
2. two modalities with genuinely different signal structure get *different*
   thresholds,
3. a modality whose null is degenerate falls back to ONSET_THRESHOLD (0.5) and
   is still emitted,
4. len(modalities) != len(dyad_signals) fails loud (ValueError),
5. per-modality thresholds equal the global pooled threshold computed on that
   modality's signals alone (code-audit-symmetry: per-modality <-> global),
6. the bridge's ``"session_pooled"`` path resolves EDA != ECG thresholds and
   records them in ``InferenceInputs.thresholds_by_modality``.
"""
from __future__ import annotations

import numpy as np
import pytest

from multisync.session_threshold import (
    compute_session_pooled_threshold,
    compute_session_pooled_thresholds_by_modality,
)
from multisync.feature_definitions import ONSET_THRESHOLD
from multisync.pipeline_bridge import records_to_inference_inputs


# ---------------------------------------------------------------------------
# synthetic signal helpers
# ---------------------------------------------------------------------------

def _smooth(n: int = 240, seed: int = 0) -> np.ndarray:
    """Very low-frequency (smooth) signal -> high surrogate WCC null."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    base = np.sin(2 * np.pi * 1.0 * t)  # ~1 period over the whole series
    return (base + rng.normal(0, 0.05, n)).astype(float)


def _white(n: int = 240, seed: int = 0) -> np.ndarray:
    """White noise -> low surrogate WCC null."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n).astype(float)


def _constant(n: int = 240) -> np.ndarray:
    """Flat signal -> degenerate surrogate WCC (NaN) -> fallback."""
    return np.full(n, 0.5, dtype=float)


def _dyads(maker, n_dyads: int = 4, n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_dyads):
        a = maker(n, seed=seed * 100 + i)
        # partner: same structure, independent draw -> weak real coupling
        b = maker(n, seed=seed * 100 + i + n_dyads)
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# 1. one key per distinct modality
# ---------------------------------------------------------------------------

def test_by_modality_returns_one_key_per_modality():
    eda = _dyads(_smooth, n_dyads=3, seed=1)
    ecg = _dyads(_white, n_dyads=3, seed=2)
    sigs = eda + ecg
    mods = ["EDA"] * 3 + ["ECG"] * 3
    thr = compute_session_pooled_thresholds_by_modality(
        sigs, mods, hz=1.0, wcc_window_size=24, surrogate_n=50, seed=0
    )
    assert set(thr.keys()) == {"EDA", "ECG"}
    assert all(isinstance(v, float) for v in thr.values())


# ---------------------------------------------------------------------------
# 2. two modalities get DIFFERENT thresholds
# ---------------------------------------------------------------------------

def test_by_modality_two_modalities_get_different_thresholds():
    eda = _dyads(_smooth, n_dyads=4, seed=11)
    ecg = _dyads(_white, n_dyads=4, seed=22)
    sigs = eda + ecg
    mods = ["EDA"] * 4 + ["ECG"] * 4
    thr = compute_session_pooled_thresholds_by_modality(
        sigs, mods, hz=1.0, wcc_window_size=24, surrogate_n=100, seed=42
    )
    # Smooth (EDA-like) signals carry higher spurious correlation than white
    # (ECG-like) noise, so the per-modality 95th-percentile nulls differ.
    assert thr["EDA"] > thr["ECG"], (
        f"expected EDA null > ECG null; got EDA={thr['EDA']:.3f}, ECG={thr['ECG']:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. degenerate modality falls back to ONSET_THRESHOLD (0.5)
# ---------------------------------------------------------------------------

def test_by_modality_fallback_on_degenerate_modality():
    # white-noise (ECG-like) modality stays well under the 0.90 sanity ceiling
    # and gets a derived threshold; constant (degenerate) modality falls back.
    good = _dyads(_white, n_dyads=4, seed=5)
    bad = [(_constant(240), _constant(240)) for _ in range(3)]
    sigs = good + bad
    mods = ["ECG"] * 4 + ["CONST"] * 3
    thr = compute_session_pooled_thresholds_by_modality(
        sigs, mods, hz=1.0, wcc_window_size=24, surrogate_n=50, seed=7
    )
    # degenerate modality still emitted, at the fixed fallback
    assert "CONST" in thr
    assert thr["CONST"] == pytest.approx(ONSET_THRESHOLD)
    # healthy modality gets a derived (non-fallback) threshold
    assert thr["ECG"] != pytest.approx(ONSET_THRESHOLD)


# ---------------------------------------------------------------------------
# 4. length mismatch fails loud
# ---------------------------------------------------------------------------

def test_by_modality_length_mismatch_raises():
    sigs = _dyads(_smooth, n_dyads=2, seed=1)
    mods = ["EDA", "EDA", "ECG"]  # one too many
    with pytest.raises(ValueError):
        compute_session_pooled_thresholds_by_modality(
            sigs, mods, hz=1.0, wcc_window_size=24, surrogate_n=20, seed=0
        )


# ---------------------------------------------------------------------------
# 5. symmetry: per-modality == global pooled on each group
# ---------------------------------------------------------------------------

def test_by_modality_matches_global_per_group():
    eda = _dyads(_smooth, n_dyads=3, seed=3)
    ecg = _dyads(_white, n_dyads=3, seed=4)
    sigs = eda + ecg
    mods = ["EDA"] * 3 + ["ECG"] * 3
    thr = compute_session_pooled_thresholds_by_modality(
        sigs, mods, hz=1.0, wcc_window_size=24, surrogate_n=60, seed=9
    )
    # Each per-modality threshold must equal the global pooled threshold
    # computed on that modality's signals alone (per-modality <-> global
    # symmetry).
    for mod, group in (("EDA", eda), ("ECG", ecg)):
        global_thr, _ = compute_session_pooled_threshold(
            group, hz=1.0, wcc_window_size=24, surrogate_n=60, seed=9
        )
        assert thr[mod] == pytest.approx(global_thr, rel=1e-9), mod


# ---------------------------------------------------------------------------
# 6. bridge "session_pooled" resolves EDA != ECG and records them
# ---------------------------------------------------------------------------

class _FakeRecord:
    def __init__(self, dyad, modality, condition, a, b, hz=1.0):
        self.dyad_label = dyad
        self.modality = modality
        self.condition = condition
        self.person_a = a
        self.person_b = b
        self.target_hz = hz
        self.incomplete = False
        self.discontinuity_mask = None


def test_bridge_session_pooled_returns_per_modality_thresholds():
    recs = []
    for i in range(3):
        recs.append(_FakeRecord(f"d{i}", "EDA", "rest",
                                _smooth(seed=100 + i), _smooth(seed=200 + i)))
        recs.append(_FakeRecord(f"d{i}", "ECG", "rest",
                                _white(seed=300 + i), _white(seed=400 + i)))

    inputs = records_to_inference_inputs(
        recs, hz=1.0, window_size=24, onset_threshold="session_pooled"
    )
    assert inputs.thresholds_by_modality is not None
    assert set(inputs.thresholds_by_modality.keys()) == {"EDA", "ECG"}
    assert inputs.thresholds_by_modality["EDA"] > inputs.thresholds_by_modality["ECG"]


def test_bridge_fixed_threshold_forwarded_unchanged():
    recs = [_FakeRecord("d0", "EDA", "rest", _smooth(seed=1), _smooth(seed=2))]
    inputs = records_to_inference_inputs(
        recs, hz=1.0, window_size=24, onset_threshold=0.42
    )
    # fixed numeric threshold -> no per-modality computation
    assert inputs.thresholds_by_modality is None
    assert len(inputs.features_df) == 1
    assert inputs.features_df["modality"].iloc[0] == "EDA"

"""Unit tests for the ``multisync.realtest`` loaders (no real data required).

These tests pin the honest-propagation contracts introduced when the loaders
were hardened against fabricated signals:

1. Lerique filename parsing (``_parse_filename``) accepts the canonical
   ``pce<NN>_P<1|2>_<Rest|Trial><K>.mat`` pattern and rejects malformed names.
2. ``_interp_outlier_ibi`` raises ``ValueError`` when 100% of beats are
   artifacts (no genuine IBI) instead of returning a fabricated constant.
3. ``_preprocess_ecg`` returns an all-``False`` mask (not a real mask) when
   R-peak detection fails, so the placeholder trace is gated out downstream
   rather than correlated as a genuine signal.
4. Gordon ``gordon_record_to_multisync_dyad`` builds a discontinuity mask
   that is False wherever either person's channel is non-finite.

The data-dependent loaders (``load_*_dataset``) are exercised separately in
integration/smoke scripts; here we only test the pure-logic pieces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multisync.realtest import lerique_2024 as L
from multisync.realtest import gordon_2025 as G


# ---------------------------------------------------------------------------
# 1. Lerique filename parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,dyad,person,cond,seg",
    [
        ("pce02_P1_Rest1.mat", "02", "1", "Rest", "1"),
        ("pce32_P2_Trial18.mat", "32", "2", "Trial", "18"),
        ("pce01_P2_Rest4.mat", "01", "2", "Rest", "4"),
    ],
)
def test_parse_filename_valid(name, dyad, person, cond, seg):
    meta = L._parse_filename(name)
    assert meta is not None
    assert meta["dyad"] == dyad
    assert meta["person"] == person
    assert meta["cond"] == cond
    assert meta["seg"] == seg


@pytest.mark.parametrize(
    "name",
    [
        "pce2_P1_Rest1.mat",      # dyad not zero-padded to 2 digits
        "pce02_P3_Rest1.mat",     # person must be 1 or 2
        "pce02_P1_rest1.mat",     # cond case-sensitive
        "pce02_P1_Rest1.csv",     # wrong extension
        "other_P1_Rest1.mat",     # wrong prefix
        "pce02_P1_Rest1.mat.bak", # trailing junk
    ],
)
def test_parse_filename_invalid(name):
    assert L._parse_filename(name) is None


# ---------------------------------------------------------------------------
# 2. IBI all-artifact branch raises (no fabricated constant)
# ---------------------------------------------------------------------------

def test_interp_outlier_ibi_all_bad_raises():
    # Every IBI outside [_IBI_MIN_SEC, _IBI_MAX_SEC] => nothing genuine.
    ibi = np.array([5.0, 9.0, 0.1])
    rpeak_t = np.array([0.0, 5.0, 9.0, 0.1])
    with pytest.raises(ValueError, match="100%"):
        L._interp_outlier_ibi(ibi, rpeak_t)


def test_interp_outlier_ibi_partial_bad_interpolates():
    # One bad beat in the middle is linearly interpolated from good beats.
    ibi = np.array([0.8, 5.0, 0.9])           # middle beat is an artifact
    rpeak_t = np.array([0.0, 0.8, 5.0, 5.9])
    clean, mid = L._interp_outlier_ibi(ibi, rpeak_t)
    assert np.isfinite(clean).all()
    assert clean[0] == pytest.approx(0.8)
    assert clean[2] == pytest.approx(0.9)
    # Interpolated middle value lies between its good neighbours.
    assert 0.8 < clean[1] < 0.9
    assert len(mid) == len(ibi)


# ---------------------------------------------------------------------------
# 3. ECG R-peak failure => mask all False (placeholder gated out)
# ---------------------------------------------------------------------------

def test_preprocess_ecg_too_few_rpeaks_mask_all_false(monkeypatch):
    # Force neurokit2 to report < 2 R-peaks regardless of input.
    class _FakeNK:
        @staticmethod
        def ecg_peaks(filtered, sampling_rate):
            return None, {"ECG_R_Peaks": np.array([100])}

    monkeypatch.setitem(__import__("sys").modules, "neurokit2", _FakeNK)

    raw = np.zeros(10_000, dtype=np.float32)  # 10 s at 1000 Hz
    sig, mask = L._preprocess_ecg(raw, raw_fs=1000.0, target_fs=1.0)
    assert sig.shape == mask.shape
    # Placeholder signal is present (shape contract) but mask is ALL False.
    assert not mask.any(), "failed R-peak trace must be gated out, not correlated"


# ---------------------------------------------------------------------------
# 4. Gordon discontinuity mask from non-finite channels
# ---------------------------------------------------------------------------

def _gordon_record(a_vals, b_vals):
    n = len(a_vals)
    t = np.arange(n, dtype=float) / G.DEFAULT_TARGET_HZ
    pa = pd.DataFrame({"time": t, "motion_intensity": np.asarray(a_vals, float)})
    pb = pd.DataFrame({"time": t, "motion_intensity": np.asarray(b_vals, float)})
    return G.GordonDyadCondition(
        dyad_id="p1_p2__exp1",
        pair_label="p1_p2",
        condition="exp1",
        person_a=pa,
        person_b=pb,
        target_hz=G.DEFAULT_TARGET_HZ,
        n_samples=n,
        duration_sec=float(t[-1] - t[0]),
    )


def test_gordon_mask_false_where_either_person_nan():
    a = [1.0, np.nan, 3.0, 4.0]
    b = [1.0, 2.0, np.nan, 4.0]
    rec = _gordon_record(a, b)
    dyad = G.gordon_record_to_multisync_dyad(rec)
    mask = np.asarray(dyad.discontinuity_mask)
    # index 1 (a NaN), index 2 (b NaN) gated; 0 and 3 usable.
    assert mask.tolist() == [True, False, False, True]


def test_gordon_mask_all_true_when_clean():
    rec = _gordon_record([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    dyad = G.gordon_record_to_multisync_dyad(rec)
    assert np.asarray(dyad.discontinuity_mask).all()

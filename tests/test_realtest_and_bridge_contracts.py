"""Contract tests for the real-data loader + pipeline-bridge path.

These are the "fuses" that Blind Spot B of the v1.0 review asked for:
the loader -> bridge -> inference chain is exactly where the
``discontinuity_mask`` single-sided-drop bug lived, yet it had *zero*
dedicated tests.  A later refactor of WCC defaults or the loaders could
silently corrupt the numbers behind the paper without any test catching
it.  These tests pin the contract so such regressions fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multisync.pipeline_bridge import records_to_inference_inputs
from multisync.realtest.gordon_2025 import (
    GordonDyadCondition,
    gordon_record_to_multisync_dyad,
    load_gordon_dataset,
)
from multisync.realtest.lerique_2024 import (
    LeriqueDyadCondition,
    _resample_mask_to_target,
    lerique_record_to_multisync_dyad,
    load_lerique_dataset,
)

N = 50
HZ = 5.0
T = np.arange(N) / HZ


def _lerique_rec(mask: np.ndarray) -> LeriqueDyadCondition:
    a = pd.DataFrame({"time": T, "value": np.sin(T)})
    b = pd.DataFrame({"time": T, "value": np.cos(T)})
    return LeriqueDyadCondition(
        dyad_id="pce01__ecg__trials_concat",
        dyad_label="pce01",
        modality="ecg",
        condition="trials_concat",
        person_a=a,
        person_b=b,
        target_hz=HZ,
        n_samples=N,
        duration_sec=N / HZ,
        incomplete=False,
        discontinuity_mask=mask,
    )


def _gordon_rec() -> GordonDyadCondition:
    a = pd.DataFrame({
        "time": T,
        "motion_intensity": np.sin(T),
        "R": np.ones(N),
        "theta": np.zeros(N),
    })
    b = pd.DataFrame({
        "time": T,
        "motion_intensity": np.cos(T),
        "R": np.ones(N),
        "theta": np.zeros(N),
    })
    return GordonDyadCondition(
        dyad_id="p1_p2__exp1",
        pair_label="p1_p2",
        condition="exp1",
        person_a=a,
        person_b=b,
        target_hz=HZ,
        n_samples=N,
        duration_sec=N / HZ,
    )


# ---------------------------------------------------------------------------
# Lerique record -> Dyad bridge: the discontinuity_mask must survive the
# conversion. This is the *bridge* path; the loader-side AND logic is covered
# separately by test_lerique_loader_combines_person_masks* below.
# ---------------------------------------------------------------------------
def test_lerique_record_to_dyad_passes_discontinuity_mask():
    mask = np.zeros(N, dtype=bool)
    mask[10:20] = True  # arbitrary internal-segment flag
    rec = _lerique_rec(mask)
    dyad = lerique_record_to_multisync_dyad(rec)
    assert dyad.discontinuity_mask is not None
    assert np.array_equal(dyad.discontinuity_mask, mask)


def test_lerique_loader_rejects_incomplete():
    rec = _lerique_rec(np.ones(N, dtype=bool))
    rec = rec.__class__(**{**rec.__dict__, "incomplete": True})
    with pytest.raises(ValueError):
        lerique_record_to_multisync_dyad(rec)


# ---------------------------------------------------------------------------
# Gordon record -> Dyad bridge: yields the expected cross-person modalities.
# (The real loader, load_gordon_dataset, is exercised by
# test_gordon_loader_returns_records below.)
# ---------------------------------------------------------------------------
def test_gordon_record_to_dyad_produces_modalities():
    dyad = gordon_record_to_multisync_dyad(_gordon_rec())
    assert "motion_intensity_a" in dyad.modalities
    assert "motion_intensity_b" in dyad.modalities


# ---------------------------------------------------------------------------
# Lerique loader: exercises load_lerique_dataset end-to-end on a synthetic
# on-disk dataset. This is the ORIGINAL single-sided mask-drop bug site
# (the P1/P2 discontinuity_mask AND logic at lerique_2024.py:906-918 and
# :888-901) -- the previous bridge-only test never reached it.
# ---------------------------------------------------------------------------
def _write_lerique_mat(path: Path, n_samples: int) -> None:
    """Write a minimal Lerique-style .mat (shape (1, N), float32)."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    savemat(str(path), {"sig": rng.random((1, n_samples)).astype(np.float32)})


def test_lerique_loader_combines_person_masks(tmp_path):
    """preprocess=False: the loader must AND-combine P1/P2 boundary masks,
    not silently drop one side. P1 and P2 have *different* segment seams but
    the same total length, so any single-sided drop would be visible.
    """
    root = tmp_path / "lerique"
    ecg = root / "ECG"
    ecg.mkdir(parents=True)
    pce = ecg / "pce01"
    pce.mkdir()
    # P1: three equal 100-sample segments -> seams at idx 100, 200
    for seg, n in (("Rest2", 100), ("Rest3", 100), ("Rest4", 100)):
        _write_lerique_mat(pce / f"pce01_P1_{seg}.mat", n)
    # P2: same total length, different split -> seams at idx 50, 250
    for seg, n in (("Rest2", 50), ("Rest3", 200), ("Rest4", 50)):
        _write_lerique_mat(pce / f"pce01_P2_{seg}.mat", n)

    recs = load_lerique_dataset(
        root, modalities=["ECG"], condition_units=["rest_postblock"],
        preprocess=False, drop_short_duration=False,
    )
    assert len(recs) == 1
    mask = recs[0].discontinuity_mask
    assert mask.shape == (300,)

    a_mask = np.ones(300, dtype=bool)
    a_mask[100] = a_mask[200] = False
    b_mask = np.ones(300, dtype=bool)
    b_mask[50] = b_mask[250] = False
    # Combined mask must equal the elementwise AND of both persons.
    assert np.array_equal(mask, a_mask & b_mask)
    # Regression guard: a seam present in only ONE person still forces the
    # combined mask False -> neither side was silently kept alone.
    assert not mask[100]   # P1 seam, P2 usable here
    assert not mask[50]    # P2 seam, P1 usable here
    assert mask[150]       # usable for both -> stays True


def test_lerique_loader_combines_person_masks_preprocessed(tmp_path):
    """preprocess=True (EDA, scipy-only, no neurokit2): same AND-mask
    invariant, but through the resampled target_fs path."""
    raw_fs, target_fs = 1000.0, 1.0
    root = tmp_path / "lerique2"
    eda = root / "EDA"
    eda.mkdir(parents=True)
    pce = eda / "pce01"
    pce.mkdir()
    # P1: three 50000-sample segments -> raw seams at 50000, 100000
    for seg, n in (("Rest2", 50000), ("Rest3", 50000), ("Rest4", 50000)):
        _write_lerique_mat(pce / f"pce01_P1_{seg}.mat", n)
    # P2: same total but different split -> raw seams at 25000, 125000
    for seg, n in (("Rest2", 25000), ("Rest3", 100000), ("Rest4", 25000)):
        _write_lerique_mat(pce / f"pce01_P2_{seg}.mat", n)

    recs = load_lerique_dataset(
        root, modalities=["EDA"], condition_units=["rest_postblock"],
        preprocess=True, raw_fs=raw_fs, target_fs=target_fs,
    )
    assert len(recs) == 1
    mask = recs[0].discontinuity_mask
    assert mask.size > 0

    n_tgt = mask.size
    p1_boundary = np.ones(150000, dtype=bool)
    p1_boundary[50000] = p1_boundary[100000] = False
    p2_boundary = np.ones(150000, dtype=bool)
    p2_boundary[25000] = p2_boundary[125000] = False
    a_rs = _resample_mask_to_target(p1_boundary, raw_fs, target_fs, n_tgt)
    b_rs = _resample_mask_to_target(p2_boundary, raw_fs, target_fs, n_tgt)
    # Combined mask = AND of each person's resampled mask.
    assert np.array_equal(mask, a_rs & b_rs)
    assert not np.array_equal(mask, a_rs)   # P1 side not kept alone
    assert not np.array_equal(mask, b_rs)   # P2 side not kept alone


def test_gordon_loader_returns_records(tmp_path):
    """Smoke test that exercises load_gordon_dataset end-to-end on a
    synthetic CSV tree (the real loader, not the record -> dyad bridge)."""
    root = tmp_path / "gordon"
    bdata = root / "behavior data"
    bdata.mkdir(parents=True)
    pair = bdata / "p1_p2"
    pair.mkdir()
    n = 60
    t = np.arange(n) / 10.0
    df = pd.DataFrame({
        "time_p1": t,
        "R_p1": np.ones(n),
        "theta_p1": np.linspace(0.0, 1.0, n),
        "time_p2": t,
        "R_p2": np.ones(n),
        "theta_p2": np.linspace(0.0, 1.0, n),
    })
    df.to_csv(pair / "exp1.csv", index=False)
    recs = load_gordon_dataset(root, target_hz=10.0)
    assert len(recs) >= 1
    rec = recs[0]
    assert "motion_intensity" in rec.person_a.columns
    assert "motion_intensity" in rec.person_b.columns
    assert len(rec.person_a) > 0
    assert len(rec.person_b) > 0


# ---------------------------------------------------------------------------
# pipeline_bridge: no KeyError + mask propagates into InferenceInputs
# (re-guards the prior single-sided mask-drop bug)
# ---------------------------------------------------------------------------
def test_bridge_propagates_discontinuity_mask():
    mask = np.zeros(N, dtype=bool)
    mask[5:15] = True
    rec = _lerique_rec(mask)
    inputs = records_to_inference_inputs(
        [rec], hz=HZ, window_size=20,
    )
    assert inputs.discontinuity_mask is not None
    key = "pce01__ecg__trials_concat"
    assert key in inputs.discontinuity_mask
    assert np.array_equal(inputs.discontinuity_mask[key], mask)
    assert not inputs.features_df.empty


def test_bridge_handles_missing_mask_gracefully():
    # A record whose discontinuity_mask attribute is None must not crash
    # and must round-trip as None in the per-key map.
    rec = SimpleNamespace(
        dyad_label="pce02",
        modality="ecg",
        condition="rest1",
        target_hz=HZ,
        incomplete=False,
        person_a=np.sin(T),
        person_b=np.cos(T),
        discontinuity_mask=None,
    )
    inputs = records_to_inference_inputs([rec], hz=HZ, window_size=20)
    key = "pce02__ecg__rest1"
    assert inputs.discontinuity_mask is not None
    assert inputs.discontinuity_mask.get(key) is None

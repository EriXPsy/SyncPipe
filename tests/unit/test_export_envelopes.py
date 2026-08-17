"""Unit tests for the envelope exporter's pure conversion function.

These tests do NOT require the OSF mirror: they build synthetic
``LeriqueDyadCondition`` records directly and verify that
``records_to_long_table`` flattens them correctly (alignment, mask
propagation, incomplete-record skipping) without touching the loader's
`.mat` machinery (which is covered separately in test_pipeline_io.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from syncpipe.realtest.lerique_2024 import LeriqueDyadCondition
from scripts.export_envelopes import records_to_long_table


def _make_record(dyad="pce01", modality="EDA", condition="rest1",
                 n=10, incomplete=False, a=None, b=None, mask=None) -> LeriqueDyadCondition:
    t = np.arange(n, dtype=np.float64)
    if a is None:
        a = pd.DataFrame({"time": t, "value": np.arange(n, dtype=np.float64)})
    if b is None:
        b = pd.DataFrame({"time": t, "value": np.arange(n, dtype=np.float64) + 0.5})
    if mask is None:
        mask = np.ones(n, dtype=bool)
    return LeriqueDyadCondition(
        dyad_id=f"{dyad}__{modality}__{condition}",
        dyad_label=dyad,
        modality=modality,
        condition=condition,
        person_a=a,
        person_b=b,
        target_hz=1.0,
        n_samples=n,
        duration_sec=float(n),
        incomplete=incomplete,
        discontinuity_mask=mask,
        meta={"p1_segment_paths": [], "p2_segment_paths": []},
    )


def test_records_to_long_table_basic():
    rec = _make_record(n=5)
    df = records_to_long_table([rec])
    assert list(df.columns) == ["dyad", "modality", "condition", "time",
                                "person_a", "person_b", "mask"]
    assert len(df) == 5
    assert (df["dyad"] == "pce01").all()
    assert (df["modality"] == "EDA").all()
    assert (df["mask"] == 1).all()


def test_records_to_long_table_propagates_mask():
    mask = np.ones(6, dtype=bool)
    mask[2] = False  # a segment seam
    rec = _make_record(n=6, mask=mask)
    df = records_to_long_table([rec])
    assert df.loc[2, "mask"] == 0
    assert df.loc[0, "mask"] == 1


def test_records_to_long_table_skips_incomplete():
    good = _make_record(dyad="pce01")
    bad = _make_record(dyad="pce02", incomplete=True, b=None)
    df = records_to_long_table([good, bad])
    assert (df["dyad"] == "pce01").all()
    assert "pce02" not in set(df["dyad"])


def test_records_to_long_table_aligns_to_shortest():
    # person_b shorter than person_a + mask -> all truncated to min length.
    from dataclasses import replace
    rec = _make_record(n=8)
    rec = replace(
        rec,
        person_b=rec.person_b.iloc[:5],
        discontinuity_mask=np.ones(5, dtype=bool),
        n_samples=5,
        duration_sec=5.0,
    )
    df = records_to_long_table([rec])
    assert len(df) == 5


def test_records_to_long_table_sorted_deterministically():
    r1 = _make_record(dyad="pce02", condition="trials_concat", n=3)
    r2 = _make_record(dyad="pce01", condition="rest1", n=3)
    df = records_to_long_table([r1, r2])
    # sorted by (modality, dyad, condition, time)
    first = df.iloc[0]
    assert first["dyad"] == "pce01" and first["condition"] == "rest1"

"""Regression tests for gstack Finding 6 (corrected): L1 silent miscount.

A WCC too short for a valid L1 surrogate test must be reported as
"test not applicable" and EXCLUDED from the L1 denominator (l1_total /
n_l1), not silently counted as "L1 not significant".  Counting it as not
significant deflates the reported L1 pass rate in datasets that contain a
non-trivial fraction of short traces.
"""

import numpy as np
import pandas as pd
import pytest

from multisync.inference_pipeline import InferencePipeline
from multisync.dynamic_features import wcc_surrogate_test, sliding_window_wcc
from multisync.feature_definitions import extract_features


def _make_signals(rng, coupling, n=300):
    t = np.arange(n)
    a = np.sin(2 * np.pi * 0.05 * t) + coupling * np.sin(2 * np.pi * 0.05 * t + 0.3)
    b = np.sin(2 * np.pi * 0.05 * t + 0.3) + coupling * np.sin(2 * np.pi * 0.05 * t)
    a = a + 0.1 * rng.standard_normal(n)
    b = b + 0.1 * rng.standard_normal(n)
    return a, b


def test_wcc_level_early_return_is_not_applicable():
    rng = np.random.default_rng(0)
    # A 10-point WCC is below min_wcc_points (30) -> early return.
    short_wcc = rng.standard_normal(10)
    res = wcc_surrogate_test(
        short_wcc, hz=4.0, surrogate_n=5, seed=0, raw_signals=None,
        null_model="state_shuffle",
    )
    assert res.get("applicable", True) is False
    assert "too short" in res.get("notes", "").lower()


def test_wcc_level_normal_return_is_applicable():
    rng = np.random.default_rng(1)
    a, b = _make_signals(rng, 0.5, n=300)
    wcc = sliding_window_wcc(a, b, window_size=40, hz=4.0)
    res = wcc_surrogate_test(
        wcc, hz=4.0, surrogate_n=5, seed=0, raw_signals=None,
        null_model="state_shuffle",
    )
    assert res.get("applicable", False) is True


def _build_split_dataset():
    """Four dyads x two conditions. Odd-indexed dyads (d1, d3) use short
    signals so their WCC is below min_wcc_points -> L1 'not applicable'.
    Even dyads (d0, d2) use long signals."""
    rng = np.random.default_rng(3)
    window = 40
    hz = 4.0

    def sig(coupling, n):
        return _make_signals(rng, coupling, n=n)

    wcc_dict, raw_dict, rows = {}, {}, []
    for di in range(4):
        short = (di % 2 == 1)
        n = 50 if short else 400
        for cond, coup in (("rest", 0.6), ("task", 0.7)):
            a, b = sig(coup, n)
            wcc = sliding_window_wcc(a, b, window_size=window, hz=hz)
            feat = extract_features(wcc, hz=hz, wcc_window_sec=window / hz)
            label = f"d{di}__{cond}"
            wcc_dict[label] = wcc
            raw_dict[label] = (a, b)
            rows.append(
                {"dyad_id": f"d{di}", "condition": cond, "label": label,
                 **feat.to_dict()}
            )
    df = pd.DataFrame(rows)
    return df, wcc_dict, raw_dict, window, hz


@pytest.mark.slow
def test_run_full_cascade_excludes_inapplicable_l1_from_denominator():
    df, wcc_dict, raw_dict, window, hz = _build_split_dataset()
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=5, seed=1)

    result = pipe.run_full_cascade(
        raw_signals_dict=raw_dict,
        wcc_dict=wcc_dict,
        wcc_window_size=window,
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=20,
    )

    # L1: only d0/d2's four labels are applicable; d1/d3's four short-WCC
    # labels are excluded from the denominator (not counted as
    # L1-not-significant).
    assert result["l1_summary"]["total"] == 4, (
        "short-WCC dyads must be excluded from l1_total, not counted as "
        "L1-not-significant"
    )
    # L0 is unaffected: all eight labels have raw signals and run L0.
    assert result["l0_summary"]["total"] == 8


def test_summarize_excludes_inapplicable_l1():
    wcc_n = sliding_window_wcc(*_make_signals(np.random.default_rng(5), 0.6, n=400),
                               window_size=40, hz=4.0)
    short_wcc = np.random.default_rng(5).standard_normal(10)

    pipe = InferencePipeline(features_df=pd.DataFrame(), hz=4.0, surrogate_n=5, seed=1)
    pipe.test_l1_structure(wcc_n, label="normal")
    pipe.test_l1_structure(short_wcc, label="short")

    summary = pipe.summarize()
    # denominator must be 1 (only the applicable dyad), not 2.
    assert "L1 (WCC-level IAAFT): 0/1 significant" in summary, (
        f"summarize must exclude inapplicable L1 dyad from denominator; got:\n{summary}"
    )

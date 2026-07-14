"""Regression tests for Claude review findings #1 (v1b) and #4 (fix ①).

#1 v1b: ``nan_fraction`` (discontinuity-masked WCC fraction) is reported
alongside ``dwell_time`` / ``switching_rate`` as a mandatory diagnostic.

#4 ①: ``morphology_feature_table`` / ``MorphologyAnalyzer.feature_table``
forward ``onset_threshold`` so the extracted SyncPipe features share the
main pipeline's surrogate-derived threshold instead of silently falling back
to the locked 0.5 default.
"""

import numpy as np

from multisync import dynamic_features as df_mod
from multisync.feature_definitions import DynamicFeatures
from multisync.morphology import morphology_feature_table, MorphologyAnalyzer


def _make_wcc(n=200, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    w = 0.5 + 0.3 * rng.normal(0, 1, n)
    if nan_frac > 0:
        k = int(n * nan_frac)
        w[:k] = np.nan
    return w


def _make_bimodal_wcc(n=300, seed=0):
    # WCC with a MID run (~0.6, above 0.5 but below 0.7) and a HIGH run
    # (~0.9, above both). At threshold 0.5 both runs count toward dwell;
    # at threshold 0.7 only the HIGH run counts, so dwell_time(0.7) <
    # dwell_time(0.5) — proving the threshold reaches the feature math.
    rng = np.random.default_rng(seed)
    base = np.full(n, 0.3)
    base[50:120] = 0.6    # mid run (len 70)
    base[150:180] = 0.9   # high run (len 30)
    return base + 0.02 * rng.normal(0, 1, n)


# ---------------------------------------------------------------------------
# Finding #1 (v1b): nan_fraction diagnostic
# ---------------------------------------------------------------------------

def test_nan_fraction_recorded_by_extract_dynamic_features():
    w = _make_wcc(n=200, seed=1, nan_frac=0.1)  # exactly 20 NaN of 200
    feats = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0)
    assert isinstance(feats, DynamicFeatures)
    assert abs(feats.nan_fraction - 0.1) < 1e-9


def test_nan_fraction_in_to_dict():
    w = _make_wcc(n=200, seed=2, nan_frac=0.25)
    d = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0).to_dict()
    assert "nan_fraction" in d
    assert abs(d["nan_fraction"] - 0.25) < 1e-9


def test_nan_fraction_round_trip_via_from_dict():
    w = _make_wcc(n=200, seed=3, nan_frac=0.2)
    f = df_mod.extract_dynamic_features(w, hz=1.0, wcc_window_sec=1.0)
    f2 = DynamicFeatures.from_dict(f.to_dict())
    assert abs(float(f2.nan_fraction) - 0.2) < 1e-9


def test_morphology_table_surfaces_nan_fraction():
    traces = [
        _make_wcc(n=200, seed=4, nan_frac=0.2),
        _make_wcc(n=200, seed=5, nan_frac=0.0),
    ]
    tbl = morphology_feature_table(traces, hz=1.0)
    assert "nan_fraction" in tbl.columns
    # nan_fraction travels in the same report as dwell_time / switching_rate
    assert "dwell_time" in tbl.columns
    assert "switching_rate" in tbl.columns
    vals = tbl["nan_fraction"].to_numpy()
    assert abs(vals[0] - 0.2) < 1e-9
    assert abs(vals[1] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Finding #4 (fix ①): onset_threshold forwarded into morphology features
# ---------------------------------------------------------------------------

def test_morphology_feature_table_threads_onset_threshold():
    traces = [_make_bimodal_wcc(n=300, seed=7)]
    tbl_05 = morphology_feature_table(traces, hz=1.0, onset_threshold=0.5)
    tbl_07 = morphology_feature_table(traces, hz=1.0, onset_threshold=0.7)
    d05 = tbl_05["dwell_time"].iloc[0]
    d07 = tbl_07["dwell_time"].iloc[0]
    # A higher threshold trims the above-run -> shorter dwell time, proving the
    # threshold (not the silent 0.5 default) reached the feature computation.
    assert d07 < d05, (d05, d07)
    assert not np.isnan(d05) and not np.isnan(d07)


def test_morphologyanalyzer_feature_table_forwards_threshold():
    traces = [_make_bimodal_wcc(n=300, seed=9)]
    m = MorphologyAnalyzer(traces, hz=1.0)
    t05 = m.feature_table(onset_threshold=0.5)["dwell_time"].iloc[0]
    t07 = m.feature_table(onset_threshold=0.7)["dwell_time"].iloc[0]
    assert t07 < t05

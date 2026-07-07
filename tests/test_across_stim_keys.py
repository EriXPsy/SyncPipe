"""Regression test for the across-stim shuffle dict-key bug.

The loop variable `k` (feature name) was being overwritten by an integer
Phipson-Smyth count and then used as the results dict key, so the returned
dict was keyed by integers (e.g. [1, 11, 18]) instead of feature names.
"""
from __future__ import annotations

import numpy as np

from multisync.validation.across_stim_shuffle import across_stim_shuffle_test


def _fake_wcc(p1, p2):
    return np.correlate(p1, p2, mode="full")


def _fake_feats(wcc):
    return {
        "mean_synchrony": float(np.mean(wcc)),
        "peak_amplitude": float(np.max(wcc)),
        "switching_rate": float(np.std(wcc)),
    }


def _make_segments(n_seg=4, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (f"seg{i}", rng.normal(size=n), rng.normal(size=n))
        for i in range(n_seg)
    ]


def test_across_stim_keys_are_feature_names():
    segs = _make_segments()
    feats = ["mean_synchrony", "peak_amplitude", "switching_rate"]
    res = across_stim_shuffle_test(
        segs,
        wcc_func=_fake_wcc,
        feature_func=_fake_feats,
        n_surr=200,
        seed=1,
        feature_names=feats,
    )
    # The critical fix: keys must be the requested feature names, not
    # integers derived from the Phipson-Smyth count.
    assert set(res.keys()) == set(feats), f"unexpected keys: {list(res.keys())}"
    for k in feats:
        assert isinstance(res[k], dict)
        assert "real" in res[k]
        assert "p_value" in res[k]


def test_across_stim_no_silent_overwrite():
    """Two features that happen to yield the same extreme-count must not
    silently overwrite one another — both must survive keyed by name."""
    segs = _make_segments(n_seg=6, seed=3)
    feats = ["mean_synchrony", "peak_amplitude", "switching_rate", "dwell_time"]
    res = across_stim_shuffle_test(
        segs,
        wcc_func=_fake_wcc,
        feature_func=_fake_feats,
        n_surr=300,
        seed=7,
        feature_names=feats,
    )
    assert set(res.keys()) == set(feats)
    assert len(res) == len(feats)

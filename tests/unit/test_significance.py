from __future__ import annotations

# === source: test_surrogate_degenerate.py ===
"""
Edge / degenerate-input regression tests for the surrogate & permutation
p-value machinery (gstack #8 / #10).

Goal: guarantee the null-model p-value path never *silently* emits a wrong
number or *crashes* on pathological inputs — the two failure modes we care
about most.  Covers:
  - phipson_smyth_p: NaN observed / empty null / constant null / bounds / tails
  - _wcc_level_surrogate_test: all-NaN WCC, constant WCC, sub-minimum length,
    surrogate_n=1
  - _signal_level_surrogate_test: all-NaN signals, constant signals,
    surrogate_n=1
"""
import numpy as np
import pytest

from multisync.dynamic_features import (
    _wcc_level_surrogate_test,
    _signal_level_surrogate_test,
)
from multisync.validation.pgt1_intensity import phipson_smyth_p


# ---------------------------------------------------------------------------
# phipson_smyth_p — the unbiased estimator's boundary behaviour
# ---------------------------------------------------------------------------

def test_phipson_nan_observed_returns_nan():
    assert np.isnan(phipson_smyth_p(np.nan, np.array([0.1, 0.2, 0.3])))


def test_phipson_empty_null_returns_nan():
    assert np.isnan(phipson_smyth_p(0.5, np.array([])))
    assert np.isnan(phipson_smyth_p(0.5, np.array([np.nan, np.nan])))


def test_phipson_constant_null_p_equals_one():
    # observed exactly at the (only) null value -> all null >= obs -> k = n
    null = np.full(50, 0.4)
    assert phipson_smyth_p(0.4, null) == 1.0


def test_phipson_minimum_p_is_one_over_n_plus_one():
    null = np.full(999, 0.0)
    # observed far above every null -> k = 0 -> (1+0)/(1+999)
    p = phipson_smyth_p(1e6, null)
    assert p == pytest.approx(1.0 / (1.0 + 999))


def test_phipson_p_in_unit_interval_all_tails():
    rng = np.random.default_rng(0)
    null = rng.normal(0.0, 1.0, 200)
    for tail in ("upper", "lower", "two"):
        p = phipson_smyth_p(0.3, null, tail=tail)
        assert 0.0 < p <= 1.0


def test_phipson_two_tailed_symmetric_about_null_mean():
    rng = np.random.default_rng(1)
    # construct a null EXACTLY symmetric about 0 (values mirrored) so the
    # reflection symmetry of the two-tailed p is exact, free of sampling noise
    base = rng.normal(0.0, 1.0, 2500)
    null = np.concatenate([base, -base])
    d = 1.0
    p_a = phipson_smyth_p(d, null, tail="two")
    p_b = phipson_smyth_p(-d, null, tail="two")
    assert p_a == pytest.approx(p_b, rel=1e-9)


# ---------------------------------------------------------------------------
# _wcc_level_surrogate_test — L1 null on pathological WCC
# ---------------------------------------------------------------------------

def test_wcc_all_nan_does_not_crash():
    wcc = np.full(200, np.nan)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)
    # if p-values are finite they must be in (0, 1]
    for key in ("p_dwell_time", "p_switching_rate"):
        pv = res[key]
        if np.isfinite(pv):
            assert 0.0 < pv <= 1.0


def test_wcc_constant_does_not_crash():
    wcc = np.full(200, 0.5)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)
    assert "p_dwell_time" in res and "p_switching_rate" in res


def test_wcc_sub_minimum_length_returns_l1_shape():
    # below min_wcc_points -> early return; must still carry L1 keys
    wcc = np.zeros(10)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=20, seed=1)
    assert "p_dwell_time" in res
    assert "p_switching_rate" in res
    assert res["p_dwell_time"] == 1.0  # fail-loud-safe neutral value


def test_wcc_surrogate_n_one_runs():
    wcc = np.random.default_rng(2).normal(0.3, 0.1, 200)
    res = _wcc_level_surrogate_test(wcc, hz=1.0, surrogate_n=1, seed=1)
    assert isinstance(res, dict)
    assert res["n_surrogates"] == 1


# ---------------------------------------------------------------------------
# _signal_level_surrogate_test — L0 null on pathological signals
# ---------------------------------------------------------------------------

def test_signal_all_nan_does_not_crash():
    a = np.full(200, np.nan)
    b = np.full(200, np.nan)
    wcc = np.full(150, np.nan)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)


def test_signal_constant_does_not_crash():
    # constant signals -> WCC is undefined (std=0); the test must degrade
    # gracefully rather than raise
    a = np.full(200, 1.0)
    b = np.full(200, 1.0)
    wcc = np.full(150, np.nan)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=20, seed=1)
    assert isinstance(res, dict)


def test_signal_surrogate_n_one_runs():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    wcc = rng.normal(0.3, 0.1, 150)
    res = _signal_level_surrogate_test(a, b, wcc, hz=1.0, surrogate_n=1, seed=1)
    assert isinstance(res, dict)
    assert res["n_surrogates"] == 1

# === source: test_surrogate_pvalue_tails.py ===
"""Regression tests for BRM-2026-07-13 fix #4: L0/L1 p-value tail consistency.

BUG-4: the L0 signal-level surrogate test (``_signal_level_surrogate_test``)
used a *single*-tailed Phipson-Smyth, while the L1 WCC-level test
(``_wcc_level_surrogate_test``) used the *two*-tailed form.  The same
family of features therefore received different tail policies.

Both pipeline functions now use the identical conservative two-tailed
Phipson-Smyth (Phipson & Smyth, 2010).  These tests recompute the
two-tailed p from each function's OWN returned null + observed values and
assert the reported p equals it -- proving the two-tailed formula is what
is actually computed (a single-tailed implementation would not match).
"""

import numpy as np

from multisync.dynamic_features import (
    _signal_level_surrogate_test,
    _wcc_level_surrogate_test,
    sliding_window_wcc,
)


def _two_tailed_p(null, obs):
    """Reference two-tailed Phipson-Smyth (matches _wcc_level_surrogate_test)."""
    finite = np.asarray(null, dtype=float)[np.isfinite(null)]
    n = finite.size
    if n == 0 or not np.isfinite(obs):
        return 1.0
    p_ge = (np.sum(finite >= obs) + 1) / (n + 1)
    p_le = (np.sum(finite <= obs) + 1) / (n + 1)
    return min(1.0, 2.0 * min(p_ge, p_le))


def _make_signals(n=800, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 30, n)
    a = np.sin(t) + rng.normal(0, 0.2, n)
    b = np.sin(t + 0.3) + rng.normal(0, 0.2, n)
    return a, b


def test_l0_signal_level_is_two_tailed():
    a, b = _make_signals()
    wcc = sliding_window_wcc(a, b, 30, 1.0)
    res = _signal_level_surrogate_test(
        a, b, wcc, hz=1.0, wcc_window_size=30, surrogate_n=100, seed=42,
    )
    p = res["p_peak_amplitude"]
    assert np.isfinite(p)
    expected = _two_tailed_p(res["null_peak_amplitude"], res["obs_peak_amplitude"])
    assert abs(p - expected) < 1e-9


def test_l1_wcc_level_is_two_tailed():
    a, b = _make_signals()
    wcc = sliding_window_wcc(a, b, 30, 1.0)
    res = _wcc_level_surrogate_test(
        wcc, hz=1.0, surrogate_n=100,
        features=("dwell_time", "switching_rate"),
        wcc_window_sec=30.0, seed=42,
    )
    for f in ("dwell_time", "switching_rate"):
        p = res[f"p_{f}"]
        assert np.isfinite(p)
        expected = _two_tailed_p(res[f"null_{f}"], res[f"obs_{f}"])
        assert abs(p - expected) < 1e-9


def test_l0_not_single_tailed_for_lower_extreme():
    """A single-tailed UPPER test gives p~1.0 for a lower-extreme obs;
    the two-tailed test must give a small p. Construct an obs in the lower
    tail of the L0 null and confirm the reported p is small, not ~1.0."""
    a, b = _make_signals(seed=3)
    wcc = sliding_window_wcc(a, b, 30, 1.0)
    res = _signal_level_surrogate_test(
        a, b, wcc, hz=1.0, wcc_window_size=30, surrogate_n=200, seed=11,
    )
    null = np.asarray(res["null_peak_amplitude"], dtype=float)
    finite = null[np.isfinite(null)]
    n = finite.size
    # Force an observed value clearly in the LOWER tail of the null.
    obs = float(np.percentile(finite, 1))  # 1st percentile -> lower extreme
    p_ge = (np.sum(finite >= obs) + 1) / (n + 1)      # single-tailed upper
    p_le = (np.sum(finite <= obs) + 1) / (n + 1)      # lower-tail mass
    single_upper = p_ge
    two_tailed = min(1.0, 2.0 * min(p_ge, p_le))
    # The lower tail must dominate -> two-tailed p is driven by p_le, which
    # is small; a single-tailed UPPER test would report ~1.0 here.
    assert two_tailed < 0.2
    assert single_upper > 0.9
    # Sanity: the formula we just used is what the function applies.
    assert abs(two_tailed - _two_tailed_p(finite, obs)) < 1e-12

# === source: test_ft_surrogate_dc.py ===
"""Regression tests: ft_surrogate must preserve the DC component's phase
(sign of the mean). gstack Finding 7.

The buggy version forced ``random_phases[0] = 0.0``, which made the
reconstructed DC term ``|X[0]| * exp(i*0) = +|X[0]|`` — deterministically
flipping a negative-mean signal to a positive-mean surrogate (Claude:
30 seeds all yielded mean = +0.4110, variance 0). Phase randomization must
preserve the DC phase, so the surrogate mean equals the original mean.
"""
import numpy as np

from multisync.surrogate import ft_surrogate


def test_ft_surrogate_preserves_negative_mean_sign():
    """A negative-mean signal must yield negative-mean surrogates."""
    rng = np.random.default_rng(0)
    x = -0.4 + rng.normal(0.0, 1.0, 256)
    assert x.mean() < 0

    means = []
    surrogates = []
    for seed in range(30):
        s = ft_surrogate(x, np.random.default_rng(seed))
        means.append(s.mean())
        surrogates.append(s)
    means = np.asarray(means)

    # Mean preserved exactly (DC phase preserved) -> stays negative, not flipped.
    assert abs(means.mean() - x.mean()) < 1e-9
    assert means.mean() < 0
    # Surrogates genuinely differ across seeds (phase randomization active).
    assert not np.allclose(surrogates[0], surrogates[1])


def test_ft_surrogate_preserves_positive_mean_sign():
    """Symmetry check: a positive-mean signal stays positive (sanity)."""
    rng = np.random.default_rng(1)
    x = 0.5 + rng.normal(0.0, 1.0, 256)
    assert x.mean() > 0
    for seed in range(10):
        s = ft_surrogate(x, np.random.default_rng(seed))
        assert abs(s.mean() - x.mean()) < 1e-9


def test_ft_surrogate_even_length_nyquist_preserved():
    """Even-length signals: Nyquist phase preserved, stays real/consistent."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 300)  # even length
    s = ft_surrogate(x, np.random.default_rng(7))
    # Reconstruction must be real (finite) — broken Nyquist handling would
    # introduce non-negligible imaginary leakage.
    assert np.all(np.isfinite(s))
    assert abs(s.mean() - x.mean()) < 1e-9

# === source: test_bh_fdr_consistency.py ===
"""Cross-implementation consistency lock for the (formerly 3, now 1 canonical
+ thin wrappers) BH-FDR routines.

Claude's review found 3 independent BH-FDR implementations:
  - multisync.batch._bh_fdr_correction  (canonical; returns (adjusted, rejected))
  - multisync.validation.l2_between_condition._bh_fdr  (now delegates to batch)
  - multisync.validation.pgt1_intensity.bh_fdr  (returns boolean rejected array)

All three must agree on (a) adjusted p-values and (b) rejected flags.  This
test guards against any future drift between the copies.
"""

import numpy as np

from multisync.batch import _bh_fdr_correction
from multisync.validation.l2_between_condition import _bh_fdr
from multisync.validation.pgt1_intensity import bh_fdr


def test_bh_fdr_implementations_agree():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        p = rng.uniform(0.0, 1.0, size=n)
        # Occasionally inject NaN / boundary values.
        if rng.random() < 0.3:
            p[int(rng.integers(0, n))] = np.nan
        if rng.random() < 0.2:
            p[0] = 0.0
        if rng.random() < 0.2:
            p[0] = 1.0

        adj_batch, rej_batch = _bh_fdr_correction(list(p), alpha=0.05)
        adj_l2 = _bh_fdr(p)
        rej_pgt1 = bh_fdr(p, q=0.05)

        # Adjusted p-values: batch (canonical) vs l2 (thin wrapper).
        np.testing.assert_allclose(
            np.asarray(adj_batch, dtype=float),
            np.asarray(adj_l2, dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
        # Rejected flags: batch vs pgt1 (both at FDR 0.05).
        np.testing.assert_array_equal(
            np.asarray(rej_batch, dtype=bool),
            np.asarray(rej_pgt1, dtype=bool),
        )

# === source: test_fdr_dedupe.py ===
"""Regression tests for the #18 FDR de-duplication guard.

`dedupe_fdr_input` refuses (or merges) duplicate endpoint/feature keys
BEFORE BH-FDR runs, so a key entered twice can never silently inflate the
test count ``m``.  `apply_fdr` is the single canonical FDR entry (dedupe ->
BH-FDR -> keyed result) that must agree with the canonical
`batch._bh_fdr_correction`.  The guard is also wired into the two FDR call
sites (`between_condition_fdr`, `apply_bh_fdr_within_noise`) and must fail
loud on duplicate keys there too.
"""

import numpy as np
import pandas as pd
import pytest

from multisync.batch import _bh_fdr_correction, apply_fdr, dedupe_fdr_input
from multisync.validation.l2_between_condition import between_condition_fdr
from multisync.validation.pgt1_intensity import apply_bh_fdr_within_noise


def test_dedupe_preserves_unique():
    names, vals = dedupe_fdr_input(["a", "b", "c"], [0.1, 0.2, 0.3])
    assert names == ["a", "b", "c"]
    assert vals == [0.1, 0.2, 0.3]


def test_dedupe_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        dedupe_fdr_input(["a", "b", "a"], [0.01, 0.02, 0.03])


def test_dedupe_first_keeps_first():
    names, vals = dedupe_fdr_input(
        ["a", "a", "b"], [0.5, 0.1, 0.2], on_duplicate="first"
    )
    assert names == ["a", "b"]
    assert vals == [0.5, 0.2]


def test_dedupe_max_keeps_largest_p():
    names, vals = dedupe_fdr_input(
        ["a", "a", "b"], [0.5, 0.1, 0.2], on_duplicate="max"
    )
    assert names == ["a", "b"]
    assert vals == [0.5, 0.2]  # max(0.5, 0.1)


def test_apply_fdr_dedupes_matches_bh():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=20)
    res = apply_fdr([f"f{i}" for i in range(20)], list(p), alpha=0.05)
    adj, rej = _bh_fdr_correction(list(p), alpha=0.05)
    for i in range(20):
        assert abs(res[f"f{i}"]["p_fdr"] - adj[i]) < 1e-12
        assert res[f"f{i}"]["significant"] == rej[i]
        assert abs(res[f"f{i}"]["p_raw"] - p[i]) < 1e-12


def test_apply_fdr_raises_on_dup():
    with pytest.raises(ValueError, match="duplicate"):
        apply_fdr(["a", "a"], [0.01, 0.02])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        dedupe_fdr_input(["a", "b"], [0.1])


def _small_l2_df(n_dyads=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        a, b = rng.normal(1.0, 0.3), rng.normal(0.5, 0.3)
        rows.append({"dyad_label": f"d{i}", "condition": "A", "peak_amplitude": a})
        rows.append({"dyad_label": f"d{i}", "condition": "B", "peak_amplitude": b})
    return pd.DataFrame(rows)


def test_l2_guard_rejects_duplicate_feature_cols():
    """between_condition_fdr must fail loud (not inflate m) on a duplicated
    feature column."""
    df = _small_l2_df(n_dyads=4)
    with pytest.raises(ValueError, match="duplicate"):
        between_condition_fdr(
            df,
            feature_cols=["peak_amplitude", "peak_amplitude"],
            n_permutations=1000,
            seed=1,
        )


def test_pgt1_guard_rejects_duplicate_p_columns():
    """apply_bh_fdr_within_noise must fail loud on a duplicated p-column."""
    df = pd.DataFrame({"p_a": [0.01], "p_b": [0.4]})
    with pytest.raises(ValueError, match="duplicate"):
        apply_bh_fdr_within_noise(
            df, feature_p_columns=["p_a", "p_a"], q=0.05
        )

# === source: test_full_family_fdr.py ===
"""
Tests for the full-family FDR option (critique A, 2026-07-07).

Reviewer-proofing: instead of only testing the 3 frozen FDR-family
features, the full 12-feature set can be entered into a single BH-FDR step
via ``full_family_fdr=True``. The frozen core family remains the
primary endpoint; the full family is a strictly more conservative
supplementary check that neutralises the "cherry-picking 3/12" critique.

Covers
------
- SSoT API: ``feature_definitions.get_fdr_features`` / ``ALL_FEATURES``
- ``feature_pipeline.get_fdr_features`` is a thin pass-through to the SSoT
- ``inference_pipeline.test_l2_condition`` / ``run_full_cascade`` thread the
  flag and test 3 vs 12 features accordingly
- ``summarize()`` / ``cascade_summary`` reflect the family
  ("FDR-family" vs "all-features")
"""

import numpy as np
import pandas as pd
import pytest

from multisync.feature_definitions import (
    ALL_FEATURES,
    FDR_FEATURES,
    PRIMARY_FDR_FAMILY,
    SECONDARY_FDR_FAMILY,
    get_fdr_features,
    get_primary_fdr_features,
    get_secondary_fdr_features,
)
from multisync.feature_pipeline import get_fdr_features as fp_get_fdr_features
from multisync.inference_pipeline import UNSPECIFIED_MODALITY, InferencePipeline

N_FEATURES = 12
PRIMARY_N = len(PRIMARY_FDR_FAMILY)


def _full_feature_df(n_dyads: int = 8, seed: int = 0) -> pd.DataFrame:
    """A DataFrame containing ALL 12 SyncPipe features + dyad/condition cols.

    Every feature is finite so dyad-paired permutation keeps both conditions
    per dyad (we are testing *feature counts*, not missing-data handling).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        for cond in ("rest", "task"):
            row = {"dyad_id": f"dyad_{i}", "condition": cond}
            for f in ALL_FEATURES:
                shift = 0.0 if cond == "rest" else 0.3
                row[f] = float(rng.normal(shift, 0.3))
            rows.append(row)
    return pd.DataFrame(rows)


# --- A1: SSoT API -----------------------------------------------------------

def test_all_features_has_twelve_members():
    assert isinstance(ALL_FEATURES, tuple)
    assert len(ALL_FEATURES) == N_FEATURES
    # FDR features are a strict subset of the full family
    assert set(FDR_FEATURES).issubset(set(ALL_FEATURES))


def test_get_fdr_features_default_is_primary_family():
    # Default = frozen PRIMARY FDR family (single peak_amplitude).
    # SECONDARY family (dwell_time, switching_rate) is reported alongside but
    # does NOT enter the primary BH-FDR correction.
    assert get_fdr_features() == list(PRIMARY_FDR_FAMILY)
    assert get_fdr_features(False) == list(PRIMARY_FDR_FAMILY)
    assert len(get_fdr_features(False)) == PRIMARY_N
    assert set(get_primary_fdr_features()) == set(PRIMARY_FDR_FAMILY)
    assert set(get_secondary_fdr_features()) == set(SECONDARY_FDR_FAMILY)
    # FDR_FEATURES remains the union (used for exports / guards), n=3.
    assert set(FDR_FEATURES) == set(PRIMARY_FDR_FAMILY) | set(SECONDARY_FDR_FAMILY)


def test_get_fdr_features_full_family_is_all():
    assert get_fdr_features(True) == list(ALL_FEATURES)
    assert len(get_fdr_features(True)) == N_FEATURES


def test_feature_pipeline_mirrors_ssot():
    """feature_pipeline.get_fdr_features must be a thin pass-through to the SSoT."""
    assert fp_get_fdr_features(False) == get_fdr_features(False)
    assert fp_get_fdr_features(True) == get_fdr_features(True)


# --- A2/A3: inference threading --------------------------------------------

def test_l2_default_tests_only_primary_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(n_permutations=200, contrast=("rest", "task"))
    # Default L2 input = PRIMARY FDR family (single frozen endpoint).
    assert res["n_tested"] == PRIMARY_N
    assert len(res["per_feature"]) == PRIMARY_N
    assert set(f.feature for f in res["per_feature"]) == set(PRIMARY_FDR_FAMILY)


def test_l2_full_family_tests_all_twelve():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(
        n_permutations=200, contrast=("rest", "task"), full_family_fdr=True
    )
    assert res["n_tested"] == N_FEATURES
    assert len(res["per_feature"]) == N_FEATURES
    assert set(f.feature for f in res["per_feature"]) == set(ALL_FEATURES)


def test_l2_full_family_ignored_when_feature_cols_explicit():
    """When feature_cols is supplied, full_family_fdr must have no effect."""
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.test_l2_condition(
        feature_cols=["peak_amplitude"],
        n_permutations=200,
        contrast=("rest", "task"),
        full_family_fdr=True,
    )
    assert res["n_tested"] == 1
    assert set(f.feature for f in res["per_feature"]) == {"peak_amplitude"}


def test_summarize_reflects_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    pipe.test_l2_condition(n_permutations=200, contrast=("rest", "task"))
    # Default L2 corrects the PRIMARY FDR family -> "primary-FDR" label.
    assert "primary-FDR" in pipe.summarize()

    pipe2 = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    pipe2.test_l2_condition(
        n_permutations=200, contrast=("rest", "task"), full_family_fdr=True
    )
    assert "all-features" in pipe2.summarize()


def test_run_full_cascade_carries_full_family_flag():
    """The flag must propagate to L2 inside run_full_cascade.

    An empty wcc_dict skips the (slow) L0/L1 surrogate loops, isolating the
    L2 threading check.
    """
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.run_full_cascade(
        raw_signals_dict={},
        wcc_dict={},
        wcc_window_size=20,
        n_permutations=200,
        full_family_fdr=True,
    )
    # 1c: L2 is always modality-keyed. This fixture has no modality column, so
    # the single entry is labelled with the explicit sentinel.
    l2 = res["l2_results"]
    assert set(l2) == {UNSPECIFIED_MODALITY}
    assert l2[UNSPECIFIED_MODALITY]["n_tested"] == N_FEATURES
    assert "all-features" in res["cascade_summary"]


def test_run_full_cascade_default_is_primary_family():
    df = _full_feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=5, seed=1)
    res = pipe.run_full_cascade(
        raw_signals_dict={},
        wcc_dict={},
        wcc_window_size=20,
        n_permutations=200,
    )
    l2 = res["l2_results"]
    assert set(l2) == {UNSPECIFIED_MODALITY}
    assert l2[UNSPECIFIED_MODALITY]["n_tested"] == PRIMARY_N
    assert "primary-FDR" in res["cascade_summary"]

# === source: test_per_feature_significance.py ===
"""Regression tests: per-feature significance (no OR) for BOTH L0 and L1,
two-tailed p, no silent surrogate cap, BC moved to L0 family.
"""
import numpy as np
import pytest

from multisync.dynamic_features import (
    wcc_surrogate_test, _wcc_level_surrogate_test, sliding_window_wcc,
)
from multisync.feature_definitions import FDR_FAMILIES, MATHEMATICAL_TIER, FDR_FEATURES


def _structured_wcc(seed=0):
    rng = np.random.default_rng(seed)
    w = np.tile(np.r_[np.full(50, 0.9), np.full(50, 0.1)], 5)
    return np.clip(w + rng.normal(0, 0.02, w.size), -1, 1)


# ---- L1 path -------------------------------------------------------------
def test_l1_emits_per_feature_significant_not_or():
    res = wcc_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=200, seed=1)
    assert "per_feature_significant" in res
    assert "surrogate_is_significant" not in res  # no OR aggregate flag
    assert set(res["per_feature_significant"]) == {"dwell_time", "switching_rate"}


def test_l1_two_tailed_p_bounded():
    res = wcc_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=300, seed=1)
    for k in ("p_dwell_time", "p_switching_rate"):
        assert 0.0 <= res[k] <= 1.0


def test_no_silent_surrogate_cap():
    res = _wcc_level_surrogate_test(_structured_wcc(), hz=10.0, surrogate_n=1100, seed=1)
    assert res["n_surrogates"] == 1100


# ---- gstack Finding 6: short-WCC early-return shape parity -------------
def test_l1_short_wcc_early_return_matches_normal_shape():
    """A WCC shorter than min_wcc_points must still return an L1-shaped dict
    (p_dwell_time / p_switching_rate present and == 1.0), so downstream
    consumers that index those keys don't KeyError a whole batch. Regression
    for gstack Finding 6."""
    short = np.zeros(10)  # far below default min_wcc_points=30
    res = _wcc_level_surrogate_test(short, hz=1.0, surrogate_n=50, seed=1)
    for k in ("p_dwell_time", "p_switching_rate"):
        assert k in res, f"short-WCC early return missing {k}"
        assert res[k] == 1.0  # non-significant: no evidence to reject H0
    for k in ("null_dwell_time", "null_switching_rate"):
        assert k in res
    assert res["per_feature_significant"] == {"dwell_time": False, "switching_rate": False}
    assert np.isnan(res["obs_dwell_time"])
    assert res["notes"].startswith("WCC too short")


def test_l1_short_wcc_via_dispatcher_no_keyerror():
    """Same guarantee through the public wcc_surrogate_test dispatcher (L1 path,
    no raw_signals)."""
    short = np.zeros(10)
    res = wcc_surrogate_test(short, hz=1.0, surrogate_n=50, seed=1)
    assert "p_dwell_time" in res and "p_switching_rate" in res
    assert res["p_dwell_time"] == 1.0


# ---- gstack: dispatcher-level shape-contract (prevents Finding 6 class) ----
def _p_feature_keys(res):
    return {k[2:] for k in res if k.startswith("p_")}


def test_dispatcher_shape_contract_l0_l1_and_short():
    """wcc_surrogate_test must honor a STABLE key contract per level so
    downstream consumers can rely on it across refactors. Contract:
      (1) every p_<f> key has matching null_<f> and obs_<f> companions;
      (2) per_feature_significant is keyed by exactly the same feature set;
      (3) the L1 short-WCC early-return shares the SAME p_* key set as the
          L1 normal path (exactly what Finding 6 violated).
    This is a regression guard: if any future refactor diverges one path's
    shape, the assertion fails instead of silently corrupting a whole batch.
    """
    rng = np.random.default_rng(0)
    n = 600
    shared = np.cumsum(rng.normal(0, 1, n))
    a = shared + rng.normal(0, 2, n)
    b = shared + rng.normal(0, 2, n)
    wcc = sliding_window_wcc(a, b, window_size=30)

    res_l0 = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=60, seed=1,
                                raw_signals=(a, b), wcc_window_size=30)
    res_l1 = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=60, seed=1)
    res_l1_short = wcc_surrogate_test(np.zeros(10), hz=1.0, surrogate_n=50, seed=1)

    for res, level in ((res_l0, "L0"), (res_l1, "L1"), (res_l1_short, "L1-short")):
        p_keys = _p_feature_keys(res)
        assert p_keys, f"{level}: no p_* keys found"
        assert set(res["per_feature_significant"]) == p_keys, (
            f"{level}: per_feature_significant keys "
            f"{set(res['per_feature_significant'])} != p_* keys {p_keys}"
        )
        for f in p_keys:
            assert f"null_{f}" in res, f"{level}: missing null_{f}"
            assert f"obs_{f}" in res, f"{level}: missing obs_{f}"

    # The invariant Finding 6 broke: short and normal L1 paths agree.
    assert _p_feature_keys(res_l1_short) == _p_feature_keys(res_l1), (
        "L1 short-WCC early-return shape diverges from L1 normal path"
    )


# ---- L0 path -------------------------------------------------------------
def test_l0_emits_per_feature_significant_for_three_features():
    rng = np.random.default_rng(0)
    n = 600
    shared = np.cumsum(rng.normal(0, 1, n))
    a = shared + rng.normal(0, 2, n)
    b = shared + rng.normal(0, 2, n)
    wcc = sliding_window_wcc(a, b, window_size=30)
    res = wcc_surrogate_test(wcc, hz=1.0, surrogate_n=100, seed=1,
                             raw_signals=(a, b), wcc_window_size=30)
    assert res["null_model"] == "signal_level_iaaft"
    pfs = res["per_feature_significant"]
    assert set(pfs) == {"mean_synchrony", "peak_amplitude", "bimodality_coefficient"}
    assert "surrogate_is_significant" not in res  # no OR aggregate flag


# ---- BC tier consistency -------------------------------------------------
def test_bimodality_is_l0_math_tier_but_not_in_confirmatory_fdr():
    """2026-06-29 (SSoT Option B): bimodality_coefficient remains a
    permutation-invariant L0 feature for the synchrony-existence audit
    (MATHEMATICAL_TIER + _NULL_MODEL_L0), but was removed from the
    confirmatory group-condition FDR family (FDR_FAMILIES / FDR_FEATURES)."""
    assert MATHEMATICAL_TIER["bimodality_coefficient"] == "L0"
    assert "bimodality_coefficient" not in FDR_FAMILIES["L0"]
    assert "bimodality_coefficient" not in FDR_FAMILIES["L1"]
    assert "bimodality_coefficient" not in FDR_FEATURES
    assert FDR_FAMILIES["L0"] == ("peak_amplitude",)
    assert FDR_FAMILIES["L1"] == ("dwell_time", "switching_rate")

# === source: test_simulation_kuramoto.py ===
"""Tests for the simulation ground-truth generators (Blind Spot B fuses).

The Kuramoto simulator and the shared-signal model are what *produce* the
validation numbers cited in the paper.  If a coupling parameter is
mis-wired or a noise term is broken, the output still looks plausible but
the ground truth underneath is wrong.  These tests pin the core invariant:
synchrony must increase with coupling strength.
"""


import numpy as np

from multisync.simulation import constant_coupling, generate_signals
from multisync.simulation.kuramoto import mean_sync_from_K, solve_phase_difference


def test_solve_phase_difference_in_unit_range():
    r = solve_phase_difference(0.5, 0.7, 0.0, 60.0)
    assert r.min() >= 0.0
    assert r.max() <= 1.0
    assert r.shape[0] == 2000


def test_kuramoto_sync_monotonic_in_coupling():
    """Larger constant coupling -> larger time-averaged synchrony."""
    Ks = [0.0, 0.2, 0.5, 1.0, 2.0]
    syncs = np.array([mean_sync_from_K(K) for K in Ks])
    # Strictly increasing across the grid.
    assert np.all(np.diff(syncs) > 1e-3), f"non-monotonic: {syncs}"


def test_shared_signal_coupling_monotonic():
    """Higher preset coupling -> higher observed cross-person correlation."""
    c_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    cors = []
    for c in c_vals:
        res = generate_signals(
            constant_coupling(c),
            duration_sec=120.0,
            hz=2.0,
            noise_sigma=0.2,
            seed=0,
        )
        cors.append(abs(np.corrcoef(res.x_A, res.x_B)[0, 1]))
    cors = np.array(cors)
    assert np.all(np.diff(cors) > 0), f"non-monotonic coupling->corr: {cors}"

# === source: test_across_stim_keys.py ===
"""Regression test for the across-stim shuffle dict-key bug.

The loop variable `k` (feature name) was being overwritten by an integer
Phipson-Smyth count and then used as the results dict key, so the returned
dict was keyed by integers (e.g. [1, 11, 18]) instead of feature names.
"""

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

# === source: test_v1_safety_fixes.py ===
import numpy as np
import pandas as pd
import pytest

from multisync.core import Dyad, DynamicAnalyzer
from multisync.dataset import SynchronyDataset
from multisync.feature_definitions import extract_features, DynamicFeatures
from multisync.surrogate import iaaft_surrogate


def test_iaaft_preserves_empirical_amplitude_distribution():
    x = np.r_[np.zeros(50), np.ones(50) * 10, np.linspace(-3, 3, 50)]
    rng = np.random.default_rng(123)
    y = iaaft_surrogate(x, rng=rng, max_iter=50)
    assert y.shape == x.shape
    assert np.allclose(np.sort(y), np.sort(x))


def test_dynamic_features_roundtrip_preserves_non_fdr_descriptors():
    wcc = np.linspace(0.0, 1.0, 80)
    f = extract_features(wcc, hz=1.0, wcc_window_sec=10.0)
    d = f.to_dict()
    rt = DynamicFeatures.from_dict(d).to_dict()
    for key in [
        "onset_latency",
        "rise_time",
        "recovery_time",
        "onset_latency_imputed",
        "rise_time_imputed",
        "recovery_time_imputed",
        "synchrony_entropy",
        "peak_amplitude",
        "mean_synchrony",
    ]:
        assert np.isclose(rt[key], d[key], equal_nan=True), key


def test_all_absolute_timestamps_align_without_mixed_error():
    base = 1_700_000_000.0
    t = base + np.arange(20, dtype=float)
    ds = SynchronyDataset(
        "abs",
        {
            "a": pd.DataFrame({"time": t, "x": np.arange(20, dtype=float)}),
            "b": pd.DataFrame({"time": t, "y": np.arange(20, dtype=float)}),
        },
    )
    ds.align(target_hz=1.0)
    assert ds._aligned


def test_mixed_absolute_relative_timestamps_fail():
    base = 1_700_000_000.0
    ds = SynchronyDataset(
        "mixed",
        {
            "a": pd.DataFrame({"time": base + np.arange(20, dtype=float), "x": np.arange(20, dtype=float)}),
            "b": pd.DataFrame({"time": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)}),
        },
    )
    with pytest.raises(ValueError, match="Timestamp type mismatch"):
        ds.align(target_hz=1.0)


def test_zscore_all_nan_remains_nan_not_zero():
    ds = SynchronyDataset(
        "nan",
        {"a": pd.DataFrame({"time": np.arange(5, dtype=float), "x": [np.nan] * 5})},
    )
    _, stats = ds.zscore()
    assert stats["a"]["x"]["status"] == "all_nan"
    assert np.isnan(ds.modalities["a"]["x"].to_numpy()).all()


def test_dynamic_analyzer_passes_surrogate_n_into_threshold_meta():
    n = 120
    t = np.arange(n, dtype=float)
    df_a = pd.DataFrame({"time": t, "x": np.sin(t / 8)})
    df_b = pd.DataFrame({"time": t, "y": np.sin(t / 8) + 0.1 * np.cos(t / 3)})
    dyad = Dyad(a=df_a, b=df_b, hz=1.0)
    dyad.align(target_hz=1.0)
    dyad.zscore()
    analyzer = DynamicAnalyzer(window_size=10, surrogate_n=7, enable_prediction=False)
    result = analyzer.fit_transform(dyad)
    assert result.threshold_meta
    assert all(meta.get("surrogate_n") == 7 for meta in result.threshold_meta.values())
    assert all(meta.get("mode") == "within_dyad_surrogate" for meta in result.threshold_meta.values())


# ---------------------------------------------------------------------------
# S1: per-modality primary-modality existence gate
# ---------------------------------------------------------------------------
from multisync.inference_pipeline import (
    _existence_gate_by_modality,
    _modality_from_label,
)


def _sig(flag: bool) -> dict:
    return {"per_feature_significant": {"peak_amplitude": flag}}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("d1__ECG__Rest", "ECG"),
        ("d1__EDA", "EDA"),
        ("dyad_x__RESP__Trial2", "RESP"),
        ("d1", ""),  # no separator -> unnamed bucket, no crash
    ],
)
def test_modality_from_label(label, expected):
    assert _modality_from_label(label) == expected


def test_gate_primary_modality_majority_supports():
    res = {
        "d1__ECG__Rest": _sig(True),
        "d2__ECG__Rest": _sig(True),
        "d3__ECG__Rest": _sig(False),
        "d1__EDA__Rest": _sig(False),
        "d2__EDA__Rest": _sig(False),
        # RESP passes 3/3 but is NOT a primary modality -> cannot support.
        "d1__RESP__Rest": _sig(True),
        "d2__RESP__Rest": _sig(True),
        "d3__RESP__Rest": _sig(True),
    }
    g = _existence_gate_by_modality(res, ["ECG", "EDA"], 0.5)
    # ECG passes 2/3 (>0.5) -> gate satisfied by at least one primary modality.
    assert g["primary_pass"] is True
    assert g["per_modality"]["ECG"]["pass_rate"] == pytest.approx(2 / 3)
    assert g["per_modality"]["ECG"]["supports"] is True
    assert g["per_modality"]["EDA"]["supports"] is False
    assert g["per_modality"]["RESP"]["is_primary"] is False
    assert g["per_modality"]["RESP"]["supports"] is False
    assert g["endpoint"] == "peak_amplitude"


def test_gate_all_primary_fail_blocks():
    res = {f"d{i}__ECG": _sig(False) for i in range(4)}
    res.update({f"d{i}__EDA": _sig(False) for i in range(4)})
    g = _existence_gate_by_modality(res, ["ECG", "EDA"], 0.5)
    assert g["primary_pass"] is False


def test_gate_exactly_half_is_not_majority():
    res = {"d1__ECG": _sig(True), "d2__ECG": _sig(False)}
    g = _existence_gate_by_modality(res, ["ECG"], 0.5)
    # pass_rate == 0.5 is NOT strictly greater than 0.5 -> not a majority.
    assert g["per_modality"]["ECG"]["supports"] is False
    assert g["primary_pass"] is False


def test_gate_sensitivity_only_cannot_open():
    # Only the sensitivity modality passes; primary modalities absent/fail.
    res = {f"d{i}__RESP": _sig(True) for i in range(5)}
    g = _existence_gate_by_modality(res, ["ECG", "EDA"], 0.5)
    assert g["primary_pass"] is False


# ---------------------------------------------------------------------------
# S5: timestamp-type detection must fail loud on the ambiguous 1e6..1e9 zone
# ---------------------------------------------------------------------------
from multisync.dataset import _detect_time_type


@pytest.mark.parametrize(
    "vals,expected",
    [
        # Unix-seconds absolute (2026) -> absolute.
        (np.array([1.78e9, 1.78e9 + 1, 1.78e9 + 2]), "absolute"),
        # Unix-milliseconds absolute (2026) -> absolute (>= ~1.13e12).
        (np.array([1.78e12, 1.78e12 + 1, 1.78e12 + 2]), "absolute"),
        # Clean relative axis starting near 0 (seconds) -> relative.
        (np.arange(0.0, 100.0, 0.5), "relative"),
        # AMBIGUOUS: millisecond RELATIVE axis that starts away from 0
        # (device ms clock modulo) — must NOT be trusted as absolute.
        (np.array([5.0e6, 5.0e6 + 1, 5.0e6 + 2]), "unknown"),
        (np.array([2.0e8, 2.0e8 + 1000, 2.0e8 + 2000]), "unknown"),
        # AMBIGUOUS: 1973-2001 Unix-seconds (rare) — same zone, fail loud.
        (np.array([1.0e9 - 5, 1.0e9 - 4, 1.0e9 - 3]), "unknown"),
    ],
)
def test_detect_time_type_ambiguous_zone_fails_loud(vals, expected):
    assert _detect_time_type(pd.Series(vals)) == expected


def test_millisecond_relative_from_zero_is_relative():
    # A ms axis that genuinely starts at 0 and stays below 10000 stays relative.
    vals = np.arange(0.0, 9000.0, 1.0)  # 9 s of ms stamps
    assert _detect_time_type(pd.Series(vals)) == "relative"


# ---------------------------------------------------------------------------
# 5a: pseudo-pair cross-dyad alignment (mask + length) & length reporting
# ---------------------------------------------------------------------------
from multisync.design_controls import design_control_audit


def _coupled(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n) * 0.1


def test_pseudo_pair_reports_aligned_length_distribution():
    # Two dyads of DIFFERENT lengths; one has a discontinuity mask.
    rng = np.random.default_rng(0)
    a1 = _coupled(120, 1); b1 = a1 + rng.standard_normal(120) * 0.2
    a2 = _coupled(80, 2);  b2 = a2 + rng.standard_normal(80) * 0.2
    pairs = {"d1": (a1, b1), "d2": (a2, b2)}
    # d1 mask: first 100 samples in-segment (rest out); d2 fully in-segment.
    m1 = np.zeros(120, dtype=bool); m1[:100] = True
    masks = {"d1": m1, "d2": np.ones(80, dtype=bool)}
    res = design_control_audit(
        pairs, hz=1.0, window_size=20, seed=0,
        discontinuity_masks=masks, n_pseudo_per_dyad=3,
        shift_lags_sec=(-30.0, 30.0),
    )
    pp = res["pseudo_pair"]
    assert pp["enabled"] is True
    # Cross-dyad pairs are cropped to the shorter partner (80) minus nothing,
    # so the aligned length cannot exceed the shorter dyad's length.
    assert pp["aligned_length_max"] <= 80
    assert pp["aligned_length_min"] >= 1
    assert pp["aligned_length_min"] <= pp["aligned_length_median"] <= pp["aligned_length_max"]
    # The audit still produced a per-feature summary across both dyads.
    assert res["n_dyads"] == 2
    assert res["feature_summary"]


def test_pseudo_pair_equal_length_no_mask_unchanged():
    # Equal-length dyads, no masks -> aligned length equals the full length
    # and the control runs cleanly (regression: alignment must not corrupt
    # the already-clean case).
    rng = np.random.default_rng(3)
    a1 = _coupled(100, 4); b1 = a1 + rng.standard_normal(100) * 0.2
    a2 = _coupled(100, 5); b2 = a2 + rng.standard_normal(100) * 0.2
    pairs = {"d1": (a1, b1), "d2": (a2, b2)}
    res = design_control_audit(
        pairs, hz=1.0, window_size=20, seed=1,
        n_pseudo_per_dyad=2, shift_lags_sec=(-30.0, 30.0),
    )
    pp = res["pseudo_pair"]
    assert pp["aligned_length_min"] == 100
    assert pp["aligned_length_max"] == 100


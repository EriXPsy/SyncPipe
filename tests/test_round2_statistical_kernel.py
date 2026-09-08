"""Round-2 adversarial regression tests (2026-09-08 red/blue review).

Encodes the two statistical-kernel findings and one clean-bill check:

- BUG-4: under ``null_model='state_shuffle'`` both L1 statistics
  (dwell_time, switching_rate) are invariant by construction — the test
  documents the degeneracy and guards the protocol default (iaaft) that
  restores L1 discriminative power.
- BUG-3: per-pair derived seeds remove the cross-dyad surrogate correlation
  that a shared master seed introduced (group-null spread inflation).
- L2/BH kernel: BH matches a textbook step-up reference; the L2 paired
  permutation p (Monte-Carlo) agrees with exact enumeration.
"""

import inspect

import numpy as np
import pytest

from syncpipe.batch import _bh_fdr_correction
from syncpipe.dynamic_features import _wcc_level_surrogate_test
from syncpipe.inference_pipeline import InferencePipeline, _pair_seed
from syncpipe.validation.l2_between_condition import (
    _permutation_p,
    _signflip_null,
)


def _structured_wcc(n: int = 600, rng: np.random.Generator | None = None):
    """Square-wave WCC trace: elevated episodes of fixed run length."""
    rng = rng or np.random.default_rng(5)
    return np.where((np.arange(n) % 200) < 60, 0.75, 0.2) + 0.05 * rng.normal(size=n)


# --------------------------------------------------------------------------
# BUG-4: state_shuffle invariance vs iaaft discriminative power
# --------------------------------------------------------------------------

def test_state_shuffle_is_degenerate_for_both_l1_statistics():
    res = _wcc_level_surrogate_test(
        _structured_wcc(), surrogate_n=199, null_model="state_shuffle",
        seed=42, hz=4.0, wcc_window_sec=15.0,
    )
    assert res["p_dwell_time"] == pytest.approx(1.0)
    assert res["p_switching_rate"] == pytest.approx(1.0)


def test_state_shuffle_emits_degeneracy_warning():
    with pytest.warns(UserWarning, match="invariant"):
        _wcc_level_surrogate_test(
            _structured_wcc(), surrogate_n=19, null_model="state_shuffle",
            seed=1, hz=4.0, wcc_window_sec=15.0,
        )


def test_iaaft_l1_null_is_discriminative_on_structured_trace():
    """The protocol null (WCC-level IAAFT) must NOT be degenerate: on a
    trace with clean fixed-length episodes it should reject structure
    destruction (small p_dwell) rather than return 1.0."""
    res = _wcc_level_surrogate_test(
        _structured_wcc(), surrogate_n=99, null_model="iaaft",
        seed=42, hz=4.0, wcc_window_sec=15.0,
    )
    assert res["p_dwell_time"] < 0.05
    # switching_rate likewise loses its degenerate 1.0
    assert res["p_switching_rate"] < 1.0


def test_l1_pipeline_default_is_iaaft():
    sig = inspect.signature(InferencePipeline.test_l1_structure)
    assert sig.parameters["null_model"].default == "iaaft"


# --------------------------------------------------------------------------
# BUG-3: per-pair seeds remove cross-dyad null correlation
# --------------------------------------------------------------------------

def _null_peaks(x, y, seed, n_draws=120, w=60):
    from syncpipe.feature_definitions import compute_peak_amplitude, smoothed_wcc
    from syncpipe.surrogate import iaaft_surrogate

    rng = np.random.default_rng(seed)
    peaks = []
    for _ in range(n_draws):
        sx = iaaft_surrogate(x, rng)
        sy = iaaft_surrogate(y, rng)
        wcc = np.array([
            np.corrcoef(sx[i:i + w], sy[i:i + w])[0, 1]
            for i in range(len(x) - w + 1)
        ])
        p, _ = compute_peak_amplitude(smoothed_wcc(wcc))
        peaks.append(p)
    return np.asarray(peaks)


def _ar1(rng, n, phi):
    e = rng.normal(size=n)
    v = np.empty(n)
    v[0] = e[0]
    for i in range(1, n):
        v[i] = phi * v[i - 1] + e[i]
    return v


@pytest.mark.slow
def test_derived_seeds_remove_cross_dyad_null_correlation():
    """Heterogeneous dyads audited with label-derived seeds must produce
    mutually uncorrelated null-peak arrays (shared master seed gave
    r ≈ +0.33 and a 1.46x group-null spread inflation)."""
    rng = np.random.default_rng(2026)
    n = 600
    dyads = [
        (_ar1(rng, n, 0.4), _ar1(rng, n, 0.4)),
        (_ar1(rng, n, 0.7), _ar1(rng, n, 0.7)),
        (_ar1(rng, n, 0.1), _ar1(rng, n, 0.1)),
    ]
    labels = ["dyad_001__ECG", "dyad_002__ECG", "dyad_003__ECG"]
    master = 42
    nulls = [
        _null_peaks(x, y, _pair_seed(master, lab), n_draws=120)
        for (x, y), lab in zip(dyads, labels)
    ]
    cors = [
        np.corrcoef(nulls[i], nulls[j])[0, 1]
        for i in range(len(nulls))
        for j in range(i + 1, len(nulls))
    ]
    assert max(abs(c) for c in cors) < 0.2, (
        f"cross-dyad null-peak correlation {np.round(cors, 3)} exceeds 0.2; "
        "per-pair seed derivation failed"
    )


# --------------------------------------------------------------------------
# L2 / BH kernel reference checks
# --------------------------------------------------------------------------

def test_bh_matches_textbook_step_up_reference():
    def bh_ref(p):
        p = np.asarray(p, float)
        n = len(p)
        order = np.argsort(p)
        adj = p[order] * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(adj, 0.0, 1.0)
        return out

    rng = np.random.default_rng(0)
    for case in (
        rng.uniform(0, 1, 7),
        [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212,
         0.216, 0.222, 0.251, 0.269, 0.275, 0.34, 0.341, 0.421],
        [1.0, np.nan, 0.02, 0.5],
    ):
        got, _ = _bh_fdr_correction(list(case))
        exp = bh_ref([p if not np.isnan(p) else 1.0 for p in case])
        np.testing.assert_allclose(got, exp, atol=1e-12)


def test_l2_monte_carlo_p_agrees_with_exact_enumeration():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.3, 1.0, size=10)
    null_mc = _signflip_null(diffs, 20000, rng)
    null_ex = _signflip_null(diffs, 0, rng)
    p_mc = _permutation_p(float(np.median(diffs)), null_mc, exhaustive=False)
    p_ex = _permutation_p(float(np.median(diffs)), null_ex, exhaustive=True)
    assert abs(p_mc - p_ex) < 0.02
    assert 0.0 < p_ex <= 1.0

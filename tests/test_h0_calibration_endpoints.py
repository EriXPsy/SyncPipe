"""H0 calibration smoke tests, one per statistical endpoint (P1-3, 2026-09-09).

Institutionalizes the null-calibration contract across the three pipelines:
every statistical endpoint must reject at ~nominal alpha under its own H0.
A default (or a fix) that shifts an endpoint's size is a protocol defect even
when every unit test passes — this family exists so that regression is caught
at the calibration level, not only at the unit level (lesson from BUG-4: the
round-1 guard test froze an invalid default; see tests/test_v1_defaults_guard).

Endpoints and their nulls
-------------------------
- L0 per-pair existence audit (``synchrony_existence_audit``): H0 = two
  INDEPENDENT autocorrelated signals; the signal-level IAAFT null must make
  the audited per-feature p-values ~ U(0,1), i.e. FPR ≈ alpha.
- L1 WCC-level surrogate test (``wcc_surrogate_test``, null_model="iaaft"):
  H0 = WCC trace of independent signals; dwell_time / switching_rate
  p-values ~ U(0,1) (regression check for the BUG-4 family at calibration
  level).
- L2 dyad-paired group permutation (``run_group_condition_inference``):
  H0 = within-dyad condition deltas are pure noise; raw p ~ U(0,1) and the
  BH-FDR rejection rate ≈ alpha.

These are smoke-tier checks (fast marker, moderate Monte-Carlo, wide binomial
tolerances). High-power calibrations live in
``test_existence_gate_nan_sensitivity`` (second-order group gate, slow) and
``test_round2_statistical_kernel`` (seed independence, L2 MC vs exact).
"""

import numpy as np
import pandas as pd
import pytest

from syncpipe.design_controls import synchrony_existence_audit
from syncpipe.dynamic_features import wcc_surrogate_test, _sliding_window_wcc_cumsum
from syncpipe.inference_pipeline import InferencePipeline

ALPHA = 0.05


def _ar(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) signal — autocorrelated, independent across the two partners."""
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def _fpr_in(p_values: list[float], lo: float, hi: float, n: int,
            endpoint: str) -> None:
    """Assert the observed rejection rate is inside the smoke tolerance.

    Tolerances are wide smoke-tier binomial bands around alpha=0.05; a value
    above ``hi`` means the endpoint is anti-conservative under H0 (false
    positives), below ``lo`` for large ``hi-lo`` bands would mean broken
    uniformity (e.g. p pinned to 1.0 — the BUG-4 signature).
    """
    p = np.asarray([v for v in p_values if np.isfinite(v)], dtype=float)
    assert p.size == n, (
        f"{endpoint}: {n - p.size}/{n} replicates produced a non-finite or "
        "undefined p-value; the endpoint silently loses samples under H0."
    )
    fpr = float(np.mean(p < ALPHA))
    assert lo <= fpr <= hi, (
        f"{endpoint}: H0 rejection rate = {fpr:.3f} over {n} replicates "
        f"({int(np.sum(p < ALPHA))} rejections); expected ~{ALPHA} within "
        f"[{lo}, {hi}]. Above {hi} = anti-conservative size; a degenerate "
        "p-mass at 1.0 = invariant statistic (BUG-4 signature)."
    )


# ---------------------------------------------------------------------------
# L0: per-pair existence audit under independent-signal H0
# ---------------------------------------------------------------------------

def test_l0_existence_audit_h0_size_per_feature():
    """Signal-level IAAFT audit on independent AR partners: FPR ≈ alpha."""
    rng = np.random.default_rng(20260909)
    n_rep = 80
    collected: dict[str, list[float]] = {
        "peak_amplitude": [],
        "mean_synchrony": [],
        "bimodality_coefficient": [],
    }
    for rep in range(n_rep):
        res = synchrony_existence_audit(
            _ar(150, 0.7, rng), _ar(150, 0.7, rng),
            hz=1.0, window_size=10, surrogate_n=99, seed=1000 + rep,
        )
        assert res["status"] == "ok"
        for feat in collected:
            collected[feat].append(res["p_values"][feat])
    for feat, ps in collected.items():
        # n=80, alpha=0.05: binomial 95% CI ≈ [0.006, 0.104]; smoke band 0..0.13.
        _fpr_in(ps, 0.0, 0.13, n_rep, f"L0/{feat}")


# ---------------------------------------------------------------------------
# L1: WCC-level IAAFT surrogate test under independent-signal H0
# ---------------------------------------------------------------------------

def test_l1_wcc_surrogate_h0_size_dwell_switching():
    """WCC-level IAAFT (protocol default): dwell/switching FPR ≈ alpha.

    This is the calibration-level regression guard for the BUG-4 family:
    under H0 the two L1 statistics must stay stochastic, never pinned to
    p=1.0 (state_shuffle degeneracy).
    """
    rng = np.random.default_rng(20260910)
    n_rep = 60
    dwell_ps: list[float] = []
    switching_ps: list[float] = []
    for rep in range(n_rep):
        wcc = _sliding_window_wcc_cumsum(
            _ar(300, 0.6, rng), _ar(300, 0.6, rng), window_size=20,
        )
        res = wcc_surrogate_test(
            wcc, hz=1.0, surrogate_n=99, seed=2000 + rep,
            raw_signals=None, wcc_window_size=20, wcc_window_sec=20.0,
        )
        dwell_ps.append(res["p_dwell_time"])
        switching_ps.append(res["p_switching_rate"])
    _fpr_in(dwell_ps, 0.0, 0.15, n_rep, "L1/dwell_time")
    _fpr_in(switching_ps, 0.0, 0.15, n_rep, "L1/switching_rate")


# ---------------------------------------------------------------------------
# L2: dyad-paired group permutation under no-condition-effect H0
# ---------------------------------------------------------------------------

def _no_effect_features_df(rng: np.random.Generator, n_dyads: int = 20,
                           feature: str = "peak_amplitude") -> pd.DataFrame:
    """Within-dyad condition deltas that are pure noise (no real effect)."""
    rows = []
    for d in range(n_dyads):
        baseline = rng.normal(0.4, 0.15)  # dyad-level heterogeneity
        for cond in ("A", "B"):
            rows.append({
                "dyad_id": f"dyad_{d:03d}",
                "condition": cond,
                "modality": "ECG",
                feature: baseline + 0.5 * cond_delta_noise(rng),
            })
    return pd.DataFrame(rows)


def cond_delta_noise(rng: np.random.Generator) -> float:
    return float(rng.normal(0.0, 0.1))


def test_l2_group_permutation_h0_size_bh_fdr():
    """Permutation L2 with no condition effect: BH-FDR rejection ≈ alpha.

    With a single tested feature BH-FDR equals the raw test, so the
    significance flag must reject at ~5% over repeated no-effect datasets.
    """
    rng = np.random.default_rng(20260911)
    n_rep = 100
    rejections = 0
    for rep in range(n_rep):
        df = _no_effect_features_df(rng)
        pipe = InferencePipeline(df, hz=1.0, surrogate_n=10, seed=3000 + rep)
        result = pipe.run_group_condition_inference(
            feature_cols=["peak_amplitude"],
            n_permutations=200,
            contrast=("A", "B"),
            n_min_dyads=10,
        )
        l2 = result["ECG"]
        per_feature = l2["per_feature"]
        assert len(per_feature) == 1
        row = per_feature[0]
        assert (row.feature if hasattr(row, "feature")
                else row["feature"]) == "peak_amplitude"
        sig = row.significant_05 if hasattr(row, "significant_05") \
            else row["significant_05"]
        if bool(sig):
            rejections += 1
    fpr = rejections / n_rep
    assert 0.0 <= fpr <= 0.12, (
        f"L2/BH-FDR: H0 rejection rate = {fpr:.3f} over {n_rep} no-effect "
        f"datasets ({rejections} rejections); expected ~{ALPHA} within "
        "[0.0, 0.12]. Above 0.12 = the permutation null or the FDR step is "
        "anti-conservative."
    )


@pytest.mark.slow
def test_l0_l1_pvalues_approximately_uniform_ks():
    """Slow-tier: pooled H0 p-values are ~uniform (KS test), not just size-ok.

    A size-only check can miss non-uniformity that is benign at 0.05 but
    distorts tail behaviour (e.g. a p-mass near alpha). Pooling replicates
    gives the resolution a Kolmogorov-Smirnov test needs.
    """
    from scipy import stats

    rng = np.random.default_rng(20260912)
    l0_ps: list[float] = []
    for rep in range(150):
        res = synchrony_existence_audit(
            _ar(150, 0.7, rng), _ar(150, 0.7, rng),
            hz=1.0, window_size=10, surrogate_n=99, seed=4000 + rep,
        )
        l0_ps.append(res["p_values"]["peak_amplitude"])
    ks = stats.kstest(np.asarray(l0_ps), "uniform")
    assert ks.pvalue > 0.01, (
        f"L0 peak_amplitude p-values are not uniform under H0 "
        f"(KS D={ks.statistic:.3f}, p={ks.pvalue:.4f} over {len(l0_ps)} "
        "replicates)."
    )

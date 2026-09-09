"""Sensitivity check for the second-order group surrogate existence gate.

Red-team concern (2026-09-07 review, item T-3): the gate aggregates per-dyad
surrogate peaks draw-wise with ``min(width)`` truncation and NaN-masked
means. Real IAAFT runs produce ragged draw counts and sporadic NaN (short
segments, invalid surrogate windows). This test verifies the aggregation
stays calibrated under H0 when dyads are heterogeneous in:

- null-distribution scale (autocorrelation heterogeneity across dyads),
- number of surrogate draws (ragged stacks),
- sporadic NaN draws,

including the observed-mean-vs-null-subset mismatch identified in review.
Under H0 the gate should reject at ~alpha (two-tailed Phipson-Smyth + BH
across two primary modalities).
"""

import numpy as np
import pytest

from syncpipe.inference_pipeline import _existence_gate_by_modality

ALPHA = 0.05


def _make_results(
    rng: np.random.Generator,
    n_dyads: int = 20,
    ragged: bool = True,
    nan_rate: float = 0.10,
    modality: str = "ECG",
) -> dict:
    """Synthetic existence results under H0 (no coupling).

    Per dyad d: surrogate peaks ~ N(mu_d, sd_d) with heterogeneous sd_d;
    the observed peak is an independent draw from the SAME distribution,
    so the group observed mean and the group null are exchangeable.
    """
    results = {}
    for d in range(n_dyads):
        mu = rng.uniform(-0.2, 0.2)          # dyad baseline heterogeneity
        sd = rng.uniform(0.03, 0.30)         # autocorrelation heterogeneity
        n_draws = int(rng.integers(50, 100)) if ragged else 99
        null = rng.normal(mu, sd, size=n_draws)
        nan_mask = rng.random(n_draws) < nan_rate
        null[nan_mask] = np.nan
        obs = float(rng.normal(mu, sd))
        if rng.random() < 0.05:
            obs = np.nan                     # observed peak lost (segment)
        results[f"dyad_{d:03d}__{modality}"] = {
            "observed": {"peak_amplitude": obs},
            "null_peak_amplitude": null,
            "per_feature_significant": {"peak_amplitude": False},
        }
    return results


def _gate_rejects(rng: np.random.Generator) -> bool:
    results = _make_results(rng)
    gate = _existence_gate_by_modality(
        results,
        primary_modalities=("ECG", "EDA"),
        alpha=ALPHA,
    )
    return bool(gate["primary_pass"])


@pytest.mark.slow
def test_gate_type_one_error_heterogeneous_nan_ragged():
    """Under H0 with ragged/NaN/heterogeneous inputs, FPR must be ~alpha."""
    rng = np.random.default_rng(20260907)
    n_rep = 300
    rejections = sum(_gate_rejects(rng) for _ in range(n_rep))
    fpr = rejections / n_rep
    # Binomial 95% CI half-width at alpha=0.05, n=300 is ~0.025.
    assert 0.02 <= fpr <= 0.08, (
        f"Gate FPR under H0 = {fpr:.3f} over {n_rep} replicates "
        f"({rejections} rejections); expected ~{ALPHA} within [0.02, 0.08]. "
        "A value above 0.08 means the draw-wise NaN-mean aggregation is "
        "anti-conservative under heterogeneous missingness."
    )


@pytest.mark.slow
def test_gate_is_conservative_when_one_dyad_has_few_draws():
    """A single dyad with very few surrogate draws truncates the group null.

    Truncation must reduce power (min attainable p rises), never inflate
    rejection. Verify the recorded min attainable p respects 1/(n+1).
    """
    rng = np.random.default_rng(42)
    results = _make_results(rng, n_dyads=10, ragged=True, nan_rate=0.0)
    # Force one dyad to have only 5 draws.
    first_key = next(iter(results))
    results[first_key]["null_peak_amplitude"] = results[first_key][
        "null_peak_amplitude"
    ][:5]
    gate = _existence_gate_by_modality(
        results, primary_modalities=("ECG",), alpha=ALPHA
    )
    mod = gate["per_modality"]["ECG"]
    # Group null width = min over dyads = 5 finite draws -> min attainable
    # two-sided p = 2/(5+1) = 0.333..., so the gate cannot pass at 0.05.
    assert mod["n_null_draws"] == 5
    assert mod["min_attainable_two_sided_p"] == pytest.approx(2 / 6)
    assert gate["primary_pass"] is False


def test_gate_reports_explicit_three_way_status():
    """Round-4: the raw gate dict must distinguish fail vs not-evaluable.

    - A gate whose modalities are all outside the registered primary set
      did NOT adjudicate: gate_status="not_evaluable" (the typed evidence
      chain maps this to INCONCLUSIVE, not NOT_SUPPORTED).
    - A gate with a registered, testable primary that did not pass is a
      genuine "fail".
    """
    rng = np.random.default_rng(7)
    results = _make_results(rng, modality="motion")
    gate = _existence_gate_by_modality(
        results, primary_modalities=("ECG", "EDA"), alpha=ALPHA
    )
    assert gate["primary_pass"] is False
    assert gate["gate_status"] == "not_evaluable"

    results_ecg = _make_results(rng, modality="ECG", nan_rate=0.0, ragged=False)
    gate2 = _existence_gate_by_modality(
        results_ecg, primary_modalities=("ECG",), alpha=ALPHA
    )
    assert gate2["gate_status"] == ("pass" if gate2["primary_pass"] else "fail")

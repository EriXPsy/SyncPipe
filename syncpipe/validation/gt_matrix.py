"""分层 GT 矩阵：白盒数值核对、灰盒生成机制和统计校准不能互相替代。"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..design_controls import synchrony_existence_audit
from ..simulation import constant_coupling, generate_signals, smooth_trapezoidal_coupling
from ..wcc import sliding_window_wcc
from .discriminant import generate_discriminant_pair


@dataclass(frozen=True)
class GTScenario:
    """生成场景及其已知机制；不是构念真值。"""

    name: str
    family: str
    x_A: np.ndarray
    x_B: np.ndarray
    truth: Dict[str, object]
    intended_claim: str


@dataclass(frozen=True)
class GTValidationResult:
    """合同检查不等于数值或统计验证通过。"""

    scenario: str
    numerical_correctness: str
    type_i_calibration: str
    power: str
    definedness: str
    claimability: str
    notes: Tuple[str, ...] = ()


GT_SCENARIOS = {
    "null": {"family": "null", "truth": "independent_ar1", "generator": "independent_ar1"},
    "zero_lag_linear": {"family": "coupling", "truth": "shared_signal_mixture"},
    "lead_lag": {"family": "coupling", "truth": "lagged_copy"},
    "shared_driver_slow_drift": {"family": "rival", "truth": "common_drift", "generator": "common_drift"},
    "missingness": {"family": "robustness", "truth": "shared_mixture_with_missingness"},
    "nonstationary_episode": {"family": "structure", "truth": "episode_mixture"},
    "independent_autocorr_null": {"family": "null", "truth": "independent_ar1", "generator": "independent_ar1"},
    "reciprocal_var": {"family": "coupling", "truth": "direct_reciprocal_var", "generator": "reciprocal_var"},
    "shared_driver_no_direct_coupling": {"family": "rival", "truth": "shared_stimulus", "generator": "shared_stimulus"},
}

EXTENDED_SCENARIOS = GT_SCENARIOS


_DISCRIMINANT_SCENARIOS = {"null", "independent_autocorr_null", "shared_driver_slow_drift",
                           "reciprocal_var", "shared_driver_no_direct_coupling"}


def _lead_lag_pair(n_samples: int, seed: int, lag: int = 8):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float)
    a = np.sin(2 * np.pi * 0.04 * t) + 0.4 * np.sin(2 * np.pi * 0.11 * t)
    a += 0.15 * rng.normal(size=n_samples)
    b = np.empty_like(a)
    b[:lag] = rng.normal(size=lag)
    b[lag:] = a[:-lag] + 0.15 * rng.normal(size=n_samples - lag)
    return a, b


def make_gt_scenario(name: str, *, n_samples: int = 240, seed: int = 42,
                     missing_fraction: float = 0.15) -> GTScenario:
    """复用现有生成器；混合权重不解释为相关系数或直接因果耦合。"""
    if name not in GT_SCENARIOS:
        raise ValueError(f"unknown GT scenario {name!r}; choose from {sorted(GT_SCENARIOS)}")
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must lie in [0, 1)")
    mechanism = str(GT_SCENARIOS[name]["truth"])
    if name in _DISCRIMINANT_SCENARIOS:
        generator = str(GT_SCENARIOS[name]["generator"])
        a, b = generate_discriminant_pair(generator, n_samples=n_samples, seed=seed)
        truth = {"direct_coupling": name == "reciprocal_var",
                 "independence_null": generator == "independent_ar1"}
        if generator != "reciprocal_var":
            truth["ar_phi"] = 0.9
        if name == "null":
            truth["coupling"] = 0.0
        if name == "reciprocal_var":
            truth.update(own_coefficient=0.62, cross_coefficient=0.22)
    elif name == "lead_lag":
        a, b = _lead_lag_pair(n_samples, seed)
        truth = {"coupling": 1.0, "lag_samples": 8, "direction": "A_leads_B",
                 "direct_coupling": True, "independence_null": False}
    else:
        coupling = constant_coupling(0.75)
        if name == "nonstationary_episode":
            coupling = smooth_trapezoidal_coupling(onset_delay=40.0, rise_duration=30.0,
                                                  plateau_duration=60.0, decay_duration=30.0)
        result = generate_signals(coupling, duration_sec=float(n_samples), hz=1.0, seed=seed)
        a, b = result.x_A.copy(), result.x_B.copy()
        truth = {"coupling": result.c_t, "coupling_meaning": "shared_mixture_weight_not_correlation",
                 "direct_coupling": False, "independence_null": False, "noise_sigma": 0.3}
        if name == "missingness":
            missing = np.random.default_rng(seed + 1).random(n_samples) < missing_fraction
            a[missing] = np.nan
            b[missing] = np.nan
            truth.update(missing_fraction=missing_fraction, missing_count=int(missing.sum()),
                         missing_mechanism="paired_MCAR_no_time_compression")
    truth.update(mechanism=mechanism, seed=seed, n_samples=n_samples)
    return GTScenario(name, str(GT_SCENARIOS[name]["family"]), a, b, truth,
                      "synthetic measurement behavior only; no external-validity claim")


def validate_gt_contract(scenario: GTScenario, *, observed_defined: bool | None = None):
    """形状检查不冒充白盒数值核对；有限样本数不保证 IAAFT 可测。"""
    same_shape = scenario.x_A.shape == scenario.x_B.shape and scenario.x_A.ndim == 1
    defined = "not_tested" if observed_defined is None else ("pass" if observed_defined else "undefined")
    return GTValidationResult(
        scenario.name, "not_tested" if same_shape else "fail",
        "required" if scenario.family == "null" else "not_applicable",
        "required" if scenario.family not in ("null", "rival") else "not_applicable",
        defined, "not_claimable", (scenario.intended_claim, "shape_pass" if same_shape else "shape_fail"),
    )


def wilson_interval(successes: int, trials: int):
    """有效独立重复的双侧 95% Wilson 区间；零分母不报告零率。"""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return None, None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


GT_CELLS = tuple((name, 240, 30) for name in GT_SCENARIOS) + tuple(
    (name, n, w) for name in ("null", "reciprocal_var")
    for n, w in ((240, 60), (480, 30)))


def run_gt_matrix(*, n_replicates: int = 20, surrogate_n: int = 99, seed: int = 42,
                  cells=None):
    """九场景及 null/直接耦合的单因素窗口、长度对照，不做全因子。"""
    if n_replicates < 1 or surrogate_n < 20 or seed < 0:
        raise ValueError("require n_replicates >= 1, surrogate_n >= 20, seed >= 0")
    cells = GT_CELLS if cells is None else tuple(cells)
    if not cells or any(tuple(cell) not in GT_CELLS for cell in cells):
        raise ValueError("cells must be a nonempty subset of GT_CELLS")
    rows = []
    for name, n, window in cells:
        # 同场景的敏感性单元使用相同 seed；区间仅在单元内计算，不跨单元合并。
        index = list(GT_SCENARIOS).index(name)
        for replicate in range(n_replicates):
            data_seed = seed + index * 100_000 + replicate
            audit_seed = data_seed + 10_000_000
            scenario = make_gt_scenario(name, n_samples=n, seed=data_seed)
            a, b = scenario.x_A, scenario.x_B
            wcc = sliding_window_wcc(a, b, window_size=window, hz=1.0)
            reference = []
            for start in range(n - window + 1):
                x, y = a[start:start + window], b[start:start + window]
                valid = np.isfinite(x) & np.isfinite(y)
                reference.append(float(np.corrcoef(x[valid], y[valid])[0, 1])
                                 if valid.sum() >= max(2, window * 0.5) else np.nan)
            reference = np.asarray(reference)
            numerical = bool(np.allclose(wcc, reference, atol=1e-10, rtol=1e-10, equal_nan=True))
            audit = synchrony_existence_audit(a, b, hz=1.0, window_size=window,
                                             surrogate_n=surrogate_n, seed=audit_seed)
            p = audit.get("p_values", {}).get("peak_amplitude", np.nan)
            valid = audit.get("status") == "ok" and bool(np.isfinite(p))
            rate_kind = ("statistical_fpr" if scenario.truth["independence_null"] else
                         "wrong_interpersonal_interpretation_risk" if scenario.family == "rival" else
                         "power_under_declared_generator")
            rows.append(dict(scenario=name, n_samples=n, window_size=window, replicate=replicate,
                             seed=data_seed, audit_seed=audit_seed, surrogate_n=surrogate_n,
                             mechanism=scenario.truth["mechanism"],
                             direct_coupling=scenario.truth["direct_coupling"],
                             numerical_correctness="pass" if numerical else "fail",
                             generation_audit="declared_mechanism_not_construct_truth",
                             status=audit.get("status"), reason=audit.get("reason", ""),
                             effective=valid, p_peak_amplitude=float(p) if valid else None,
                             rejected=bool(p < 0.05) if valid else None, rate_kind=rate_kind,
                             finite_pairs=int((np.isfinite(a) & np.isfinite(b)).sum())))
    replicates = pd.DataFrame(rows)
    summaries = []
    for (name, n, w), group in replicates.groupby(["scenario", "n_samples", "window_size"], sort=False):
        effective = group[group.effective]
        count = len(effective)
        rejected = int(effective.rejected.sum())
        low, high = wilson_interval(rejected, count)
        summaries.append(dict(scenario=name, n_samples=n, window_size=w,
                              n_requested=n_replicates, n_effective=count,
                              n_unmeasurable=n_replicates - count, n_rejected=rejected,
                              rate=rejected / count if count else None,
                              wilson95_lower=low, wilson95_upper=high,
                              rate_kind=group.rate_kind.iloc[0],
                              numerical_passes=int((group.numerical_correctness == "pass").sum()),
                              statistical_calibration="not_established" if count else "unmeasurable",
                              claimability="not_claimable"))
    return replicates, pd.DataFrame(summaries)


__all__ = ["GTScenario", "GTValidationResult", "GT_SCENARIOS", "EXTENDED_SCENARIOS",
           "GT_CELLS", "make_gt_scenario", "validate_gt_contract", "wilson_interval",
           "run_gt_matrix"]

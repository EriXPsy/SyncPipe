"""
multisync.batch — Group-level statistical analysis.

Design: (1) no for-loops, (2) BH-FDR, (3) effect sizes, (4) non-parametric default.

Usage: batch_analyze(configs_A) → results_A; group_comparison(results_A, results_B) → report.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .core import Dyad, DynamicAnalyzer, AnalysisResults


# ---------------------------------------------------------------------------
# BatchConfig — per-dyad configuration record
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """
    Configuration for a single dyad in a batch analysis.

    Parameters
    ----------
    dyad_id : str
        Unique identifier for this dyad.
    hz : float
        Target sampling rate.
    preprocessing : dict
        Preprocessing kwargs. Keys: 'clip_outliers', 'median_filter', 'zscore'.
        Example: {'clip_outliers': {'factor': 3.0, 'method': 'mad'},
                  'median_filter': {'kernel_size': 5},
                  'zscore': {'method': 'robust', 'clip_sigma': 3.5}}
    context_labels : list of dict
        Context window definitions.
        Example: [{'start': 0, 'end': 300, 'label': 'Task'}]
    **modalities : pd.DataFrame
        Modality DataFrames passed as keyword arguments (same as Dyad).
    """
    dyad_id: str
    modalities: Dict[str, pd.DataFrame] = field(default_factory=dict)
    hz: float = 1.0
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "clip_outliers": {"factor": 3.0, "method": "mad"},
        "median_filter": {"kernel_size": 5},
        "zscore": {"method": "robust", "clip_sigma": 3.5},
    })
    context_labels: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DyadResult — single dyad's analysis outcome for batch reporting
# ---------------------------------------------------------------------------

@dataclass
class DyadResult:
    """Extracted scalar metrics from one dyad's AnalysisResults.

    Notes (v3)
    ----------
    v2 fields (``mean_peak_sync`` etc.) retained for backward compatibility.
    v3 canonical fields: ``mean_peak_amplitude``, ``mean_rise_time``,
    ``mean_recovery_time``, ``mean_synchrony``, ``mean_synchrony_entropy``,
    plus definedness rates.
    """
    dyad_id: str
    # Association metrics
    frac_significant_edges: float = float("nan")
    mean_peak_lag_sec: float = float("nan")
    driver_scores: Dict[str, float] = field(default_factory=dict)
    # ── v2 (backward-compat aliases) ──
    mean_onset_latency: float = float("nan")
    mean_peak_sync: float = float("nan")
    mean_build_up_rate: float = float("nan")
    mean_breakdown_rate: float = float("nan")
    # ── v3 (canonical) ──
    mean_peak_amplitude: float = float("nan")
    mean_rise_time: float = float("nan")
    mean_recovery_time: float = float("nan")
    mean_dwell_time: float = float("nan")
    mean_switching_rate: float = float("nan")
    mean_synchrony: float = float("nan")
    mean_synchrony_entropy: float = float("nan")
    # Definedness rates (across modality pairs within the dyad)
    onset_defined_rate: float = float("nan")
    recovery_defined_rate: float = float("nan")
    # Prediction
    mean_dynamic_auc: float = float("nan")
    mean_delta_auc: float = float("nan")
    # Quality
    n_diagnostics: int = 0
    raw: Optional[AnalysisResults] = field(default=None, repr=False)

    @classmethod
    def from_analysis_results(
        cls, results: AnalysisResults
    ) -> "DyadResult":
        """Extract scalar summary from AnalysisResults.

        Populates both v2 (legacy) and v3 (canonical) feature fields."""
        # Cross-modal association/cascade was removed from the v1 pipeline,
        # so these fields are no longer populated.
        n_total = 0
        n_sig = 0
        mean_lag = float("nan")
        driver_scores: Dict[str, float] = {}

        # Dynamic features
        feat = results.dynamic_features
        def _mean_feat(key: str) -> float:
            vals = []
            for pair_f in feat.values():
                # Support both dict-style and dataclass-style entries.
                if hasattr(pair_f, key):
                    v = getattr(pair_f, key)
                elif isinstance(pair_f, dict):
                    v = pair_f.get(key)
                else:
                    v = None
                if v is not None and not (
                    isinstance(v, float) and np.isnan(v)
                ):
                    vals.append(float(v))
            return float(np.mean(vals)) if vals else float("nan")

        def _defined_rate(key: str) -> float:
            """Fraction of modality pairs where ``key`` is defined.

            Strategy
            --------
            1. If the feature object exposes the explicit ``<key>``
               flag (1.0 / 0.0), use it directly.
            2. Otherwise, fall back to the SEMANTIC equivalence used by
               ``recovery._extract_six_features``: a feature is
               "defined" iff its paired value-bearing feature is
               non-NaN.  ``onset_defined`` is mirrored by
               ``onset_latency``; ``recovery_defined`` by
               ``recovery_time``.
            """
            mirror = {
                "onset_defined": "onset_latency",
                "recovery_defined": "recovery_time",
            }
            value_key = mirror.get(key)

            n_total_pairs = 0
            n_defined = 0
            for pair_f in feat.values():
                n_total_pairs += 1
                # Try the explicit flag first.
                if hasattr(pair_f, key):
                    v = getattr(pair_f, key)
                elif isinstance(pair_f, dict) and key in pair_f:
                    v = pair_f.get(key)
                else:
                    v = None

                if v is None and value_key is not None:
                    # Fallback: defined ⟺ paired value is non-NaN.
                    if hasattr(pair_f, value_key):
                        vv = getattr(pair_f, value_key)
                    elif isinstance(pair_f, dict):
                        vv = pair_f.get(value_key)
                    else:
                        vv = None
                    if vv is None or (isinstance(vv, float) and np.isnan(vv)):
                        continue  # not defined
                    n_defined += 1
                    continue

                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                n_defined += 1 if float(v) > 0.5 else 0

            if n_total_pairs == 0:
                return float("nan")
            return n_defined / n_total_pairs

        # v3 canonical names (and v2 fallback aliases)
        v3_peak    = _mean_feat("peak_amplitude")
        v3_rise    = _mean_feat("rise_time")
        v3_rec     = _mean_feat("recovery_time")
        v3_dwell   = _mean_feat("dwell_time")
        v3_switch  = _mean_feat("switching_rate")
        v3_mean    = _mean_feat("mean_synchrony")
        v3_entropy = _mean_feat("synchrony_entropy")
        v3_onset   = _mean_feat("onset_latency")

        # Backward-compat v2 mirrors (v3 canonical names preferred)
        v2_peak       = v3_peak if not np.isnan(v3_peak) else _mean_feat("peak_sync_value")
        v2_buildup    = (1.0 / v3_rise) if (np.isfinite(v3_rise) and v3_rise > 0) \
                        else _mean_feat("build_up_rate")
        v2_breakdown  = (1.0 / v3_rec) if (np.isfinite(v3_rec) and v3_rec > 0) \
                        else _mean_feat("breakdown_rate")

        # Prediction
        pred = results.prediction
        aucs = [v["mean_dynamic_auc"] for v in pred.values()
                if v.get("mean_dynamic_auc") is not None
                and not np.isnan(float(v["mean_dynamic_auc"]))]
        deltas = [v["mean_delta_auc"] for v in pred.values()
                  if v.get("mean_delta_auc") is not None
                  and not np.isnan(float(v["mean_delta_auc"]))]

        return cls(
            dyad_id=results.dyad_id,
            frac_significant_edges=n_sig / n_total if n_total > 0 else float("nan"),
            mean_peak_lag_sec=mean_lag,
            driver_scores=driver_scores,
            # v2
            mean_onset_latency=v3_onset,
            mean_peak_sync=v2_peak,
            mean_build_up_rate=v2_buildup,
            mean_breakdown_rate=v2_breakdown,
            # v3
            mean_peak_amplitude=v3_peak,
            mean_rise_time=v3_rise,
            mean_recovery_time=v3_rec,
            mean_dwell_time=v3_dwell,
            mean_switching_rate=v3_switch,
            mean_synchrony=v3_mean,
            mean_synchrony_entropy=v3_entropy,
            onset_defined_rate=_defined_rate("onset_defined"),
            recovery_defined_rate=_defined_rate("recovery_defined"),
            mean_dynamic_auc=float(np.mean(aucs)) if aucs else float("nan"),
            mean_delta_auc=float(np.mean(deltas)) if deltas else float("nan"),
            n_diagnostics=len(results.diagnostics),
            raw=results,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dyad_id": self.dyad_id,
            "frac_significant_edges": self.frac_significant_edges,
            "mean_peak_lag_sec": self.mean_peak_lag_sec,
            "driver_scores": self.driver_scores,
            # v2 (backward-compat)
            "mean_onset_latency": self.mean_onset_latency,
            "mean_peak_sync": self.mean_peak_sync,
            "mean_build_up_rate": self.mean_build_up_rate,
            "mean_breakdown_rate": self.mean_breakdown_rate,
            # v3 (canonical)
            "mean_peak_amplitude": self.mean_peak_amplitude,
            "mean_rise_time": self.mean_rise_time,
            "mean_recovery_time": self.mean_recovery_time,
            "mean_dwell_time": self.mean_dwell_time,
            "mean_switching_rate": self.mean_switching_rate,
            "mean_synchrony": self.mean_synchrony,
            "mean_synchrony_entropy": self.mean_synchrony_entropy,
            "onset_defined_rate": self.onset_defined_rate,
            "recovery_defined_rate": self.recovery_defined_rate,
            "mean_dynamic_auc": self.mean_dynamic_auc,
            "mean_delta_auc": self.mean_delta_auc,
            "n_diagnostics": self.n_diagnostics,
        }


# ---------------------------------------------------------------------------
# Group comparison statistics
# ---------------------------------------------------------------------------

@dataclass
class MetricTestResult:
    """Statistical test result for one metric."""
    metric: str
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    # Raw (uncorrected) p-value
    p_raw: float
    # BH-corrected p-value (filled in after group_comparison runs all tests)
    p_fdr: float = float("nan")
    # Effect size
    effect_size: float = float("nan")
    effect_size_type: str = "cohens_d"  # "cohens_d" or "rank_biserial_r"
    # Test details
    test_name: str = "mann_whitney_u"
    statistic: float = float("nan")
    n_a: int = 0
    n_b: int = 0
    significant_fdr: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mean_a": self.mean_a, "mean_b": self.mean_b,
            "median_a": self.median_a, "median_b": self.median_b,
            "p_raw": self.p_raw, "p_fdr": self.p_fdr,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "n_a": self.n_a, "n_b": self.n_b,
            "significant_fdr": self.significant_fdr,
        }


@dataclass
class GroupComparisonReport:
    """Full group comparison report."""
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    alpha: float
    test_results: List[MetricTestResult] = field(default_factory=list)
    dyad_results_a: List[DyadResult] = field(default_factory=list)
    dyad_results_b: List[DyadResult] = field(default_factory=list)

    def significant_metrics(self, use_fdr: bool = True) -> List[MetricTestResult]:
        """Return only significant results."""
        return [r for r in self.test_results if r.significant_fdr]

    def summary_table(self) -> str:
        """Pretty-print summary table."""
        lines = []
        W = 90
        lines.append("\n" + "=" * W)
        lines.append(f"  Group Comparison: {self.label_a} (n={self.n_a}) vs "
                     f"{self.label_b} (n={self.n_b})")
        lines.append(f"  Alpha={self.alpha}, FDR correction: Benjamini-Hochberg")
        lines.append("=" * W)

        header = (f"  {'Metric':<28} {'Mean A':>8} {'Mean B':>8} "
                  f"{'p_raw':>8} {'p_FDR':>8} {'Effect':>8} {'Sig?':>6}")
        lines.append(header)
        lines.append("-" * W)

        for r in sorted(self.test_results, key=lambda x: x.p_raw):
            sig = "*" if r.significant_fdr else " "
            lines.append(
                f"  {r.metric:<28} "
                f"{r.mean_a:>8.3f} {r.mean_b:>8.3f} "
                f"{r.p_raw:>8.4f} {r.p_fdr:>8.4f} "
                f"{r.effect_size:>8.3f} {sig:>6}"
            )

        lines.append("=" * W)
        n_sig = sum(1 for r in self.test_results if r.significant_fdr)
        lines.append(f"  {n_sig}/{len(self.test_results)} metrics significant "
                     f"after FDR correction (alpha={self.alpha})")
        lines.append("=" * W + "\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "alpha": self.alpha,
            "test_results": [r.to_dict() for r in self.test_results],
            "dyad_results_a": [d.to_dict() for d in self.dyad_results_a],
            "dyad_results_b": [d.to_dict() for d in self.dyad_results_b],
        }

    def to_json(self, filepath: str) -> str:
        """Export report to JSON."""
        def _sanitize(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_sanitize(self.to_dict()), f, indent=2)
        return str(p)

    def to_dataframe(self) -> pd.DataFrame:
        """Export test results as a pandas DataFrame (for further stats)."""
        return pd.DataFrame([r.to_dict() for r in self.test_results])


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using pooled SD (ignores NaN)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt((np.var(a, ddof=1) * (len(a) - 1) +
                         np.var(b, ddof=1) * (len(b) - 1)) /
                        (len(a) + len(b) - 2))
    if pooled_sd < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def _rank_biserial_r(u_stat: float, n_a: int, n_b: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    if n_a == 0 or n_b == 0:
        return float("nan")
    return float(1.0 - (2.0 * u_stat) / (n_a * n_b))


def _bh_fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Benjamini-Hochberg FDR correction.

    Returns corrected p-values and significance flags.
    Handles NaN p-values gracefully (treated as 1.0).
    """
    n = len(p_values)
    if n == 0:
        return [], []

    p_arr = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    ranked = np.argsort(p_arr)
    p_corrected = np.ones(n)

    # BH formula: p_adj[i] = p[i] * n / rank[i]
    # Applied in rank order, enforcing monotonicity from right
    cummin = np.inf
    for rank, idx in enumerate(reversed(ranked), start=1):
        rank_from_top = n - rank + 1
        corrected = p_arr[idx] * n / rank_from_top
        cummin = min(cummin, corrected)
        p_corrected[idx] = cummin

    p_corrected = np.clip(p_corrected, 0.0, 1.0)
    significant = p_corrected < alpha
    return p_corrected.tolist(), significant.tolist()


def _extract_metric_arrays(
    dyad_results: List[DyadResult],
) -> Dict[str, np.ndarray]:
    """Build metric arrays from DyadResults (v2 and v3 fields)."""
    metrics: Dict[str, List[float]] = {
        "frac_significant_edges": [],
        "mean_peak_lag_sec": [],
        # v2 mirrors
        "mean_onset_latency": [],
        "mean_peak_sync": [],
        "mean_build_up_rate": [],
        "mean_breakdown_rate": [],
        # v3 canonical
        "mean_peak_amplitude": [],
        "mean_rise_time": [],
        "mean_recovery_time": [],
        "mean_synchrony": [],
        "mean_synchrony_entropy": [],
        "onset_defined_rate": [],
        "recovery_defined_rate": [],
        # prediction
        "mean_dynamic_auc": [],
        "mean_delta_auc": [],
    }
    for dr in dyad_results:
        metrics["frac_significant_edges"].append(dr.frac_significant_edges)
        metrics["mean_peak_lag_sec"].append(dr.mean_peak_lag_sec)
        # v2
        metrics["mean_onset_latency"].append(dr.mean_onset_latency)
        metrics["mean_peak_sync"].append(dr.mean_peak_sync)
        metrics["mean_build_up_rate"].append(dr.mean_build_up_rate)
        metrics["mean_breakdown_rate"].append(dr.mean_breakdown_rate)
        # v3
        metrics["mean_peak_amplitude"].append(dr.mean_peak_amplitude)
        metrics["mean_rise_time"].append(dr.mean_rise_time)
        metrics["mean_recovery_time"].append(dr.mean_recovery_time)
        metrics["mean_synchrony"].append(dr.mean_synchrony)
        metrics["mean_synchrony_entropy"].append(dr.mean_synchrony_entropy)
        metrics["onset_defined_rate"].append(dr.onset_defined_rate)
        metrics["recovery_defined_rate"].append(dr.recovery_defined_rate)
        # prediction
        metrics["mean_dynamic_auc"].append(dr.mean_dynamic_auc)
        metrics["mean_delta_auc"].append(dr.mean_delta_auc)

    return {k: np.array(v, dtype=float) for k, v in metrics.items()}


def batch_analyze(
    configs: List[BatchConfig],
    analyzer_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> List[DyadResult]:
    """
    Run the full SyncPipe pipeline on a list of dyads.

    Parameters
    ----------
    configs : list of BatchConfig
        One config per dyad. Each config specifies modalities, Hz,
        preprocessing, and context labels.
    analyzer_kwargs : dict or None
        Kwargs forwarded to DynamicAnalyzer (e.g., surrogate_n, window_size).
    verbose : bool
        Print progress.

    Returns
    -------
    list of DyadResult
        One DyadResult per successfully analyzed dyad.

    Notes
    -----
    Failed dyads (preprocessing errors, insufficient data) are skipped with
    a warning rather than aborting the entire batch.
    """
    if analyzer_kwargs is None:
        analyzer_kwargs = {}

    analyzer = DynamicAnalyzer(**analyzer_kwargs)
    results: List[DyadResult] = []

    for i, cfg in enumerate(configs):
        if verbose:
            print(f"  [{i + 1}/{len(configs)}] Analyzing dyad: {cfg.dyad_id}")

        try:
            dyad = Dyad(hz=cfg.hz, dyad_id=cfg.dyad_id, **cfg.modalities)
            dyad.align(target_hz=cfg.hz)

            # Poka-yoke preprocessing (use defaults from config)
            pp = cfg.preprocessing
            if "clip_outliers" in pp:
                dyad.clip_outliers(**pp["clip_outliers"])
            if "median_filter" in pp:
                dyad.median_filter(**pp["median_filter"])
            if "zscore" in pp:
                dyad.zscore(**pp["zscore"])

            # Context labels
            for ctx in cfg.context_labels:
                dyad.add_context(**ctx)

            ar = analyzer.fit_transform(dyad)
            dr = DyadResult.from_analysis_results(ar)
            results.append(dr)

        except Exception as exc:
            warnings.warn(
                f"Dyad '{cfg.dyad_id}' failed with {type(exc).__name__}: {exc}. "
                "Skipping.",
                UserWarning,
            )

    if verbose:
        print(f"  Done. {len(results)}/{len(configs)} dyads analyzed successfully.")

    return results


def group_comparison(
    group_a: List[DyadResult],
    group_b: List[DyadResult],
    label_a: str = "Group A",
    label_b: str = "Group B",
    alpha: float = 0.05,
    test: str = "mann_whitney",
) -> GroupComparisonReport:
    """
    Compare two groups of dyads on all scalar synchrony metrics.

    Applies Benjamini-Hochberg FDR correction across all comparisons.

    Parameters
    ----------
    group_a, group_b : list of DyadResult
        Results from batch_analyze() for each group.
    label_a, label_b : str
        Display names for the two groups.
    alpha : float
        FDR threshold (default 0.05).
    test : str
        'mann_whitney' (default, non-parametric) or 'welch_t' (parametric).

    Returns
    -------
    GroupComparisonReport

    Notes
    -----
    * Metrics with <3 valid values in either group are skipped.
    * All p-values are two-tailed.
    * Effect sizes:
        - Mann-Whitney: rank-biserial r
        - Welch's t: Cohen's d
    """
    if len(group_a) < 2 or len(group_b) < 2:
        warnings.warn(
            "Each group should have ≥ 2 dyads for meaningful statistics. "
            f"Got: group_a={len(group_a)}, group_b={len(group_b)}.",
            UserWarning,
        )

    arr_a = _extract_metric_arrays(group_a)
    arr_b = _extract_metric_arrays(group_b)

    raw_results: List[MetricTestResult] = []

    for metric in arr_a:
        a = arr_a[metric]
        b = arr_b[metric]

        # Drop NaN
        a_valid = a[~np.isnan(a)]
        b_valid = b[~np.isnan(b)]

        if len(a_valid) < 3 or len(b_valid) < 3:
            raw_results.append(MetricTestResult(
                metric=metric,
                mean_a=float(np.nanmean(a)) if len(a_valid) else float("nan"),
                mean_b=float(np.nanmean(b)) if len(b_valid) else float("nan"),
                median_a=float(np.nanmedian(a)) if len(a_valid) else float("nan"),
                median_b=float(np.nanmedian(b)) if len(b_valid) else float("nan"),
                p_raw=float("nan"),
                n_a=len(a_valid), n_b=len(b_valid),
                test_name="skipped_insufficient_n",
            ))
            continue

        mean_a = float(np.mean(a_valid))
        mean_b = float(np.mean(b_valid))
        med_a = float(np.median(a_valid))
        med_b = float(np.median(b_valid))

        if test == "welch_t":
            t_stat, p_val = stats.ttest_ind(a_valid, b_valid, equal_var=False)
            d = _cohens_d(a_valid, b_valid)
            raw_results.append(MetricTestResult(
                metric=metric,
                mean_a=mean_a, mean_b=mean_b,
                median_a=med_a, median_b=med_b,
                p_raw=float(p_val),
                effect_size=d, effect_size_type="cohens_d",
                test_name="welch_t",
                statistic=float(t_stat),
                n_a=len(a_valid), n_b=len(b_valid),
            ))
        else:
            # Mann-Whitney U (default)
            u_stat, p_val = stats.mannwhitneyu(
                a_valid, b_valid, alternative="two-sided"
            )
            r = _rank_biserial_r(float(u_stat), len(a_valid), len(b_valid))
            raw_results.append(MetricTestResult(
                metric=metric,
                mean_a=mean_a, mean_b=mean_b,
                median_a=med_a, median_b=med_b,
                p_raw=float(p_val),
                effect_size=r, effect_size_type="rank_biserial_r",
                test_name="mann_whitney_u",
                statistic=float(u_stat),
                n_a=len(a_valid), n_b=len(b_valid),
            ))

    # BH FDR correction (only on non-NaN p-values)
    valid_indices = [i for i, r in enumerate(raw_results)
                     if not np.isnan(r.p_raw)]
    valid_p = [raw_results[i].p_raw for i in valid_indices]

    if valid_p:
        corrected_p, sig_flags = _bh_fdr_correction(valid_p, alpha=alpha)
        for idx, corr_p, sig in zip(valid_indices, corrected_p, sig_flags):
            raw_results[idx].p_fdr = corr_p
            raw_results[idx].significant_fdr = sig

    return GroupComparisonReport(
        label_a=label_a,
        label_b=label_b,
        n_a=len(group_a),
        n_b=len(group_b),
        alpha=alpha,
        test_results=raw_results,
        dyad_results_a=group_a,
        dyad_results_b=group_b,
    )
    
def residualize_features(
    df: pd.DataFrame,
    features: List[str],
    baseline: str = "mean_synchrony",
    suffix: str = "_residual",
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Remove linear contribution of baseline (mean_synchrony) from each feature.

    Implements the deconfounding step described in DIMENSIONAL_MODEL.md §2:
        f_residual = f − β̂·mean_synchrony

    Parameters
    ----------
    df : DataFrame containing feature columns and baseline column.
         Typically the output of DyadResult.to_dict() stacked into a DataFrame.
    features : list of feature column names to residualize.
    baseline : column to regress out (default: "mean_synchrony").
    suffix : appended to each feature name for the new residual column.
    min_n : minimum valid rows required; returns NaN column if not met.

    Returns
    -------
    df copy with new columns  ``{feature}{suffix}``  for each feature.

    Notes
    -----
    Uses bivariate OLS (β = Cov(X,Y) / Var(X)).  Only rows where both
    feature and baseline are finite are used to estimate β; residuals
    are then computed for all rows (NaN where either is NaN).
    """
    out = df.copy()
    baseline_vals = df[baseline].to_numpy(dtype=float)

    for feat in features:
        col = f"{feat}{suffix}"
        if feat not in df.columns:
            out[col] = np.nan
            continue

        feat_vals = df[feat].to_numpy(dtype=float)
        mask = np.isfinite(feat_vals) & np.isfinite(baseline_vals)

        if mask.sum() < min_n:
            out[col] = np.nan
            continue

        x = baseline_vals[mask]
        y = feat_vals[mask]
        beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        alpha = y.mean() - beta * x.mean()

        out[col] = feat_vals - (alpha + beta * baseline_vals)

    return out
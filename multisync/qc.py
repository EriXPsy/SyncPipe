"""
Data Quality Check — 3-stage pre-analysis gate.

Stages: (1) temporal alignment, (2) NaN integrity, (3) sampling uniformity.

Mandatory: must pass before analysis. WARN → warning; FAIL → DataQualityError.

Usage: run_quality_check(dataset) → DataQualityReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class StageVerdict:
    """Nominal verdict for a single QC stage."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Data Quality Error (raised on FAIL)
# ---------------------------------------------------------------------------

class DataQualityError(ValueError):
    """Raised when quality check fails at FAIL level.

    Analysis should NOT proceed when this is raised.  The message
    contains a human-readable summary of all failures.
    """
    pass


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result for a single QC stage."""
    stage: str
    verdict: str                    # "PASS" | "WARN" | "FAIL"
    details: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "verdict": self.verdict,
            "details": self.details,
            "message": self.message,
        }


def _suggest_actions_for_detail(detail: Dict[str, Any]) -> List[str]:
    """Return short, user-facing repair suggestions for a QC detail."""
    dtype = detail.get("type")
    if dtype == "length_mismatch":
        return ["Run dataset.align(target_hz=...) before analysis."]
    if dtype == "time_divergence":
        return ["Confirm all modalities share the same clock; realign or use absolute timestamps."]
    if dtype == "non_monotonic":
        return ["Sort the affected file by time and remove duplicate/backward timestamps."]
    if dtype == "high_nan_rate":
        return [
            "Inspect raw sensor dropout/artifacts for the affected channel.",
            "Interpolate only short gaps with handle_nan(..., max_gap_sec=...).",
            "Exclude the channel or participant if dropout is structural.",
        ]
    if dtype == "long_nan_gap":
        return [
            "Treat long missing stretches as sensor dropout; avoid interpolating across long gaps.",
            "Add exclusion intervals or remove the affected segment/channel.",
        ]
    if dtype == "irregular_sampling":
        return ["Resample to a uniform grid before WCC/IAAFT analysis."]
    if dtype == "zero_mean_isi":
        return ["Fix the time column; timestamps must be strictly increasing."]
    return []


@dataclass
class DataQualityReport:
    """Aggregated data quality check report.

    Attributes
    ----------
    dyad_id : str
        Dyad identifier.
    passed : bool
        True if NO stage returned FAIL.
    stages : list of StageResult
        Per-stage results in execution order.
    overall_verdict : str
        "PASS" if all stages pass; "WARN" if any stage warns (but none fail);
        "FAIL" if any stage fails.
    n_warnings : int
        Number of WARN-level issues across all stages.
    n_failures : int
        Number of FAIL-level issues across all stages.
    warnings : list of str
        Human-readable warning messages (for frontend display).
    failures : list of str
        Human-readable failure messages.
    """
    dyad_id: str
    stages: List[StageResult] = field(default_factory=list)
    n_warnings: int = 0
    n_failures: int = 0
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_failures == 0

    @property
    def overall_verdict(self) -> str:
        if self.n_failures > 0:
            return StageVerdict.FAIL
        if self.n_warnings > 0:
            return StageVerdict.WARN
        return StageVerdict.PASS

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        lines = [f"Data Quality Report — dyad '{self.dyad_id}'"]
        lines.append(f"  Overall: {self.overall_verdict}")
        lines.append(f"  Warnings: {self.n_warnings} | Failures: {self.n_failures}")
        for st in self.stages:
            lines.append(f"  [{st.verdict}] {st.stage}: {st.message}")
        if self.failures:
            lines.append("  --- FAILURES ---")
            for f in self.failures:
                lines.append(f"    {f}")
        if self.warnings:
            lines.append("  --- WARNINGS ---")
            for w in self.warnings:
                lines.append(f"    {w}")
        return "\n".join(lines)

    def suggested_actions(self) -> List[str]:
        """Actionable, de-duplicated suggestions derived from stage details."""
        actions: List[str] = []
        for stage in self.stages:
            for detail in stage.details:
                actions.extend(_suggest_actions_for_detail(detail))
        seen = set()
        out = []
        for action in actions:
            if action not in seen:
                out.append(action)
                seen.add(action)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dyad_id": self.dyad_id,
            "overall_verdict": self.overall_verdict,
            "passed": self.passed,
            "n_warnings": self.n_warnings,
            "n_failures": self.n_failures,
            "warnings": self.warnings,
            "failures": self.failures,
            "suggested_actions": self.suggested_actions(),
            "stages": [st.to_dict() for st in self.stages],
        }


# ---------------------------------------------------------------------------
# Default thresholds (configurable)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Stage 1: Temporal Alignment
    "max_time_offset_sec": 0.5,        # max allowed clock offset between modalities (seconds)
    "require_same_length": True,        # all modalities must have same N after alignment
    # Stage 2: NaN Integrity
    "max_nan_rate": 0.05,              # max fraction of NaN per modality/feature (5%)
    "max_consecutive_nan_sec": 5.0,    # max consecutive NaN gap (seconds)
    # Stage 3: Sampling Uniformity
    "max_isi_cv": 0.10,               # max coefficient of variation of inter-sample intervals
    "min_effective_samples": 50,       # minimum total samples after alignment
}


# ---------------------------------------------------------------------------
# Stage 1 — Temporal Alignment
# ---------------------------------------------------------------------------

def _check_temporal_alignment(dataset: Any, config: Dict[str, Any]) -> StageResult:
    """Check that all modalities share a common, monotonic time axis.

    Verifies:
    1. All modalities have the same number of samples after alignment.
    2. Time vectors are identical (within floating-point tolerance).
    3. No time reversal (monotonicity).

    This is the most critical check: misaligned clocks produce artificial
    lags in CCF that masquerade as synchrony.
    """
    details: List[Dict[str, Any]] = []
    failures: List[str] = []

    if not hasattr(dataset, "modalities") or not dataset.modalities:
        return StageResult(
            stage="temporal_alignment",
            verdict=StageVerdict.FAIL,
            message="No modalities found in dataset.",
        )

    modality_names = list(dataset.modalities.keys())
    time_vectors: Dict[str, np.ndarray] = {}
    lengths: Dict[str, int] = {}

    for name in modality_names:
        df = dataset.modalities[name]
        if "time" not in df.columns:
            return StageResult(
                stage="temporal_alignment",
                verdict=StageVerdict.FAIL,
                message=f"Modality '{name}' missing 'time' column.",
            )
        t = df["time"].values.astype(float)
        time_vectors[name] = t
        lengths[name] = len(t)

    # Check 1: same length
    if config.get("require_same_length", True):
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            detail = {
                "type": "length_mismatch",
                "lengths": {k: v for k, v in lengths.items()},
            }
            details.append(detail)
            failures.append(
                f"Modalities have different sample counts: {lengths}. "
                f"Run dataset.align() to resample to a common time axis."
            )

    # Check 2: identical time vectors
    if len(time_vectors) >= 2:
        names_list = list(time_vectors.keys())
        ref_t = time_vectors[names_list[0]]
        tol = 1e-6
        for name in names_list[1:]:
            t = time_vectors[name]
            if len(t) != len(ref_t) or not np.allclose(t, ref_t, atol=tol):
                details.append({
                    "type": "time_divergence",
                    "ref_modality": names_list[0],
                    "divergent_modality": name,
                    "max_diff": float(np.max(np.abs(t - ref_t[:len(t)]))) if len(t) == len(ref_t) else float("inf"),
                })
                failures.append(
                    f"Time vectors differ between '{names_list[0]}' and '{name}'. "
                    f"Alignment may have failed silently."
                )

    # Check 3: monotonicity
    for name, t in time_vectors.items():
        if len(t) < 2:
            continue
        diffs = np.diff(t)
        if np.any(diffs <= 0):
            details.append({
                "type": "non_monotonic",
                "modality": name,
                "backward_steps": int(np.sum(diffs <= 0)),
            })
            failures.append(
                f"Time axis for '{name}' is non-monotonic "
                f"({int(np.sum(diffs <= 0))} backward steps). "
                f"Sort by time before alignment."
            )

    if failures:
        return StageResult(
            stage="temporal_alignment",
            verdict=StageVerdict.FAIL,
            details=details,
            message=f"{len(failures)} temporal alignment issue(s).",
        )

    return StageResult(
        stage="temporal_alignment",
        verdict=StageVerdict.PASS,
        message=f"All {len(modality_names)} modalities share a common time axis "
                f"({lengths[modality_names[0]]} samples).",
    )


# ---------------------------------------------------------------------------
# Stage 2 — NaN Integrity
# ---------------------------------------------------------------------------

def _check_nan_integrity(dataset: Any, config: Dict[str, Any]) -> StageResult:
    """Assess NaN rate per modality/feature and detect long NaN gaps.

    High NaN rates inflate WCC variance and produce unstable dynamic
    feature estimates.  Long NaN gaps indicate sensor dropout that
    should be handled in preprocessing, not silently interpolated.
    """
    details: List[Dict[str, Any]] = []
    warnings_list: List[str] = []
    failures_list: List[str] = []

    max_nan_rate = config.get("max_nan_rate", 0.05)
    max_consecutive_nan_sec = config.get("max_consecutive_nan_sec", 5.0)

    if not hasattr(dataset, "feature_columns"):
        return StageResult(
            stage="nan_integrity",
            verdict=StageVerdict.FAIL,
            message="Dataset has no feature_columns attribute.",
        )

    feat_cols = dataset.feature_columns
    hz = getattr(dataset, "target_hz", 1.0) or 1.0
    dt = 1.0 / hz if hz > 0 else 1.0
    max_consecutive_samples = int(np.ceil(max_consecutive_nan_sec / dt))

    total_features = 0
    high_nan_features: List[str] = []
    gap_features: List[str] = []

    for mod_name, cols in feat_cols.items():
        if mod_name not in dataset.modalities:
            continue
        df = dataset.modalities[mod_name]
        for col in cols:
            if col not in df.columns:
                continue
            total_features += 1
            vals = df[col].values.astype(float)
            n_total = len(vals)
            n_nan = int(np.isnan(vals).sum())
            nan_rate = n_nan / n_total if n_total > 0 else 1.0

            if nan_rate > max_nan_rate:
                key = f"{mod_name}/{col}"
                high_nan_features.append(key)
                details.append({
                    "type": "high_nan_rate",
                    "modality": mod_name,
                    "feature": col,
                    "nan_rate": round(nan_rate, 4),
                    "n_nan": n_nan,
                    "n_total": n_total,
                })
                if nan_rate > 0.20:
                    failures_list.append(
                        f"{key}: {nan_rate:.1%} NaN ({n_nan}/{n_total}). "
                        f"Exceeds critical threshold (20%)."
                    )
                else:
                    warnings_list.append(
                        f"{key}: {nan_rate:.1%} NaN ({n_nan}/{n_total}). "
                        f"Exceeds warning threshold ({max_nan_rate:.0%})."
                    )

            # Check for long consecutive NaN gaps
            if n_nan > 0:
                isnan = np.isnan(vals)
                # Find runs of consecutive NaN
                gap_starts = np.where(np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == 1)[0]
                gap_ends = np.where(np.diff(np.concatenate([[0], isnan.astype(int), [0]])) == -1)[0]
                max_gap_samples = max(
                    [(gap_ends[i] - gap_starts[i]) for i in range(len(gap_starts))],
                    default=0,
                )
                if max_gap_samples > max_consecutive_samples:
                    max_gap_sec = max_gap_samples * dt
                    key = f"{mod_name}/{col}"
                    gap_features.append(key)
                    details.append({
                        "type": "long_nan_gap",
                        "modality": mod_name,
                        "feature": col,
                        "max_gap_sec": round(max_gap_sec, 1),
                        "max_gap_samples": int(max_gap_samples),
                    })
                    warnings_list.append(
                        f"{key}: longest NaN gap = {max_gap_sec:.1f}s "
                        f"({max_gap_samples} samples). Sensor dropout suspected."
                    )

    if total_features == 0:
        return StageResult(
            stage="nan_integrity",
            verdict=StageVerdict.WARN,
            message="No numeric features found in dataset.",
        )

    if failures_list:
        verdict = StageVerdict.FAIL
        message = f"{len(failures_list)} feature(s) exceed critical NaN threshold."
    elif warnings_list:
        verdict = StageVerdict.WARN
        message = f"{len(warnings_list)} NaN issue(s) across {total_features} features."
    else:
        verdict = StageVerdict.PASS
        message = f"NaN rates acceptable across {total_features} features."

    return StageResult(
        stage="nan_integrity",
        verdict=verdict,
        details=details,
        message=message,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Sampling Uniformity
# ---------------------------------------------------------------------------

def _check_sampling_uniformity(dataset: Any, config: Dict[str, Any]) -> StageResult:
    """Check for irregular sampling intervals and insufficient sample count.

    Irregular sampling (e.g., variable-rate PPG) violates the uniform-
    sampling assumption underlying FFT-based methods (CCF, IAAFT).
    Insufficient sample count makes statistical inference unreliable.
    """
    details: List[Dict[str, Any]] = []
    failures_list: List[str] = []

    max_isi_cv = config.get("max_isi_cv", 0.10)
    min_effective_samples = config.get("min_effective_samples", 50)

    if not hasattr(dataset, "modalities") or not dataset.modalities:
        return StageResult(
            stage="sampling_uniformity",
            verdict=StageVerdict.FAIL,
            message="No modalities to check.",
        )

    # Check overall sample count
    first_mod = next(iter(dataset.modalities.values()))
    n_samples = len(first_mod)

    if n_samples < min_effective_samples:
        failures_list.append(
            f"Only {n_samples} samples (minimum: {min_effective_samples}). "
            f"Statistical inference will be unreliable."
        )

    # Check inter-sample interval uniformity for each modality's time axis
    for name, df in dataset.modalities.items():
        if "time" not in df.columns:
            continue
        t = df["time"].values.astype(float)
        if len(t) < 3:
            continue

        isi = np.diff(t)
        mean_isi = float(np.mean(isi))
        std_isi = float(np.std(isi, ddof=1))

        if mean_isi <= 0:
            details.append({
                "type": "zero_mean_isi",
                "modality": name,
            })
            failures_list.append(f"'{name}' has zero/negative mean inter-sample interval.")
            continue

        cv = std_isi / mean_isi

        if cv > max_isi_cv:
            details.append({
                "type": "irregular_sampling",
                "modality": name,
                "isi_cv": round(cv, 4),
                "mean_isi_sec": round(mean_isi, 4),
                "std_isi_sec": round(std_isi, 4),
            })
            if cv > 0.30:
                failures_list.append(
                    f"'{name}' has highly irregular sampling (ISI CV = {cv:.2f}). "
                    f"FFT-based methods (CCF, IAAFT) assume uniform sampling."
                )
            else:
                # WARN level — mild irregularity
                pass

    if failures_list:
        return StageResult(
            stage="sampling_uniformity",
            verdict=StageVerdict.FAIL,
            details=details,
            message=f"{len(failures_list)} sampling uniformity issue(s).",
        )

    return StageResult(
        stage="sampling_uniformity",
        verdict=StageVerdict.PASS,
        message=f"Sampling is uniform ({n_samples} samples, "
                f"effective rate ~{1.0 / np.mean(np.diff(first_mod['time'].values)):.2f} Hz).",
    )


def format_qc_report(report: DataQualityReport) -> str:
    """Return a concise user-facing QC message.

    Intended for CLI/user-facing logs.  The structured version remains
    available via ``DataQualityReport.to_dict()``.
    """
    verdict = report.overall_verdict
    if verdict == StageVerdict.PASS:
        header = f"QC: PASS — {report.dyad_id}; analysis can proceed."
    elif verdict == StageVerdict.WARN:
        header = f"QC: WARN — {report.dyad_id}; analysis can proceed, but review warnings."
    else:
        header = f"QC: FAIL — {report.dyad_id}; analysis should stop before WCC computation."

    lines = [header]
    for stage in report.stages:
        if stage.verdict == StageVerdict.PASS:
            continue
        lines.append(f"- {stage.stage}: {stage.message}")
        for detail in stage.details[:5]:
            dtype = detail.get("type", "issue")
            mod = detail.get("modality")
            feat = detail.get("feature")
            where = "/".join(str(x) for x in (mod, feat) if x is not None)
            suffix = f" ({where})" if where else ""
            lines.append(f"  • {dtype}{suffix}")
    actions = report.suggested_actions()
    if actions:
        lines.append("Suggested fixes:")
        for i, action in enumerate(actions, start=1):
            lines.append(f"  {i}. {action}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_quality_check(
    dataset: Any,
    config: Optional[Dict[str, Any]] = None,
    raise_on_fail: bool = False,
) -> DataQualityReport:
    """Run the 3-stage data quality check pipeline.

    Parameters
    ----------
    dataset : SynchronyDataset
        The dataset to check.  Should already be aligned.
    config : dict, optional
        Override default thresholds.  Keys: see DEFAULT_CONFIG.
    raise_on_fail : bool
        If True, raise DataQualityError when overall verdict is FAIL.
        Default False (caller inspects the report).

    Returns
    -------
    DataQualityReport
        Structured report with per-stage verdicts and details.

    Raises
    ------
    DataQualityError
        If ``raise_on_fail=True`` and any stage returns FAIL.

    Examples
    --------
    >>> report = run_quality_check(dataset)
    >>> if not report.passed:
    ...     print(report.summary())
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    dyad_id = getattr(dataset, "dyad_id", "unknown")

    stages: List[StageResult] = []

    # Stage 1
    st1 = _check_temporal_alignment(dataset, cfg)
    stages.append(st1)

    # Stage 2
    st2 = _check_nan_integrity(dataset, cfg)
    stages.append(st2)

    # Stage 3
    st3 = _check_sampling_uniformity(dataset, cfg)
    stages.append(st3)

    # Aggregate
    all_warnings: List[str] = []
    all_failures: List[str] = []

    for st in stages:
        # Collect human-readable messages
        if st.verdict == StageVerdict.FAIL:
            all_failures.append(f"[{st.stage}] {st.message}")
        elif st.verdict == StageVerdict.WARN:
            all_warnings.append(f"[{st.stage}] {st.message}")
        # Collect per-detail warnings/failures
        for d in st.details:
            # Pass through detail-level messages
            pass

    report = DataQualityReport(
        dyad_id=dyad_id,
        stages=stages,
        n_warnings=len(all_warnings),
        n_failures=len(all_failures),
        warnings=all_warnings,
        failures=all_failures,
    )

    if raise_on_fail and not report.passed:
        raise DataQualityError(report.summary())

    return report

"""
External-facing v1 feature status table.

This module is the communication SSoT for manuscript/demo feature status.  It is
intentionally simpler than the implementation-level labels in
``feature_definitions.py``.  The purpose is to make Table 1 informative and
honest: what each descriptor is computed from, what incremental information it
adds beyond raw signals / mean synchrony, which inferential audit is appropriate,
and where interpretation is fragile.

Important boundary:
    This file does NOT implement feature math.  Feature computation remains in
    ``feature_definitions.py``.  Proposed descriptors can appear here before
    entering the mathematical SSoT, but they must be marked as proposed.
"""

from __future__ import annotations

from typing import Dict, List


FEATURE_STATUS_ROWS: List[Dict[str, object]] = [
    {
        "feature": "raw_signal_quality",
        "source_level": "raw/aligned signal",
        "computed_from": "time series before WCC",
        "dimension": "measurement validity",
        "incremental_information": "whether downstream synchrony estimates are interpretable at all",
        "order_sensitive": True,
        "paradigm": "all",
        "status": "required-qc",
        "enters_primary_fdr": False,
        "default_audit_or_test": "QC gates: alignment, missingness, sampling uniformity, clock-offset warning",
        "recommended_use": "report before any synchrony result",
        "main_risk": "bad alignment or preprocessing can create false lag/synchrony before feature extraction begins",
        "implemented_in_ssot": False,
    },
    {
        "feature": "wcc_trace",
        "source_level": "synchrony trace",
        "computed_from": "aligned person_a/person_b signals",
        "dimension": "measurement substrate",
        "incremental_information": "time-local association rather than one full-session scalar",
        "order_sensitive": True,
        "paradigm": "all, with window-size justification",
        "status": "measurement-substrate",
        "enters_primary_fdr": False,
        "default_audit_or_test": "window-size sensitivity; signal-level null for derived descriptors",
        "recommended_use": "inspect and export; all WCC-derived descriptors depend on this trace",
        "main_risk": "window overlap, lag choice, nonstationarity, and shared task timing shape all downstream features",
        "implemented_in_ssot": False,
    },
    {
        "feature": "mean_synchrony",
        "source_level": "WCC distribution",
        "computed_from": "all finite WCC values",
        "dimension": "intensity",
        "incremental_information": "reference average coupling level",
        "order_sensitive": False,
        "paradigm": "all",
        "status": "reference",
        "enters_primary_fdr": False,
        "default_audit_or_test": "signal-level IAAFT as synchrony-existence audit; design controls for coupling interpretation",
        "recommended_use": "baseline comparator for average synchrony level",
        "main_risk": "collapses all temporal structure into one scalar and is vulnerable to shared-stimulus elevation",
        "implemented_in_ssot": True,
    },
    {
        "feature": "peak_amplitude",
        "source_level": "WCC distribution",
        "computed_from": "smoothed WCC trace maximum",
        "dimension": "intensity",
        "incremental_information": "strongest momentary synchrony beyond the mean",
        "order_sensitive": False,
        "paradigm": "all",
        "status": "primary-workhorse",
        "enters_primary_fdr": True,
        "default_audit_or_test": "signal-level IAAFT; pseudo-pair/time-shift before dyad-specific interpretation",
        "recommended_use": "primary synchrony-existence descriptor in v1",
        "main_risk": "sensitive to window size, smoothing, and isolated high-correlation periods",
        "implemented_in_ssot": True,
    },
    {
        "feature": "fraction_above_threshold",
        "source_level": "threshold-state occupancy",
        "computed_from": "proportion of finite WCC values above threshold",
        "dimension": "occupancy",
        "incremental_information": "how much of the interaction is spent in an above-threshold state, regardless of episode ordering",
        "order_sensitive": False,
        "paradigm": "all, with threshold justification",
        "status": "exploratory-secondary",
        "enters_primary_fdr": False,
        "default_audit_or_test": "descriptive plus design controls; not in primary FDR in v1",
        "recommended_use": "interpretable coverage of above-threshold synchrony; report alongside mean/peak and threshold metadata",
        "main_risk": "threshold dependence and redundancy with mean/peak synchrony",
        "implemented_in_ssot": True,
    },
    {
        "feature": "dwell_time",
        "source_level": "threshold-state sequence",
        "computed_from": "run lengths of consecutive above-threshold WCC states",
        "dimension": "structure / stability",
        "incremental_information": "whether synchrony is sustained in long episodes rather than brief bursts",
        "order_sensitive": True,
        "paradigm": "continuous or event blocks, with threshold justification",
        "status": "primary-structure",
        "enters_primary_fdr": True,
        "default_audit_or_test": "WCC-level IAAFT null (Family L1) plus design controls; enters the primary group-condition FDR family",
        "recommended_use": "confirmatory structure descriptor; describe sustained synchrony episodes after existence/design audits",
        "main_risk": "threshold dependence; undefined or unstable when few episodes exist",
        "implemented_in_ssot": True,
    },
    {
        "feature": "switching_rate",
        "source_level": "threshold-state sequence",
        "computed_from": "transitions between below/above-threshold WCC states",
        "dimension": "structure / flexibility",
        "incremental_information": "how often dyads enter and exit synchrony states, beyond total occupancy",
        "order_sensitive": True,
        "paradigm": "continuous or long event blocks",
        "status": "primary-structure",
        "enters_primary_fdr": True,
        "default_audit_or_test": "WCC-level IAAFT null (Family L1) plus design controls; enters the primary group-condition FDR family",
        "recommended_use": "confirmatory structure descriptor; describe intermittent versus stable coordination patterns",
        "main_risk": "sensitive to jitter, smoothing, hysteresis, and WCC overlap",
        "implemented_in_ssot": True,
    },
    {
        "feature": "bimodality_coefficient",
        "source_level": "WCC distribution shape",
        "computed_from": "skewness and kurtosis of finite WCC values",
        "dimension": "distribution-shape / state separability",
        "incremental_information": "whether WCC values resemble two high/low states rather than one broad distribution",
        "order_sensitive": False,
        "paradigm": "all, if enough WCC samples",
        "status": "exploratory",
        "enters_primary_fdr": False,
        "default_audit_or_test": "signal-level IAAFT if tested; currently report cautiously",
        "recommended_use": "diagnostic for two-state high/low synchrony distributions",
        "main_risk": "sample-size sensitivity and weak direct psychological construct validity",
        "implemented_in_ssot": True,
    },
    {
        "feature": "synchrony_entropy",
        "source_level": "WCC distribution shape",
        "computed_from": "histogram entropy of finite WCC values",
        "dimension": "distribution diversity",
        "incremental_information": "diversity of WCC values visited, not their temporal order",
        "order_sensitive": False,
        "paradigm": "all, if enough WCC samples",
        "status": "exploratory",
        "enters_primary_fdr": False,
        "default_audit_or_test": "descriptive; check collinearity with mean/variance",
        "recommended_use": "diagnostic for spread/diversity of synchrony states",
        "main_risk": "often collinear with mean/variance and difficult to interpret psychologically",
        "implemented_in_ssot": True,
    },
    {
        "feature": "onset_latency",
        "source_level": "event-anchored threshold sequence",
        "computed_from": "first sustained threshold crossing after event/condition onset",
        "dimension": "event timing",
        "incremental_information": "when synchrony first emerges relative to a meaningful external anchor",
        "order_sensitive": True,
        "paradigm": "event only",
        "status": "exploratory-event-only",
        "enters_primary_fdr": False,
        "default_audit_or_test": "event-mode sensitivity; peak-timing null under development (block bootstrap), deferred to v2",
        "recommended_use": "time to first sustained synchrony after a meaningful event onset",
        "main_risk": "requires a true baseline and event anchor; not meaningful in unanchored interaction",
        "implemented_in_ssot": True,
    },
    {
        "feature": "rise_time",
        "source_level": "event morphology",
        "computed_from": "25–75% rise segment before dominant WCC peak",
        "dimension": "event morphology / build-up",
        "incremental_information": "shape of build-up toward a dominant synchrony peak",
        "order_sensitive": True,
        "paradigm": "event / single-peak morphology only",
        "status": "exploratory-event-only",
        "enters_primary_fdr": False,
        "default_audit_or_test": "event-mode sensitivity; peak-timing null under development (block bootstrap), deferred to v2",
        "recommended_use": "build-up speed for SCR-like single-peak synchrony responses",
        "main_risk": "imports a single-peak SCR assumption into non-SCR synchrony traces",
        "implemented_in_ssot": True,
    },
    {
        "feature": "recovery_time",
        "source_level": "event morphology",
        "computed_from": "time from dominant WCC peak to half-recovery level",
        "dimension": "event morphology / decay",
        "incremental_information": "whether a dominant synchrony peak dissolves quickly or slowly",
        "order_sensitive": True,
        "paradigm": "event / single-peak morphology only",
        "status": "exploratory-event-only",
        "enters_primary_fdr": False,
        "default_audit_or_test": "event-mode sensitivity; peak-timing null under development (block bootstrap), deferred to v2",
        "recommended_use": "decay time after a dominant synchrony peak",
        "main_risk": "on oscillatory traces may measure a half-cycle rather than recovery",
        "implemented_in_ssot": True,
    },
    {
        "feature": "first_peak_time",
        "source_level": "morphology-agnostic peak timing",
        "computed_from": "first prominent local WCC peak above threshold",
        "dimension": "timing",
        "incremental_information": "first prominent synchrony burst without requiring baseline→rise→recovery morphology",
        "order_sensitive": True,
        "paradigm": "event or long continuous, with peak-definition caveat",
        "status": "exploratory-secondary",
        "enters_primary_fdr": False,
        "default_audit_or_test": "descriptive plus design controls; not in primary FDR in v1; report definedness rate",
        "recommended_use": "morphology-agnostic timing descriptor; carries information beyond mean/peak (artifact audit max|r|=0.24) but defined in ~0.39-1.00 of traces depending on dataset",
        "main_risk": "depends on prominence threshold and may be undefined in sustained or short traces",
        "implemented_in_ssot": True,
    },
    {
        "feature": "inter_peak_cv",
        "source_level": "morphology-agnostic peak sequence",
        "computed_from": "coefficient of variation of inter-peak intervals",
        "dimension": "rhythm / metastability",
        "incremental_information": "regular versus irregular recurrence of synchrony bursts",
        "order_sensitive": True,
        "paradigm": "long continuous traces with >= 3 prominent peaks",
        "status": "exploratory-secondary",
        "enters_primary_fdr": False,
        "default_audit_or_test": "descriptive plus design controls; not in primary FDR in v1; report definedness rate",
        "recommended_use": "descriptor of rhythmic/metastable synchrony recurrence; carries information beyond mean/peak (artifact audit max|r|=0.42) but defined in only ~0.05-1.00 of traces depending on trace length",
        "main_risk": "undefined with too few peaks; defined in only 5% of short Gordon traces",
        "implemented_in_ssot": True,
    },
]


def feature_status_table(as_dataframe: bool = True):
    """Return the external-facing v1 feature status table.

    Parameters
    ----------
    as_dataframe : bool
        If True and pandas is available, return a ``pandas.DataFrame``.
        Otherwise return a list of dictionaries.
    """
    rows = [dict(row) for row in FEATURE_STATUS_ROWS]
    if not as_dataframe:
        return rows
    try:
        import pandas as pd
    except Exception:
        return rows
    return pd.DataFrame(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def feature_status_latex(
    path: str | None = None,
    *,
    caption: str = "SyncPipe v1 feature-status table.",
    label: str = "tab:multisync_feature_status",
) -> str:
    """Return a manuscript-ready LaTeX longtable for Table 1.

    The LaTeX table is intentionally compressed relative to the full CSV:
    it focuses on the columns most useful for a Methods paper.
    """
    columns = [
        ("feature", "Feature"),
        ("source_level", "Source"),
        ("dimension", "Dimension"),
        ("incremental_information", "Incremental information"),
        ("paradigm", "Paradigm"),
        ("status", "Status"),
        ("default_audit_or_test", "Default audit/test"),
        ("main_risk", "Main risk"),
    ]
    lines = [
        r"\begin{longtable}{p{0.12\textwidth}p{0.12\textwidth}p{0.11\textwidth}p{0.17\textwidth}p{0.12\textwidth}p{0.10\textwidth}p{0.13\textwidth}p{0.13\textwidth}}",
        f"\\caption{{{_latex_escape(caption)}}}\\label{{{label}}}\\\\",
        r"\toprule",
        " & ".join(_latex_escape(header) for _, header in columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(_latex_escape(header) for _, header in columns) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in FEATURE_STATUS_ROWS:
        lines.append(" & ".join(_latex_escape(row[key]) for key, _ in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    latex = "\n".join(lines) + "\n"
    if path is not None:
        from pathlib import Path
        Path(path).write_text(latex, encoding="utf-8")
    return latex


__all__ = ["FEATURE_STATUS_ROWS", "feature_status_table", "feature_status_latex"]

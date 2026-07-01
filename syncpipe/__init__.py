"""SyncPipe — preferred public namespace.

Use ``import syncpipe as sp`` for the clean public API.
``import multisync`` is the legacy alias and remains available.
"""
from multisync import (  # noqa: F401  — re-export v1 public API
    # Core
    Dyad,
    DynamicAnalyzer,
    AnalysisResults,
    SynchronyDataset,
    ContextLabel,
    # Pipelines
    ComputationPipeline,
    BatchComputationPipeline,
    quick_compute,
    batch_compute,
    InferencePipeline,
    # Feature governance
    FDR_FEATURES,
    REFERENCE_FEATURE,
    ONSET_THRESHOLD,
    feature_status_table,
    feature_status_latex,
    explain_feature,
    # QC and audit
    run_quality_check,
    synchrony_existence_audit,
    design_control_audit,
    # Session thresholds
    compute_session_pooled_threshold,
    compute_condition_pooled_thresholds,
)
from multisync.__about__ import __version__  # noqa: F401

"""
multisync — public v1 API for SyncPipe.

The top-level namespace is intentionally small.  Import advanced, experimental,
or dataset-specific utilities from their submodules (for example
``multisync.synthetic`` or ``multisync.morphology``) rather than treating them as
stable v1 public API.
"""

from .__about__ import __version__

# Core user objects
from .core import AnalysisResults, Dyad, DynamicAnalyzer
from .dataset import ContextLabel, SynchronyDataset

# Computation and inference pipelines
from .computation_pipeline import (
    BatchComputationPipeline,
    ComputationPipeline,
    batch_compute,
    quick_compute,
)
from .inference_pipeline import InferencePipeline

# Feature/status governance
from .feature_definitions import FDR_FEATURES, ONSET_THRESHOLD, REFERENCE_FEATURE
from .feature_status import FEATURE_STATUS_ROWS, feature_status_latex, feature_status_table
from .feature_pipeline import explain_feature

# Quality-control and audit layer
from .qc import (
    DataQualityError,
    DataQualityReport,
    StageResult,
    StageVerdict,
    format_qc_report,
    run_quality_check,
)
from .design_controls import (
    DEFAULT_AUDIT_FEATURES,
    design_control_audit,
    extract_pair_features,
    synchrony_existence_audit,
)
from .session_threshold import (
    compute_condition_pooled_thresholds,
    compute_session_pooled_threshold,
)

__all__ = [
    "__version__",
    # Core user objects
    "Dyad",
    "DynamicAnalyzer",
    "AnalysisResults",
    "SynchronyDataset",
    "ContextLabel",
    # Computation and inference
    "ComputationPipeline",
    "BatchComputationPipeline",
    "quick_compute",
    "batch_compute",
    "InferencePipeline",
    # Feature/status governance
    "FDR_FEATURES",
    "REFERENCE_FEATURE",
    "ONSET_THRESHOLD",
    "FEATURE_STATUS_ROWS",
    "feature_status_table",
    "feature_status_latex",
    "explain_feature",
    # QC and audits
    "run_quality_check",
    "format_qc_report",
    "DataQualityReport",
    "StageResult",
    "StageVerdict",
    "DataQualityError",
    "DEFAULT_AUDIT_FEATURES",
    "design_control_audit",
    "extract_pair_features",
    "synchrony_existence_audit",
    "compute_session_pooled_threshold",
    "compute_condition_pooled_thresholds",
]

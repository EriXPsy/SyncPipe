"""
syncpipe — public v1 API for SyncPipe.

The top-level namespace is intentionally small.  Import advanced, experimental,
or dataset-specific utilities from their submodules (for example
``syncpipe.synthetic`` or ``syncpipe.morphology``) rather than treating them as
stable v1 public API.
"""

from .__about__ import (
    __version__,
    PACKAGE_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
)

# Core user objects
from .core import AnalysisResults, Dyad, DynamicAnalyzer
from .dataset import ContextLabel, SynchronyDataset

# Computation and inference pipelines
from .computation_pipeline import (
    BatchComputationPipeline,
    ComputationPipeline,
    PairResult,
    batch_compute,
    compute_pair_pipeline,
    quick_compute,
)
from .inference_pipeline import InferencePipeline
from .pipeline_bridge import InferenceInputs, records_to_inference_inputs
from .canonical_runner import (
    CanonicalResult,
    DEFAULT_CONFIG,
    ManifestRecord,
    SyncPipeConfig,
    parse_config,
    parse_manifest,
    run_canonical,
)

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
    compute_session_pooled_thresholds_by_modality,
)

__all__ = [
    "__version__",
    "PACKAGE_VERSION",
    "ANALYSIS_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    # Core user objects
    "Dyad",
    "DynamicAnalyzer",
    "AnalysisResults",
    "SynchronyDataset",
    "ContextLabel",
    # Computation and inference
    "ComputationPipeline",
    "BatchComputationPipeline",
    "compute_pair_pipeline",
    "PairResult",
    "quick_compute",
    "batch_compute",
    "InferencePipeline",
    # Data-layer -> pipeline bridge (reviewer entry point)
    "records_to_inference_inputs",
    "InferenceInputs",
    # Canonical scientific runner (Gate 1 single entry point)
    "run_canonical",
    "parse_manifest",
    "parse_config",
    "ManifestRecord",
    "SyncPipeConfig",
    "CanonicalResult",
    "DEFAULT_CONFIG",
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
    "compute_session_pooled_thresholds_by_modality",
    "compute_condition_pooled_thresholds",
]

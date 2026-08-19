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
    EVIDENCE_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
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
from .preparation import (
    PreparedCohort, PreparedObservation, PreparationExclusion, SignalGeometry,
)
from .contracts import AnalysisSpec, EndpointSpec, ModalitySpec, NullSpec
from .evidence import ClaimDecision, EvidenceChain, EvidenceStageResult, EvidenceStatus
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
from .external_validation import audit_external_bundle, create_external_validation_kit
from .migration import (
    detect_config_contract,
    detect_manifest_contract,
    migrate_config_v1_to_v2,
    migrate_manifest_v1_to_v2,
    migrate_v1_project,
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
    "MANIFEST_SCHEMA_VERSION",
    "PREPROCESSING_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
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
    "PreparedObservation",
    "PreparedCohort",
    "PreparationExclusion",
    "SignalGeometry",
    "EvidenceStatus",
    "EvidenceStageResult",
    "EvidenceChain",
    "ClaimDecision",
    # Canonical scientific runner (Gate 1 single entry point)
    "run_canonical",
    "parse_manifest",
    "parse_config",
    "ManifestRecord",
    "AnalysisSpec",
    "SyncPipeConfig",
    "EndpointSpec",
    "NullSpec",
    "ModalitySpec",
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
    "create_external_validation_kit",
    "audit_external_bundle",
    "detect_manifest_contract",
    "detect_config_contract",
    "migrate_manifest_v1_to_v2",
    "migrate_config_v1_to_v2",
    "migrate_v1_project",
    "compute_session_pooled_threshold",
    "compute_session_pooled_thresholds_by_modality",
    "compute_condition_pooled_thresholds",
]

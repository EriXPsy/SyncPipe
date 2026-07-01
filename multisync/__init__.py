"""
multisync — Episode-based dynamic feature extraction for interpersonal synchrony.

Three pipelines (use what you need):

    Pipeline 1 — Feature:  understand and select features
        import multisync as ms
        ms.list_features()
        ms.explain_feature("dwell_time")
        ms.recommend_features("intensity")

    Pipeline 2 — Computation:  load data, compute WCC, extract features
        pipe = ms.ComputationPipeline(hz=4.0, window_size=40)
        pipe.run(sig_a, sig_b)
        df = pipe.to_dataframe()

    Pipeline 3 — Inference:  L0 → L1 → L2 statistical validation
        pipe = ms.InferencePipeline(features_df, hz=4.0, wcc_window_sec=10.0)
        pipe.test_l2_condition("condition", "dyad_id")
        print(pipe.summarize())
"""

# High-level API
from .core import Dyad, DynamicAnalyzer, AnalysisResults

# Dataset container
from .dataset import SynchronyDataset, ContextLabel

# (Cross-modal temporal association analysis was moved to experimental/ in v1.)

# Dynamic features
from .dynamic_features import (
    sliding_window_wcc,
    extract_dynamic_features,
    extract_features_all_pairs,
    extract_features_segmented,
    DynamicFeatures,
)

# Simple v1 feature status table
from .feature_status import FEATURE_STATUS_ROWS, feature_status_latex, feature_status_table

# Feature definitions (SSoT) — two-axis classification
from .feature_definitions import (
    # Primary: functional tier
    FEATURE_TIER,
    FDR_FEATURES,
    REFERENCE_FEATURE,
    CORE_FEATURES,
    CONDITIONAL_FEATURES,
    # Secondary: informational tier
    INTENSITY_FEATURES, # Measure Synchrony Epoch's intensity
    STRUCTURE_FEATURES, # Measure Synchrony Epoch's structure
    TEMPORAL_FEATURES,  # Measure Synchrony Epoch's temporal pattern
    # Constants
    ONSET_THRESHOLD,
)

# Prediction
from .prediction import (
    rolling_origin_cv,
    cross_modal_prediction,
    check_group_consistency,
    lodo_cv,  # backward-compatible alias
    PredictionResult,
    FoldResult,
)

# Synthetic data
from .synthetic import generate_ground_truth_dyad, generate_multimodal_dyad

# Batch / Group-level analysis
from .batch import (
    BatchConfig,
    DyadResult,
    GroupComparisonReport,
    MetricTestResult,
    batch_analyze,
    group_comparison,
)

# Data import
from .importer import DataImporter

# Report generation
from .report import ReportGenerator

# Data Quality Check
from .qc import (
    run_quality_check,
    DataQualityReport,
    StageResult,
    StageVerdict,
    DataQualityError,
)

# Treur dyad simulator (ground-truth validation)
from .simulation.treur_dyad import (
    TreurDyadResult,
    TreurDyadSimulator,
    scenario_constant_high_sync,
    scenario_frequent_switching,
    scenario_leader_follower,
    scenario_gradual_emergence,
    scenario_isc_confound,
)
from .simulation.treur_dyad_v2 import scenario_emergent_sync

# ---- Three pipelines (v1.0) ----------------------------------------------
# Feature pipeline: understand and select features
from .feature_pipeline import (
    list_features,
    explain_feature,
    get_fdr_features,
    get_core_features,
    get_conditional_features,
    get_reference_feature,
    recommend_features,
    print_feature_table,
    FeatureInfo,
)

# Computation pipeline: load → compute WCC → extract features
from .computation_pipeline import (
    ComputationPipeline,
    BatchComputationPipeline,
    quick_compute,
    batch_compute,
)

# Session-level pooled surrogate threshold
from .session_threshold import (
    compute_session_pooled_threshold,
    compute_condition_pooled_thresholds,
)

# Design-level controls (pseudo-pair / time-shift / synchrony-existence audit)
from .design_controls import (
    DEFAULT_AUDIT_FEATURES,
    design_control_audit,
    extract_pair_features,
    synchrony_existence_audit,
)

# WCLR backend
from .wclr import (
    windowed_cross_lagged_regression,
    wclr_coupling_trace,
)

# Morphology analysis
from .morphology import (
    MorphologyAnalyzer,
    trace_shape_cluster,
    episode_archetype_cluster,
    morphology_feature_table,
    collinearity_report,
    incremental_value,
    matched_mean_contrast,
)

# Inference pipeline: L0 → L1 → L2 statistical validation
from .inference_pipeline import (
    InferencePipeline,
)

# WCC trace export (for morphology clustering / inter-peak-CV analysis)
from .wcc_export import export_wcc_traces, wcc_traces_to_frame

# Feature collinearity / VIF diagnostics (for FDR family validation)
from .feature_vif_test import (
    feature_correlation,
    feature_vif,
    collinearity_report,
)


__all__ = [
    # High-level
    "Dyad",
    "DynamicAnalyzer",
    "AnalysisResults",
    # Dataset
    "SynchronyDataset",
    "ContextLabel",
    # Features
    "sliding_window_wcc",
    "extract_dynamic_features",
    "extract_features_all_pairs",
    "extract_features_segmented",
    "DynamicFeatures",
    # Feature definitions (SSoT) — two-axis classification
    "FEATURE_TIER",
    "FDR_FEATURES",
    "REFERENCE_FEATURE",
    "CORE_FEATURES",
    "CONDITIONAL_FEATURES",
    "INTENSITY_FEATURES",
    "STRUCTURE_FEATURES",
    "TEMPORAL_FEATURES",
    "FEATURE_STATUS_ROWS",
    "feature_status_table",
    "feature_status_latex",
    # Constants
    "ONSET_THRESHOLD",
    # Prediction
    "rolling_origin_cv",
    "cross_modal_prediction",
    "check_group_consistency",
    "lodo_cv",  # backward-compatible alias
    "PredictionResult",
    "FoldResult",
    # Synthetic
    "generate_ground_truth_dyad",
    "generate_multimodal_dyad",
    # Batch
    "BatchConfig",
    "DyadResult",
    "GroupComparisonReport",
    "MetricTestResult",
    "batch_analyze",
    "group_comparison",
    # Import
    "DataImporter",
    # Report
    "ReportGenerator",
    # QC
    "run_quality_check",
    "DataQualityReport",
    "StageResult",
    "StageVerdict",
    "DataQualityError",
    # Treur dyad simulator
    "TreurDyadResult",
    "TreurDyadSimulator",
    "scenario_constant_high_sync",
    "scenario_frequent_switching",
    "scenario_leader_follower",
    "scenario_gradual_emergence",
    "scenario_isc_confound",
    "scenario_emergent_sync",
    # Pipeline 1: Feature
    "list_features",
    "explain_feature",
    "get_fdr_features",
    "get_core_features",
    "get_conditional_features",
    "get_reference_feature",
    "recommend_features",
    "print_feature_table",
    "FeatureInfo",
    # Pipeline 2: Computation
    "ComputationPipeline",
    "BatchComputationPipeline",
    "quick_compute",
    "batch_compute",
    # Pipeline 3: Inference
    "InferencePipeline",
    # Session-level pooled threshold
    "compute_session_pooled_threshold",
    "compute_condition_pooled_thresholds",
    # Design controls
    "DEFAULT_AUDIT_FEATURES",
    "design_control_audit",
    "extract_pair_features",
    "synchrony_existence_audit",
    # WCLR backend
    "windowed_cross_lagged_regression",
    "wclr_coupling_trace",
    # Morphology analysis
    "MorphologyAnalyzer",
    "trace_shape_cluster",
    "episode_archetype_cluster",
    "morphology_feature_table",
    "collinearity_report",
    "incremental_value",
    "matched_mean_contrast",
    # WCC export
    "export_wcc_traces",
    "wcc_traces_to_frame",
    # Feature VIF diagnostics
    "feature_correlation",
    "feature_vif",
    "collinearity_report",
]

__version__ = "1.0.0"

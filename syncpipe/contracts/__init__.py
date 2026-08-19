"""Stable typed contracts for canonical SyncPipe analyses."""
from .analysis import (
    AnalysisSpec,
    ENDPOINT_SPECS,
    EndpointSpec,
    ModalitySpec,
    NullSpec,
    SIGNAL_LEVEL_IAAFT_PEAK,
    SyncPipeConfig,
    analysis_spec_from_mapping,
    resolve_endpoint_spec,
)

__all__ = [
    "AnalysisSpec", "SyncPipeConfig", "EndpointSpec", "NullSpec",
    "ModalitySpec", "ENDPOINT_SPECS", "SIGNAL_LEVEL_IAAFT_PEAK",
    "analysis_spec_from_mapping", "resolve_endpoint_spec",
]

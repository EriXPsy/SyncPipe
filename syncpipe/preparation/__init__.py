"""Immutable preparation objects shared across pipeline stages."""
from .observation import (
    PreparedCohort,
    PreparedObservation,
    Segment,
    SignalGeometry,
    contiguous_segments,
    resolve_signal_geometry,
)

__all__ = [
    "PreparedCohort", "PreparedObservation", "Segment", "SignalGeometry",
    "contiguous_segments", "resolve_signal_geometry",
]

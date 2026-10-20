"""Immutable preparation objects shared across pipeline stages."""
from .loading import LoaderRecord, load_manifest_record, iter_record_groups, load_record_group, validate_trial_alignment
from .observation import (
    PreparedCohort,
    PreparedObservation,
    PreparationExclusion,
    Segment,
    SignalGeometry,
    contiguous_segments,
    resolve_signal_geometry,
)

__all__ = [
    "LoaderRecord", "load_manifest_record",
    "PreparedCohort", "PreparedObservation", "PreparationExclusion",
    "Segment", "SignalGeometry",
    "contiguous_segments", "resolve_signal_geometry",
]

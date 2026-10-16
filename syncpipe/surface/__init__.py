"""Synchrony surfaces: the time x lag object every SyncPipe result is derived from.

The public surface of this package is deliberately small.  See :class:`SyncSurface` for the
conventions (window alignment, lag sign) and :func:`wcc_surface` for the native estimator.
"""
from __future__ import annotations

from .engine import wcc_surface
from .io import read_surface, write_surface
from .model import (
    LAG_SIGN,
    SURFACE_SCHEMA_VERSION,
    Z_CLIP,
    SurfaceSettings,
    SurfaceSummary,
    SyncSurface,
    r_to_z,
)

__all__ = [
    "LAG_SIGN",
    "SURFACE_SCHEMA_VERSION",
    "Z_CLIP",
    "SurfaceSettings",
    "SurfaceSummary",
    "SyncSurface",
    "r_to_z",
    "read_surface",
    "wcc_surface",
    "write_surface",
]

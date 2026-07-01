"""SyncPipe — preferred public namespace.

Use ``import syncpipe as sp`` for the clean public API.
``import multisync`` is the legacy alias and remains available.
"""
from multisync import (  # noqa: F401  — re-export v1 public API
    Dyad,
    DynamicAnalyzer,
    InferencePipeline,
    feature_status_table,
)
from multisync.__about__ import __version__  # noqa: F401

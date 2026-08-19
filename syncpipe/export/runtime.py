"""Strict JSON conversion and runtime provenance capture."""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..__about__ import __version__, CONFIG_SCHEMA_VERSION, ANALYSIS_SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]

def _json_safe(obj: Any) -> Any:
    """Recursively convert results to JSON-safe structures.

    Non-finite floats -> null; dataclasses -> dict; ndarray -> list. Mirrors the
    sanitizer in ``InferencePipeline.to_json`` so report files are strict-JSON.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    try:
        from dataclasses import is_dataclass, asdict

        if is_dataclass(obj) and not isinstance(obj, type):
            return _json_safe(asdict(obj))
    except Exception:
        pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    return obj


def _safe_float(x: Any) -> Optional[float]:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if xf != xf or xf in (float("inf"), float("-inf")):
        return None
    return xf


# ──────────────────────────────────────────────────────────────────────────
# Environment + claimability helpers
# ──────────────────────────────────────────────────────────────────────────
def _environment(seed: int) -> Dict[str, Any]:
    git_hash = "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=10,
        )
        if out.returncode == 0:
            git_hash = out.stdout.strip()
    except Exception:
        pass
    try:
        import pandas as _pandas
        import scipy as _scipy
        import sklearn as _sklearn
        dependency_versions = {
            "pandas": _pandas.__version__,
            "scipy": _scipy.__version__,
            "scikit_learn": _sklearn.__version__,
        }
    except Exception:
        dependency_versions = {}
    return {
        "python_version": platform.python_version(),
        "syncpipe_version": __version__,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "numpy_version": np.__version__,
        "dependency_versions": dependency_versions,
        "platform": platform.platform(),
        "git_hash": git_hash,
        "seed": seed,
    }



json_safe = _json_safe
safe_float = _safe_float
capture_environment = _environment

__all__ = ["json_safe", "safe_float", "capture_environment"]

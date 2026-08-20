"""Load canonical manifest records into bridge-compatible records."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from ..contracts.manifest import ManifestRecord

@dataclass
class LoaderRecord:
    dyad_label: str
    modality: str
    condition: str
    person_a: Any
    person_b: Any
    target_hz: float
    discontinuity_mask: Optional[np.ndarray] = None
    incomplete: bool = False

def _load_mask(path: str) -> np.ndarray:
    """Load a discontinuity mask CSV (single 0/1 or boolean column)."""
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"mask file {path} has no numeric column.")
    values = num.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"mask file {path} must contain only finite 0/1 values")
    return values.astype(bool)


def _validate_aligned_signal_frames(a_df: pd.DataFrame, b_df: pd.DataFrame, *, hz: float, key: str) -> None:
    """Validate the time axes before the bridge discards them.

    Guards against manufacturing spurious synchrony when person A/B are not
    sampled on the same time grid (Gate 1 residual, vulnerability B).
    """
    if "time" not in a_df.columns or "time" not in b_df.columns:
        raise ValueError(f"{key}: both signal files must contain a 'time' column")
    ta = a_df["time"].to_numpy(dtype=float)
    tb = b_df["time"].to_numpy(dtype=float)
    if ta.ndim != 1 or tb.ndim != 1 or ta.size != tb.size:
        raise ValueError(f"{key}: person A/B time axes must be one-dimensional and equal length")
    if ta.size < 2 or not np.isfinite(ta).all() or not np.isfinite(tb).all():
        raise ValueError(f"{key}: time axes must contain at least two finite samples")
    if not (np.all(np.diff(ta) > 0) and np.all(np.diff(tb) > 0)):
        raise ValueError(f"{key}: time axes must be strictly increasing")
    if not np.allclose(ta, tb, rtol=0.0, atol=max(1e-6, 0.1 / hz)):
        raise ValueError(f"{key}: person A/B time axes are not aligned")
    dt = float(np.median(np.diff(ta)))
    if not np.isclose(dt, 1.0 / hz, rtol=0.01, atol=1e-9):
        raise ValueError(f"{key}: time step {dt:g} does not match manifest hz={hz:g}")


def _manifest_record_to_loader_record(
    rec: ManifestRecord, base_dir: Optional[Union[str, Path]] = None
) -> LoaderRecord:
    """Resolve paths and load signals/mask for one manifest row."""
    from ..io import load_csv  # local import avoids core/preparation cycle

    r = rec.resolve(base_dir)
    a_df = load_csv(r.person_a_path)
    b_df = load_csv(r.person_b_path)
    key = f"{r.dyad_id}__{r.modality}__{r.condition}"
    _validate_aligned_signal_frames(a_df, b_df, hz=r.hz, key=key)
    mask = _load_mask(r.mask_path) if r.mask_path else None
    if mask is not None and mask.size != len(a_df):
        raise ValueError(f"{key}: mask length {mask.size} does not match signal length {len(a_df)}")
    return LoaderRecord(
        dyad_label=r.dyad_id,
        modality=r.modality,
        condition=r.condition,
        person_a=a_df,
        person_b=b_df,
        target_hz=r.hz,
        discontinuity_mask=mask,
        incomplete=False,
    )


# ──────────────────────────────────────────────────────────────────────────
# JSON sanitizer (mirrors InferencePipeline.to_json)

load_manifest_record = _manifest_record_to_loader_record
__all__ = ["LoaderRecord", "load_manifest_record"]

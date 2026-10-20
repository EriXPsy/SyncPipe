"""Load canonical manifest records into bridge-compatible records."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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
    # Set only for series assembled from trial rows (ManifestRecord.trial_id,
    # 2026-10-16): per-trial ids/stimulus keys and lengths, in stack order.
    trial_ids: Optional[tuple] = None
    stimulus_keys: Optional[tuple] = None
    trial_lengths: Optional[tuple] = None

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


def iter_record_groups(records: Sequence[ManifestRecord]) -> List[List[ManifestRecord]]:
    """Group manifest rows into series, preserving first-seen order.

    Rows that all carry a ``trial_id`` and share
    ``(dyad_id, modality, condition)`` form one series. Everything else
    stays a single-row group, so a plain duplicate key still reaches the
    downstream duplicate check exactly as it did before trials existed.
    """
    by_key: Dict[tuple, List[ManifestRecord]] = {}
    for rec in records:
        by_key.setdefault((rec.dyad_id, rec.modality, rec.condition), []).append(rec)
    groups: List[List[ManifestRecord]] = []
    for rows in by_key.values():
        if len(rows) > 1 and all(r.trial_id is not None for r in rows):
            groups.append(rows)
        else:
            groups.extend([r] for r in rows)
    return groups


def _stack_trials(
    loaded: Sequence[LoaderRecord], group: Sequence[ManifestRecord]
) -> LoaderRecord:
    """Stack the trials of one series into a single seam-aware record.

    Trial signals are concatenated in manifest order; the discontinuity
    mask marks every trial boundary sample False so no downstream window
    can straddle a seam, and a trial's own mask (if any) is honoured
    inside its span. All trials must share the sampling rate; equal
    lengths are NOT required here but ARE required for the stranger
    control (checked in validate_trial_alignment).
    """
    first = loaded[0]
    if len({lr.target_hz for lr in loaded}) != 1:
        raise ValueError(
            f"trial stack {first.dyad_label}/{first.modality}: trials must share one hz"
        )
    # Trials of one series must present the SAME columns (a different schema
    # would make the concatenation ambiguous).
    ref_cols = list(first.person_a.columns)
    for lr in loaded[1:]:
        if list(lr.person_a.columns) != ref_cols or list(lr.person_b.columns) != ref_cols:
            raise ValueError(
                f"trial stack {first.dyad_label}/{first.modality}: trials must "
                "have the same columns"
            )
    a_parts, b_parts, mask_parts = [], [], []
    t_offset = 0.0
    dt = 1.0 / first.target_hz
    for lr, rec in zip(loaded, group):
        n = len(lr.person_a)
        if len(lr.person_b) != n:
            raise ValueError(
                f"trial {rec.trial_id} of {first.dyad_label}/{first.modality}: "
                "person_a and person_b lengths differ"
            )
        trial_mask = (
            np.asarray(lr.discontinuity_mask, dtype=bool)
            if lr.discontinuity_mask is not None
            else np.ones(n, dtype=bool)
        )
        if trial_mask.size != n:
            raise ValueError(
                f"trial {rec.trial_id}: mask length {trial_mask.size} != signal length {n}"
            )
        trial_mask = trial_mask.copy()
        if t_offset > 0:
            trial_mask[0] = False  # seam at each TRIAL BOUNDARY (not the first start)
        # Continuous clock: each trial's time column continues from the end
        # of the previous one (a stacked series is ONE series on one axis).
        a_df = lr.person_a.copy()
        b_df = lr.person_b.copy()
        if "time" in a_df.columns and "time" in b_df.columns:
            a_df["time"] = a_df["time"].to_numpy() + t_offset
            b_df["time"] = b_df["time"].to_numpy() + t_offset
        t_offset += n * dt
        a_parts.append(a_df)
        b_parts.append(b_df)
        mask_parts.append(trial_mask)
    import pandas as pd  # local: avoid top-level cycle in odd import orders
    return LoaderRecord(
        dyad_label=first.dyad_label,
        modality=first.modality,
        condition=first.condition,
        person_a=pd.concat(a_parts, ignore_index=True),
        person_b=pd.concat(b_parts, ignore_index=True),
        target_hz=first.target_hz,
        discontinuity_mask=np.concatenate(mask_parts),
        incomplete=any(lr.incomplete for lr in loaded),
        trial_ids=tuple(r.trial_id for r in group),
        stimulus_keys=(
            tuple(r.stimulus_key for r in group)
            if any(r.stimulus_key is not None for r in group) else None
        ),
        trial_lengths=tuple(len(lr.person_a) for lr in loaded),
    )


def load_record_group(
    group: Sequence[ManifestRecord], base_dir: Optional[Union[str, Path]] = None
) -> LoaderRecord:
    """Load one series: a single row as-is, or a trial stack (seam-aware)."""
    if len(group) == 1:
        return _manifest_record_to_loader_record(group[0], base_dir)
    loaded = [_manifest_record_to_loader_record(r, base_dir) for r in group]
    return _stack_trials(loaded, group)


def validate_trial_alignment(loader_records: Sequence[LoaderRecord]) -> None:
    """Stranger-control prerequisites for trial-stacked series.

    The stranger control pairs trials BY POSITION, so every dyad that
    stacks trials for a (modality, condition) must show equal trial
    counts and equal per-position lengths. Where stimulus keys are
    declared they must agree across dyads (parse_manifest already
    enforces same-order; this catches load-level divergence).
    """
    per_cell: Dict[tuple, Dict[str, tuple]] = {}
    per_cell_keys: Dict[tuple, Dict[str, tuple]] = {}
    unkeyed_cells: set = set()
    for lr in loader_records:
        if lr.trial_lengths is None:
            continue
        cell = (lr.modality, lr.condition)
        per_cell.setdefault(cell, {})[lr.dyad_label] = lr.trial_lengths
        if lr.stimulus_keys is not None:
            per_cell_keys.setdefault(cell, {})[lr.dyad_label] = lr.stimulus_keys
        else:
            # A series without stimulus keys makes no positional-pairing
            # claim: its cell is exempt from the length check entirely.
            unkeyed_cells.add(cell)
    # Length equality is only claimed (and checked) where stimulus keys are
    # declared: keys make trials positionally pairable, unkeyed series make
    # no such claim.
    for cell in sorted(set(per_cell_keys) & set(per_cell) - unkeyed_cells):
        per_dyad = per_cell[cell]
        reference_dyad, reference = next(iter(per_dyad.items()))
        for dyad, lengths in per_dyad.items():
            if lengths != reference:
                raise ValueError(
                    f"modality={cell[0]}, condition={cell[1]}: trial lengths differ "
                    f"between dyads ({dyad!r}: {lengths} vs {reference_dyad!r}: "
                    f"{reference}); keyed stimuli pair trials by position, so "
                    "per-position lengths must match"
                )
    for cell, per_dyad in per_cell_keys.items():
        refs = set(per_dyad.values())
        if len(refs) > 1:
            raise ValueError(
                f"stimulus keys for modality={cell[0]}, condition={cell[1]} diverge "
                f"across dyads: {sorted(refs)}"
            )


__all__ = [
    "LoaderRecord", "load_manifest_record", "iter_record_groups",
    "load_record_group", "validate_trial_alignment",
]

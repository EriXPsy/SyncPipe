"""Immutable prepared-observation geometry shared by all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

Segment = Tuple[int, int]


def _readonly_float(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    if arr.ndim != 1:
        raise ValueError("prepared signals must be one-dimensional")
    arr.setflags(write=False)
    return arr


def _readonly_bool(x) -> np.ndarray:
    arr = np.asarray(x, dtype=bool).copy()
    if arr.ndim != 1:
        raise ValueError("prepared masks must be one-dimensional")
    arr.setflags(write=False)
    return arr


def contiguous_segments(mask: np.ndarray) -> Tuple[Segment, ...]:
    """Return half-open contiguous True runs from one boolean mask."""
    valid = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


@dataclass(frozen=True)
class SignalGeometry:
    """One authoritative definition of finite/discontinuity geometry."""

    source_mask: np.ndarray
    joint_finite_mask: np.ndarray
    analysis_mask: np.ndarray
    segments: Tuple[Segment, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.source_mask), len(self.joint_finite_mask), len(self.analysis_mask)
        }
        if len(lengths) != 1:
            raise ValueError("prepared masks must have identical lengths")

    def segments_at_least(self, minimum: int) -> Tuple[Segment, ...]:
        if minimum < 1:
            raise ValueError("minimum segment length must be >= 1")
        return tuple((s, e) for s, e in self.segments if e - s >= minimum)

    def window_mask(self, window_size: int) -> np.ndarray:
        """Boolean WCC-resolution mask; True iff every raw sample is eligible."""
        if int(window_size) != window_size or window_size < 2:
            raise ValueError("window_size must be an integer >= 2")
        n_out = max(0, len(self.analysis_mask) - int(window_size) + 1)
        if n_out == 0:
            return np.zeros(0, dtype=bool)
        values = self.analysis_mask.astype(np.int64)
        csum = np.concatenate(([0], np.cumsum(values)))
        sums = csum[window_size:] - csum[:-window_size]
        out = sums == window_size
        out.setflags(write=False)
        return out


def resolve_signal_geometry(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    discontinuity_mask: Optional[np.ndarray] = None,
) -> SignalGeometry:
    """Resolve the shared mask and segments without deleting any sample."""
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise ValueError("prepared observation requires equal-length 1-D signals")
    if discontinuity_mask is None:
        source = np.ones(a.size, dtype=bool)
    else:
        source = np.asarray(discontinuity_mask, dtype=bool)
        if source.ndim != 1 or source.size != a.size:
            raise ValueError("discontinuity mask must match signal length")
    joint = np.isfinite(a) & np.isfinite(b)
    analysis = source & joint
    return SignalGeometry(
        source_mask=_readonly_bool(source),
        joint_finite_mask=_readonly_bool(joint),
        analysis_mask=_readonly_bool(analysis),
        segments=contiguous_segments(analysis),
    )


@dataclass(frozen=True)
class PreparedObservation:
    """Immutable data handed from preparation to computation and inference."""

    key: str
    dyad_id: str
    modality: str
    condition: str
    hz: float
    signal_a: np.ndarray
    signal_b: np.ndarray
    geometry: SignalGeometry

    @classmethod
    def from_signals(
        cls, *, key: str, dyad_id: str, modality: str, condition: str,
        hz: float, signal_a, signal_b,
        discontinuity_mask: Optional[np.ndarray] = None,
    ) -> "PreparedObservation":
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("prepared observation hz must be finite and positive")
        a = _readonly_float(signal_a)
        b = _readonly_float(signal_b)
        geometry = resolve_signal_geometry(a, b, discontinuity_mask)
        return cls(
            key=str(key), dyad_id=str(dyad_id), modality=str(modality),
            condition=str(condition), hz=float(hz), signal_a=a, signal_b=b,
            geometry=geometry,
        )

    def threshold_eligible(self, window_size: int) -> bool:
        """Whether any shared contiguous segment can contribute threshold WCC."""
        return bool(self.geometry.segments_at_least(window_size))

    def diagnostics(self, window_size: int) -> Dict[str, object]:
        window_mask = self.geometry.window_mask(window_size)
        return {
            "key": self.key,
            "n_samples": int(self.signal_a.size),
            "n_jointly_finite": int(self.geometry.joint_finite_mask.sum()),
            "n_analysis_eligible": int(self.geometry.analysis_mask.sum()),
            "eligible_fraction": float(self.geometry.analysis_mask.mean()) if self.signal_a.size else 0.0,
            "n_segments": len(self.geometry.segments),
            "segment_lengths": [end - start for start, end in self.geometry.segments],
            "n_wcc_eligible": int(window_mask.sum()),
            "threshold_eligible": self.threshold_eligible(window_size),
        }


@dataclass(frozen=True)
class PreparationExclusion:
    """Typed reason an observation did not enter the prepared cohort."""

    key: str
    dyad_id: str
    modality: str
    condition: str
    code: str
    stage: str
    detail: str
    claim_effect: str = "observation_excluded"

    @property
    def reason(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code

    def to_dict(self) -> Dict[str, str]:
        return {
            "key": self.key,
            "dyad_id": self.dyad_id,
            "modality": self.modality,
            "condition": self.condition,
            "code": self.code,
            "stage": self.stage,
            "detail": self.detail,
            "claim_effect": self.claim_effect,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PreparedCohort:
    observations: Tuple[PreparedObservation, ...]
    exclusions: Tuple[PreparationExclusion, ...] = ()

    def __post_init__(self) -> None:
        keys = [obs.key for obs in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("prepared cohort contains duplicate observation keys")

    def by_key(self) -> Dict[str, PreparedObservation]:
        return {obs.key: obs for obs in self.observations}

    def diagnostics(self, window_size: int) -> Dict[str, Dict[str, object]]:
        return {obs.key: obs.diagnostics(window_size) for obs in self.observations}

    def exclusion_records(self) -> Tuple[Dict[str, str], ...]:
        return tuple(exclusion.to_dict() for exclusion in self.exclusions)


__all__ = [
    "Segment", "SignalGeometry", "PreparedObservation", "PreparedCohort",
    "PreparationExclusion",
    "contiguous_segments", "resolve_signal_geometry",
]

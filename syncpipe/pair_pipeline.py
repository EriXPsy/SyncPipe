"""Canonical per-pair computation API."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd
from .dynamic_features import _apply_discontinuity_mask, extract_dynamic_features
from .coupling_pipeline import compute_coupling_trace
from .feature_definitions import DynamicFeatures

@dataclass
class PairResult:
    wcc: np.ndarray
    features: DynamicFeatures
    hz: float
    window_size: int
    label: Optional[str] = None
    discontinuity_mask: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def features_dict(self) -> Dict[str, float]:
        return self.features.to_dict()

    def to_dataframe(self) -> pd.DataFrame:
        row: Dict[str, object] = {}
        if self.label is not None:
            row["label"] = self.label
        row.update(self.metadata)
        row.update(self.features_dict)
        return pd.DataFrame([row])


def compute_pair_pipeline(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    hz: float,
    window_size: int,
    onset_threshold: Optional[float] = None,
    wcc: Optional[np.ndarray] = None,
    discontinuity_mask: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    window_type: str = "rect",
    normalize: bool = True,
    backend: str = "wcc",
    wclr_max_lag_samples: int = 2,
    wclr_metric: str = "beta",
    **metadata,
) -> PairResult:
    if wcc is None:
        wcc_arr = compute_coupling_trace(
            np.asarray(sig_a, dtype=float), np.asarray(sig_b, dtype=float),
            hz, window_size, backend, "cumsum", normalize, window_type,
            wclr_max_lag_samples, wclr_metric,
        )
        if discontinuity_mask is not None:
            wcc_arr = _apply_discontinuity_mask(
                wcc_arr, np.asarray(discontinuity_mask, dtype=bool), window_size
            )
        kwargs = {}
        if onset_threshold is not None:
            kwargs["onset_threshold"] = onset_threshold
        if discontinuity_mask is not None:
            kwargs["gap_policy"] = "segment"
        features = extract_dynamic_features(
            wcc_arr, hz=hz, wcc_window_sec=window_size / hz, **kwargs
        )
    else:
        wcc_arr = np.asarray(wcc, dtype=float)
        if discontinuity_mask is not None:
            wcc_arr = _apply_discontinuity_mask(
                wcc_arr, np.asarray(discontinuity_mask, dtype=bool), window_size
            )
        kwargs = {"onset_threshold": onset_threshold} if onset_threshold is not None else {}
        if discontinuity_mask is not None:
            kwargs["gap_policy"] = "segment"
        features = extract_dynamic_features(
            wcc_arr, hz=hz, wcc_window_sec=window_size / hz, **kwargs
        )
    return PairResult(wcc_arr, features, hz, window_size, label,
                      discontinuity_mask, dict(metadata))

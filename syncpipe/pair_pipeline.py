"""Canonical per-pair computation API."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd
from .dynamic_features import extract_dynamic_features
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


def compute_pair_pipeline(sig_a: np.ndarray, sig_b: np.ndarray, *, hz: float,
                          window_size: int, onset_threshold: Optional[float] = None,
                          wcc: Optional[np.ndarray] = None,
                          discontinuity_mask: Optional[np.ndarray] = None,
                          label: Optional[str] = None, window_type: str = "rect",
                          normalize: bool = True, backend: str = "wcc",
                          wclr_max_lag_samples: int = 2, wclr_metric: str = "beta",
                          **metadata) -> PairResult:
    if wcc is None:
        from .computation_pipeline import ComputationPipeline
        pipe = ComputationPipeline(
            hz=hz, window_size=window_size, onset_threshold=onset_threshold,
            backend=backend, wclr_max_lag_samples=wclr_max_lag_samples,
            wclr_metric=wclr_metric, window_type=window_type,
        )
        pipe.load_signals(sig_a, sig_b, label=label,
                          discontinuity_mask=discontinuity_mask, **metadata)
        pipe.compute_wcc()
        features, wcc_arr = pipe.extract_features(), pipe.wcc
    else:
        wcc_arr = np.asarray(wcc, dtype=float)
        kwargs = {"onset_threshold": onset_threshold} if onset_threshold is not None else {}
        features = extract_dynamic_features(
            wcc_arr, hz=hz, wcc_window_sec=window_size / hz, **kwargs
        )
    return PairResult(wcc_arr, features, hz, window_size, label,
                      discontinuity_mask, dict(metadata))

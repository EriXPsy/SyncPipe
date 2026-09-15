from __future__ import annotations

from typing import Any, Optional

import logging
import warnings
import numpy as np

from .feature_definitions import (
    DynamicFeatures,
    extract_features as _ssot_extract_features,
    ONSET_THRESHOLD,
)

logger = logging.getLogger(__name__)


def extract_dynamic_features(
    wcc: np.ndarray,
    hz: float = 1.0,
    onset_threshold: Optional[float] = None,
    onset_k: float = 2.0,
    max_nan_ratio: float = 0.2,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    aggregation: str = "mean",
    return_raw_profiles: bool = False,
    wcc_window_sec: float = 1.0,
    gap_policy: Optional[str] = None,
) -> Any:
    """Extract features from a WCC series via the SSoT."""
    if onset_k != 2.0:
        warnings.warn(
            "`onset_k` is deprecated and ignored: data-driven onset "
            "thresholds were removed in v1.0.0 (DECISION-04). "
            "Pass `onset_threshold` explicitly if you need a non-default "
            "threshold for sensitivity analysis.",
            DeprecationWarning, stacklevel=2,
        )
    if height is not None or distance is not None or prominence is not None:
        warnings.warn(
            "`height`, `distance`, and `prominence` are deprecated and "
            "ignored: feature math is no longer peak-detection-based "
            "in v1.0.0 (DECISION-08).  See "
            "syncpipe.feature_definitions for the locked-in definitions.",
            DeprecationWarning, stacklevel=2,
        )
    if aggregation != "mean":
        warnings.warn(
            "`aggregation` is deprecated and ignored: each WCC series now "
            "maps to a single DynamicFeatures via the SSoT; multi-peak "
            "aggregation no longer occurs at this layer.",
            DeprecationWarning, stacklevel=2,
        )
    wcc_arr = np.asarray(wcc, dtype=float)
    valid = np.isfinite(wcc_arr)
    nan_ratio = 1.0 if wcc_arr.size == 0 else 1.0 - float(valid.mean())
    if nan_ratio > max_nan_ratio or int(valid.sum()) < 5:
        logger.warning(
            "extract_dynamic_features: NaN ratio %.3f exceeds "
            "max_nan_ratio %.3f (or only %d valid points < 5) — returning "
            "all-NaN DynamicFeatures for this WCC series.",
            nan_ratio, max_nan_ratio, int(valid.sum()),
        )
        features = DynamicFeatures.from_dict({
            name: float("nan") for name in (
                "onset_latency", "rise_time", "peak_amplitude", "recovery_time",
                "dwell_time", "switching_rate", "mean_synchrony", "synchrony_entropy",
            )
        })
        features.nan_fraction = nan_ratio
        return (features, []) if return_raw_profiles else features
    threshold = ONSET_THRESHOLD if onset_threshold is None else float(onset_threshold)
    features = _ssot_extract_features(
        wcc_arr, hz=hz, wcc_window_sec=wcc_window_sec,
        threshold=threshold, gap_policy=gap_policy,
    )
    features.nan_fraction = nan_ratio
    return (features, []) if return_raw_profiles else features

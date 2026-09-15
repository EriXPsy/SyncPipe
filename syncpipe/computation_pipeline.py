"""Calculate a sliding-window correlation trace and its summary values.

Inputs must already be aligned, preprocessed one-dimensional signals. This
module does not preprocess raw physiological recordings.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .dynamic_features import (
    _apply_discontinuity_mask,
    extract_dynamic_features,
)
from .coupling_pipeline import compute_coupling_trace
from .feature_definitions import DynamicFeatures
from .session_threshold import (
    compute_session_pooled_threshold,
    compute_session_pooled_thresholds_by_modality,
)
from .feature_definitions import ONSET_THRESHOLD


class ComputationPipeline:
    """End-to-end computation: load → WCC → features → DataFrame.

    Parameters
    ----------
    hz : float
        Sampling rate of the input signals (Hz).
    window_size : int
        WCC sliding window size in samples.
    onset_threshold : float or None
        WCC threshold for episode onset detection.
        Default 0.5. Use None for surrogate-derived thresholds
        (computed externally via feature_pipeline or inference_pipeline).

    Examples
    --------
    >>> pipe = ComputationPipeline(hz=4.0, window_size=40)
    >>> pipe.load_signals(sig_a, sig_b)
    >>> pipe.compute_wcc()
    >>> pipe.extract_features()
    >>> df = pipe.to_dataframe()
    """

    def __init__(
        self,
        hz: float,
        window_size: int = 40,
        onset_threshold: Optional[float] = 0.5,
        backend: str = "wcc",
        wclr_max_lag_samples: int = 2,
        wclr_metric: str = "beta",
        window_type: str = "rect",
    ):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError(f"hz must be finite and positive, got {hz!r}")
        if int(window_size) != window_size or window_size < 2:
            raise ValueError(f"window_size must be an integer >= 2, got {window_size!r}")
        self.hz = float(hz)
        self.window_size = int(window_size)
        self.onset_threshold = onset_threshold
        self.backend = backend.lower()
        self.window_type = window_type.lower()
        self.wclr_max_lag_samples = wclr_max_lag_samples
        self.wclr_metric = wclr_metric
        if self.backend not in ("wcc", "wclr"):
            raise ValueError(f"backend must be 'wcc' or 'wclr', got {backend!r}")
        if self.wclr_metric not in ("beta", "r2"):
            raise ValueError(f"wclr_metric must be 'beta' or 'r2', got {wclr_metric!r}")

        self._sig_a: Optional[np.ndarray] = None
        self._sig_b: Optional[np.ndarray] = None
        self._wcc: Optional[np.ndarray] = None
        self._features: Optional[DynamicFeatures] = None
        self._metadata: Dict[str, object] = {}
        # Per-sample segment-boundary mask (True = internal to a segment).
        # When set, cross-boundary coupling windows are NaN'd so features
        # and the surrogate null skip them.  Signal-resolution (same length
        # as the loaded signals); None means "no boundaries to gate".
        self._discontinuity_mask: Optional[np.ndarray] = None

    # ---- data loading ----------------------------------------------------

    def load_signals(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
        label: Optional[str] = None,
        discontinuity_mask: Optional[np.ndarray] = None,
        **metadata,
    ):
        """Load two pre-processed 1-D signal arrays.

        Parameters
        ----------
        sig_a, sig_b : np.ndarray
            Two aligned time series (e.g. Person A and Person B EDA).
        label : str or None
            Optional label (e.g. condition name).
        discontinuity_mask : np.ndarray of bool or None
            Per-sample validity flag marking segment boundaries / seams
            (True = internal to a segment, False = at a discontinuity).
            When provided, coupling windows straddling a boundary are set
            NaN.  Must be at the same sampling resolution as ``sig_a`` /
            ``sig_b`` (signal-level).
        **metadata
            Arbitrary key-value metadata stored with the results.
        """
        self._sig_a = np.asarray(sig_a, dtype=float)
        self._sig_b = np.asarray(sig_b, dtype=float)
        if self._sig_a.ndim != 1 or self._sig_b.ndim != 1:
            raise ValueError(
                f"sig_a and sig_b must be 1-D arrays; got "
                f"{self._sig_a.shape} and {self._sig_b.shape}."
            )
        if self._sig_a.size != self._sig_b.size:
            raise ValueError(
                f"sig_a and sig_b must have equal length; got "
                f"{self._sig_a.size} and {self._sig_b.size}."
            )
        if np.isinf(self._sig_a).any() or np.isinf(self._sig_b).any():
            raise ValueError(
                "sig_a and sig_b must not contain +/-Inf. Use finite values or "
                "NaN for explicitly governed missing samples."
            )
        if discontinuity_mask is not None:
            dm = np.asarray(discontinuity_mask, dtype=bool)
            if dm.ndim != 1 or dm.size != self._sig_a.size:
                raise ValueError(
                    "discontinuity_mask must be a 1-D boolean array with "
                    f"length {self._sig_a.size}, got shape {dm.shape}."
                )
            discontinuity_mask = dm
        self._discontinuity_mask = discontinuity_mask
        self._metadata = {"label": label, **metadata}
        self._wcc = None
        self._features = None

    # ---- WCC computation ------------------------------------------------

    def compute_wcc(
        self,
        method: str = "cumsum",
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute the coupling time series (WCC or WCLR).

        Parameters
        ----------
        method : str
            "cumsum" (default, O(n)) or "stride" (legacy). Only used when
            backend is "wcc".
        normalize : bool
            If True, apply min-max normalization before WCC. For WCLR backend,
            the resulting standardized beta trace is min-max normalized to
            [-1, 1] so the same threshold machinery can be used.

        Returns
        -------
        wcc : np.ndarray
            Coupling time series (length = n_samples - window_size + 1).
        """
        if self._sig_a is None or self._sig_b is None:
            raise ValueError("Call load_signals() first.")

        self._wcc = compute_coupling_trace(
            self._sig_a, self._sig_b, self.hz, self.window_size,
            self.backend, method, normalize, self.window_type,
            self.wclr_max_lag_samples, self.wclr_metric,
        )
        # Gate out coupling windows that straddle a segment-boundary seam.
        # NaN windows are skipped by feature extraction (isfinite) and by the
        # surrogate nulls, keeping observed features and null consistent.
        if self._discontinuity_mask is not None:
            self._wcc = _apply_discontinuity_mask(
                self._wcc, self._discontinuity_mask, self.window_size
            )

        return self._wcc

    # ---- feature extraction ---------------------------------------------

    def extract_features(
        self,
    ) -> DynamicFeatures:
        """Extract all dynamic features from the WCC/WCLR coupling series.

        Surrogate testing is handled by the InferencePipeline; this method
        only computes feature values from the coupling trace.

        Returns
        -------
        features : DynamicFeatures
            The SSoT feature dataclass (see ``feature_definitions``). Call
            ``.to_dict()`` for a plain ``{name: value}`` mapping.
        """
        if self._wcc is None:
            raise ValueError("Call compute_wcc() first.")

        kwargs = {}
        if self.onset_threshold is not None:
            kwargs["onset_threshold"] = self.onset_threshold
        # P1-R2: when a discontinuity mask gated this pair's WCC, do not glue
        # elevated runs across seams (use segment gap policy). Unmasked pairs
        # keep the SSoT default merge_valid (short dropout bridging).
        if self._discontinuity_mask is not None:
            kwargs["gap_policy"] = "segment"

        self._features = extract_dynamic_features(
            self._wcc,
            hz=self.hz,
            wcc_window_sec=self.window_size / self.hz,
            **kwargs,
        )
        return self._features

    # ---- quick-run shortcut ---------------------------------------------

    def run(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
        label: Optional[str] = None,
        **metadata,
    ) -> Dict[str, float]:
        """One-shot: load → compute WCC → extract features.

        Equivalent to calling load_signals(), compute_wcc(), extract_features()
        in sequence.
        """
        self.load_signals(sig_a, sig_b, label=label, **metadata)
        self.compute_wcc()
        return self.extract_features()

    # ---- output --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a single-row DataFrame."""
        if self._features is None:
            raise ValueError("Call extract_features() first.")
        feature_dict = self._features.to_dict()
        row = {**self._metadata, **feature_dict}
        n_signal = int(self._sig_a.size) if self._sig_a is not None else 0
        n_wcc = int(self._wcc.size) if self._wcc is not None else 0
        n_valid = int(np.isfinite(self._wcc).sum()) if self._wcc is not None else 0
        row["n_signal_samples"] = n_signal
        row["n_wcc_points"] = n_wcc
        row["n_valid_wcc_points"] = n_valid
        row["valid_wcc_fraction"] = n_valid / n_wcc if n_wcc else float("nan")
        row["wcc_observation_sec"] = n_wcc / self.hz if self.hz > 0 else float("nan")
        return pd.DataFrame([row])

    @property
    def wcc(self) -> Optional[np.ndarray]:
        """The computed coupling series (or None if not yet computed)."""
        return self._wcc

    @property
    def features(self) -> Optional[Dict[str, float]]:
        """The extracted feature dictionary (or None)."""
        if self._features is None:
            return None
        return self._features.to_dict()

    @property
    def duration_sec(self) -> Optional[float]:
        """Duration of the WCC series in seconds."""
        if self._wcc is not None:
            return len(self._wcc) / self.hz
        return None


from .batch_pipeline import BatchComputationPipeline


# ---- canonical per-pair entry point (dual API) -------------------------

@dataclass
class PairResult:
    """Stateless result of a single dyad/pair computation.

    Returned by :func:`compute_pair_pipeline`.  Aggregates the coupling
    series and the extracted feature object so downstream code can reuse a
    pre-computed WCC without re-running WCC.
    """

    wcc: np.ndarray
    features: DynamicFeatures  # SSoT feature object exposing .to_dict()
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
    """Compute WCC features for one pair — from signals OR a pre-computed WCC.

    This is the single canonical entry point for per-pair feature
    extraction.  ``quick_compute`` / ``batch_compute`` delegate here.

    Dual input mode (the "double API"):
    * ``wcc is None`` — signals are loaded and the WCC coupling series is
      computed internally, reusing :class:`ComputationPipeline`'s cumsum /
      WCLR / discontinuity-mask machinery.
    * ``wcc is not None`` — the supplied coupling series is used directly,
      skipping WCC recomputation.  Feature extraction is identical to the
      signals path, so a given WCC always yields the same features.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Aligned time series (only used when ``wcc is None``).
    hz : float
        Sampling rate (Hz).
    window_size : int
        WCC window size in samples.
    onset_threshold : float or None
        Episode onset threshold forwarded to feature extraction; ``None``
        uses the feature default.
    wcc : np.ndarray or None
        Pre-computed coupling series.  When given, ``sig_a``/``sig_b`` are
        ignored.
    discontinuity_mask, label, window_type, normalize, backend,
    wclr_max_lag_samples, wclr_metric : forwarded to WCC computation when
        ``wcc is None``.
    **metadata
        Arbitrary key-value metadata merged into :meth:`PairResult.to_dataframe`.

    Returns
    -------
    PairResult
    """
    if wcc is None:
        pipe = ComputationPipeline(
            hz=hz,
            window_size=window_size,
            onset_threshold=onset_threshold,
            backend=backend,
            wclr_max_lag_samples=wclr_max_lag_samples,
            wclr_metric=wclr_metric,
            window_type=window_type,
        )
        pipe.load_signals(
            sig_a, sig_b, label=label,
            discontinuity_mask=discontinuity_mask, **metadata,
        )
        pipe.compute_wcc()
        features = pipe.extract_features()
        wcc_arr = pipe.wcc
    else:
        wcc_arr = np.asarray(wcc, dtype=float)
        extract_kwargs: Dict[str, float] = {}
        if onset_threshold is not None:
            extract_kwargs["onset_threshold"] = onset_threshold
        features = extract_dynamic_features(
            wcc_arr,
            hz=hz,
            wcc_window_sec=window_size / hz,
            **extract_kwargs,
        )

    return PairResult(
        wcc=wcc_arr,
        features=features,
        hz=hz,
        window_size=window_size,
        label=label,
        discontinuity_mask=discontinuity_mask,
        metadata=dict(metadata),
    )


# ---- module-level convenience (thin wrappers over compute_pair_pipeline) -


def quick_compute(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    hz: float = 4.0,
    window_size: int = 40,
    onset_threshold: float = 0.5,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """One-liner: compute WCC and extract features, return DataFrame.

    Delegates to :func:`compute_pair_pipeline` (canonical per-pair entry).

    Examples
    --------
    >>> df = quick_compute(eda_person_a, eda_person_b, hz=4.0)
    """
    result = compute_pair_pipeline(
        sig_a, sig_b, hz=hz, window_size=window_size,
        onset_threshold=onset_threshold, label=label,
    )
    return result.to_dataframe()


def batch_compute(
    dyad_signals: List[Tuple[np.ndarray, np.ndarray]],
    hz: float = 4.0,
    window_size: int = 40,
    onset_threshold: float = 0.5,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute features for multiple dyads, returning a combined DataFrame.

    Each dyad is delegated to :func:`compute_pair_pipeline`.

    Returns
    -------
    pd.DataFrame with one row per dyad.
    """
    if labels is None:
        labels = [None] * len(dyad_signals)

    frames = []
    for i, ((sig_a, sig_b), label) in enumerate(zip(dyad_signals, labels)):
        result = compute_pair_pipeline(
            sig_a, sig_b, hz=hz, window_size=window_size,
            onset_threshold=onset_threshold, label=label, dyad_id=i,
        )
        frames.append(result.to_dataframe())

    return pd.concat(frames, ignore_index=True)


# Compatibility exports for the canonical per-pair module.
from .pair_pipeline import PairResult as PairResult
from .pair_pipeline import compute_pair_pipeline as compute_pair_pipeline

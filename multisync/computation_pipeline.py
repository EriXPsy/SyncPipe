"""
Pipeline 2: Computation pipeline.

Purpose: Load data, compute WCC, extract features, return a clean DataFrame.

This is the "how to compute" layer — users feed in raw time series and get
back structured feature data ready for analysis or the inference pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_features import (
    sliding_window_wcc,
    extract_dynamic_features,
)
from .importer import DataImporter
from .wclr import wclr_coupling_trace
from .session_threshold import (
    compute_session_pooled_threshold,
    compute_condition_pooled_thresholds,
)


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
    ):
        self.hz = hz
        self.window_size = window_size
        self.onset_threshold = onset_threshold
        self.backend = backend.lower()
        self.wclr_max_lag_samples = wclr_max_lag_samples
        self.wclr_metric = wclr_metric
        if self.backend not in ("wcc", "wclr"):
            raise ValueError(f"backend must be 'wcc' or 'wclr', got {backend!r}")
        if self.wclr_metric not in ("beta", "r2"):
            raise ValueError(f"wclr_metric must be 'beta' or 'r2', got {wclr_metric!r}")

        self._sig_a: Optional[np.ndarray] = None
        self._sig_b: Optional[np.ndarray] = None
        self._wcc: Optional[np.ndarray] = None
        self._features: Optional[Dict[str, float]] = None
        self._metadata: Dict[str, object] = {}

    # ---- data loading ----------------------------------------------------

    def load_signals(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
        label: Optional[str] = None,
        **metadata,
    ):
        """Load two pre-processed 1-D signal arrays.

        Parameters
        ----------
        sig_a, sig_b : np.ndarray
            Two aligned time series (e.g. Person A and Person B EDA).
        label : str or None
            Optional label (e.g. condition name).
        **metadata
            Arbitrary key-value metadata stored with the results.
        """
        self._sig_a = np.asarray(sig_a, dtype=float)
        self._sig_b = np.asarray(sig_b, dtype=float)
        self._metadata = {"label": label, **metadata}
        self._wcc = None
        self._features = None

    def load_from_files(
        self,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
        column_a: str = "signal",
        column_b: str = "signal",
        label: Optional[str] = None,
        **metadata,
    ):
        """Load signals from CSV files.

        Parameters
        ----------
        path_a, path_b : str or Path
            Paths to CSV files.
        column_a, column_b : str
            Column names containing the signal data.
        """
        importer = DataImporter()
        sig_a = importer.load_signal(path_a, column=column_a)
        sig_b = importer.load_signal(path_b, column=column_b)
        self.load_signals(sig_a, sig_b, label=label, **metadata)

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

        if self.backend == "wclr":
            trace = wclr_coupling_trace(
                self._sig_a, self._sig_b,
                window_size=self.window_size,
                hz=self.hz,
                max_lag_samples=self.wclr_max_lag_samples,
                metric=self.wclr_metric,
            )
            self._wcc = trace
            return self._wcc

        if method == "cumsum":
            from .dynamic_features import _sliding_window_wcc_cumsum

            if normalize:
                a_min, a_max = np.nanmin(self._sig_a), np.nanmax(self._sig_a)
                b_min, b_max = np.nanmin(self._sig_b), np.nanmax(self._sig_b)
                sig_a_n = (self._sig_a - a_min) / max(a_max - a_min, 1e-10)
                sig_b_n = (self._sig_b - b_min) / max(b_max - b_min, 1e-10)
            else:
                sig_a_n, sig_b_n = self._sig_a, self._sig_b

            self._wcc = _sliding_window_wcc_cumsum(
                sig_a_n, sig_b_n, self.window_size
            )
        else:
            if normalize:
                a_min, a_max = np.nanmin(self._sig_a), np.nanmax(self._sig_a)
                b_min, b_max = np.nanmin(self._sig_b), np.nanmax(self._sig_b)
                sig_a_n = (self._sig_a - a_min) / max(a_max - a_min, 1e-10)
                sig_b_n = (self._sig_b - b_min) / max(b_max - b_min, 1e-10)
            else:
                sig_a_n, sig_b_n = self._sig_a, self._sig_b
            self._wcc = sliding_window_wcc(
                sig_a_n, sig_b_n, self.window_size, hz=self.hz
            )

        return self._wcc

    # ---- feature extraction ---------------------------------------------

    def extract_features(
        self,
    ) -> Dict[str, float]:
        """Extract all dynamic features from the WCC/WCLR coupling series.

        Surrogate testing is handled by the InferencePipeline; this method
        only computes feature values from the coupling trace.

        Returns
        -------
        features : dict
            Feature name → value mapping.
        """
        if self._wcc is None:
            raise ValueError("Call compute_wcc() first.")

        kwargs = {}
        if self.onset_threshold is not None:
            kwargs["onset_threshold"] = self.onset_threshold

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


class BatchComputationPipeline:
    """Batch computation with a shared, session-level surrogate threshold.

    This pipeline addresses the cross-dyad comparability problem of per-dyad
    surrogate thresholds. It first pools all surrogate WCC values across all
    dyads, computes a single percentile threshold, and then extracts features
    for every dyad using that shared threshold.

    Parameters
    ----------
    hz : float
        Sampling rate.
    window_size : int
        WCC window size in samples.
    onset_threshold : float, str, or None
        Either a fixed numeric threshold (e.g. 0.5), or "session_pooled" to
        compute a session-level pooled surrogate threshold (default).
    surrogate_n : int
        Number of surrogates per dyad when threshold is pooled.
    surrogate_percentile : float
        Percentile of the pooled surrogate WCC distribution (default 95).
    surrogate_seed : int
        RNG seed for reproducibility.
    backend : {"wcc", "wclr"}
        Computational backend (default "wcc").
    """

    def __init__(
        self,
        hz: float,
        window_size: int = 40,
        onset_threshold: Union[float, str, None] = "session_pooled",
        surrogate_n: int = 200,
        surrogate_percentile: float = 95.0,
        surrogate_seed: int = 42,
        backend: str = "wcc",
        wclr_max_lag_samples: int = 2,
        wclr_metric: str = "beta",
    ):
        self.hz = hz
        self.window_size = window_size
        self.onset_threshold = onset_threshold
        self.surrogate_n = surrogate_n
        self.surrogate_percentile = surrogate_percentile
        self.surrogate_seed = surrogate_seed
        self.backend = backend.lower()
        self.wclr_max_lag_samples = wclr_max_lag_samples
        self.wclr_metric = wclr_metric
        if self.backend not in ("wcc", "wclr"):
            raise ValueError(f"backend must be 'wcc' or 'wclr', got {backend!r}")
        if self.wclr_metric not in ("beta", "r2"):
            raise ValueError(f"wclr_metric must be 'beta' or 'r2', got {wclr_metric!r}")

        self._dyad_signals: List[Tuple[np.ndarray, np.ndarray]] = []
        self._labels: List[Optional[str]] = []
        self._metadata: List[Dict[str, object]] = []
        self._threshold: Optional[float] = None
        self._threshold_meta: Optional[Dict] = None

    def add_dyad(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
        label: Optional[str] = None,
        **metadata,
    ):
        """Add one dyad to the batch."""
        self._dyad_signals.append((np.asarray(sig_a, dtype=float),
                                   np.asarray(sig_b, dtype=float)))
        self._labels.append(label)
        self._metadata.append(metadata)

    def _compute_threshold(self) -> float:
        """Compute the shared threshold for the batch."""
        if isinstance(self.onset_threshold, (int, float)):
            self._threshold = float(self.onset_threshold)
            self._threshold_meta = {"mode": "fixed", "threshold": self._threshold}
            return self._threshold

        if self.onset_threshold == "session_pooled":
            threshold, meta = compute_session_pooled_threshold(
                self._dyad_signals,
                hz=self.hz,
                wcc_window_size=self.window_size,
                surrogate_n=self.surrogate_n,
                percentile=self.surrogate_percentile,
                seed=self.surrogate_seed,
                surrogate_method="iaaft",
                backend=self.backend,
                wclr_max_lag_samples=self.wclr_max_lag_samples,
            )
            self._threshold = threshold
            self._threshold_meta = meta
            return self._threshold

        if self.onset_threshold is None:
            from .feature_definitions import ONSET_THRESHOLD
            self._threshold = ONSET_THRESHOLD
            self._threshold_meta = {"mode": "default_fallback"}
            return self._threshold

        raise ValueError(f"Unrecognized onset_threshold: {self.onset_threshold!r}")

    def run(self) -> pd.DataFrame:
        """Run the full batch pipeline and return a DataFrame."""
        if not self._dyad_signals:
            raise ValueError("No dyads added. Call add_dyad() first.")

        threshold = self._compute_threshold()
        frames = []
        for i, ((sig_a, sig_b), label, meta) in enumerate(
            zip(self._dyad_signals, self._labels, self._metadata)
        ):
            pipe = ComputationPipeline(
                hz=self.hz,
                window_size=self.window_size,
                onset_threshold=threshold,
                backend=self.backend,
                wclr_max_lag_samples=self.wclr_max_lag_samples,
                wclr_metric=self.wclr_metric,
            )
            pipe.run(sig_a, sig_b, label=label, dyad_id=i, **meta)
            row = pipe.to_dataframe()
            row["threshold_mode"] = self._threshold_meta.get("mode", "unknown")
            row["threshold_value"] = threshold
            row["threshold_fallback"] = self._threshold_meta.get("fallback_used", False)
            frames.append(row)

        return pd.concat(frames, ignore_index=True)

    @property
    def threshold_meta(self) -> Optional[Dict]:
        """Metadata about the computed threshold."""
        return self._threshold_meta


# ---- module-level convenience -------------------------------------------


def quick_compute(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    hz: float = 4.0,
    window_size: int = 40,
    onset_threshold: float = 0.5,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """One-liner: compute WCC and extract features, return DataFrame.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Aligned time series.
    hz : float
        Sampling rate.
    window_size : int
        WCC window size in samples (default 40 → 10 s at 4 Hz).
    onset_threshold : float
        WCC threshold for episode detection.
    label : str or None
        Optional label.

    Returns
    -------
    pd.DataFrame with one row of features.

    Examples
    --------
    >>> df = quick_compute(eda_person_a, eda_person_b, hz=4.0)
    """
    pipe = ComputationPipeline(hz=hz, window_size=window_size, onset_threshold=onset_threshold)
    pipe.run(sig_a, sig_b, label=label)
    return pipe.to_dataframe()


def batch_compute(
    dyad_signals: List[Tuple[np.ndarray, np.ndarray]],
    hz: float = 4.0,
    window_size: int = 40,
    onset_threshold: float = 0.5,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute features for multiple dyads, returning a combined DataFrame.

    Parameters
    ----------
    dyad_signals : list of (sig_a, sig_b) tuples
    hz, window_size, onset_threshold : passed to ComputationPipeline.
    labels : list of str or None
        Optional labels for each dyad.

    Returns
    -------
    pd.DataFrame with one row per dyad.
    """
    if labels is None:
        labels = [None] * len(dyad_signals)

    pipe = ComputationPipeline(hz=hz, window_size=window_size, onset_threshold=onset_threshold)
    frames = []
    for i, ((sig_a, sig_b), label) in enumerate(zip(dyad_signals, labels)):
        pipe.run(sig_a, sig_b, label=label, dyad_id=i)
        frames.append(pipe.to_dataframe())

    return pd.concat(frames, ignore_index=True)

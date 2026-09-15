from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .feature_definitions import ONSET_THRESHOLD
from .session_threshold import compute_session_pooled_thresholds_by_modality


class BatchComputationPipeline:
    """Batch computation with a per-modality pooled surrogate threshold.

    This pipeline addresses the cross-dyad comparability problem of per-dyad
    surrogate thresholds. With the canonical default ``onset_threshold=
    "session_pooled"`` it groups dyads by modality, pools each modality's
    surrogate WCC values, and computes one percentile threshold per modality
    (per-modality null distribution). Features are then extracted for every
    dyad using its modality's shared threshold.

    ``self._thresholds`` is a ``modality -> value`` mapping (one pooled
    surrogate threshold per modality); see :meth:`_compute_threshold`.

    Parameters
    ----------
    hz : float
        Sampling rate.
    window_size : int
        WCC window size in samples.
    onset_threshold : float, str, or None
        Either a fixed numeric threshold (e.g. 0.5), or "session_pooled" to
        compute a per-modality pooled surrogate threshold (canonical default).
        With "session_pooled" each modality gets its own pooled surrogate
        threshold; a fixed numeric value applies the same threshold to every
        modality.
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
        self._modalities: List[Optional[str]] = []
        self._metadata: List[Dict[str, object]] = []
        self._thresholds: Dict[str, float] = {}
        self._threshold: Optional[float] = None
        self._threshold_meta: Optional[Dict] = None
        # Authoritative per-modality fallback flags (populated in the
        # session_pooled branch of _compute_threshold from the threshold meta).
        self._fallback_by_modality: Dict[str, bool] = {}
        self._fallback_reason_by_modality: Dict[str, Optional[str]] = {}

    def add_dyad(
        self,
        sig_a: np.ndarray,
        sig_b: np.ndarray,
        label: Optional[str] = None,
        modality: Optional[str] = None,
        **metadata,
    ):
        """Add one dyad to the batch.

        Parameters
        ----------
        modality : str, optional
            Modality label for this dyad (e.g. "EDA", "ECG"). When
            ``onset_threshold="session_pooled"`` dyads are grouped by modality
            so each modality gets its own pooled surrogate threshold. Dyads
            added without a modality are grouped under a single shared key.
        """
        self._dyad_signals.append((np.asarray(sig_a, dtype=float),
                                   np.asarray(sig_b, dtype=float)))
        self._labels.append(label)
        self._modalities.append(modality)
        self._metadata.append(metadata)

    def _modality_of(self, i: int) -> str:
        """Resolve the grouping key for dyad ``i`` (None -> "None" sentinel)."""
        mod = self._modalities[i] if i < len(self._modalities) else None
        return str(mod) if mod is not None else "None"

    def _compute_threshold(self) -> float:
        """Compute the shared threshold(s) for the batch.

        Populates ``self._thresholds`` (modality-key -> threshold). For a fixed
        or ``None`` onset threshold every modality key maps to the same value;
        for ``"session_pooled"`` each modality gets its own pooled surrogate
        threshold (per-modality null distribution).
        """
        if isinstance(self.onset_threshold, (int, float)):
            value = float(self.onset_threshold)
            self._thresholds = {
                self._modality_of(i): value for i in range(len(self._dyad_signals))
            }
            self._threshold = value
            self._threshold_meta = {"mode": "fixed", "threshold": value}
            return self._threshold

        if self.onset_threshold == "session_pooled":
            modalities = [self._modality_of(i) for i in range(len(self._dyad_signals))]
            thresholds_with_meta = compute_session_pooled_thresholds_by_modality(
                self._dyad_signals,
                modalities,
                hz=self.hz,
                wcc_window_size=self.window_size,
                surrogate_n=self.surrogate_n,
                percentile=self.surrogate_percentile,
                seed=self.surrogate_seed,
                surrogate_method="iaaft",
                backend=self.backend,
                wclr_max_lag_samples=self.wclr_max_lag_samples,
                return_meta=True,
            )
            self._thresholds = {
                mod: thr for mod, (thr, _meta) in thresholds_with_meta.items()
            }
            # Use the authoritative per-modality meta (fallback_used/reason) rather
            # than re-inferring a fallback by float-equality to 0.5, which would
            # misreport a genuine surrogate threshold that happens to land on 0.5.
            self._fallback_by_modality = {
                mod: bool(meta.get("fallback_used", False))
                for mod, (_thr, meta) in thresholds_with_meta.items()
            }
            self._fallback_reason_by_modality = {
                mod: meta.get("reason")
                for mod, (_thr, meta) in thresholds_with_meta.items()
            }
            any_fallback = any(self._fallback_by_modality.values())
            self._threshold = next(iter(self._thresholds.values()), ONSET_THRESHOLD)
            self._threshold_meta = {
                "mode": "session_pooled_by_modality",
                "backend": self.backend,
                "fallback_used": any_fallback,
                "fallback_by_modality": dict(self._fallback_by_modality),
                "fallback_reason_by_modality": dict(self._fallback_reason_by_modality),
                "thresholds_by_modality": dict(self._thresholds),
                "n_modalities": len(self._thresholds),
            }
            return self._threshold

        if self.onset_threshold is None:
            self._threshold = ONSET_THRESHOLD
            self._thresholds = {
                self._modality_of(i): ONSET_THRESHOLD
                for i in range(len(self._dyad_signals))
            }
            self._threshold_meta = {"mode": "default_fallback"}
            return self._threshold

        raise ValueError(f"Unrecognized onset_threshold: {self.onset_threshold!r}")

    def run(self) -> pd.DataFrame:
        """Run the full batch pipeline and return a DataFrame."""
        if not self._dyad_signals:
            raise ValueError("No dyads added. Call add_dyad() first.")

        self._compute_threshold()
        # Prefer authoritative per-modality fallback flags from the threshold
        # meta (session_pooled path). For fixed/default modes no surrogate
        # fallback concept applies, so all modalities are non-fallback.
        if self._fallback_by_modality:
            fallback_by_modality = dict(self._fallback_by_modality)
        else:
            fallback_by_modality = {k: False for k in self._thresholds}
        frames = []
        for i, ((sig_a, sig_b), label, meta) in enumerate(
            zip(self._dyad_signals, self._labels, self._metadata)
        ):
            mod_key = self._modality_of(i)
            threshold = self._thresholds.get(mod_key, ONSET_THRESHOLD)
            from .computation_pipeline import ComputationPipeline

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
            row["threshold_fallback"] = fallback_by_modality.get(
                mod_key, self._threshold_meta.get("fallback_used", False)
            )
            row["modality"] = self._modalities[i] if i < len(self._modalities) else None
            frames.append(row)

        return pd.concat(frames, ignore_index=True)

    @property
    def threshold_meta(self) -> Optional[Dict]:
        """Metadata about the computed threshold."""
        return self._threshold_meta


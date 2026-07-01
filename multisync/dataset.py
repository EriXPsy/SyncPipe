"""
SynchronyDataset — Multi-modal dyadic data container.

Handles: multi-Hz alignment, within-dyad Z-score, NaN imputation,
and context (episode/score) annotation.

Design target: replace fragile CSV wrangling with a single typed object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate as sp_interp
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Time type detection helper
# ---------------------------------------------------------------------------

def _detect_time_type(time_series: pd.Series) -> str:
    """
    Detect if a time column is ABSOLUTE or RELATIVE.

    Detection logic:
    - ABSOLUTE: values look like Unix timestamps (> 1e9, i.e., > 2001-09-09)
      or ISO-like strings containing 'T' or ':'.
    - RELATIVE: values are small floats starting near 0 (typical CSV export
      with "Time" column starting at 0.00).

    Returns
    -------
    str : "absolute" | "relative" | "unknown"
    """
    if pd.api.types.is_string_dtype(time_series):
        # Check for ISO-like strings
        sample = time_series.dropna().iloc[0] if len(time_series.dropna()) > 0 else ""
        if isinstance(sample, str) and ("T" in sample or ":" in sample):
            return "absolute"
        return "unknown"

    vals = time_series.dropna().values
    if len(vals) == 0:
        return "unknown"

    min_val = float(np.min(vals))
    max_val = float(np.max(vals))

    # Unix timestamp heuristic: values > 1e9 (year 2001+) are likely absolute
    if min_val > 1e9:
        return "absolute"

    # Values > 1e6 might be millisecond timestamps
    if min_val > 1e6:
        return "absolute"

    # Small values starting near 0 → relative
    if min_val >= 0 and max_val < 10000:
        return "relative"

    return "unknown"


# ---------------------------------------------------------------------------
# Context annotation
# ---------------------------------------------------------------------------

@dataclass
class ContextLabel:
    """A scored / labelled episode annotation (the psycho context layer)."""
    start_sec: float
    end_sec: float
    label: str
    score: float = 0.0  # optional continuous score (e.g. rapport rating)


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class SynchronyDataset:
    """
    Container for one dyad's multi-modal time-series.

    Each modality is stored as a ``pandas.DataFrame`` with a ``time`` column
    (seconds) and one or more feature columns.  Modalities may have different
    native sampling rates; :meth:`align` resamples everything to a common rate.

    Parameters
    ----------
    dyad_id : str
        Identifier for this dyad pair.
    modalities : dict[str, DataFrame]
        Mapping of modality name → DataFrame.  Each DataFrame must contain a
        ``time`` column (monotonically increasing, seconds) plus at least one
        numeric feature column.
    """

    def __init__(
        self,
        dyad_id: str,
        modalities: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.dyad_id = dyad_id
        self.modalities: Dict[str, pd.DataFrame] = {}
        self.context_labels: List[ContextLabel] = []
        self._aligned: bool = False
        self._normalized: bool = False
        self.target_hz: float = 1.0

        if modalities:
            for name, df in modalities.items():
                self.add_modality(name, df)

    # ------------------------------------------------------------------
    # Modality management
    # ------------------------------------------------------------------

    def add_modality(self, name: str, df: pd.DataFrame) -> "SynchronyDataset":
        """Register a modality DataFrame.  Must contain a 'time' column."""
        df = df.copy()
        if "time" not in df.columns:
            raise ValueError(
                f"Modality '{name}' must have a 'time' column. "
                f"Got columns: {list(df.columns)}"
            )
        if not pd.api.types.is_numeric_dtype(df["time"]):
            raise ValueError(f"'time' column in '{name}' must be numeric.")

        # Sort by time, drop duplicate timestamps
        df = df.sort_values("time").drop_duplicates(subset="time")
        df = df.reset_index(drop=True)

        self.modalities[name] = df
        self._aligned = False
        return self

    @property
    def modality_names(self) -> List[str]:
        return list(self.modalities.keys())

    @property
    def feature_columns(self) -> Dict[str, List[str]]:
        """Return {modality: [feature_cols]} for each modality."""
        out: Dict[str, List[str]] = {}
        for name, df in self.modalities.items():
            cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
            # Ensure float dtype for downstream ops
            for c in cols:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype(float)
            out[name] = cols
        return out

    # ------------------------------------------------------------------
    # Context annotation (psycho layer)
    # ------------------------------------------------------------------

    def add_context(
        self, start: float, end: float, label: str, score: float = 0.0
    ) -> "SynchronyDataset":
        """Add a scored episode annotation."""
        if start >= end:
            raise ValueError(f"start ({start}) must be < end ({end}).")
        self.context_labels.append(ContextLabel(start, end, label, score))
        # Sort by start time
        self.context_labels.sort(key=lambda c: c.start_sec)
        return self

    def get_context_at(self, t: float) -> Optional[ContextLabel]:
        """Return the context label active at time *t*, if any."""
        for ctx in self.context_labels:
            if ctx.start_sec <= t < ctx.end_sec:
                return ctx
        return None

    # ------------------------------------------------------------------
    # Alignment — resample all modalities to a common rate
    # ------------------------------------------------------------------

    def align(
        self,
        target_hz: float = 1.0,
        method: str = "linear",
        require_co_start: bool = False,
    ) -> "SynchronyDataset":
        """
        Resample all modalities to *target_hz*.

        CRITICAL: If using relative timestamps with devices
        that started at different times, CCF will show a
        FALSE LAG equal to the clock offset.
        Use absolute timestamps or set require_co_start=True.

        For heterogeneous data (e.g. EEG 250Hz + EDA 4Hz):
        compute second-level feature envelopes FIRST,
        then align to 1-10 Hz. Do NOT feed raw
        heterogeneous signals directly to align().

        Parameters
        ----------
        target_hz : float
            Target sampling rate (Hz). Default 1.0.
        method : str
            'linear' | 'nearest' | 'cubic'.
        require_co_start : bool
            If True, raise when all modalities use
            relative timestamps.
        """
        if not self.modalities:
            raise ValueError("No modalities to align.")

        # --- Time synchronization check ---
        # Detect if modalities use absolute or relative timestamps.
        # If ALL are relative, warn the user about potential clock offset.
        time_types = {}
        for name, df in self.modalities.items():
            time_types[name] = _detect_time_type(df["time"])

        all_relative = all(tt == "relative" for tt in time_types.values())
        any_absolute = any(tt == "absolute" for tt in time_types.values())
        mixed = any_absolute and not all_relative

        if mixed:
            # Some files have absolute timestamps, some don't — very dangerous
            abs_names = [n for n, t in time_types.items() if t == "absolute"]
            rel_names = [n for n, t in time_types.items() if t == "relative"]
            raise ValueError(
                f"Timestamp type mismatch! Modalities {abs_names} use absolute "
                f"timestamps while {rel_names} use relative timestamps. "
                f"Either convert all to absolute or ensure co-starting."
            )

        if all_relative and require_co_start:
            raise ValueError(
                "All modalities use RELATIVE timestamps (starting near 0). "
                "Cross-device synchrony analysis requires PRECISE time "
                "synchronization. Please either: (1) use absolute timestamps "
                "(Unix timestamp or ISO 8601), or (2) confirm all devices "
                "were started at the exact same moment (co-started). "
                "If co-started, set require_co_start=False to proceed."
            )

        if all_relative and not require_co_start:
            warnings.warn(
                "All modalities use relative timestamps (starting near 0). "
                "If recording devices started at DIFFERENT times, CCF will "
                "show a FALSE LAG equal to the start-time difference. "
                "Please confirm all devices were co-started, or use absolute "
                "timestamps (Unix/ISO). Set require_co_start=True to block "
                "this check.",
                UserWarning,
            )

        self.target_hz = target_hz
        feat_cols = self.feature_columns

        # Common time span = intersection of all modality ranges
        t_starts = [df["time"].iloc[0] for df in self.modalities.values()]
        t_ends = [df["time"].iloc[-1] for df in self.modalities.values()]
        t_min = max(t_starts)
        t_max = min(t_ends)

        if t_min >= t_max:
            raise ValueError(
                "Modalities have no overlapping time range. "
                f"Ranges: {dict(zip(self.modality_names, zip(t_starts, t_ends)))}"
            )

        n_samples = int(np.floor((t_max - t_min) * target_hz)) + 1
        common_time = np.linspace(t_min, t_max, n_samples)

        for name, df in self.modalities.items():
            cols = feat_cols[name]
            original_time = df["time"].values.astype(float)

            new_df = pd.DataFrame({"time": common_time})
            for col in cols:
                valid = ~np.isnan(df[col].values)
                if valid.sum() < 2:
                    # Not enough data to interpolate; fill with NaN
                    new_df[col] = np.nan
                    continue

                if method == "nearest":
                    kind = "nearest"
                elif method == "cubic" and valid.sum() >= 4:
                    kind = "cubic"
                else:
                    kind = "linear"

                interp_func = sp_interp.interp1d(
                    original_time[valid],
                    df[col].values[valid],
                    kind=kind,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                new_df[col] = interp_func(common_time)

            self.modalities[name] = new_df

        self._aligned = True
        return self

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    def handle_nan(
        self,
        strategy: str = "ffill",
        max_gap_sec: Optional[float] = None,
    ) -> "SynchronyDataset":
        """
        Fill or drop NaN values.

        Parameters
        ----------
        strategy : str
            'ffill' — forward fill (default)
            'drop_window' — drop windows where ANY modality has NaN
            'interpolate' — linear interpolation within gaps
        max_gap_sec : float or None
            If set, NaN gaps longer than this (seconds) are NOT filled
            and remain NaN (prevents imputing over long signal dropouts).
        """
        if not self._aligned:
            warnings.warn(
                "Data not yet aligned. Call align() first for reliable results.",
                UserWarning,
            )

        dt = 1.0 / self.target_hz if self.target_hz > 0 else 1.0
        max_gap_samples = int(max_gap_sec / dt) if max_gap_sec else None

        feat_cols = self.feature_columns

        for name in self.modality_names:
            df = self.modalities[name]
            for col in feat_cols[name]:
                series = df[col].copy()

                if strategy == "ffill":
                    if max_gap_samples:
                        # Only ffill gaps <= max_gap_samples
                        nan_groups = (series.isna() != series.isna().shift()).cumsum()
                        for grp_id in nan_groups[series.isna()].unique():
                            mask = nan_groups == grp_id
                            if mask.sum() <= max_gap_samples:
                                series[mask] = series.ffill()[mask]
                            # else: leave as NaN
                    else:
                        series = series.ffill().bfill()

                elif strategy == "interpolate":
                    if max_gap_samples:
                        nan_groups = (series.isna() != series.isna().shift()).cumsum()
                        for grp_id in nan_groups[series.isna()].unique():
                            mask = nan_groups == grp_id
                            if mask.sum() <= max_gap_samples:
                                series[mask] = series.interpolate(method="linear")[mask]
                    else:
                        series = series.interpolate(method="linear").ffill().bfill()

                elif strategy == "drop_window":
                    # Mark rows where ANY feature in ANY modality is NaN
                    pass  # handled below

                else:
                    raise ValueError(f"Unknown NaN strategy: {strategy}")

                df[col] = series
            self.modalities[name] = df

        if strategy == "drop_window":
            # Build a valid mask across all modalities
            common_len = len(next(iter(self.modalities.values())))
            valid_mask = np.ones(common_len, dtype=bool)
            for name in self.modality_names:
                df = self.modalities[name]
                for col in feat_cols[name]:
                    valid_mask &= ~df[col].isna().values
            # Trim all modalities to valid rows
            for name in self.modality_names:
                self.modalities[name] = self.modalities[name][valid_mask].reset_index(drop=True)

        return self

    # ------------------------------------------------------------------
    # Within-dyad Z-score normalization
    # ------------------------------------------------------------------

    def zscore(
        self, method: str = "standard", clip_sigma: Optional[float] = None,
    ) -> Tuple["SynchronyDataset", Dict]:
        """
        Within-dyad Z-score normalization (required before analysis).

        Each column independently z-scored to comparable scale.

        Parameters
        ----------
        method : str
            'standard' (mean/std) or 'robust' (median/IQR).
        clip_sigma : float or None
            Post-normalization Winsorization (+-clip_sigma).
            To protect stats from outliers, call clip_outliers() FIRST.

        Returns
        -------
        self, stats : tuple
            self (for chaining) + pre-normalization stats dict.
        """
        feat_cols = self.feature_columns
        stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        for name in self.modality_names:
            df = self.modalities[name]
            stats[name] = {}
            for col in feat_cols[name]:
                vals = df[col].values.astype(float)

                if method == "robust":
                    median = float(np.nanmedian(vals))
                    q75 = float(np.nanpercentile(vals, 75))
                    q25 = float(np.nanpercentile(vals, 25))
                    iqr = q75 - q25
                    stats[name][col] = {
                        "median": median, "iqr": iqr,
                        "q25": q25, "q75": q75,
                    }
                    if iqr > 0:
                        df[col] = (vals - median) / iqr
                    else:
                        df[col] = 0.0
                else:
                    # standard (mean/std)
                    mu = float(np.nanmean(vals))
                    sigma = float(np.nanstd(vals, ddof=1))  # ddof=1: sample std (psychology standard)
                    stats[name][col] = {"mean": mu, "std": sigma}
                    if sigma > 0:
                        df[col] = (vals - mu) / sigma
                    else:
                        df[col] = 0.0

                # Post-normalization Winsorization (clip extreme tails)
                if clip_sigma is not None:
                    df[col] = df[col].clip(-clip_sigma, clip_sigma)

            self.modalities[name] = df

        self._normalized = True
        return self, stats

    # ------------------------------------------------------------------
    # Outlier clipping (IQR-based Winsorization)
    # ------------------------------------------------------------------

    def clip_outliers(
        self,
        factor: float = 3.0,
        method: str = "iqr",
    ) -> Tuple["SynchronyDataset", Dict]:
        """
        Detect and clip outliers in-place.

        A single sensor glitch (e.g., sneeze → EDA spike) can dominate
        CCF and inflate std, compressing all normal fluctuations.  This
        method clips extreme values BEFORE z-score to prevent that.

        Parameters
        ----------
        factor : float
            IQR multiplier for outlier bounds.  Default 3.0 means values
            beyond Q1 - 3*IQR or Q3 + 3*IQR are clipped.
        method : str
            'iqr' — IQR-based bounds (default).
            'mad' — Median Absolute Deviation based (more robust to
              extreme outlier clustering).

        Returns
        -------
        self (in-place) and a dict reporting per-feature clipping stats.
        """
        feat_cols = self.feature_columns
        report: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for name in self.modality_names:
            df = self.modalities[name]
            report[name] = {}
            for col in feat_cols[name]:
                vals = df[col].values.astype(float)
                valid = vals[~np.isnan(vals)]
                if len(valid) < 10:
                    report[name][col] = {"clipped": 0, "total": len(vals)}
                    continue

                if method == "mad":
                    med = np.median(valid)
                    mad = np.median(np.abs(valid - med))
                    # Scale MAD to std equivalent: σ ≈ 1.4826 × MAD
                    k = 1.4826
                    lower = med - factor * k * mad
                    upper = med + factor * k * mad
                else:
                    q25 = np.percentile(valid, 25)
                    q75 = np.percentile(valid, 75)
                    iqr = q75 - q25
                    lower = q25 - factor * iqr
                    upper = q75 + factor * iqr

                n_clipped = int(np.sum((vals < lower) | (vals > upper)))
                df[col] = df[col].clip(lower, upper)
                report[name][col] = {
                    "clipped": n_clipped,
                    "total": len(vals),
                    "lower": float(lower),
                    "upper": float(upper),
                }

            self.modalities[name] = df

        return self, report

    # ------------------------------------------------------------------
    # Median filter (pulse noise removal)
    # ------------------------------------------------------------------

    def median_filter(
        self,
        kernel_size: int = 5,
    ) -> Tuple["SynchronyDataset", Dict]:
        """
        Apply sliding median filter to remove pulse-type noise.

        Median filters are ideal for removing short-duration spikes
        (sensor glitches, movement artifacts) while preserving
        edge shapes and slow trends.  Unlike low-pass filters, they
        do NOT introduce phase distortion.

        Parameters
        ----------
        kernel_size : int
            Window size in samples (must be odd).  Default 5.
            At 1 Hz target rate, kernel_size=5 covers 5 seconds.
            Recommended: 3-7 for physiological signals at ~1 Hz.

        Returns
        -------
        self (in-place) and a dict of filter parameters applied.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        feat_cols = self.feature_columns
        report: Dict[str, Dict[str, int]] = {}

        for name in self.modality_names:
            df = self.modalities[name]
            for col in feat_cols[name]:
                vals = df[col].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.sum() < kernel_size:
                    continue
                # Only filter non-NaN regions (NaN boundaries preserved)
                filtered = vals.copy()
                if mask.all():
                    filtered = median_filter(vals, size=kernel_size)
                else:
                    # Find contiguous non-NaN segments
                    diff = np.diff(mask.astype(int))
                    starts = np.where(diff == 1)[0] + 1
                    ends = np.where(diff == -1)[0] + 1
                    if mask[0]:
                        starts = np.concatenate([[0], starts])
                    if mask[-1]:
                        ends = np.concatenate([ends, [len(vals)]])
                    for s, e in zip(starts, ends):
                        seg = vals[s:e]
                        if len(seg) >= kernel_size:
                            filtered[s:e] = median_filter(seg, size=kernel_size)
                df[col] = filtered
            report[name] = {"kernel_size": kernel_size}
            self.modalities[name] = df

        return self, report

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_aligned_array(
        self, modality: str, feature: str
    ) -> Optional[np.ndarray]:
        """Return a 1-D numpy array for one modality+feature."""
        if modality not in self.modalities:
            return None
        df = self.modalities[modality]
        if feature not in df.columns:
            return None
        return df[feature].values.astype(float)

    def time_vector(self) -> np.ndarray:
        """Return the common time vector (requires prior align())."""
        if not self.modalities:
            return np.array([])
        # All modalities share the same time after align()
        return self.modalities[next(iter(self.modalities))]["time"].values

    # ------------------------------------------------------------------
    # Exclusion mask (seconds-axis)
    # ------------------------------------------------------------------

    def add_exclusion_interval(self, start_sec: float, end_sec: float, reason: str = "") -> "SynchronyDataset":
        """
        Mark a time interval to be excluded from all subsequent analyses.

        Parameters
        ----------
        start_sec, end_sec : float
            Time interval **in seconds** to exclude.
        reason : str
            Optional label (e.g. "movement_artifact", "break").
            Stored in ``diagnostics`` for the Viewer's Data Exclusion panel.

        Returns
        -------
        self (for chaining).
        """
        if start_sec >= end_sec:
            raise ValueError(f"exclusion start ({start_sec}) must be < end ({end_sec})")
        if not hasattr(self, "_exclusion_intervals"):
            self._exclusion_intervals: List[Dict[str, Any]] = []
        self._exclusion_intervals.append({
            "start": float(start_sec),
            "end": float(end_sec),
            "reason": str(reason),
        })
        return self

    def get_exclusion_mask(self) -> np.ndarray:
        """
        Return a boolean mask of length = common time vector.

        ``True``  = this sample is **excluded** (inside any exclusion interval).
        ``False`` = this sample is **kept**.

        Returns an all-``False`` mask if no intervals have been registered.
        """
        if not hasattr(self, "_exclusion_intervals") or not self._exclusion_intervals:
            return np.zeros(len(self.time_vector()), dtype=bool)

        t = self.time_vector()
        mask = np.zeros(len(t), dtype=bool)
        for iv in self._exclusion_intervals:
            mask |= (t >= iv["start"]) & (t < iv["end"])
        return mask

    def get_exclusion_report(self) -> List[Dict[str, Any]]:
        """Return the list of registered exclusion intervals (for JSON export)."""
        if not hasattr(self, "_exclusion_intervals"):
            return []
        return list(self._exclusion_intervals)

    # ------------------------------------------------------------------
    # Convenience: full preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        outlier_factor: float = 3.0,
        outlier_method: str = "iqr",
        median_kernel: Optional[int] = None,
        zscore_method: str = "standard",
        clip_sigma: Optional[float] = None,
    ) -> Tuple["SynchronyDataset", Dict]:
        """
        Run the recommended signal-preprocessing pipeline in one call.

        Executes in this order::

            clip_outliers(factor, method)
                ↓
            [optional] median_filter(kernel_size)
                ↓
            zscore(method, clip_sigma)

        Rationale
        ---------
        * **clip_outliers first** — prevents sensor glitches (e.g., an EDA
          sneeze spike) from inflating ``std``, which would compress all
          normal fluctuations after z-scoring.
        * **median_filter second** (optional) — removes short-duration pulse
          noise without phase distortion; must run before z-score so the
          filter acts on raw-unit signal.
        * **zscore last** — only after outliers are contained can the mean/std
          reliably represent the "typical" baseline.

        Parameters
        ----------
        outlier_factor : float
            Multiplier for outlier bounds.  Default 3.0 (IQR-based: Q1-3*IQR
            to Q3+3*IQR).
        outlier_method : str
            'iqr' (default) or 'mad' (more robust to outlier clustering).
        median_kernel : int or None
            If set, apply a median filter with this kernel size (samples)
            after outlier clipping.  Must be odd; auto-incremented if even.
            ``None`` skips this step.
        zscore_method : str
            'standard' (mean/std, default) or 'robust' (median/IQR).
        clip_sigma : float or None
            Post-z-score output cap (Winsorization).  ``None`` skips.

        Returns
        -------
        self : SynchronyDataset
            In-place mutation, returned for chaining.
        report : dict
            Keys: 'outliers', 'median_filter' (if applied), 'zscore_stats'.
        """
        report: Dict[str, Any] = {}

        # Step 1: clip outliers
        _, outlier_report = self.clip_outliers(factor=outlier_factor, method=outlier_method)
        report["outliers"] = outlier_report

        # Step 2: optional median filter
        if median_kernel is not None:
            _, mf_report = self.median_filter(kernel_size=median_kernel)
            report["median_filter"] = mf_report

        # Step 3: z-score
        _, zscore_stats = self.zscore(method=zscore_method, clip_sigma=clip_sigma)
        report["zscore_stats"] = zscore_stats

        return self, report

    def summary(self) -> str:
        lines = [f"SynchronyDataset '{self.dyad_id}'"]
        lines.append(f"  Aligned: {self._aligned} | Normalized: {self._normalized}")
        lines.append(f"  Target Hz: {self.target_hz}")
        for name, df in self.modalities.items():
            cols = self.feature_columns.get(name, [])
            n_nan = sum(df[c].isna().sum() for c in cols)
            lines.append(
                f"  {name}: {len(df)} samples, {len(cols)} features, "
                f"{n_nan} NaNs, t=[{df['time'].iloc[0]:.1f}, {df['time'].iloc[-1]:.1f}]s"
            )
        lines.append(f"  Context labels: {len(self.context_labels)}")
        return "\n".join(lines)

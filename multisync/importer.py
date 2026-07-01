"""
multisync.importer — Multi-format data import (CSV, OpenFace, EDF, BioSignalsPlux).

Auto-detects delimiter, time column, signal columns. Returns SynchronyDataset-compatible dict.

Usage: DataImporter().load_csv(...), .load_openface(...), .load_edf(...), .load_opensignals(...)
"""

from __future__ import annotations

import csv
import io
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: dtype detection
# ---------------------------------------------------------------------------

def _infer_hz(time_col: np.ndarray) -> float:
    """Estimate sampling rate from a time vector (seconds)."""
    diffs = np.diff(time_col)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    median_dt = float(np.median(diffs))
    return 1.0 / median_dt if median_dt > 0 else 1.0


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically identify the time column."""
    candidates = ["time", "timestamp", "t", "time_sec", "seconds", "sample_time"]
    for c in candidates:
        if c.lower() in [col.lower() for col in df.columns]:
            return next(col for col in df.columns if col.lower() == c.lower())
    # Fallback: first monotone numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].is_monotonic_increasing:
                return col
    return None


# ---------------------------------------------------------------------------
# DataImporter
# ---------------------------------------------------------------------------

class DataImporter:
    """
    Multi-format physiological data importer.

    Returns DataFrames in SyncPipe's expected format:
        - 'time' column (seconds, zero-based or absolute)
        - One or more signal columns

    Parameters
    ----------
    default_hz : float
        Fallback sampling rate if it cannot be inferred from the data.
    force_zero_start : bool
        If True, shift time to start at 0.0.
    """

    def __init__(
        self,
        default_hz: float = 1.0,
        force_zero_start: bool = True,
    ) -> None:
        self.default_hz = default_hz
        self.force_zero_start = force_zero_start

    # ------------------------------------------------------------------
    # Generic CSV / TSV
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filepath: str,
        time_col: Optional[str] = None,
        signal_cols: Optional[List[str]] = None,
        person_a_cols: Optional[List[str]] = None,
        person_b_cols: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        skip_rows: int = 0,
        encoding: str = "utf-8",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a generic CSV/TSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        time_col : str or None
            Name of the time column. Auto-detected if None.
        signal_cols : list of str or None
            Signal columns to load. All numeric columns if None.
        person_a_cols, person_b_cols : list of str or None
            If provided, build a dyad DataFrame with 'person_a' and 'person_b'
            columns from these column subsets.
        delimiter : str or None
            Auto-detected from file if None.
        skip_rows : int
            Number of header rows to skip.
        encoding : str
            File encoding.

        Returns
        -------
        dict mapping modality name → DataFrame with 'time' + signal columns.
        """
        filepath = str(filepath)

        # Auto-detect delimiter
        if delimiter is None:
            delimiter = self._detect_delimiter(filepath, encoding)

        df = pd.read_csv(
            filepath,
            sep=delimiter,
            skiprows=skip_rows,
            encoding=encoding,
            low_memory=False,
        )

        # Standardize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Identify time column
        if time_col is None:
            time_col = _detect_time_column(df)
        if time_col is None:
            # No time column found: generate synthetic time
            warnings.warn(
                f"No time column found in {filepath}. "
                "Generating synthetic time at default_hz.",
                UserWarning,
            )
            df["time"] = np.arange(len(df)) / self.default_hz
            time_col = "time"

        time_arr = df[time_col].values.astype(float)
        if self.force_zero_start:
            time_arr = time_arr - time_arr[0]

        # Build output DataFrame(s)
        if signal_cols is None:
            signal_cols = [c for c in df.columns
                           if c != time_col
                           and pd.api.types.is_numeric_dtype(df[c])]

        if person_a_cols and person_b_cols:
            # Build dyad DataFrame: columns are 'time', 'person_a', 'person_b'
            # Caller is responsible for passing matching-length arrays
            out = {"signal": pd.DataFrame({
                "time": time_arr,
                "person_a": df[person_a_cols[0]].values.astype(float),
                "person_b": df[person_b_cols[0]].values.astype(float),
            })}
            return out

        result = {}
        for col in signal_cols:
            result[col] = pd.DataFrame({
                "time": time_arr,
                col: df[col].values.astype(float),
            })
        return result

    def load_signal(
        self,
        filepath: str,
        column: str = "signal",
        time_col: Optional[str] = None,
        delimiter: Optional[str] = None,
        skip_rows: int = 0,
        encoding: str = "utf-8",
    ) -> np.ndarray:
        """Load one numeric signal column from a CSV/TSV file.

        This small helper backs ``ComputationPipeline.load_from_files``.  It
        deliberately returns only the requested 1-D signal array; alignment and
        time handling are the caller's responsibility in that low-level API.
        """
        loaded = self.load_csv(
            filepath,
            time_col=time_col,
            signal_cols=[column],
            delimiter=delimiter,
            skip_rows=skip_rows,
            encoding=encoding,
        )
        if column not in loaded:
            raise ValueError(f"Column {column!r} was not loaded from {filepath!r}.")
        return loaded[column][column].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # OpenFace AU CSV
    # ------------------------------------------------------------------

    def load_openface(
        self,
        filepath: str,
        au_cols: Optional[List[str]] = None,
        person_id: str = "person_a",
        use_confidence: bool = True,
        min_confidence: float = 0.8,
    ) -> pd.DataFrame:
        """
        Load OpenFace Action Unit output CSV.

        Parameters
        ----------
        filepath : str
            OpenFace output CSV (columns: frame, face_id, timestamp, confidence,
            success, AU01_r, AU01_c, ...).
        au_cols : list of str or None
            AU columns to extract (e.g., ['AU06_r', 'AU12_r']).
            Defaults to all AU_r (intensity) columns.
        person_id : str
            Person identifier column name in output (default 'person_a').
        use_confidence : bool
            If True, NaN-out rows with confidence < min_confidence.
        min_confidence : float
            Minimum face detection confidence (0–1).

        Returns
        -------
        pd.DataFrame with columns: time, [au_cols] → for use as a modality.
        """
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Time column
        time_col = next(
            (c for c in ["timestamp", "time", " timestamp"] if c in df.columns),
            None,
        )
        if time_col is None:
            # Infer from frame
            fps = 25.0
            if "frame" in df.columns:
                df["time"] = df["frame"].values / fps
            else:
                df["time"] = np.arange(len(df)) / fps
            time_col = "time"

        time_arr = df[time_col].values.astype(float)
        if self.force_zero_start:
            time_arr = time_arr - time_arr[0]

        # Confidence masking
        low_conf_mask = np.zeros(len(df), dtype=bool)
        if use_confidence and "confidence" in df.columns:
            conf = df["confidence"].values.astype(float)
            low_conf_mask = conf < min_confidence

        # AU columns
        if au_cols is None:
            au_cols = [c for c in df.columns if c.endswith("_r")
                       and c.startswith("AU")]

        if not au_cols:
            raise ValueError(
                f"No Action Unit columns found in {filepath}. "
                "OpenFace AU columns should end with '_r' (e.g., AU06_r)."
            )

        result = {"time": time_arr}
        for col in au_cols:
            vals = df[col].values.astype(float)
            vals[low_conf_mask] = np.nan
            result[col] = vals

        out_df = pd.DataFrame(result)
        return out_df

    # ------------------------------------------------------------------
    # OpenSignals / BioSignalsPlux TXT
    # ------------------------------------------------------------------

    def load_opensignals(
        self,
        filepath: str,
        channel_map: Optional[Dict[str, str]] = None,
        person_id: str = "person_a",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load BioSignalsPlux / OpenSignals TXT file.

        OpenSignals files begin with a JSON header (prefixed with #) followed
        by tab-separated data.

        Parameters
        ----------
        filepath : str
            Path to the .txt file.
        channel_map : dict or None
            Mapping from original channel name (e.g., 'CH1') to modality
            name (e.g., 'eda'). Auto-detects if None.
        person_id : str
            Person identifier column name in output.

        Returns
        -------
        dict mapping modality → DataFrame with 'time' + person_id columns.
        """
        filepath = str(filepath)

        header_lines: List[str] = []
        data_lines: List[str] = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line[1:].strip())
                else:
                    data_lines.append(line)

        # Parse JSON header to extract hz and channel info
        hz = self.default_hz
        channel_names: List[str] = []

        for hline in header_lines:
            try:
                meta = json.loads(hline)
                # OpenSignals header: {"device_id": {..., "sampling rate": 1000, "channels": [...]}}
                for device_id, device_info in meta.items():
                    if isinstance(device_info, dict):
                        hz = float(device_info.get("sampling rate",
                               device_info.get("sr", hz)))
                        channel_names = device_info.get("channels", [])
                        if isinstance(channel_names, list):
                            channel_names = [f"CH{i+1}" for i in range(len(channel_names))]
                break
            except (json.JSONDecodeError, AttributeError):
                continue

        # Parse data
        data_str = "".join(data_lines)
        try:
            df = pd.read_csv(io.StringIO(data_str), sep="\t", header=None)
        except Exception as exc:
            raise ValueError(
                f"Could not parse data section of {filepath}: {exc}"
            ) from exc

        # Assign column names: first col = sample index, remaining = channels
        if channel_names:
            col_names = ["sample"] + channel_names + [
                f"CH{i}" for i in range(len(channel_names) + 1, len(df.columns))
            ]
        else:
            col_names = [f"col_{i}" for i in range(len(df.columns))]
            channel_names = col_names[1:]

        df.columns = col_names[:len(df.columns)]

        # Generate time column from sample index
        sample_col = df.columns[0]
        time_arr = df[sample_col].values.astype(float) / hz
        if self.force_zero_start:
            time_arr = time_arr - time_arr[0]

        # Apply channel map
        if channel_map is None:
            channel_map = {cn: cn.lower() for cn in channel_names}

        result: Dict[str, pd.DataFrame] = {}
        for ch_name, modality_name in channel_map.items():
            if ch_name in df.columns:
                result[modality_name] = pd.DataFrame({
                    "time": time_arr,
                    person_id: df[ch_name].values.astype(float),
                })
            else:
                warnings.warn(
                    f"Channel '{ch_name}' not found in {filepath}. "
                    f"Available: {list(df.columns)}",
                    UserWarning,
                )

        return result

    # ------------------------------------------------------------------
    # EDF / EDF+ (requires pyEDFlib)
    # ------------------------------------------------------------------

    def load_edf(
        self,
        filepath: str,
        channel_names: Optional[List[str]] = None,
        person_id: str = "person_a",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load EDF/EDF+ physiological data file.

        Requires the ``pyEDFlib`` package (``pip install pyEDFlib``).

        Parameters
        ----------
        filepath : str
            Path to the .edf or .bdf file.
        channel_names : list of str or None
            Signal labels to extract. Loads all channels if None.
        person_id : str
            Person identifier for the output columns.

        Returns
        -------
        dict mapping channel_name → DataFrame with 'time' + person_id columns.
        """
        try:
            import pyedflib
        except ImportError:
            raise ImportError(
                "pyEDFlib is required for EDF import. "
                "Install with: pip install pyEDFlib"
            )

        filepath = str(filepath)
        result: Dict[str, pd.DataFrame] = {}

        with pyedflib.EdfReader(filepath) as f:
            available_signals = f.getSignalLabels()
            n_samples = f.getNSamples()
            sample_rates = f.getSampleFrequencies()

            if channel_names is None:
                channel_names = list(available_signals)

            for ch_name in channel_names:
                # Case-insensitive match
                match = next(
                    (i for i, label in enumerate(available_signals)
                     if label.strip().lower() == ch_name.strip().lower()),
                    None,
                )
                if match is None:
                    warnings.warn(
                        f"Channel '{ch_name}' not found in {filepath}. "
                        f"Available: {available_signals}",
                        UserWarning,
                    )
                    continue

                hz = float(sample_rates[match])
                data = f.readSignal(match)
                n = len(data)
                time_arr = np.arange(n) / hz

                result[ch_name.strip()] = pd.DataFrame({
                    "time": time_arr,
                    person_id: data.astype(float),
                })

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_multisync_csv(
        self,
        filepath: str,
        modality_name: str = "signal",
    ) -> pd.DataFrame:
        """
        Load a SyncPipe native export CSV.

        Expected columns: time_sec, person_a, person_b (or similar).

        Returns a DataFrame ready to pass directly to Dyad().
        """
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]

        time_col = _detect_time_column(df)
        if time_col and time_col != "time":
            df = df.rename(columns={time_col: "time"})

        if "time" not in df.columns:
            df.insert(0, "time", np.arange(len(df)) / self.default_hz)

        return df

    def merge_person_files(
        self,
        file_a: str,
        file_b: str,
        modality: str = "signal",
        time_col: Optional[str] = None,
        signal_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge two single-person CSV files into one dyad DataFrame.

        Useful when person A and person B are stored in separate files.
        Time alignment is done by inner-join on nearest time values.

        Parameters
        ----------
        file_a, file_b : str
            Paths to the two CSV files.
        modality : str
            Name of the signal column to merge.
        time_col : str or None
            Time column name (auto-detected if None).
        signal_col : str or None
            Signal column to extract (auto-detected if None).

        Returns
        -------
        pd.DataFrame with columns: time, person_a, person_b
        """
        def _load_person(fpath: str) -> pd.DataFrame:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            tcol = time_col or _detect_time_column(df)
            if tcol is None:
                df["time"] = np.arange(len(df)) / self.default_hz
                tcol = "time"
            scol = signal_col or next(
                (c for c in df.columns if c != tcol
                 and pd.api.types.is_numeric_dtype(df[c])), None
            )
            if scol is None:
                raise ValueError(f"No numeric signal column found in {fpath}")
            t = df[tcol].values.astype(float)
            if self.force_zero_start:
                t = t - t[0]
            return pd.DataFrame({"time": t, "signal": df[scol].values.astype(float)})

        df_a = _load_person(file_a).rename(columns={"signal": "person_a"})
        df_b = _load_person(file_b).rename(columns={"signal": "person_b"})

        # Merge on nearest time (tolerance = 2 samples at default_hz)
        merged = pd.merge_asof(
            df_a.sort_values("time"),
            df_b.sort_values("time"),
            on="time",
            tolerance=2.0 / self.default_hz,
            direction="nearest",
        )
        return merged[["time", "person_a", "person_b"]]

    @staticmethod
    def _detect_delimiter(filepath: str, encoding: str = "utf-8") -> str:
        """Auto-detect CSV delimiter from first 4096 bytes."""
        try:
            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except Exception:
            return ","

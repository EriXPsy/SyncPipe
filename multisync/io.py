"""
I/O utilities — CSV loading and Viewer JSON export.

The exported JSON schema is designed for "zero-computation on the frontend":
all peaks, p-values, graph edges, and context summaries are pre-computed
by the Python core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .core import AnalysisResults


def load_csv(
    filepath: str,
    time_col: str = "time",
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single CSV file as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    time_col : str
        Name of the time column.
    value_col : str or None
        If specified, only keep this column (plus time).
    """
    df = pd.read_csv(filepath)
    if time_col not in df.columns:
        raise ValueError(f"CSV must have '{time_col}' column. Got: {list(df.columns)}")
    if value_col:
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found. Got: {list(df.columns)}")
        df = df[[time_col, value_col]]
    return df


def load_multimodal_csv(
    modality_files: Dict[str, str],
    time_col: str = "time",
    value_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files, one per modality.

    Parameters
    ----------
    modality_files : dict
        {modality_name: filepath}.
    time_col : str
        Name of the time column in each file.
    value_col : str or None
        If specified, only keep this column.

    Returns
    -------
    dict of {modality_name: DataFrame}.
    """
    result = {}
    for name, path in modality_files.items():
        result[name] = load_csv(path, time_col, value_col)
    return result


def export_viewer_json(
    results: AnalysisResults,
    filepath: str,
) -> str:
    """
    Export analysis results to Viewer-ready JSON.

    This is the single decoupling bridge between Python Core and React Viewer.
    The Viewer must perform ZERO computation on this data.
    """
    return results.export_viewer_json(filepath)


def load_analysis_results(filepath: str) -> Dict[str, Any]:
    """Load a previously exported Viewer JSON (for inspection/testing)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

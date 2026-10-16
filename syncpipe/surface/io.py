"""On-disk exchange format for surfaces (schema 0.1): a tidy CSV plus a JSON sidecar.

``<stem>.surface.csv``   one row per cell: ``window_start_sample`` (0-based), ``tau`` (samples), ``value``.
``<stem>.surface.json``  schema version, settings, grid size and the content hash.

The hash is verified on read, so a surface that was edited, truncated or re-ordered by another
tool is rejected instead of silently analysed.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .model import SURFACE_SCHEMA_VERSION, SurfaceSettings, SyncSurface

PathLike = str | pathlib.Path
_CSV_COLUMNS = ("window_start_sample", "tau", "value")


def _paths(stem: PathLike) -> tuple[pathlib.Path, pathlib.Path]:
    base = pathlib.Path(stem)
    return base.with_name(base.name + ".surface.csv"), base.with_name(base.name + ".surface.json")


def write_surface(surface: SyncSurface, stem: PathLike) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``surface`` next to ``stem``; returns the (csv, json) paths."""
    csv_path, json_path = _paths(stem)
    cells = surface.tidy()
    frame = pd.DataFrame({
        "window_start_sample": surface.window_start_samples[cells["window_index"].to_numpy()],
        "tau": cells["tau"].to_numpy(),
        "value": cells["value"].to_numpy(),
    })
    # 17 significant digits + the round-trip parser: pandas' default float I/O drops the last bit,
    # which would make a freshly written surface fail its own hash check on read.
    frame.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.17g")
    meta = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "settings": surface.settings.to_dict(),
        "n_lags": int(surface.values.shape[0]),
        "n_windows": int(surface.values.shape[1]),
        "content_hash": surface.content_hash(),
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def read_surface(stem: PathLike) -> SyncSurface:
    """Read a surface written by :func:`write_surface` (or by a conforming engine adapter).

    Raises
    ------
    ValueError
        On an unknown schema major version, an incomplete grid, or a content hash that does
        not match the recorded one.
    """
    csv_path, json_path = _paths(stem)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    version = str(meta.get("schema_version", ""))
    if version.split(".")[0] != SURFACE_SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"unsupported surface schema {version!r}; this reader understands {SURFACE_SCHEMA_VERSION!r}")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    if tuple(frame.columns) != _CSV_COLUMNS:
        raise ValueError(f"surface CSV columns must be {_CSV_COLUMNS}, got {tuple(frame.columns)}")
    lags = pd.unique(frame["tau"].to_numpy())
    starts = pd.unique(frame["window_start_sample"].to_numpy())
    if len(frame) != lags.size * starts.size:
        raise ValueError(f"incomplete grid: {len(frame)} cells for {lags.size} lags x {starts.size} windows")
    values = np.full((lags.size, starts.size), np.nan)
    row = {int(t): k for k, t in enumerate(lags)}
    col = {int(s): k for k, s in enumerate(starts)}
    for tau, start, value in zip(frame["tau"], frame["window_start_sample"], frame["value"], strict=True):
        values[row[int(tau)], col[int(start)]] = value
    surface = SyncSurface(values, lags.astype(np.int64), starts.astype(np.int64),
                          SurfaceSettings.from_dict(meta["settings"]))
    if surface.content_hash() != meta.get("content_hash"):
        raise ValueError("surface content does not match its recorded hash (edited, truncated or re-ordered?)")
    return surface

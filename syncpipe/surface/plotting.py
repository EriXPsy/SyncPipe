"""Plotting for surfaces (matplotlib is an optional dependency, imported lazily)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .model import SyncSurface


def plot_surface(surface: SyncSurface, ax: Any = None) -> Any:
    """Heat map of ``surface`` (time x lag), or a line for a single-lag surface."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    time = surface.window_center_sec
    if surface.values.shape[0] == 1:
        ax.plot(time, surface.values[0])
        ax.set_ylabel("correlation (r)")
        ax.set_ylim(-1.0, 1.0)
    else:
        lags = surface.lags_sec
        half_step = (lags[1] - lags[0]) / 2.0
        extent = (float(time[0]), float(time[-1]), float(lags[0] - half_step), float(lags[-1] + half_step))
        image = ax.imshow(np.ma.masked_invalid(surface.values), aspect="auto", origin="lower",
                          extent=extent, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.figure.colorbar(image, ax=ax, label="correlation (r)")
        ax.set_ylabel("lag (s); positive = x leads y")
    ax.set_xlabel("window centre (s)")
    return ax

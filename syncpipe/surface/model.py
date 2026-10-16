"""The time x lag synchrony surface and the quantities derived from it.

A *surface* is the raw result of a synchrony estimator before any summarising:
one value per (window position, lag).  Keeping it is the point of this module --
every scalar a study reports (magnitude, when, at which lag) is a pure function of
the surface, so a different summary never requires re-estimating.

Conventions
-----------
* Window ``k`` covers samples ``[start_k, start_k + window_samples)`` of ``x``; at
  lag ``tau`` it is paired with ``y[start_k + tau : start_k + tau + window_samples]``.
* A **positive lag means x leads y** (y follows x), as in bsync.
* All lags share the same window positions, trimmed by ``lag_max`` at both ends.
* Correlations are stored as ``r``.  Averages are taken in Fisher-z space with the
  same clamp bsync uses (``Z_CLIP``), so the bsync-compatible summaries below can
  be compared with bsync's own output.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

SURFACE_SCHEMA_VERSION = "0.1"
LAG_SIGN = "x_leads_y_positive"
Z_CLIP = 0.9999  # |r| is clamped here before Fisher's z, matching bsync's r_to_z().

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def r_to_z(r: npt.ArrayLike) -> FloatArray:
    """Fisher's z of correlations, clamped to ``+-Z_CLIP`` so perfect fits stay finite.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isfinite(r_to_z(1.0)))
    True
    >>> round(float(r_to_z(0.0)), 12)
    0.0
    """
    clipped = np.clip(np.asarray(r, dtype=np.float64), -Z_CLIP, Z_CLIP)
    return np.arctanh(clipped)


def _nan_reduce(func: Callable[..., Any], values: FloatArray, axis: int | None = None) -> Any:
    """Apply a ``numpy.nan*`` reducer without the all-NaN ``RuntimeWarning``.

    All-NaN slices yield NaN: absence of data is reported, not hidden or warned about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(values, axis=axis)


@dataclass(frozen=True)
class SurfaceSettings:
    """Everything needed to reproduce a surface from its input series.

    Parameters
    ----------
    hz
        Sampling rate of the series, in samples per second.
    window_samples, window_increment
        Window length and step, in samples.
    lag_max_samples, lag_increment
        Lags run over ``-lag_max_samples .. +lag_max_samples`` in steps of
        ``lag_increment``; ``lag_max_samples = 0`` is the single-lag (zero-lag) surface.
    n_samples
        Length of each input series.
    estimator, engine, engine_version
        Provenance: what produced the values.
    """

    hz: float
    window_samples: int
    window_increment: int
    lag_max_samples: int
    lag_increment: int
    n_samples: int
    estimator: str = "wcc"
    engine: str = "syncpipe.native"
    engine_version: str = ""
    scale: str = "r"
    lag_sign: str = LAG_SIGN
    na_policy: str = "pairwise_delete"
    time_origin_sec: float = 0.0

    def __post_init__(self) -> None:
        if not self.hz > 0:
            raise ValueError(f"hz must be positive, got {self.hz!r}")
        for name in ("window_samples", "window_increment", "lag_increment", "n_samples"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer, got {getattr(self, name)!r}")
        if int(self.lag_max_samples) != self.lag_max_samples or self.lag_max_samples < 0:
            raise ValueError(f"lag_max_samples must be a non-negative integer, got {self.lag_max_samples!r}")
        if self.scale != "r":
            raise ValueError(f"only scale='r' is stored (derive z on demand), got {self.scale!r}")
        if self.lag_sign != LAG_SIGN:
            raise ValueError(f"unsupported lag sign convention {self.lag_sign!r}; expected {LAG_SIGN!r}")

    @property
    def window_sec(self) -> float:
        """Window length in seconds."""
        return self.window_samples / self.hz

    @property
    def lag_max_sec(self) -> float:
        """Largest lag magnitude in seconds."""
        return self.lag_max_samples / self.hz

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form used for hashing and the on-disk schema."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SurfaceSettings:
        """Rebuild settings from :meth:`to_dict` output (unknown keys are an error)."""
        return cls(**payload)


@dataclass(frozen=True)
class SurfaceSummary:
    """Scalar summaries of one surface; every field is derived, none is stored state.

    ``mean_abs_z`` and ``peak_abs_z`` follow bsync's definitions exactly (mean of
    ``|z|`` over cells; per-window maximum of ``|z|`` over lags, averaged over
    windows).  ``mean_z`` and ``peak_pos_z`` are the signed, positive-direction
    counterparts: negative correlation is not counted as synchrony.
    """

    n_windows: int
    n_lags: int
    n_cells_valid: int
    fraction_valid: float
    mean_z: float
    mean_abs_z: float
    peak_pos_z: float
    peak_abs_z: float
    global_peak_r: float
    lag_of_peak_sec: float
    lag_profile_peak_z: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form."""
        return asdict(self)


@dataclass(frozen=True, eq=False)
class SyncSurface:
    """A synchrony surface: correlations over (lag, window position).

    Parameters
    ----------
    values
        Array of shape ``(n_lags, n_windows)`` of correlations ``r``; NaN marks a cell
        that could not be estimated.
    lags_samples
        Lag of each row, in samples (positive = x leads y).
    window_start_samples
        0-based start sample of each column.
    settings
        The :class:`SurfaceSettings` that produced it.

    Examples
    --------
    >>> import numpy as np
    >>> from syncpipe.surface import wcc_surface
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=60)
    >>> y = np.roll(x, 2)                      # y follows x by 2 samples
    >>> surface = wcc_surface(x, y, hz=1.0, window_samples=20, lag_max_samples=5)
    >>> surface.values.shape
    (11, 31)
    >>> surface.summary().lag_of_peak_sec      # positive: x leads y
    2.0
    """

    values: FloatArray
    lags_samples: IntArray
    window_start_samples: IntArray
    settings: SurfaceSettings
    _hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64)
        lags = np.array(self.lags_samples, dtype=np.int64)
        starts = np.array(self.window_start_samples, dtype=np.int64)
        if values.ndim != 2 or values.shape != (lags.size, starts.size):
            raise ValueError(
                f"values must have shape (n_lags, n_windows) = ({lags.size}, {starts.size}); got {values.shape}"
            )
        finite = values[np.isfinite(values)]
        if finite.size and float(np.max(np.abs(finite))) > 1.0 + 1e-9:
            raise ValueError("correlation values must lie in [-1, 1]")
        for array in (values, lags, starts):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "lags_samples", lags)
        object.__setattr__(self, "window_start_samples", starts)

    # ------------------------------------------------------------------ axes
    @property
    def lags_sec(self) -> FloatArray:
        """Lag of each row in seconds (positive = x leads y)."""
        return self.lags_samples / self.settings.hz

    @property
    def window_start_sec(self) -> FloatArray:
        """Start time of each window in seconds."""
        return self.settings.time_origin_sec + self.window_start_samples / self.settings.hz

    @property
    def window_center_sec(self) -> FloatArray:
        """Centre time of each window in seconds (the window spans ``window_samples`` samples)."""
        half = (self.settings.window_samples - 1) / 2.0
        centre = self.settings.time_origin_sec + (self.window_start_samples + half) / self.settings.hz
        return np.asarray(centre, dtype=np.float64)

    # ------------------------------------------------------------ derivations
    def to_z(self) -> FloatArray:
        """The surface in Fisher-z space (clamped; NaN preserved)."""
        return r_to_z(self.values)

    def lag_profile(self) -> FloatArray:
        """Mean signed z at each lag, averaged over windows (NaN where a lag has no data)."""
        return np.asarray(_nan_reduce(np.nanmean, self.to_z(), axis=1), dtype=np.float64)

    def summary(self) -> SurfaceSummary:
        """Scalar summaries (see :class:`SurfaceSummary` for the exact definitions)."""
        z = self.to_z()
        valid = np.isfinite(z)
        profile = self.lag_profile()
        if np.isfinite(profile).any():
            best = int(np.nanargmax(profile))
            lag_sec, profile_peak = float(self.lags_sec[best]), float(profile[best])
        else:
            lag_sec = profile_peak = float("nan")
        return SurfaceSummary(
            n_windows=int(z.shape[1]),
            n_lags=int(z.shape[0]),
            n_cells_valid=int(valid.sum()),
            fraction_valid=float(valid.mean()) if z.size else float("nan"),
            mean_z=float(_nan_reduce(np.nanmean, z)),
            mean_abs_z=float(_nan_reduce(np.nanmean, np.abs(z))),
            peak_pos_z=float(_nan_reduce(np.nanmean, _nan_reduce(np.nanmax, z, axis=0))),
            peak_abs_z=float(_nan_reduce(np.nanmean, _nan_reduce(np.nanmax, np.abs(z), axis=0))),
            global_peak_r=float(_nan_reduce(np.nanmax, self.values)),
            lag_of_peak_sec=lag_sec,
            lag_profile_peak_z=profile_peak,
        )

    # --------------------------------------------------------- tabular views
    def _cells(self) -> dict[str, Any]:
        """Column arrays for every cell, lag-major (window varies fastest, as in bsync)."""
        n_lags, n_windows = self.values.shape
        lag_idx = np.repeat(np.arange(n_lags), n_windows)
        win_idx = np.tile(np.arange(n_windows), n_lags)
        return {
            "window_index": win_idx,
            "i": self.window_start_samples[win_idx] + 1,  # 1-based start sample, as bsync's `i`
            "window_start_sec": self.window_start_sec[win_idx],
            "window_center_sec": self.window_center_sec[win_idx],
            "tau": self.lags_samples[lag_idx],
            "lag_sec": self.lags_sec[lag_idx],
            "value": self.values[lag_idx, win_idx],
        }

    def tidy(self) -> pd.DataFrame:
        """One row per (window, lag) cell."""
        return pd.DataFrame(self._cells())

    def to_bsync_frame(self) -> pd.DataFrame:
        """The cells in bsync's ``tidy()`` layout: columns ``i``, ``tau`` and the estimator name."""
        cells = self._cells()
        return pd.DataFrame({"i": cells["i"], "tau": cells["tau"], self.settings.estimator: cells["value"]})

    def glance(self) -> pd.DataFrame:
        """One row: key settings, summaries and the content hash (stack rows to compare surfaces)."""
        s, summ = self.settings, self.summary()
        return pd.DataFrame([{
            "estimator": s.estimator, "engine": s.engine, "engine_version": s.engine_version,
            "hz": s.hz, "window_samples": s.window_samples, "window_sec": s.window_sec,
            "window_increment": s.window_increment, "lag_max_samples": s.lag_max_samples,
            "lag_max_sec": s.lag_max_sec, "lag_increment": s.lag_increment, "n_samples": s.n_samples,
            **summ.to_dict(), "surface_hash": self.content_hash(),
        }])

    # ------------------------------------------------------------- provenance
    def content_hash(self) -> str:
        """SHA-256 over the values and settings; identical inputs give an identical hash."""
        if self._hash is None:
            canonical = np.where(np.isnan(self.values), np.nan, self.values).astype("<f8")
            digest = hashlib.sha256()
            digest.update(json.dumps(self.settings.to_dict(), sort_keys=True).encode("utf-8"))
            digest.update(self.lags_samples.astype("<i8").tobytes())
            digest.update(self.window_start_samples.astype("<i8").tobytes())
            digest.update(np.ascontiguousarray(canonical).tobytes())
            object.__setattr__(self, "_hash", digest.hexdigest())
        return str(self._hash)

    # --------------------------------------------------------------- plotting
    def plot(self, ax: Any = None) -> Any:
        """Draw the surface as a heat map (a line for a single-lag surface); returns the axes.

        Requires matplotlib, which is an optional dependency.
        """
        from .plotting import plot_surface

        return plot_surface(self, ax=ax)

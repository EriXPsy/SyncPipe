"""Native WCC surface engine: the reference estimator behind :class:`SyncSurface`.

The grid is bsync-compatible so surfaces from the two engines can be compared cell by
cell: all lags share the same window positions, the series is trimmed by ``lag_max`` at
both ends, and the number of windows is ``floor((n - window + 1 - 2*lag_max) / increment)``.
The correlation itself is SyncPipe's validated, unit-invariant ``sliding_window_wcc``.
"""
from __future__ import annotations


import numpy as np
import numpy.typing as npt

from ..wcc import sliding_window_wcc
from .model import SurfaceSettings, SyncSurface

_WHOLE_SAMPLE_TOL = 1e-6
_MIN_WINDOW_SAMPLES = 3


def _to_samples(name: str, samples: int | None, seconds: float | None, hz: float) -> int | None:
    """Resolve a quantity given either in samples or in seconds (never both)."""
    if samples is not None and seconds is not None:
        raise ValueError(f"give {name} either in samples or in seconds, not both")
    if seconds is None:
        return None if samples is None else int(samples)
    exact = seconds * hz
    rounded = int(round(exact))
    if abs(exact - rounded) > _WHOLE_SAMPLE_TOL:
        raise ValueError(f"{name}_sec={seconds} is not a whole number of samples at {hz} Hz ({exact:.6f})")
    return rounded


def _validated_series(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if xa.size != ya.size:
        raise ValueError(f"x and y must have the same length: {xa.size} vs {ya.size}")
    if np.isinf(xa).any() or np.isinf(ya).any():
        raise ValueError("x and y must not contain infinite values (use NaN for missing samples)")
    return xa, ya


def wcc_surface(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    hz: float,
    window_samples: int | None = None,
    window_sec: float | None = None,
    lag_max_samples: int | None = None,
    lag_max_sec: float | None = None,
    window_increment: int = 1,
    lag_increment: int = 1,
    enforce_susy_limit: bool = True,
    time_origin_sec: float = 0.0,
) -> SyncSurface:
    """Windowed cross-correlation surface of two equally sampled series.

    Parameters
    ----------
    x, y
        Series of equal length; NaN marks a missing sample (pairwise deletion inside a window).
    hz
        Sampling rate in Hz.
    window_samples, window_sec
        Window length (give exactly one). Required.
    lag_max_samples, lag_max_sec
        Largest lag magnitude (give at most one); omitted means ``0``, the single-lag surface.
    window_increment, lag_increment
        Steps in samples.
    enforce_susy_limit
        Reject ``lag_max > window // 2``. Lags beyond half a window make the compared
        windows overlap too little for the correlation to be reliable (bsync enforces the
        same limit); disable it only deliberately.
    time_origin_sec
        Time of the first sample, used for the time axes.

    Returns
    -------
    SyncSurface
        Shape ``(n_lags, n_windows)``; positive lag means ``x`` leads ``y``.

    Raises
    ------
    ValueError
        On inconsistent lengths, non-integral sample counts, a lag range beyond the
        reliability limit, or a series too short for the window and lag range.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.normal(size=50)
    >>> single = wcc_surface(x, x, hz=1.0, window_samples=10)        # zero-lag, one row
    >>> single.values.shape
    (1, 41)
    >>> bool(np.allclose(single.values, 1.0))
    True
    """
    xa, ya = _validated_series(x, y)
    n = int(xa.size)
    window = _to_samples("window", window_samples, window_sec, hz)
    if window is None:
        raise ValueError("a window length is required (window_samples or window_sec)")
    lag_max = _to_samples("lag_max", lag_max_samples, lag_max_sec, hz) or 0
    if window < _MIN_WINDOW_SAMPLES:
        raise ValueError(f"window must span at least {_MIN_WINDOW_SAMPLES} samples, got {window}")
    if lag_max < 0 or window_increment < 1 or lag_increment < 1:
        raise ValueError("lag_max must be >= 0 and the increments must be >= 1")
    if enforce_susy_limit and lag_max > window // 2:
        raise ValueError(
            f"lag_max={lag_max} samples exceeds half the window ({window // 2}); correlations at larger "
            "lags compare windows that overlap too little to be reliable. Shorten the lag range, "
            "lengthen the window, or pass enforce_susy_limit=False deliberately."
        )
    n_windows = (n - window + 1 - 2 * lag_max) // window_increment
    if n_windows < 1:
        raise ValueError(f"series too short: need at least {window + 2 * lag_max} samples for window={window} "
                         f"and lag_max={lag_max}, got {n}")

    lags = np.arange(-lag_max, lag_max + 1, lag_increment, dtype=np.int64)
    starts = lag_max + np.arange(n_windows, dtype=np.int64) * window_increment
    x_core = xa[lag_max:n - lag_max]
    rows = [
        sliding_window_wcc(x_core, ya[lag_max + tau:n - lag_max + tau], window)[::window_increment][:n_windows]
        for tau in lags
    ]
    from .. import __version__

    settings = SurfaceSettings(
        hz=float(hz), window_samples=window, window_increment=int(window_increment),
        lag_max_samples=int(lag_max), lag_increment=int(lag_increment), n_samples=n,
        engine_version=str(__version__), time_origin_sec=float(time_origin_sec),
    )
    return SyncSurface(values=np.vstack(rows), lags_samples=lags, window_start_samples=starts, settings=settings)

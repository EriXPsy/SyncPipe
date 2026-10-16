"""Cell-by-cell comparison of a native surface with another engine's output (e.g. bsync).

The comparison is its own, tested unit so that a disagreement can be attributed: either the
implementation differs, or the comparer is wrong -- never both at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import SyncSurface


@dataclass(frozen=True)
class ConformanceResult:
    """Outcome of :func:`compare_with_bsync_cells`."""

    n_cells_compared: int
    n_cells_only_in_native: int
    n_cells_only_in_other: int
    max_abs_diff: float
    atol: float

    @property
    def ok(self) -> bool:
        """True when both engines cover the same grid and agree within ``atol``."""
        return (
            self.n_cells_compared > 0
            and self.n_cells_only_in_native == 0
            and self.n_cells_only_in_other == 0
            and self.max_abs_diff <= self.atol
        )


def compare_with_bsync_cells(surface: SyncSurface, other: pd.DataFrame, *, atol: float = 1e-8) -> ConformanceResult:
    """Compare ``surface`` with a frame in bsync's ``tidy()`` layout (columns ``i``, ``tau``, ``wcc``).

    Cells are matched on ``(i, tau)`` where ``i`` is the 1-based start sample of the window.
    A cell that is NaN in both engines counts as agreeing; NaN in only one is a disagreement.

    Examples
    --------
    >>> import numpy as np
    >>> from syncpipe.surface import wcc_surface
    >>> x = np.sin(np.arange(60) / 3.0) + 0.1 * np.cos(np.arange(60) * 1.7)
    >>> s = wcc_surface(x, np.roll(x, 1), hz=1.0, window_samples=12, lag_max_samples=3)
    >>> compare_with_bsync_cells(s, s.to_bsync_frame()).ok
    True
    """
    estimator = surface.settings.estimator
    for column in ("i", "tau", estimator):
        if column not in other.columns:
            raise ValueError(f"other engine's frame needs columns i, tau, {estimator}; has {list(other.columns)}")
    mine = surface.to_bsync_frame().set_index(["i", "tau"])[estimator]
    theirs = other.set_index(["i", "tau"])[estimator].astype(float)
    if theirs.index.has_duplicates:
        raise ValueError("other engine's frame has duplicate (i, tau) cells")
    common = mine.index.intersection(theirs.index)
    a, b = mine.loc[common].to_numpy(), theirs.loc[common].to_numpy()
    if bool((np.isnan(a) != np.isnan(b)).any()):
        max_diff = float("inf")                      # NaN in only one engine is a disagreement
    else:
        valid = ~np.isnan(a)
        max_diff = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
    return ConformanceResult(
        n_cells_compared=int(len(common)),
        n_cells_only_in_native=int(len(mine.index.difference(theirs.index))),
        n_cells_only_in_other=int(len(theirs.index.difference(mine.index))),
        max_abs_diff=max_diff,
        atol=atol,
    )

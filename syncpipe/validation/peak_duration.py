"""Duration and block-resampling validation for ``peak_amplitude``.

This module validates an existing descriptor; it does not define or promote a
new feature. ``peak_amplitude`` is an extreme-value statistic, so its expected
value and repeatability depend on observation duration and trace persistence.
The routines below make that dependence visible.

The block-resampling interval is deliberately NOT called a confidence interval
for a population maximum. Ordinary non-parametric bootstrap is inconsistent
for an endpoint maximum because it cannot generate values beyond the observed
sample. It is reported only as a finite-record stability diagnostic.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..dynamic_features import sliding_window_wcc
from ..feature_definitions import compute_peak_amplitude, smoothed_wcc


def _peak(wcc: np.ndarray) -> float:
    """Compute the frozen SSoT ``peak_amplitude`` from a WCC trace."""
    value, _ = compute_peak_amplitude(smoothed_wcc(np.asarray(wcc, dtype=float)))
    return float(value)


def moving_block_peak_stability(
    wcc: np.ndarray,
    *,
    block_length: int,
    n_resamples: int = 1000,
    seed: int = 42,
    interval: Tuple[float, float] = (0.025, 0.975),
) -> Dict[str, object]:
    """Circular moving-block resampling stability for ``peak_amplitude``.

    The resampled trace has the same length as the observed trace. Blocks are
    sampled with replacement from circularly wrapped starts, preserving local
    ordering within each block while varying which portions of the finite
    record dominate the maximum.

    Returns an explicitly labelled ``stability_interval``. It must not be
    reported as a confidence interval for a population maximum.
    """
    x = np.asarray(wcc, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("wcc must be a one-dimensional trace with at least 3 points")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "moving_block_peak_stability requires a finite contiguous WCC trace; "
            "split at discontinuities before resampling"
        )
    if int(block_length) != block_length or not 3 <= block_length <= x.size:
        raise ValueError("block_length must be an integer in [3, len(wcc)]")
    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    lo, hi = map(float, interval)
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("interval must satisfy 0 <= low < high <= 1")

    rng = np.random.default_rng(seed)
    n = x.size
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    peaks = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        peaks[i] = _peak(x[indices])

    observed = _peak(x)
    q_lo, q_hi = np.quantile(peaks, [lo, hi])
    return {
        "descriptor": "peak_amplitude",
        "observed": observed,
        "block_length": int(block_length),
        "n_resamples": int(n_resamples),
        "stability_interval": [float(q_lo), float(q_hi)],
        "resampled_median": float(np.median(peaks)),
        "resampled_mad": float(np.median(np.abs(peaks - np.median(peaks)))),
        "interval_label": "block_resampling_stability_not_population_CI",
        "resampled_values": peaks,
    }


def duration_dependability_curve(
    traces: Mapping[str, np.ndarray],
    *,
    durations: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate duration-matched contiguous-block dependability across dyads.

    For each duration, every trace is divided into complete non-overlapping
    blocks. The first and last blocks are compared within dyad, and their
    rank-order agreement is evaluated across dyads. Individual WCC points are
    never randomly split.

    Returns
    -------
    block_table, summary_table
        Long block-level values and one summary row per duration.
    """
    if not traces:
        raise ValueError("traces must not be empty")
    durations = [int(d) for d in durations]
    if not durations or any(d < 3 for d in durations):
        raise ValueError("durations must contain integers >= 3 WCC points")

    rows = []
    for dyad, trace in traces.items():
        x = np.asarray(trace, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"trace {dyad!r} must be one-dimensional")
        if not np.all(np.isfinite(x)):
            raise ValueError(
                f"trace {dyad!r} is non-finite; pass finite contiguous segments"
            )
        for duration in durations:
            n_blocks = x.size // duration
            for block_index in range(n_blocks):
                start = block_index * duration
                stop = start + duration
                rows.append({
                    "dyad_id": str(dyad),
                    "duration_points": duration,
                    "block_index": block_index,
                    "start_index": start,
                    "peak_amplitude": _peak(x[start:stop]),
                })
    blocks = pd.DataFrame(rows)
    if blocks.empty:
        raise ValueError("no complete blocks are available for the requested durations")

    summaries = []
    for duration, group in blocks.groupby("duration_points", sort=True):
        first_last = []
        for _, dyad_group in group.groupby("dyad_id", sort=True):
            ordered = dyad_group.sort_values("block_index")
            if len(ordered) >= 2:
                first_last.append((
                    float(ordered.iloc[0]["peak_amplitude"]),
                    float(ordered.iloc[-1]["peak_amplitude"]),
                ))
        if first_last:
            first = np.asarray([p[0] for p in first_last])
            last = np.asarray([p[1] for p in first_last])
            median_abs_diff = float(np.median(np.abs(first - last)))
            if len(first_last) >= 3 and np.std(first) > 0 and np.std(last) > 0:
                rho = float(spearmanr(first, last).statistic)
            else:
                rho = float("nan")
        else:
            median_abs_diff = float("nan")
            rho = float("nan")
        summaries.append({
            "duration_points": int(duration),
            "n_dyads": int(group["dyad_id"].nunique()),
            "n_blocks": int(len(group)),
            "n_dyads_with_repeats": int(len(first_last)),
            "median_peak": float(group["peak_amplitude"].median()),
            "first_last_spearman": rho,
            "median_abs_first_last_diff": median_abs_diff,
        })
    return blocks, pd.DataFrame(summaries)


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie in (-1, 1)")
    x = np.empty(n, dtype=float)
    x[0] = rng.normal()
    eps = rng.normal(size=n)
    scale = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + scale * eps[i]
    return x


def simulate_peak_duration_bias(
    *,
    durations: Sequence[int] = (120, 300, 600),
    n_replicates: int = 200,
    window_size: int = 30,
    phi: float = 0.9,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate null peak inflation as independent observation length grows.

    Each replicate generates one maximum-length pair of independent AR(1)
    signals. Shorter durations use prefixes of that same pair, making duration
    comparisons paired by replicate. No interpersonal or shared-input component
    is present.
    """
    durations = sorted({int(d) for d in durations})
    if not durations or durations[0] < window_size + 2:
        raise ValueError("every duration must exceed window_size by at least 2")
    if n_replicates < 2:
        raise ValueError("n_replicates must be >= 2")
    if window_size < 3:
        raise ValueError("window_size must be >= 3")

    rng = np.random.default_rng(seed)
    rows = []
    max_n = durations[-1]
    for replicate in range(int(n_replicates)):
        a = _ar1(rng, max_n, phi)
        b = _ar1(rng, max_n, phi)
        for duration in durations:
            wcc = sliding_window_wcc(a[:duration], b[:duration], window_size)
            rows.append({
                "replicate": replicate,
                "duration_samples": duration,
                "n_wcc_points": int(wcc.size),
                "peak_amplitude": _peak(wcc),
            })
    values = pd.DataFrame(rows)
    summary = (
        values.groupby("duration_samples", as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            n_wcc_points=("n_wcc_points", "first"),
            mean_null_peak=("peak_amplitude", "mean"),
            median_null_peak=("peak_amplitude", "median"),
            q95_null_peak=("peak_amplitude", lambda x: float(np.quantile(x, 0.95))),
        )
    )
    return values, summary


__all__ = [
    "moving_block_peak_stability",
    "duration_dependability_curve",
    "simulate_peak_duration_bias",
]

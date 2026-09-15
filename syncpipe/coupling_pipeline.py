"""Internal coupling-trace computation helpers."""

import numpy as np

from .dynamic_features import sliding_window_wcc
from .wclr import wclr_coupling_trace


def compute_coupling_trace(
    sig_a,
    sig_b,
    hz,
    window_size,
    backend,
    method,
    normalize,
    window_type,
    wclr_max_lag_samples,
    wclr_metric,
):
    """Compute the WCC/WCLR coupling trace without applying discontinuity masks."""
    if backend == "wclr":
        return wclr_coupling_trace(
            sig_a, sig_b,
            window_size=window_size,
            hz=hz,
            max_lag_samples=wclr_max_lag_samples,
            metric=wclr_metric,
        )

    if method == "cumsum" and window_type == "rect":
        from .dynamic_features import _sliding_window_wcc_cumsum

        if normalize:
            a_min, a_max = np.nanmin(sig_a), np.nanmax(sig_a)
            b_min, b_max = np.nanmin(sig_b), np.nanmax(sig_b)
            sig_a = (sig_a - a_min) / max(a_max - a_min, 1e-10)
            sig_b = (sig_b - b_min) / max(b_max - b_min, 1e-10)
        return _sliding_window_wcc_cumsum(sig_a, sig_b, window_size, window_type)

    if normalize:
        a_min, a_max = np.nanmin(sig_a), np.nanmax(sig_a)
        b_min, b_max = np.nanmin(sig_b), np.nanmax(sig_b)
        sig_a = (sig_a - a_min) / max(a_max - a_min, 1e-10)
        sig_b = (sig_b - b_min) / max(b_max - b_min, 1e-10)
    return sliding_window_wcc(
        sig_a, sig_b, window_size, hz=hz, window_type=window_type
    )

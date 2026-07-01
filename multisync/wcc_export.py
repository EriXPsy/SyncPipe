"""
multisync.wcc_export — persist raw WCC traces for morphology analysis.

WCC traces are normally computed in memory (compute_wcc / sliding_window_wcc),
consumed for feature extraction, then DISCARDED. Morphology clustering and
inter-peak-CV both need the trace itself, so this module writes the WCC time
series to a CSV in the exact schema the morphology pipeline expects:

    id, modality, condition, hz, wcc_json
    pce01__RESP__rest1, RESP, rest1, 1.0, "[0.14, 0.21, ...]"

Where to wire it: wherever you currently loop over dyad×modality×condition and
call ``ComputationPipeline.compute_wcc()`` (or ``sliding_window_wcc``) before
extracting features — collect each returned WCC array into a dict keyed by id
and call :func:`export_wcc_traces`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["export_wcc_traces", "wcc_traces_to_frame"]


def wcc_traces_to_frame(
    traces: Iterable[Tuple[str, np.ndarray]],
    *,
    hz: float = 1.0,
    meta: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> pd.DataFrame:
    """Build the long-format WCC-trace DataFrame.

    Parameters
    ----------
    traces : iterable of (id, wcc_array)
        ``id`` is a unique label, conventionally ``f"{dyad}__{modality}__{cond}"``.
    hz : float
        Sampling rate of the WCC series (Hz). Constant across traces here; pass a
        per-trace value via ``meta`` if it varies.
    meta : mapping id -> {"modality": ..., "condition": ..., "hz": ...}, optional
        Optional per-id metadata. If a key parses from the id as
        ``dyad__modality__condition`` it is auto-filled when not in ``meta``.
    """
    rows = []
    for tid, wcc in traces:
        arr = np.asarray(wcc, dtype=float)
        m = dict(meta.get(tid, {})) if meta else {}
        # auto-parse "dyad__modality__condition"
        parts = str(tid).split("__")
        dyad = m.get("dyad", parts[0] if len(parts) >= 1 else "")
        modality = m.get("modality", parts[1] if len(parts) >= 2 else "")
        condition = m.get("condition", parts[2] if len(parts) >= 3 else "")
        row_hz = float(m.get("hz", hz))
        rows.append({
            "id": tid,
            "dyad": dyad,           # promoted to its own column: every downstream
                                    # step (morphology clustering, grouped CV,
                                    # dyad-level bootstrap) groups by dyad and must
                                    # not have to re-parse the id string (which
                                    # breaks if a dyad name contains "__").
            "modality": modality,
            "condition": condition,
            "hz": row_hz,
            "n_samples": int(arr.size),
            "wcc_json": json.dumps([None if not np.isfinite(x) else round(float(x), 6)
                                    for x in arr]),
        })
    return pd.DataFrame(rows)


def export_wcc_traces(
    traces: Iterable[Tuple[str, np.ndarray]],
    out_path: str | Path,
    *,
    hz: float = 1.0,
    meta: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Path:
    """Write WCC traces to ``out_path`` (CSV) and return the path.

    The output is directly consumable by the morphology pipeline:
    ``python scripts/morphology_analysis.py --traces-csv <out_path> --trace-col wcc_json``
    """
    df = wcc_traces_to_frame(traces, hz=hz, meta=meta)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

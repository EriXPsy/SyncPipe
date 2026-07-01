"""Gordon (Mayo & Gordon, 2025) dataset converter for SyncPipe.

Source
------
Mayo, O., & Gordon, I. (2025). Contextual pulls for synchrony and
segregation: An empirical test of a novel theoretical framework.
*American Psychologist*. https://doi.org/10.1037/amp0001659
OSF: 349su

Data layout (raw)
-----------------
<data_root>/behavior data/<p1>_<p2>/expN.csv     (N = 1..4)

Each ``expN.csv`` has 6 columns:
    time_p1, R_p1, theta_p1, time_p2, R_p2, theta_p2

The four expN conditions correspond to a 2x2 design of contextual pulls
(competition vs cooperation, etc.; consult the paper for the exact mapping
to the four cells).  This converter is condition-agnostic: it loads each
exp as a separate dyad-condition unit and lets downstream analysis label
them.

What this converter does
------------------------
1. Walks ``<data_root>/behavior data/`` for every ``<p1>_<p2>/`` directory.
2. For each of the four ``expN.csv`` (N=1..4), loads the time series,
   derives a single continuous "motion intensity" signal per person by
   combining ``R`` (radial distance to a target) and ``theta``
   (angular position) into angular velocity-like and radial velocity-like
   channels.  Two channels per person ensures the downstream SyncPipe
   analyzer can still build modality-pair edges if desired, while a
   simpler one-channel default (`motion_intensity = sqrt(dR^2 + (R*dtheta)^2)`)
   is also provided.
3. Returns a list of ``GordonDyadCondition`` records, each containing the
   per-person time-aligned DataFrames plus dyad / condition metadata.

Why this design
---------------
* SyncPipe's ``Dyad`` object takes ``person_a`` and ``person_b``
  DataFrames with a "time" column plus modality columns.  We respect
  that contract.
* Conditions are kept as separate "dyad_id = <pair>__exp<N>" entries so
  ``batch_analyze`` will treat them as independent observation units.
  Group analysis can then segment by condition.

Usage
-----
>>> from multisync.realtest.gordon_2025 import load_gordon_dataset
>>> records = load_gordon_dataset("/path/to/gordon_2025")
>>> for rec in records[:2]:
...     print(rec.dyad_id, rec.person_a.shape, rec.person_b.shape)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default sampling rate inferred from Mayo & Gordon (2025) (Hz).
#: The raw CSVs already contain explicit ``time_p1`` / ``time_p2`` columns,
#: so resampling rate is read from data, not assumed.
DEFAULT_TARGET_HZ: float = 10.0

#: The four condition CSVs we expect inside each dyad directory.
EXPECTED_CONDITION_FILES: Sequence[str] = (
    "exp1.csv", "exp2.csv", "exp3.csv", "exp4.csv",
)

#: Column names in the raw CSV.
RAW_COLUMNS: Sequence[str] = (
    "time_p1", "R_p1", "theta_p1",
    "time_p2", "R_p2", "theta_p2",
)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GordonDyadCondition:
    """One (dyad, condition) record.

    Attributes
    ----------
    dyad_id : str
        Composite identifier ``"<p1>_<p2>__exp<N>"``.
    pair_label : str
        Original directory name (``"<p1>_<p2>"``).
    condition : str
        ``"exp1"`` ... ``"exp4"``.
    person_a, person_b : pandas.DataFrame
        Columns: ``time`` (seconds, common axis), ``motion_intensity``
        (scalar), ``R``, ``theta``.  Both persons share the same
        time grid (resampled to ``target_hz``).
    target_hz : float
        Sampling rate after resampling.
    n_samples : int
        Number of samples on the shared time grid.
    duration_sec : float
        Total trial duration in seconds.
    meta : dict
        Free-form metadata (e.g. raw file paths).
    """
    dyad_id: str
    pair_label: str
    condition: str
    person_a: pd.DataFrame
    person_b: pd.DataFrame
    target_hz: float
    n_samples: int
    duration_sec: float
    meta: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap_unwrap_theta(theta: np.ndarray) -> np.ndarray:
    """Unwrap angular trace so velocity is well-defined.

    Raw ``theta`` is in radians and bounded by [-pi, pi].  We unwrap to
    avoid spurious 2*pi jumps confusing the time-derivative.
    """
    return np.unwrap(np.asarray(theta, dtype=float))


def _build_person_channels(
    t: np.ndarray,
    R: np.ndarray,
    theta: np.ndarray,
) -> pd.DataFrame:
    """Build a per-person DataFrame with derived motion channels.

    Channels
    --------
    R : float
        Radial distance to target.
    theta : float
        Angular position (unwrapped).
    motion_intensity : float
        ``sqrt(dR/dt^2 + (R * dtheta/dt)^2)``.  This is the magnitude of
        the velocity vector in polar coordinates and is the single most
        informative scalar channel for synchrony analysis in this dataset.
    """
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    theta = _wrap_unwrap_theta(theta)

    # Time-derivative via central differences; fall back to forward at
    # edges.  np.gradient handles non-uniform spacing if t is monotonic.
    dt = np.gradient(t)
    dt_safe = np.where(np.abs(dt) < 1e-9, np.nan, dt)
    dR = np.gradient(R) / dt_safe
    dtheta = np.gradient(theta) / dt_safe

    motion_intensity = np.sqrt(dR ** 2 + (R * dtheta) ** 2)

    return pd.DataFrame({
        "time": t,
        "R": R,
        "theta": theta,
        "motion_intensity": motion_intensity,
    })


def _resample_to_common_grid(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    target_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample both persons onto a shared uniform grid.

    Uses ``pandas.DataFrame.interpolate`` with linear method.  This is
    appropriate because the raw motion traces are smooth and densely
    sampled relative to ``target_hz`` (typical raw rate ~30+ Hz).
    """
    t_min = max(df_a["time"].iloc[0], df_b["time"].iloc[0])
    t_max = min(df_a["time"].iloc[-1], df_b["time"].iloc[-1])
    if t_max <= t_min:
        raise ValueError(
            "Person A and B time ranges do not overlap "
            f"(A: {df_a['time'].iloc[0]:.2f}-{df_a['time'].iloc[-1]:.2f}, "
            f"B: {df_b['time'].iloc[0]:.2f}-{df_b['time'].iloc[-1]:.2f})."
        )

    dt = 1.0 / float(target_hz)
    grid = np.arange(t_min, t_max + 0.5 * dt, dt)

    def _interp(df: pd.DataFrame) -> pd.DataFrame:
        out: Dict[str, np.ndarray] = {"time": grid}
        for col in ("R", "theta", "motion_intensity"):
            out[col] = np.interp(grid, df["time"].values, df[col].values)
        return pd.DataFrame(out)

    return _interp(df_a), _interp(df_b)


def _load_single_condition(
    csv_path: Path,
    pair_label: str,
    target_hz: float,
) -> Optional[GordonDyadCondition]:
    """Load one ``expN.csv`` into a GordonDyadCondition.

    Returns None on failure (with a logged warning) so a single broken
    condition doesn't abort the whole dataset load.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        # OSF export note: some Gordon CSVs have no header row, so pandas
        # treats the first data row as column names. Fall back to a 6-column
        # headerless read before giving up.
        try:
            df = pd.read_csv(csv_path, header=None, names=list(RAW_COLUMNS))
        except Exception as exc:
            logger.warning(
                "%s missing required columns %s and headerless fallback failed: %s",
                csv_path, missing, exc,
            )
            return None

        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(
                "%s missing required columns %s; skipping.",
                csv_path, missing,
            )
            return None

    # Coerce all raw columns to numeric so malformed rows become NaN and are
    # discarded by the validity filter below.
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where either person has NaN in any of its key columns.
    df = df.dropna(subset=list(RAW_COLUMNS)).reset_index(drop=True)
    if len(df) < 30:
        logger.warning(
            "%s has only %d valid rows (<30); skipping.",
            csv_path, len(df),
        )
        return None

    df_a = _build_person_channels(
        t=df["time_p1"].values,
        R=df["R_p1"].values,
        theta=df["theta_p1"].values,
    )
    df_b = _build_person_channels(
        t=df["time_p2"].values,
        R=df["R_p2"].values,
        theta=df["theta_p2"].values,
    )

    a_resampled, b_resampled = _resample_to_common_grid(
        df_a, df_b, target_hz=target_hz,
    )

    condition = csv_path.stem  # "exp1" etc.
    dyad_id = f"{pair_label}__{condition}"

    return GordonDyadCondition(
        dyad_id=dyad_id,
        pair_label=pair_label,
        condition=condition,
        person_a=a_resampled,
        person_b=b_resampled,
        target_hz=float(target_hz),
        n_samples=int(len(a_resampled)),
        duration_sec=float(
            a_resampled["time"].iloc[-1] - a_resampled["time"].iloc[0]
        ),
        meta={
            "raw_path": str(csv_path),
            "n_raw_rows": int(len(df)),
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gordon_dataset(
    data_root: str | Path,
    target_hz: float = DEFAULT_TARGET_HZ,
    pair_whitelist: Optional[Sequence[str]] = None,
    condition_whitelist: Optional[Sequence[str]] = None,
) -> List[GordonDyadCondition]:
    """Load all dyad-condition records from the Gordon (Mayo & Gordon, 2025) dataset.

    Parameters
    ----------
    data_root : str or Path
        Path to the dataset root.  Expected to contain a ``behavior data/``
        sub-directory (note the space in the original folder name).  If
        that sub-directory is missing, ``data_root`` itself is searched
        for ``<p1>_<p2>/`` directories.
    target_hz : float
        Resampling rate (default 10 Hz).
    pair_whitelist : sequence of str, optional
        If given, only load dyad directories whose name is in this set.
    condition_whitelist : sequence of str, optional
        If given, only load conditions in this set (subset of
        {"exp1", "exp2", "exp3", "exp4"}).

    Returns
    -------
    list of GordonDyadCondition
        One record per (dyad, condition).  Sorted by ``dyad_id``.

    Notes
    -----
    Every loadable record is returned; failures (missing columns,
    unreadable CSVs, too few rows) are logged as warnings and skipped.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Gordon data_root does not exist: {root}")

    behavior_root = root / "behavior data"
    if not behavior_root.exists():
        # Fall back to data_root itself if no "behavior data" subfolder.
        behavior_root = root

    records: List[GordonDyadCondition] = []
    pair_dirs = sorted(p for p in behavior_root.iterdir() if p.is_dir())

    if not pair_dirs:
        raise RuntimeError(
            f"No dyad directories found in {behavior_root}.  "
            "Expected <p1>_<p2>/ sub-folders."
        )

    for pair_dir in pair_dirs:
        pair_label = pair_dir.name
        if pair_whitelist is not None and pair_label not in pair_whitelist:
            continue

        condition_files = sorted(
            p for p in pair_dir.iterdir()
            if p.is_file() and p.name in EXPECTED_CONDITION_FILES
        )
        if not condition_files:
            logger.info("No expN.csv found under %s; skipping.", pair_dir)
            continue

        for csv_path in condition_files:
            cond_name = csv_path.stem
            if condition_whitelist is not None and cond_name not in condition_whitelist:
                continue
            rec = _load_single_condition(
                csv_path=csv_path,
                pair_label=pair_label,
                target_hz=target_hz,
            )
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.dyad_id)
    logger.info(
        "Loaded %d (dyad, condition) records from %d pair directories.",
        len(records), len(pair_dirs),
    )
    return records


def gordon_record_to_multisync_dyad(
    rec: GordonDyadCondition,
    channels: Sequence[str] = ("motion_intensity",),
):
    """Convert a GordonDyadCondition into a SyncPipe ``Dyad`` object.

    Parameters
    ----------
    rec : GordonDyadCondition
    channels : sequence of str
        Which derived channels to expose as modalities to SyncPipe.
        Default: only ``motion_intensity``.  Each channel is registered
        TWICE (once per person) using the suffix convention ``<ch>_a`` /
        ``<ch>_b``, so SyncPipe's ``extract_features_all_pairs`` will
        compute the cross-person WCC at modality position
        ``<ch>_a__<ch>_b``.

    Returns
    -------
    multisync.core.Dyad
        Ready for ``DynamicAnalyzer.fit_transform`` after ``align`` +
        ``zscore``.

    Notes
    -----
    SyncPipe's canonical Dyad API treats each *person + channel* as a
    distinct modality.  For a single-channel dyadic dataset like Gordon
    this gives exactly two modalities (``motion_intensity_a`` and
    ``_b``) and one cross-person edge.  Multi-channel mode (e.g.
    ``channels=("motion_intensity", "R")``) yields four modalities and
    six edges, of which only the cross-person ones are scientifically
    interpretable.
    """
    from multisync.core import Dyad  # local import keeps converter light

    modalities: Dict[str, pd.DataFrame] = {}
    for ch in channels:
        modalities[f"{ch}_a"] = rec.person_a[["time", ch]].rename(
            columns={ch: "value"}
        )
        modalities[f"{ch}_b"] = rec.person_b[["time", ch]].rename(
            columns={ch: "value"}
        )

    dyad = Dyad(
        hz=rec.target_hz,
        dyad_id=rec.dyad_id,
        **modalities,
    )
    return dyad


__all__ = [
    "GordonDyadCondition",
    "load_gordon_dataset",
    "gordon_record_to_multisync_dyad",
    "DEFAULT_TARGET_HZ",
    "EXPECTED_CONDITION_FILES",
    "RAW_COLUMNS",
]

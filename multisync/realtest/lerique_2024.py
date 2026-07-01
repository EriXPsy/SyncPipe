"""Lerique-47n3p dataset converter for SyncPipe.

Source
------
Lerique (2024). Multimodal peripheral physiology dyad dataset.
OSF: 47n3p

Experimental paradigm (confirmed 2026-05-24 from Participant Instruction PDF)
-----------------------------------------------------------------------------
- Two participants (P1, P2) in separate rooms, doing a shared-virtual-disc
  partner-finding task. Eyes closed during trials; haptic vibration feedback.
- Structure:
    [Rest1 180s] -> [3 training trials, dropped] ->
    [Trial 1..6][Rest2 180s] ->
    [Trial 7..12][Rest3 180s] ->
    [Trial 13..18][Rest4 180s]
- Each trial = 60s active dyadic interaction.
- Rest1 is the only pre-task baseline; Rest2/3/4 are post-block recovery
  (carry-over influenced).

Data layout (raw)
-----------------
<data_root>/
+- ECG/
|  +- pceXX{YYMMDD}/
|     +- pceXX_P{1|2}_Rest{1..4}.mat   # 4 segments x 180 s = 720 s/person
|     +- pceXX_P{1|2}_Trial{1..18}.mat # 18 segments x 60 s = 1080 s/person
+- EDA/    # same structure
+- RESP/   # same structure

Each .mat file contains a single variable named ``pce_P{1|2}_{Rest|Trial}``
with shape ``(1, N)`` and dtype ``float32``.

Sampling rate
-------------
- Rest .mat: N = 180000 -> 180000 / 180 s = 1000 Hz
- Trial .mat: N =  60000 ->  60000 /  60 s = 1000 Hz
- ``Fs = 1000 Hz`` confirmed against PDF (segment durations 3 min / 1 min).

Dyad availability (inventory 2026-05-24)
----------------------------------------
- 31 dyad directories total (pce01-pce30 + pce32; no pce31)
- 27 complete (P1 + P2, all three modalities, all segments)
- 3 dyads missing ECG P2 (pce01, pce24, pce26) -> EDA/RESP still usable
- 1 dyad missing ECG P1 (pce32) -> EDA/RESP P2 still usable

Condition units emitted by the loader
-------------------------------------
Per (dyad, modality), three condition units are emitted:

1. ``rest1``           : single Rest1 segment (180 s).
                          Pre-task baseline. Main analysis baseline.
2. ``rest_postblock``  : concat(Rest2, Rest3, Rest4) (540 s, 2 boundaries).
                          Sensitivity analysis baseline.
3. ``trials_concat``   : concat(Trial1..Trial18) (1080 s, 17 boundaries).
                          Task condition (both main and sensitivity).

Pre-registration: see ``docs/PRE_REGISTRATION_PILOTS.md`` sections 1.3,
1.3a, 1.3b for design / asymmetry mitigation / alignment caveats.

Status
------
2026-05-25: preprocessing pipeline lands (pre-reg §1.4 protocol).
``_preprocess_{ecg,eda,resp}`` produce ``(signal, mask)`` tuples on the
``target_fs`` grid; mask is propagated from raw segment-boundary mask.
ECG path uses ``neurokit2.ecg_peaks`` (added to pyproject dependencies);
EDA / RESP use scipy Butterworth (zero-phase) + resample_poly. The
``MIN_DURATION_SEC = 60 s`` floor is enforced before preprocessing so
sub-floor records never reach the filter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (pre-registration locked, corrected 2026-05-24 post-PDF)
# ---------------------------------------------------------------------------

#: Raw sampling rate (Hz). Confirmed from Participant Instruction PDF
#: ("Each Resting period lasts for three minutes" + "Each trial lasts
#: 1 minute") combined with .mat sample counts (180000 / 180 = 1000;
#: 60000 / 60 = 1000).
RAW_FS_HZ: float = 1000.0

#: SyncPipe v3 WCC target rate (DECISION-locked at the project level).
TARGET_FS_HZ: float = 1.0

#: Modalities provided by the dataset.
MODALITIES: Tuple[str, ...] = ("ECG", "EDA", "RESP")

#: Per-segment durations (seconds), confirmed from PDF.
REST_SEGMENT_DURATION_SEC: float = 180.0   # "three minutes"
TRIAL_SEGMENT_DURATION_SEC: float = 60.0   # "1 minute"

#: Expected raw sample counts per segment (at RAW_FS_HZ).
REST_SEGMENT_SAMPLES: int = int(REST_SEGMENT_DURATION_SEC * RAW_FS_HZ)   # 180000
TRIAL_SEGMENT_SAMPLES: int = int(TRIAL_SEGMENT_DURATION_SEC * RAW_FS_HZ)  # 60000

#: Total segment counts per condition class.
REST_SEGMENT_COUNT: int = 4
TRIAL_SEGMENT_COUNT: int = 18

#: Pre-registered hard floor: any condition unit shorter than this (in
#: seconds, measured on the RAW grid) is dropped from analysis.
#:
#: Rationale (locked by pre-registration §1.4 "Rest1 duration sanity"):
#: SyncPipe v3 sliding-window WCC uses 30 s windows at 10 s step. The
#: minimum trace length to estimate dyad-level scalar features with
#: any reasonable variance is ~4 WCC windows = 30 s + 3 * 10 s = 60 s.
#: Anything shorter cannot support `peak_amplitude` (needs > 1 window)
#: or `dwell_time` (fraction needs > 1 window) in any meaningful way.
#:
#: This floor admits all observed Rest1 segments (shortest ~159 s); a
#: stricter floor would amount to retroactive selection.
MIN_DURATION_SEC: float = 60.0

#: Condition units emitted per (dyad, modality). See module docstring.
CONDITION_UNITS: Tuple[str, ...] = (
    "rest1",            # Main baseline (single Rest1 segment)
    "rest_postblock",   # Sensitivity baseline (Rest2+3+4 concat)
    "trials_concat",    # Task condition (Trial1..18 concat)
)

#: Regex matching .mat filenames: pce<NN>_P<1|2>_<Rest|Trial><K>.mat
_FILENAME_RE = re.compile(
    r"^pce(?P<dyad>\d{2})_P(?P<person>[12])_"
    r"(?P<cond>Rest|Trial)(?P<seg>\d{1,2})\.mat$"
)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeriqueDyadCondition:
    """One (dyad, modality, condition_unit) record.

    Attributes
    ----------
    dyad_id : str
        Composite identifier ``"pce<NN>__<modality>__<condition_unit>"``.
    dyad_label : str
        Bare dyad code, e.g. ``"pce02"``.
    modality : str
        One of ``MODALITIES``.
    condition : str
        One of ``CONDITION_UNITS`` (``"rest1"``, ``"rest_postblock"``, or
        ``"trials_concat"``). See module docstring for definitions.
    person_a, person_b : pandas.DataFrame
        Columns: ``time`` (seconds, common axis), ``value`` (scalar signal).
        When a person is missing for a given (modality, condition_unit),
        that DataFrame is None and the record is still returned with
        ``incomplete=True`` so the downstream layer can decide whether
        to drop it.
    target_hz : float
        Sampling rate of the emitted DataFrames (RAW_FS_HZ when
        ``preprocess=False``, TARGET_FS_HZ when ``preprocess=True``).
    n_samples : int
        Number of samples on the shared time grid.
    duration_sec : float
        Total trace duration in seconds.
    incomplete : bool
        True if either person is missing OR P1/P2 lengths mismatch.
    discontinuity_mask : numpy.ndarray of bool
        Length-``n_samples`` mask, True at samples internal to a single
        segment, False at the *first* sample of each post-first segment
        (so downstream WCC can skip cross-boundary windows).
    meta : dict
        Free-form metadata (raw file paths, segment indices, alignment
        sanity flag, raw_fs_hz, etc.).
    """
    dyad_id: str
    dyad_label: str
    modality: str
    condition: str
    person_a: Optional[pd.DataFrame]
    person_b: Optional[pd.DataFrame]
    target_hz: float
    n_samples: int
    duration_sec: float
    incomplete: bool
    discontinuity_mask: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_filename(name: str) -> Optional[Dict[str, str]]:
    """Parse a Lerique .mat filename into its components.

    Returns
    -------
    dict or None
        Keys: ``dyad`` (e.g. ``"02"``), ``person`` (``"1"`` or ``"2"``),
        ``cond`` (``"Rest"`` or ``"Trial"``), ``seg`` (segment index
        string, e.g. ``"1"`` or ``"18"``). Returns None on no match.
    """
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.groupdict()


# ---------------------------------------------------------------------------
# Raw segment loading
# ---------------------------------------------------------------------------

def _load_mat_segment(path: Path) -> np.ndarray:
    """Load a single .mat file and return its 1-D signal vector.

    The variable inside is named ``pce_P{1|2}_{Rest|Trial}`` (no segment
    index). We dynamically read the first non-dunder key rather than
    hard-coding the variable name, so future Lerique releases with
    renamed variables still work.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .mat file.

    Returns
    -------
    numpy.ndarray, shape (N,), dtype float32
        The 1-D signal, squeezed from raw shape ``(1, N)``.

    Raises
    ------
    ValueError
        If the .mat file has zero or multiple non-dunder variables, or
        if the squeezed array is not 1-D.
    """
    from scipy.io import loadmat  # local import: scipy is optional

    mat = loadmat(str(path))
    payload_keys = [k for k in mat if not k.startswith("__")]
    if len(payload_keys) != 1:
        raise ValueError(
            f"{path.name}: expected exactly 1 non-dunder variable, "
            f"got {len(payload_keys)}: {payload_keys}"
        )
    arr = np.asarray(mat[payload_keys[0]]).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"{path.name}: expected 1-D signal after squeeze, "
            f"got shape {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Preprocessing (per pre-registration §1.4)
# ---------------------------------------------------------------------------

# IBI outlier window (seconds). Beats with IBI outside [_IBI_MIN, _IBI_MAX]
# are treated as artifacts and linearly interpolated. Range chosen to
# accept HR ~ [30, 200] bpm.
_IBI_MIN_SEC: float = 0.3
_IBI_MAX_SEC: float = 2.0

# Filter bands (Hz), locked per pre-reg §1.4. Order = 4 (Butterworth)
# applied via zero-phase filtfilt to avoid temporal distortion that
# would corrupt WCC alignment.
_ECG_BAND_HZ: Tuple[float, float] = (5.0, 20.0)
_EDA_BAND_HZ: Tuple[float, float] = (0.05, 5.0)
_RESP_BAND_HZ: Tuple[float, float] = (0.1, 1.0)
_FILTER_ORDER: int = 4


def _bandpass_filter(
    signal: np.ndarray,
    raw_fs: float,
    band: Tuple[float, float],
    order: int = _FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (SOS form for stability).

    Returns filtered signal with the same length as ``signal``.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * raw_fs
    lo, hi = band
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32, copy=False)


def _resample_mask_to_target(
    raw_mask: np.ndarray,
    raw_fs: float,
    target_fs: float,
    n_target: int,
) -> np.ndarray:
    """Propagate a raw-rate boundary mask down to the target-rate grid.

    For each target sample at time ``t = i / target_fs`` we look at
    the corresponding raw window ``[t - 0.5/target_fs, t + 0.5/target_fs)``
    and mark the target sample False if **any** raw sample in that
    window was False (i.e., any segment boundary fell in the window).
    """
    if n_target <= 0:
        return np.zeros(0, dtype=bool)
    if raw_mask.size == 0 or raw_mask.all():
        return np.ones(n_target, dtype=bool)

    out = np.ones(n_target, dtype=bool)
    half = 0.5 / target_fs
    raw_t = np.arange(raw_mask.size, dtype=np.float64) / raw_fs
    # Index of every False position in raw grid
    false_pos = np.where(~raw_mask)[0]
    for idx in false_pos:
        t_false = raw_t[idx]
        # Which target sample's window does this fall into?
        target_idx = int(round(t_false * target_fs))
        if 0 <= target_idx < n_target:
            out[target_idx] = False
        # Also guard the neighbour if the boundary sits within ``half`` sec
        prev_idx = target_idx - 1
        if 0 <= prev_idx < n_target and (t_false - prev_idx / target_fs) <= half:
            out[prev_idx] = False
    return out


def _resample_to_target(
    signal: np.ndarray,
    raw_fs: float,
    target_fs: float,
) -> np.ndarray:
    """Decimate / resample ``signal`` from ``raw_fs`` to ``target_fs``.

    Uses scipy.signal.resample_poly with integer up/down ratios when
    possible; falls back to numpy interp on a uniform grid otherwise.
    """
    from math import gcd
    from scipy.signal import resample_poly

    if raw_fs == target_fs:
        return signal.astype(np.float32, copy=False)

    # Integer ratio path
    if float(raw_fs).is_integer() and float(target_fs).is_integer():
        up = int(target_fs)
        down = int(raw_fs)
        g = gcd(up, down)
        up //= g
        down //= g
        out = resample_poly(signal, up, down)
        return out.astype(np.float32, copy=False)

    # Generic uniform-grid path
    n_target = int(np.floor(len(signal) * target_fs / raw_fs))
    if n_target <= 1:
        return np.zeros(max(n_target, 0), dtype=np.float32)
    t_raw = np.arange(len(signal), dtype=np.float64) / raw_fs
    t_tgt = np.arange(n_target, dtype=np.float64) / target_fs
    out = np.interp(t_tgt, t_raw, signal)
    return out.astype(np.float32, copy=False)


def _interp_outlier_ibi(
    ibi_sec: np.ndarray,
    rpeak_t_sec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace IBI values outside the physiological window via linear
    interpolation along ``rpeak_t_sec`` (midpoints between beats).

    Returns
    -------
    (ibi_clean, ibi_midpoint_t)
        Same length as input; midpoint timestamps for each IBI.
    """
    if ibi_sec.size == 0:
        return ibi_sec.copy(), np.zeros(0, dtype=np.float64)

    ibi_mid_t = 0.5 * (rpeak_t_sec[1:] + rpeak_t_sec[:-1])
    good = (ibi_sec >= _IBI_MIN_SEC) & (ibi_sec <= _IBI_MAX_SEC)

    if good.all():
        return ibi_sec.astype(np.float64, copy=True), ibi_mid_t

    if not good.any():
        # All beats flagged — bail out with a constant placeholder
        # (1.0 s = 60 bpm); a fully unusable dyad will hit downstream
        # quality gates anyway.
        logger.warning(
            "IBI artifact rate = 100%% over %d beats; using fallback "
            "constant 1.0s. This trace likely needs exclusion downstream.",
            len(ibi_sec),
        )
        return np.full_like(ibi_sec, 1.0, dtype=np.float64), ibi_mid_t

    clean = ibi_sec.astype(np.float64, copy=True)
    bad_idx = np.where(~good)[0]
    good_idx = np.where(good)[0]
    clean[bad_idx] = np.interp(ibi_mid_t[bad_idx], ibi_mid_t[good_idx], clean[good_idx])
    return clean, ibi_mid_t


def _preprocess_ecg(
    raw: np.ndarray,
    raw_fs: float,
    target_fs: float,
    boundary_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ECG → IBI trace resampled to ``target_fs``.

    Pipeline (pre-reg §1.4):
        1. Bandpass 5–20 Hz, Butterworth order 4, zero-phase (sosfiltfilt)
        2. R-peak detection via ``neurokit2.ecg_peaks``
        3. IBI = diff(R-peak times); outliers (<0.3s, >2.0s) linearly
           interpolated along midpoint timestamps
        4. Resample IBI series onto the uniform ``target_fs`` grid
           via linear interpolation (IBI sampling is non-uniform, so
           resample_poly is not applicable here)
        5. Boundary mask propagated from raw-grid to target-grid

    Note: this departs from Bizzego 2020 by omitting the 0.04 Hz
    lowpass on the IBI series (DECISION-locked at project level).
    """
    import neurokit2 as nk

    filtered = _bandpass_filter(raw, raw_fs, _ECG_BAND_HZ)
    # neurokit2 expects 1-D float; sampling_rate must be int
    _, info = nk.ecg_peaks(filtered, sampling_rate=int(round(raw_fs)))
    rpeaks_idx = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)

    if rpeaks_idx.size < 2:
        logger.warning(
            "ECG: only %d R-peaks detected in %.1fs of trace; emitting "
            "zero-mean placeholder.",
            int(rpeaks_idx.size), len(raw) / raw_fs,
        )
        n_target = int(np.floor(len(raw) * target_fs / raw_fs))
        sig = np.zeros(n_target, dtype=np.float32)
        mask = (
            _resample_mask_to_target(boundary_mask, raw_fs, target_fs, n_target)
            if boundary_mask is not None
            else np.ones(n_target, dtype=bool)
        )
        return sig, mask

    rpeak_t = rpeaks_idx.astype(np.float64) / raw_fs
    ibi_sec = np.diff(rpeak_t)
    ibi_clean, ibi_mid_t = _interp_outlier_ibi(ibi_sec, rpeak_t)

    duration_sec = len(raw) / raw_fs
    n_target = int(np.floor(duration_sec * target_fs))
    if n_target <= 1:
        return np.zeros(max(n_target, 0), dtype=np.float32), np.zeros(
            max(n_target, 0), dtype=bool,
        )
    t_target = np.arange(n_target, dtype=np.float64) / target_fs
    # Extrapolate constantly at the edges (np.interp default behaviour)
    ibi_resampled = np.interp(t_target, ibi_mid_t, ibi_clean).astype(np.float32)

    if boundary_mask is not None:
        mask_out = _resample_mask_to_target(boundary_mask, raw_fs, target_fs, n_target)
    else:
        mask_out = np.ones(n_target, dtype=bool)

    return ibi_resampled, mask_out


def _preprocess_eda(
    raw: np.ndarray,
    raw_fs: float,
    target_fs: float,
    boundary_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """EDA → bandpass-filtered SCL trace at ``target_fs``.

    Pipeline (pre-reg §1.4):
        1. Bandpass 0.05–5 Hz, Butterworth order 4, zero-phase
        2. Resample to ``target_fs`` via scipy.signal.resample_poly
        3. Boundary mask propagated from raw-grid to target-grid

    No SCR onset decomposition — keep continuous SCL trace so WCC
    captures slow co-modulation directly.
    """
    filtered = _bandpass_filter(raw, raw_fs, _EDA_BAND_HZ)
    sig_out = _resample_to_target(filtered, raw_fs, target_fs)
    n_target = len(sig_out)
    if boundary_mask is not None:
        mask_out = _resample_mask_to_target(boundary_mask, raw_fs, target_fs, n_target)
    else:
        mask_out = np.ones(n_target, dtype=bool)
    return sig_out, mask_out


def _preprocess_resp(
    raw: np.ndarray,
    raw_fs: float,
    target_fs: float,
    boundary_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """RESP → bandpass-filtered respiratory waveform at ``target_fs``.

    Pipeline (pre-reg §1.4):
        1. Bandpass 0.1–1 Hz, Butterworth order 4, zero-phase
        2. Resample to ``target_fs`` via scipy.signal.resample_poly
        3. Boundary mask propagated from raw-grid to target-grid

    Respiratory rate is **not** extracted — WCC operates on the
    bandpassed waveform so it captures both phase and amplitude
    coherence naturally.
    """
    filtered = _bandpass_filter(raw, raw_fs, _RESP_BAND_HZ)
    sig_out = _resample_to_target(filtered, raw_fs, target_fs)
    n_target = len(sig_out)
    if boundary_mask is not None:
        mask_out = _resample_mask_to_target(boundary_mask, raw_fs, target_fs, n_target)
    else:
        mask_out = np.ones(n_target, dtype=bool)
    return sig_out, mask_out


_PREPROC_DISPATCH = {
    "ECG": _preprocess_ecg,
    "EDA": _preprocess_eda,
    "RESP": _preprocess_resp,
}


# ---------------------------------------------------------------------------
# Per-(dyad, modality, condition, person) assembly
# ---------------------------------------------------------------------------

def _collect_segments_for_person(
    pce_subdir: Path,
    dyad_label: str,
    modality_name: str,
    person: str,
    cond_class: str,
    seg_indices: Sequence[int],
) -> Tuple[Optional[np.ndarray], List[Path], np.ndarray]:
    """Load and concatenate a specific list of segments for one person.

    Parameters
    ----------
    pce_subdir : Path
        Per-dyad directory like ``<root>/ECG/pce02230809/``.
    dyad_label : str
        Bare dyad code (used only for logging).
    modality_name : str
        Bare modality name (used only for logging).
    person : str
        ``"1"`` or ``"2"``.
    cond_class : str
        ``"Rest"`` or ``"Trial"`` (the .mat filename token).
    seg_indices : sequence of int
        Specific segment indices to concatenate, in the order they
        should be concatenated. E.g. ``[1]`` for ``rest1``,
        ``[2, 3, 4]`` for ``rest_postblock``, ``range(1, 19)`` for
        ``trials_concat``.

    Returns
    -------
    raw_concat : np.ndarray or None
        Concatenated raw signal at RAW_FS_HZ; None if all segments missing.
    segment_paths : list of Path
        Paths to .mat files that contributed, in concat order.
    boundary_mask : np.ndarray of bool
        Same length as ``raw_concat``. False only at the first sample of
        each segment that was concatenated after the first segment;
        True everywhere else. Empty array if ``raw_concat`` is None.
    """
    available: Dict[int, Path] = {}
    for f in pce_subdir.iterdir():
        meta = _parse_filename(f.name)
        if meta is None:
            continue
        if meta["person"] != person or meta["cond"] != cond_class:
            continue
        available[int(meta["seg"])] = f

    requested_paths: List[Path] = []
    missing_idx: List[int] = []
    for idx in seg_indices:
        if idx in available:
            requested_paths.append(available[idx])
        else:
            missing_idx.append(idx)

    if missing_idx:
        logger.info(
            "Dyad %s %s person=%s %s: missing segments %s (of %s)",
            dyad_label, modality_name, person, cond_class,
            missing_idx, list(seg_indices),
        )

    if not requested_paths:
        return None, [], np.zeros(0, dtype=bool)

    parts: List[np.ndarray] = []
    for p in requested_paths:
        try:
            parts.append(_load_mat_segment(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", p, exc)

    if not parts:
        return None, [], np.zeros(0, dtype=bool)

    concat = np.concatenate(parts)
    boundary_mask = np.ones(len(concat), dtype=bool)
    cursor = 0
    for arr in parts[:-1]:
        cursor += len(arr)
        if cursor < len(boundary_mask):
            boundary_mask[cursor] = False

    return concat, requested_paths, boundary_mask


def _segments_for_condition_unit(unit: str) -> Tuple[str, Sequence[int]]:
    """Map a condition unit name to its (.mat token, segment indices).

    Parameters
    ----------
    unit : str
        One of ``CONDITION_UNITS``.

    Returns
    -------
    (cond_class, seg_indices) : (str, sequence of int)
        ``cond_class`` is the .mat filename token (``"Rest"`` or
        ``"Trial"``); ``seg_indices`` is the list of segment indices
        composing this condition unit.
    """
    if unit == "rest1":
        return "Rest", [1]
    if unit == "rest_postblock":
        return "Rest", [2, 3, 4]
    if unit == "trials_concat":
        return "Trial", list(range(1, TRIAL_SEGMENT_COUNT + 1))
    raise ValueError(f"Unknown condition unit: {unit!r}. Valid={CONDITION_UNITS}")


def _verify_p1_p2_length_alignment(
    a_raw: Optional[np.ndarray],
    b_raw: Optional[np.ndarray],
    dyad_label: str,
    modality: str,
    unit: str,
) -> bool:
    """Sanity check that P1 and P2 traces are length-aligned.

    Lerique trial files are nominally exactly TRIAL_SEGMENT_SAMPLES
    samples each, and rest files exactly REST_SEGMENT_SAMPLES. If P1
    and P2 length differ, the trial onset/offset alignment is broken
    and the dyad must be excluded (pre-registration sanity check
    1.3b).

    Returns True if both arrays are equal length OR one is None.
    Logs a warning on mismatch but does not raise (caller decides
    whether to drop).
    """
    if a_raw is None or b_raw is None:
        return True
    if len(a_raw) != len(b_raw):
        logger.warning(
            "Length mismatch: dyad=%s modality=%s unit=%s "
            "P1=%d samples, P2=%d samples (excluding)",
            dyad_label, modality, unit, len(a_raw), len(b_raw),
        )
        return False
    return True
def _verify_min_duration(
    a_raw: Optional[np.ndarray],
    b_raw: Optional[np.ndarray],
    raw_fs: float,
    dyad_label: str,
    modality: str,
    unit: str,
    min_duration_sec: float = MIN_DURATION_SEC,
) -> bool:
    """Sanity check that both P1 and P2 traces meet the minimum
    duration floor.

    A condition unit shorter than ``min_duration_sec`` cannot support
    meaningful dyad-level scalar feature estimation (fewer than ~4 WCC
    windows). The pre-registered hard floor is locked at 60 s.

    Note: the check is on **raw** length / raw_fs (not on the
    resampled 1 Hz grid), so it does not depend on the preprocessing
    pipeline being implemented.

    Parameters
    ----------
    a_raw, b_raw : np.ndarray or None
        Per-person raw concatenated traces at ``raw_fs``.
    raw_fs : float
        Raw sampling rate (Hz).
    dyad_label, modality, unit : str
        Used only for logging.
    min_duration_sec : float
        Floor in seconds. Defaults to ``MIN_DURATION_SEC``.

    Returns
    -------
    bool
        True if both traces (or the one that is non-None) meet the
        floor; False if at least one is below.
    """
    too_short = []
    for label, arr in (("P1", a_raw), ("P2", b_raw)):
        if arr is None:
            continue
        dur = len(arr) / raw_fs
        if dur < min_duration_sec:
            too_short.append((label, dur))
    if too_short:
        details = ", ".join(f"{lbl}={dur:.2f}s" for lbl, dur in too_short)
        logger.warning(
            "Below min duration: dyad=%s modality=%s unit=%s "
            "floor=%.1fs, observed: %s (excluding)",
            dyad_label, modality, unit, min_duration_sec, details,
        )
        return False
    return True



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lerique_dataset(
    data_root: str | Path,
    modalities: Sequence[str] = MODALITIES,
    condition_units: Sequence[str] = CONDITION_UNITS,
    dyad_whitelist: Optional[Sequence[str]] = None,
    *,
    preprocess: bool = False,
    raw_fs: float = RAW_FS_HZ,
    target_fs: float = TARGET_FS_HZ,
    drop_incomplete: bool = True,
    drop_misaligned: bool = True,
    drop_short_duration: bool = True,
    min_duration_sec: float = MIN_DURATION_SEC,
) -> List[LeriqueDyadCondition]:
    """Load Lerique-47n3p records into SyncPipe-compatible dataclasses.

    Parameters
    ----------
    data_root : str or Path
        Path containing ``ECG/``, ``EDA/``, ``RESP/`` subdirectories.
    modalities : sequence of str
        Subset of ``MODALITIES`` to load.
    condition_units : sequence of str
        Subset of ``CONDITION_UNITS`` to emit. Default: all three
        (``"rest1"``, ``"rest_postblock"``, ``"trials_concat"``).
    dyad_whitelist : sequence of str, optional
        If given, only load these dyad labels (e.g. ``["pce02"]``).
    preprocess : bool
        If True, run modality-specific preprocessing per pre-reg §1.4
        (ECG → IBI via neurokit2; EDA/RESP → bandpassed waveform via
        scipy) and resample to ``target_fs``. If False, return raw
        ``raw_fs`` traces wrapped in DataFrames — useful for smoke
        testing the loader / inventory without invoking neurokit2.
    raw_fs : float
        Raw sampling rate (default 1000 Hz, confirmed from PDF).
    target_fs : float
        Target sampling rate for the SyncPipe grid (default 1 Hz, the
        v3 WCC lock).
    drop_incomplete : bool
        If True, drop records where either P1 or P2 is missing for the
        given (modality, condition_unit).
    drop_misaligned : bool
        If True, drop records where P1 / P2 raw concat lengths differ
        (pre-registration §1.3b sanity check).
    drop_short_duration : bool
        If True, drop records whose raw trace duration is below
        ``min_duration_sec`` (pre-registration §1.4 hard floor).
    min_duration_sec : float
        Minimum required duration of each (P1, P2) trace, in seconds.
        Defaults to ``MIN_DURATION_SEC`` (60 s).

    Returns
    -------
    list of LeriqueDyadCondition

    Notes
    -----
    Filename parsing is regex-based on ``pce<NN>_P<1|2>_<Rest|Trial><K>.mat``.
    Files not matching are silently skipped (logged at INFO).
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Lerique data_root does not exist: {root}")

    invalid = set(modalities) - set(MODALITIES)
    if invalid:
        raise ValueError(f"Unknown modalities: {invalid}; valid={MODALITIES}")
    invalid = set(condition_units) - set(CONDITION_UNITS)
    if invalid:
        raise ValueError(
            f"Unknown condition_units: {invalid}; valid={CONDITION_UNITS}"
        )

    records: List[LeriqueDyadCondition] = []

    for modality in modalities:
        modality_root = root / modality
        if not modality_root.exists():
            logger.warning("Missing modality dir: %s", modality_root)
            continue

        pce_dirs = sorted(d for d in modality_root.iterdir() if d.is_dir())
        for pce_dir in pce_dirs:
            dyad_label = pce_dir.name[:5]  # "pceXX" prefix
            if dyad_whitelist is not None and dyad_label not in dyad_whitelist:
                continue

            for unit in condition_units:
                cond_class, seg_indices = _segments_for_condition_unit(unit)

                a_raw, a_paths, a_mask = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="1",
                    cond_class=cond_class, seg_indices=seg_indices,
                )
                b_raw, b_paths, b_mask = _collect_segments_for_person(
                    pce_dir, dyad_label, modality, person="2",
                    cond_class=cond_class, seg_indices=seg_indices,
                )

                aligned = _verify_p1_p2_length_alignment(
                    a_raw, b_raw, dyad_label, modality, unit,
                )

                duration_ok = _verify_min_duration(
                    a_raw, b_raw, raw_fs, dyad_label, modality, unit,
                    min_duration_sec=min_duration_sec,
                )

                incomplete = (
                    (a_raw is None) or (b_raw is None)
                    or (not aligned) or (not duration_ok)
                )
                if drop_incomplete and ((a_raw is None) or (b_raw is None)):
                    logger.info(
                        "Drop incomplete: dyad=%s modality=%s unit=%s "
                        "(P1=%s, P2=%s)",
                        dyad_label, modality, unit,
                        "ok" if a_raw is not None else "missing",
                        "ok" if b_raw is not None else "missing",
                    )
                    continue
                if drop_misaligned and not aligned:
                    logger.info(
                        "Drop misaligned: dyad=%s modality=%s unit=%s",
                        dyad_label, modality, unit,
                    )
                    continue
                if drop_short_duration and not duration_ok:
                    logger.info(
                        "Drop short-duration: dyad=%s modality=%s unit=%s "
                        "(below floor %.1fs)",
                        dyad_label, modality, unit, min_duration_sec,
                    )
                    continue

                # Preprocess or pass through raw.
                if preprocess:
                    proc = _PREPROC_DISPATCH[modality]
                    if a_raw is not None:
                        a_sig, a_mask_out = proc(a_raw, raw_fs, target_fs, a_mask)
                    else:
                        a_sig, a_mask_out = None, np.zeros(0, dtype=bool)
                    if b_raw is not None:
                        b_sig, b_mask_out = proc(b_raw, raw_fs, target_fs, b_mask)
                    else:
                        b_sig, b_mask_out = None, np.zeros(0, dtype=bool)
                    fs_out = target_fs
                    # Combine P1/P2 masks: a target sample is usable only
                    # if BOTH persons have it usable at this index. Align
                    # by truncating to the shorter — they should match
                    # within ±1 sample due to floor() rounding.
                    if a_mask_out.size and b_mask_out.size:
                        n_common = min(a_mask_out.size, b_mask_out.size)
                        mask_out = a_mask_out[:n_common] & b_mask_out[:n_common]
                        # Also truncate the signals so P1/P2 share grid
                        a_sig = a_sig[:n_common] if a_sig is not None else None
                        b_sig = b_sig[:n_common] if b_sig is not None else None
                    elif a_mask_out.size:
                        mask_out = a_mask_out
                    else:
                        mask_out = b_mask_out
                else:
                    a_sig = a_raw
                    b_sig = b_raw
                    fs_out = raw_fs
                    mask_out = a_mask if a_raw is not None else b_mask

                def _to_df(sig: Optional[np.ndarray], fs: float) -> Optional[pd.DataFrame]:
                    if sig is None:
                        return None
                    t = np.arange(len(sig), dtype=np.float64) / fs
                    return pd.DataFrame({"time": t, "value": sig.astype(np.float64)})

                df_a = _to_df(a_sig, fs_out)
                df_b = _to_df(b_sig, fs_out)
                n_samples = (
                    len(df_a) if df_a is not None
                    else len(df_b) if df_b is not None
                    else 0
                )
                duration_sec = n_samples / fs_out if n_samples > 0 else 0.0

                rec = LeriqueDyadCondition(
                    dyad_id=f"{dyad_label}__{modality}__{unit}",
                    dyad_label=dyad_label,
                    modality=modality,
                    condition=unit,
                    person_a=df_a,
                    person_b=df_b,
                    target_hz=float(fs_out),
                    n_samples=int(n_samples),
                    duration_sec=float(duration_sec),
                    incomplete=incomplete,
                    discontinuity_mask=mask_out,
                    meta={
                        "p1_segment_paths": [str(p) for p in a_paths],
                        "p2_segment_paths": [str(p) for p in b_paths],
                        "preprocessed": bool(preprocess),
                        "raw_fs_hz": float(raw_fs),
                        "cond_class": cond_class,
                        "segment_indices": list(seg_indices),
                        "alignment_ok": bool(aligned),
                    },
                )
                records.append(rec)

    records.sort(key=lambda r: r.dyad_id)
    logger.info(
        "Loaded %d Lerique records (modalities=%s, units=%s).",
        len(records), list(modalities), list(condition_units),
    )
    return records


def lerique_record_to_multisync_dyad(rec: LeriqueDyadCondition):
    """Convert a LeriqueDyadCondition into a SyncPipe ``Dyad`` object.

    Each person's preprocessed scalar trace becomes one channel with
    the suffix convention ``<modality>_a`` / ``<modality>_b`` so
    SyncPipe's cross-person edge falls at the canonical position.

    Parameters
    ----------
    rec : LeriqueDyadCondition
        Must have both ``person_a`` and ``person_b`` non-None
        (i.e., not ``incomplete``).

    Returns
    -------
    multisync.core.Dyad

    Raises
    ------
    ValueError
        If ``rec`` is incomplete.
    """
    if rec.incomplete:
        raise ValueError(
            f"Cannot convert incomplete record {rec.dyad_id}: "
            "missing or misaligned person_a/person_b."
        )
    from multisync.core import Dyad  # local import keeps converter light

    ch = rec.modality.lower()  # e.g., "ecg" / "eda" / "resp"
    modalities = {
        f"{ch}_a": rec.person_a[["time", "value"]].copy(),
        f"{ch}_b": rec.person_b[["time", "value"]].copy(),
    }
    return Dyad(hz=rec.target_hz, dyad_id=rec.dyad_id, **modalities)


__all__ = [
    "LeriqueDyadCondition",
    "load_lerique_dataset",
    "lerique_record_to_multisync_dyad",
    "RAW_FS_HZ",
    "TARGET_FS_HZ",
    "MODALITIES",
    "CONDITION_UNITS",
    "REST_SEGMENT_DURATION_SEC",
    "TRIAL_SEGMENT_DURATION_SEC",
    "REST_SEGMENT_SAMPLES",
    "TRIAL_SEGMENT_SAMPLES",
    "REST_SEGMENT_COUNT",
    "TRIAL_SEGMENT_COUNT",
]

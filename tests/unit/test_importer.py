"""
Unit tests for syncpipe.importer — coverage for architecture-review fixes.

Fix 1 (finding 20): ``merge_asof`` tolerance must use the *real* sampling rate,
    not the ``default_hz`` fallback. ``_infer_hz`` must therefore recover the true
    hz from a time vector. If it silently returned 1.0 for 100 Hz data the
    tolerance would become ~2.0 s (≈200 samples) of "nearest neighbour" slack and
    a mis-aligned dyad would be produced with no error — a silent measurement
    error.
Fix 2 (finding 21): ``load_opensignals`` applies *fail-soft* channel handling.
    Real *string* channel ids are preserved; integer or missing channel indices
    fall back to CH1..CHn. Note: real OpenSignals files use integer channel
    indices (readable names live in the ``sensor``/``label`` keys, which importer
    does not currently consume), so real files still yield CHn — the fail-soft
    only guards the defensive string-id case.
Fix 3: ``load_csv`` dyad build must fail loudly (not silently ignore) when more
    than one column per person is supplied, because the implementation only uses
    ``[0]``.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from syncpipe.importer import DataImporter, _infer_hz


# ---------------------------------------------------------------------------
# Fix 1 — _infer_hz must recover the *real* sampling rate
# ---------------------------------------------------------------------------

def test_infer_hz_100hz():
    """1000 samples @ 100 Hz -> 0.00..9.99 s.

    Why it matters: this is exactly the case the bug masked. With the fallback
    ``default_hz`` the merge tolerance would be 2.0 / default_hz seconds; for a
    100 Hz stream that is ~200 samples of slack, so unrelated samples get joined
    as if synchronous. Inference must return the true 100.0.
    """
    t = np.arange(1000) / 100.0
    assert abs(_infer_hz(t) - 100.0) < 1e-6


def test_infer_hz_1hz():
    """10 samples @ 1 Hz -> 0, 1, ..., 9 s.

    Why it matters: the 1 Hz baseline must still infer 1.0 so that the existing
    regression test (synthetic 1 Hz data) keeps tolerance = 2.0 s and shows zero
    regression.
    """
    t = np.arange(10) / 1.0
    assert abs(_infer_hz(t) - 1.0) < 1e-6


def test_merge_tolerance_uses_real_hz_not_fallback():
    """Encode *why* correct inference is load-bearing.

    Build two 100 Hz streams where person B has a 1-second data GAP (t in
    (4.99, 6.00) s is missing). With the fix, the effective hz is inferred as
    100 -> tolerance = 2.0 / 100 = 0.02 s, so A-samples inside the gap have no B
    within tolerance and are correctly left NaN. Under the old fallback
    (tolerance = 2.0 / default_hz = 2.0 s) those very same samples would be
    silently matched across the gap -> fabricated dyad values with no error.
    """
    imp = DataImporter(default_hz=1.0)
    d = tempfile.mkdtemp()
    a = os.path.join(d, "a.csv")
    b = os.path.join(d, "b.csv")

    ta = np.arange(1000) / 100.0  # 0.00 .. 9.99 s @ 100 Hz
    pd.DataFrame({
        "time": ta,
        "signal": np.arange(1000, dtype=float),
    }).to_csv(a, index=False)

    # B: drop the 100 samples covering t in [5.00, 5.99]
    b_idx = np.concatenate([np.arange(500), np.arange(600, 1000)])
    tb = b_idx / 100.0
    pd.DataFrame({
        "time": tb,
        "signal": np.arange(len(b_idx), dtype=float),
    }).to_csv(b, index=False)

    merged = imp.merge_person_files(a, b)
    # All 1000 A rows are retained; the ~98 gap rows must be NaN under the fix.
    assert len(merged) == 1000
    n_nan = int(merged["person_b"].isna().sum())
    assert n_nan > 50, f"expected tight tolerance to reject gap matches, got {n_nan} NaN"


# ---------------------------------------------------------------------------
# Fix 2 — load_opensignals keeps real channel names (fail-soft)
# ---------------------------------------------------------------------------

def _write_opensignals(path, channels, n_rows=5, sr=1000):
    header = json.dumps({"00:00:00:00:00:00": {
        "sampling rate": sr,
        "channels": channels,
    }})
    lines = ["#" + header]
    for i in range(n_rows):
        # sample index + one data column per channel
        row = [str(i)] + [str(float(i) * (j + 1)) for j in range(len(channels))]
        lines.append("\t".join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_load_opensignals_keeps_real_channel_names():
    """Real string channel ids must survive, not become CH1/CH2.

    Why it matters: a user's ``channel_map`` keyed on the true metadata
    ('ECG' -> 'ecg') only stays aligned if the importer preserves those names.
    """
    imp = DataImporter(default_hz=1.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "opensignals.txt")
    _write_opensignals(p, ["ECG", "EDA"])

    result = imp.load_opensignals(p)
    # channel_map lowercases the real names -> 'ecg' / 'eda'
    assert set(result.keys()) == {"ecg", "eda"}, result.keys()


def test_load_opensignals_non_string_channels_fallback():
    """Fail-soft: non-string channel ids fall back to CH1..CHn placeholders.

    Preserves prior behaviour for malformed headers instead of raising or
    silently dropping channels.
    """
    imp = DataImporter(default_hz=1.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "opensignals.txt")
    _write_opensignals(p, [1, 2])

    result = imp.load_opensignals(p)
    assert set(result.keys()) == {"ch1", "ch2"}, result.keys()


# ---------------------------------------------------------------------------
# Fix 3 — load_csv dyad build fails loud on multi-column input
# ---------------------------------------------------------------------------

def test_load_csv_single_column_dyad_ok():
    """Single column per person is the supported path and must keep working."""
    imp = DataImporter(default_hz=1.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "dyad.csv")
    pd.DataFrame({
        "time": np.arange(10) / 1.0,
        "a": np.arange(10, dtype=float),
        "b": np.arange(10, 20, dtype=float),
    }).to_csv(p, index=False)

    out = imp.load_csv(p, person_a_cols=["a"], person_b_cols=["b"])
    sig = out["signal"]
    assert list(sig.columns) == ["time", "person_a", "person_b"]
    assert np.allclose(sig["person_a"].values, np.arange(10, dtype=float))
    assert np.allclose(sig["person_b"].values, np.arange(10, 20, dtype=float))


def test_load_csv_multi_column_dyad_raises():
    """Multi-column per person is silently ignored by the [0]-only impl.

    Why it matters: the docstring advertises "column subsets" but only [0] is
    used, so extra columns vanish without a trace. Fail loudly instead.
    """
    imp = DataImporter(default_hz=1.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "dyad.csv")
    pd.DataFrame({
        "time": np.arange(10) / 1.0,
        "a1": np.arange(10, dtype=float),
        "a2": np.arange(10, 20, dtype=float),
        "b1": np.arange(20, 30, dtype=float),
    }).to_csv(p, index=False)

    with pytest.raises(ValueError):
        imp.load_csv(p, person_a_cols=["a1", "a2"], person_b_cols=["b1"])

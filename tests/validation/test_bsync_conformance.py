"""Conformance of the native WCC surface with bsync.

Two layers, so a failure can be attributed:

* the *comparer* is tested here against deliberately wrong "engines" and always runs;
* the *comparison with real bsync output* needs golden values generated once in R
  (``tests/fixtures/bsync_golden/make_golden.R``) and is skipped, with that instruction, until they exist.
"""
from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from syncpipe.surface import wcc_surface
from syncpipe.surface.conformance import compare_with_bsync_cells

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "bsync_golden"
WINDOW, LAG_MAX = 30, 10                        # must match make_golden.R


@pytest.fixture(scope="module")
def pair():
    frame = pd.read_csv(GOLDEN / "pair.csv", float_precision="round_trip")
    return frame["x"].to_numpy(), frame["y"].to_numpy()


@pytest.fixture(scope="module")
def surface(pair):
    return wcc_surface(*pair, hz=1.0, window_samples=WINDOW, lag_max_samples=LAG_MAX)


# ------------------------------------------------------- the fixture itself ----
def test_fixture_pair_reproduces_from_its_generator_script(pair):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_pair", GOLDEN / "make_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x, y = module.make_pair()
    assert np.array_equal(pair[0], x) and np.array_equal(pair[1], y)


def test_fixture_csv_holds_plain_numbers_that_r_can_read():
    lines = (GOLDEN / "pair.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x,y" and len(lines) == 301
    assert all(re.fullmatch(r"-?\d+\.\d+(e-?\d+)?,-?\d+\.\d+(e-?\d+)?", ln) for ln in lines[1:])


def test_fixture_pair_shows_x_leading_y_within_the_estimators_resolution(surface):
    """The generating lead is 3 samples; the estimate is statistical (noisy data), so assert sign and +-1."""
    lag = surface.summary().lag_of_peak_sec
    assert lag > 0 and abs(lag - 3.0) <= 1.0


def test_r_script_parameters_match_this_test():
    script = (GOLDEN / "make_golden.R").read_text(encoding="utf-8")
    assert f"window_size = {WINDOW}L" in script and f"lag_max = {LAG_MAX}L" in script


# --------------------------------------------------------------- the comparer ----
def _as_r_would_write_it(frame: pd.DataFrame) -> pd.DataFrame:
    """R's write.csv keeps 15 significant digits; emulate that round trip."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.15g")
    return pd.read_csv(io.StringIO(buffer.getvalue()))


def test_comparer_accepts_the_same_surface_after_an_r_style_csv_round_trip(surface):
    result = compare_with_bsync_cells(surface, _as_r_would_write_it(surface.to_bsync_frame()))
    assert result.ok and result.n_cells_compared == surface.values.size and result.max_abs_diff < 1e-12


def test_comparer_detects_a_flipped_lag_sign(surface):
    other = surface.to_bsync_frame()
    other["tau"] = -other["tau"]
    result = compare_with_bsync_cells(surface, other)
    assert not result.ok and result.max_abs_diff > 0.05


def test_comparer_detects_an_off_by_one_window_index(surface):
    other = surface.to_bsync_frame()
    other["i"] = other["i"] - 1                                     # 0-based instead of bsync's 1-based
    result = compare_with_bsync_cells(surface, other)
    assert not result.ok and (result.n_cells_only_in_native > 0 or result.max_abs_diff > 0.0)


def test_comparer_detects_missing_and_extra_cells(surface):
    full = surface.to_bsync_frame()
    result = compare_with_bsync_cells(surface, full.iloc[:-5])
    assert not result.ok and result.n_cells_only_in_native == 5
    extra = pd.concat([full, pd.DataFrame({"i": [10_000], "tau": [0], "wcc": [0.1]})], ignore_index=True)
    assert compare_with_bsync_cells(surface, extra).n_cells_only_in_other == 1


def test_comparer_treats_nan_in_only_one_engine_as_a_disagreement(surface):
    other = surface.to_bsync_frame()
    other.loc[3, "wcc"] = np.nan
    assert not compare_with_bsync_cells(surface, other).ok


def test_comparer_detects_a_small_numerical_drift_beyond_tolerance(surface):
    other = surface.to_bsync_frame()
    other["wcc"] = other["wcc"] + 1e-6
    assert not compare_with_bsync_cells(surface, other, atol=1e-8).ok
    assert compare_with_bsync_cells(surface, other, atol=1e-5).ok


def test_comparer_rejects_frames_it_cannot_interpret(surface):
    with pytest.raises(ValueError, match="needs columns"):
        compare_with_bsync_cells(surface, pd.DataFrame({"i": [1], "tau": [0]}))
    dup = pd.concat([surface.to_bsync_frame().iloc[:3]] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        compare_with_bsync_cells(surface, dup)


# ------------------------------------------------------ the real comparison ----
GOLDEN_CELLS = GOLDEN / "golden_cells.csv"


@pytest.mark.skipif(
    not GOLDEN_CELLS.exists(),
    reason="bsync golden values not generated yet: run `Rscript tests/fixtures/bsync_golden/make_golden.R` "
           "(needs R + bsync) and commit the output; this test then becomes a hard gate",
)
def test_native_wcc_surface_matches_bsync_cell_by_cell(surface):
    result = compare_with_bsync_cells(surface, pd.read_csv(GOLDEN_CELLS, float_precision="round_trip"), atol=1e-8)
    assert result.ok, result


@pytest.mark.skipif(not GOLDEN_CELLS.exists(), reason="bsync golden values not generated yet (see test above)")
@pytest.mark.parametrize("statistic, native", [("mean_abs_z", "mean_abs_z"), ("peak", "peak_abs_z")])
def test_bsync_compatible_aggregates_match_bsync_glance(surface, statistic, native):
    glance = pd.read_csv(GOLDEN / f"golden_glance_{statistic}.csv", float_precision="round_trip")
    assert statistic in glance.columns, f"bsync glance has columns {list(glance.columns)}"
    assert getattr(surface.summary(), native) == pytest.approx(float(glance[statistic].iloc[0]), abs=1e-8)

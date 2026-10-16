"""Surface exchange format: exact round trip and refusal of damaged or foreign files."""
from __future__ import annotations

import json

import numpy as np
import pytest

from syncpipe.surface import SURFACE_SCHEMA_VERSION, read_surface, wcc_surface, write_surface


@pytest.fixture()
def surface():
    rng = np.random.default_rng(5)
    x = np.convolve(rng.normal(size=100), np.ones(4) / 4, mode="same")
    y = np.roll(x, 2) + 0.3 * rng.normal(size=100)
    y[30:34] = np.nan                                           # a gap: NaN cells must survive the round trip
    return wcc_surface(x, y, hz=2.0, window_samples=12, lag_max_samples=4, window_increment=2, lag_increment=2)


def test_round_trip_is_exact_including_nan(surface, tmp_path):
    write_surface(surface, tmp_path / "dyad01")
    back = read_surface(tmp_path / "dyad01")
    assert np.array_equal(back.values, surface.values, equal_nan=True)
    assert np.array_equal(back.lags_samples, surface.lags_samples)
    assert np.array_equal(back.window_start_samples, surface.window_start_samples)
    assert back.settings == surface.settings and back.content_hash() == surface.content_hash()


def test_sidecar_records_schema_grid_and_hash(surface, tmp_path):
    _, json_path = write_surface(surface, tmp_path / "s")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == SURFACE_SCHEMA_VERSION
    assert (meta["n_lags"], meta["n_windows"]) == surface.values.shape
    assert meta["content_hash"] == surface.content_hash() and meta["settings"]["lag_sign"] == "x_leads_y_positive"


def test_an_edited_value_is_detected_by_the_hash(surface, tmp_path):
    csv_path, _ = write_surface(surface, tmp_path / "s")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",0.123456"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recorded hash"):
        read_surface(tmp_path / "s")


def test_a_truncated_grid_is_refused(surface, tmp_path):
    csv_path, _ = write_surface(surface, tmp_path / "s")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    csv_path.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete grid"):
        read_surface(tmp_path / "s")


def test_an_unknown_schema_major_version_is_refused(surface, tmp_path):
    _, json_path = write_surface(surface, tmp_path / "s")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    meta["schema_version"] = "9.0"
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported surface schema"):
        read_surface(tmp_path / "s")


def test_wrong_csv_columns_are_refused(surface, tmp_path):
    csv_path, _ = write_surface(surface, tmp_path / "s")
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("window_start_sample", "start"), encoding="utf-8")
    with pytest.raises(ValueError, match="columns must be"):
        read_surface(tmp_path / "s")

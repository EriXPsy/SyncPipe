"""Smoke tests for sensitivity sweep scripts (Appendix B)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from multisync.feature_definitions import FDR_FEATURES


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "experimental" / "scripts" / "run_sensitivity_sweep.py"
    return _load_module("run_sensitivity_sweep", script_path)


def test_window_values_present(sweep_module):
    assert sweep_module.WINDOW_VALUES_SEC == (10.0, 20.0, 30.0, 45.0, 60.0)


def test_threshold_values_present(sweep_module):
    assert sweep_module.THRESHOLD_VALUES == (0.3, 0.4, 0.5, 0.6, 0.7)


def test_base_config_fields(sweep_module):
    cfg = sweep_module._base_config(
        window_sec=30.0, threshold=0.55, n_surrogates=99,
    )
    assert abs(cfg.wcc_window_sec - 30.0) < 1e-9
    assert abs(cfg.onset_threshold - 0.55) < 1e-9
    assert cfg.n_surrogates == 99
    assert cfg.couplings == (0.3,)
    assert cfg.noise_ratios == (0.3,)


def test_base_config_defaults(sweep_module):
    """Default threshold=0.5 (ONSET_THRESHOLD_DEFAULT via Level3Config)."""
    cfg = sweep_module._base_config(
        window_sec=30.0, threshold=0.5, n_surrogates=49,
    )
    # self-consistency: the config should round-trip through wcc_window_samples
    assert cfg.wcc_window_samples == max(2, int(round(30.0 * cfg.hz)))


def test_window_sweep_output_format(tmp_path, sweep_module):
    """Run window sweep with tiny N, check CSV format."""
    out_dir = tmp_path / "sensitivity"
    df = sweep_module.run_window_sweep(out_dir, n_surrogates=49)
    assert isinstance(df, pd.DataFrame)
    assert "sweep_param" in df.columns
    assert "sweep_value" in df.columns
    assert set(df["sweep_param"].unique()) == {"window_sec"}
    assert set(df["sweep_value"].unique()) == set(sweep_module.WINDOW_VALUES_SEC)

    # Check wide-format rate columns exist (summary from summarise_level3).
    # Columns are derived from FDR_FEATURES (Axis C), NOT from hard-coded list.
    expected_prefixes = [f"reject_{f}_rate" for f in FDR_FEATURES]
    for c in expected_prefixes:
        assert c in df.columns, f"Missing column: {c}"

    # CSV file exists
    csv_path = out_dir / "level3_sensitivity_window.csv"
    assert csv_path.exists()


def test_threshold_sweep_output_format(tmp_path, sweep_module):
    """Run threshold sweep with tiny N, check CSV format."""
    out_dir = tmp_path / "sensitivity"
    df = sweep_module.run_threshold_sweep(out_dir, n_surrogates=49)
    assert isinstance(df, pd.DataFrame)
    assert set(df["sweep_param"].unique()) == {"onset_threshold"}
    assert set(df["sweep_value"].unique()) == set(sweep_module.THRESHOLD_VALUES)

    csv_path = out_dir / "level3_sensitivity_threshold.csv"
    assert csv_path.exists()


def test_diagnose_runs_without_error(sweep_module, tmp_path, capsys):
    """_diagnose_rank_stability should not raise on valid summary df."""
    # Build a minimal fake summary df with correct wide-format columns.
    # Columns are derived from FDR_FEATURES (Axis C), NOT from hard-coded list.
    rate_cols = [f"reject_{f}_rate" for f in FDR_FEATURES]
    rows = []
    for w in list(sweep_module.WINDOW_VALUES_SEC)[:2]:  # just 2 values
        row = {c: 0.1 for c in rate_cols}
        row["noise_ratio"] = 0.3
        row["coupling"] = 0.3
        row["n_seeds"] = 30
        row["sweep_param"] = "window_sec"
        row["sweep_value"] = w
        rows.append(row)
    df = pd.DataFrame(rows)
    sweep_module._diagnose_rank_stability(df, "window_sec")
    captured = capsys.readouterr()
    assert "Rank-order" in captured.out

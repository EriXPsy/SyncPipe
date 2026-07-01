"""Guard: the generated feature table must match the code (no drift).

NOTE: Requires scripts/build_feature_table.py and docs/FEATURE_TABLE.csv.
Run ``python scripts/build_feature_table.py`` to regenerate the CSV before
running these tests. Skipped automatically when dependencies are missing.
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "scripts" / "build_feature_table.py").exists(),
    reason="build_feature_table.py not found — run scripts/build_feature_table.py first",
)

from multisync.feature_definitions import (
    FEATURE_TIER,
    FDR_FEATURES,
    MATHEMATICAL_TIER,
)

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "docs" / "FEATURE_TABLE.csv"


def _build_rows():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_feature_table", REPO / "scripts" / "build_feature_table.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_rows()


def test_annotations_cover_exactly_coded_features():
    """build_feature_table.build_rows() raises if annotations drift from code."""
    rows = _build_rows()  # raises SystemExit on drift
    names = {r["feature"] for r in rows}
    assert names == set(FEATURE_TIER), "table features != FEATURE_TIER"


def test_fdr_membership_matches_code():
    rows = _build_rows()
    in_fdr = {r["feature"] for r in rows if r["in_FDR_family"] == "yes"}
    assert in_fdr == set(FDR_FEATURES), (
        f"table FDR set {in_fdr} != code FDR_FEATURES {set(FDR_FEATURES)}"
    )


def test_null_model_follows_mathematical_tier():
    rows = _build_rows()
    for r in rows:
        assert r["math_tier"] == MATHEMATICAL_TIER[r["feature"]]


@pytest.mark.skipif(not CSV.exists(), reason="run scripts/build_feature_table.py first")
def test_generated_csv_is_current():
    """If the CSV exists, it must match a fresh build (catches stale commits)."""
    import csv as _csv
    rows = _build_rows()
    fresh = {r["feature"]: r for r in rows}
    with CSV.open(encoding="utf-8") as f:
        on_disk = {r["feature"]: r for r in _csv.DictReader(f)}
    assert set(fresh) == set(on_disk), "CSV feature set is stale; re-run build_feature_table.py"
    for name in fresh:
        assert fresh[name]["in_FDR_family"] == on_disk[name]["in_FDR_family"]
        assert fresh[name]["math_tier"] == on_disk[name]["math_tier"]


def test_bc_removed_from_fdr_but_retains_l0_math_tier():
    """bimodality_coefficient (SSoT decision 2026-06-29, Option B).

    BC was removed from the confirmatory group-condition FDR family because
    its membership was provisional and lacked dated, pre-decision cross-
    paradigm evidence. It is therefore in NO FDR family. It remains a
    permutation-invariant L0 distribution-shape descriptor (math tier L0)
    used by the separate synchrony-existence audit, and is still computed
    and serialized. This test guards that decoupling: math-tier L0 must NOT
    silently re-imply FDR membership.
    """
    from multisync.feature_definitions import FDR_FAMILIES, FDR_FEATURES
    bc_fdr_family = next(
        (fam for fam, members in FDR_FAMILIES.items()
         if "bimodality_coefficient" in members), None
    )
    assert bc_fdr_family is None, "bimodality_coefficient must be in no FDR family"
    assert "bimodality_coefficient" not in FDR_FEATURES
    assert MATHEMATICAL_TIER["bimodality_coefficient"] == "L0"

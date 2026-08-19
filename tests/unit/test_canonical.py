from __future__ import annotations

"""Unit tests for the v1 canonical scientific runner (Gate 1).

Covers manifest/config parsing contracts, the 12-file report bundle,
exclusion accounting, and the eligibility governance floor.
"""
import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from syncpipe.canonical_runner import (
    parse_config,
    parse_manifest,
    run_canonical,
)

EXPECTED_BUNDLE = [
    "manifest_resolved.json", "config_resolved.toml", "environment.json",
    "qc_report.json", "exclusion_report.csv", "features.csv",
    "existence_audit.json", "design_control_audit.json",
    "group_inference.json", "claimability.json", "REPORT.md",
]


def _fmt(v):
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _write_cohort(root: Path, n_dyads: int = 4) -> Path:
    sigdir = root / "data"
    sigdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    n = 240
    t = np.arange(n, dtype=float)
    rows = []
    for i in range(n_dyads):
        for cond, coup in (("rest", 0.2), ("task", 0.8)):
            shared = np.sin(np.linspace(0, 8 * np.pi, n)) + 0.3 * rng.normal(size=n)
            a = coup * shared + rng.normal(scale=0.6, size=n)
            b = coup * shared + rng.normal(scale=0.6, size=n)
            pa = sigdir / f"d{i:03d}_{cond}_a.csv"
            pb = sigdir / f"d{i:03d}_{cond}_b.csv"
            pd.DataFrame({"time": t, "val": a}).to_csv(pa, index=False)
            pd.DataFrame({"time": t, "val": b}).to_csv(pb, index=False)
            rows.append((f"d{i:03d}", "EDA", cond, str(pa), str(pb), 1.0, ""))
    man = root / "manifest.csv"
    pd.DataFrame(
        rows,
        columns=["dyad_id", "modality", "condition", "person_a_path", "person_b_path", "hz", "mask_path"],
    ).to_csv(man, index=False)
    return man


def _write_config(root: Path, **over) -> Path:
    cfg = {
        "window_size": 20,
        "contrast": ["rest", "task"],
        "primary_endpoint": "peak_amplitude",
        "primary_modalities": ["EDA"],
        "fdr_scope": "global",
        "undefined_policy": "gate",
        "observation_policy": "raise",
        "eligibility_policy": "raise",
        "n_min_dyads": 4,
        "onset_threshold": "session_pooled",
        "n_permutations": 100,
        "seed": 42,
        "surrogate_n": 10,
        "design_threshold": 0.5,
    }
    cfg.update(over)
    text = "[analysis]\n" + "\n".join(f"{k} = {_fmt(v)}" for k, v in cfg.items()) + "\n"
    p = root / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


class TestParseManifest:
    def test_requires_columns(self, tmp_path):
        bad = tmp_path / "m.csv"
        pd.DataFrame({"dyad_id": ["d1"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError):
            parse_manifest(bad)

    def test_rejects_heterogeneous_hz(self, tmp_path):
        man = tmp_path / "m.csv"
        pd.DataFrame([
            {"dyad_id": "d1", "modality": "EDA", "condition": "rest",
             "person_a_path": "a", "person_b_path": "b", "hz": 1.0},
            {"dyad_id": "d2", "modality": "EDA", "condition": "task",
             "person_a_path": "a", "person_b_path": "b", "hz": 2.0},
        ]).to_csv(man, index=False)
        with pytest.raises(ValueError):
            parse_manifest(man)

    def test_parses_ok(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=2)
        recs = parse_manifest(man)
        assert len(recs) == 4  # 2 dyads x 2 conditions
        assert recs[0].modality == "EDA"

    def test_rejects_empty_manifest(self, tmp_path):
        man = tmp_path / "empty.csv"
        pd.DataFrame(columns=["dyad_id", "modality", "condition", "person_a_path", "person_b_path", "hz"]).to_csv(man, index=False)
        with pytest.raises(ValueError, match="at least one"):
            parse_manifest(man)

    def test_rejects_duplicate_manifest_key(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=1)
        df = pd.read_csv(man)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        df.to_csv(man, index=False)
        with pytest.raises(ValueError, match="duplicate"):
            parse_manifest(man)

    def test_rejects_misaligned_signal_time_axes(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=1)
        df = pd.read_csv(man)
        b = Path(df.loc[0, "person_b_path"])
        bdf = pd.read_csv(b)
        bdf["time"] = bdf["time"] + 0.25
        bdf.to_csv(b, index=False)
        cfg = _write_config(tmp_path, n_min_dyads=4, eligibility_policy="warn")
        with pytest.raises(ValueError, match="time axes"):
            run_canonical(man, cfg, tmp_path / "out")


class TestParseConfig:
    def test_requires_contrast(self, tmp_path):
        cfg = tmp_path / "c.toml"
        cfg.write_text("[analysis]\nwindow_size = 10\n", encoding="utf-8")
        with pytest.raises(ValueError):
            parse_config(cfg)

    def test_defaults_filled(self, tmp_path):
        cfg = _write_config(tmp_path, n_min_dyads=5)
        c = parse_config(cfg)
        assert c.window_size == 20
        assert c.eligibility_policy == "raise"
        assert c.n_min_dyads == 5
        assert c.resolved_contrast() == ("rest", "task")
        assert c.resolved_primary_endpoint() == "peak_amplitude"
        assert c.resolved_primary_modalities() == ("EDA",)

    def test_requires_primary_endpoint(self, tmp_path):
        cfg = _write_config(tmp_path)
        text = cfg.read_text(encoding="utf-8").replace(
            'primary_endpoint = "peak_amplitude"\n', ""
        )
        cfg.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="primary_endpoint is required"):
            parse_config(cfg)

    def test_requires_primary_modalities(self, tmp_path):
        cfg = _write_config(tmp_path)
        text = cfg.read_text(encoding="utf-8").replace(
            'primary_modalities = ["EDA"]\n', ""
        )
        cfg.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="primary_modalities is required"):
            parse_config(cfg)

    def test_rejects_unsupported_primary_endpoint(self, tmp_path):
        cfg = _write_config(tmp_path, primary_endpoint="mean_synchrony")
        with pytest.raises(ValueError, match="must be 'peak_amplitude'"):
            parse_config(cfg)

    def test_rejects_bad_onset_string(self, tmp_path):
        cfg = _write_config(tmp_path, onset_threshold="bogus")
        with pytest.raises(ValueError):
            parse_config(cfg)

    def test_accepts_numeric_onset(self, tmp_path):
        cfg = _write_config(tmp_path, onset_threshold=0.5)
        c = parse_config(cfg)
        assert c.onset_threshold == 0.5

    def test_rejects_duplicate_contrast(self, tmp_path):
        cfg = _write_config(tmp_path, contrast=["rest", "rest"])
        with pytest.raises(ValueError, match="different"):
            parse_config(cfg)


class TestRunCanonical:
    def test_writes_full_bundle(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=4)
        cfg = _write_config(tmp_path)
        out = tmp_path / "results"
        res = run_canonical(man, cfg, out)
        for f in EXPECTED_BUNDLE:
            assert (out / f).exists(), f
        assert len(res.wcc_traces) == 8          # 4 dyads x 2 conditions
        assert len(res.claimability["per_feature"]) == 4  # FDR(3) + reference(1)
        # QC accounting correctness (included = total - exclusions)
        assert res.qc["total_rows"] == 8
        assert res.qc["included"] == 8
        assert res.qc["excluded"] == 0
        # Vulnerability A: manifest_resolved must carry resolved absolute paths
        # and content hashes, not only the original relative strings.
        manifest_payload = json.loads((out / "manifest_resolved.json").read_text())
        assert manifest_payload["base_dir"]
        row0 = manifest_payload["rows"][0]
        assert Path(row0["person_a_path"]).is_absolute()
        assert row0["person_a_sha256"]

    def test_rejects_primary_modality_absent_from_manifest(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=4)
        cfg = _write_config(tmp_path, primary_modalities=["ECG"])
        with pytest.raises(ValueError, match="absent from the manifest"):
            run_canonical(man, cfg, tmp_path / "out")

    def test_excludes_load_errors(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=6)
        # Corrupt one signal path -> that row is a load error and is excluded.
        # Its orphan condition row is kept in features_df but simply does not
        # pair in the L2 contrast (records_to_inference_inputs does not drop
        # single-condition rows); it is not a separate exclusion.
        df = pd.read_csv(man)
        df.loc[0, "person_a_path"] = str(tmp_path / "does_not_exist.csv")
        df.to_csv(man, index=False)
        cfg = _write_config(tmp_path)
        out = tmp_path / "results"
        res = run_canonical(man, cfg, out)

        assert res.qc["total_rows"] == 12
        assert res.qc["excluded"] == 1
        assert res.qc["included"] == 11
        exc = pd.read_csv(out / "exclusion_report.csv")
        assert len(exc) == 1
        assert "load_error" in str(exc.iloc[0]["reason"])
        # pair_summary exposes orphan vs paired dyads (an orphan is not a usable
        # confirmatory unit and must not masquerade as one in qc_report.json).
        assert res.qc["pair_summary"]["EDA"]["n_paired_dyads"] == 5
        assert res.qc["pair_summary"]["EDA"]["n_orphan_dyads"] == 1

    def test_eligibility_raise_blocks(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=4)  # 4 paired dyads (>= hard floor 4)
        cfg = _write_config(tmp_path, n_min_dyads=10, eligibility_policy="raise")
        out = tmp_path / "results"
        with pytest.raises(ValueError):
            run_canonical(man, cfg, out)

    def test_eligibility_warn_continues(self, tmp_path):
        man = _write_cohort(tmp_path, n_dyads=4)  # 4 paired dyads -> underpowered, not blocked
        cfg = _write_config(tmp_path, n_min_dyads=10, eligibility_policy="warn")
        out = tmp_path / "results"
        res = run_canonical(man, cfg, out)
        assert res.chain["group_condition_inference"] is not None
        for pf in res.claimability["per_feature"]:
            assert pf["eligibility_status"] == "underpowered"
            assert pf["claimable"] is False

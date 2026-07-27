from __future__ import annotations

"""Gate 1 CLI/API parity: same manifest + config through both entry points
must yield a byte-identical report bundle.

The v1 canonical runner is the single code path for paper-level analysis;
``multisync analyze`` (CLI) and ``multisync.canonical_runner.run_canonical``
(Python API) both call it, so their outputs must match file-for-file.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multisync.canonical_runner import run_canonical

_REPO_ROOT = Path(__file__).resolve().parents[2]  # syncpipe/

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
        "fdr_scope": "global",
        "undefined_policy": "gate",
        "observation_policy": "raise",
        "eligibility_policy": "raise",
        "n_min_dyads": 4,
        "onset_threshold": "session_pooled",
        "n_permutations": 200,
        "seed": 42,
        "surrogate_n": 20,
        "design_threshold": 0.5,
    }
    cfg.update(over)
    text = "[analysis]\n" + "\n".join(f"{k} = {_fmt(v)}" for k, v in cfg.items()) + "\n"
    p = root / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_cli_api_byte_parity(tmp_path):
    man = _write_cohort(tmp_path, n_dyads=4)
    cfg = _write_config(tmp_path)
    api_out = tmp_path / "api_out"
    cli_out = tmp_path / "cli_out"

    # Python API entry
    run_canonical(man, cfg, api_out)

    # CLI entry (same manifest + config)
    proc = subprocess.run(
        [sys.executable, "-m", "multisync", "analyze",
         "-m", str(man), "-c", str(cfg), "-o", str(cli_out)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr

    # Compare the complete bundle, including exclusion CSV and every WCC trace.
    api_files = sorted(p.relative_to(api_out) for p in api_out.rglob("*") if p.is_file())
    cli_files = sorted(p.relative_to(cli_out) for p in cli_out.rglob("*") if p.is_file())
    assert api_files == cli_files
    expected_top = {Path(f) for f in EXPECTED_BUNDLE}
    assert expected_top.issubset(set(api_files))
    for rel in api_files:
        assert (api_out / rel).read_bytes() == (cli_out / rel).read_bytes(), f"byte mismatch: {rel}"

"""Regression tests for Finding 9: per-modality RNG seed in
``between_condition_by_modality`` must be reproducible across Python
processes / machines.

The bug: ``seed + hash(mod) % 10000`` used the built-in ``hash()``, which
is randomized per process by default (PYTHONHASHSEED). The fix derives the
offset from ``hashlib.md5(modality)`` instead, which is process-stable.

The bug is inert for n_dyads <= _ENUM_THRESHOLD (exact enumeration, no RNG),
so the determinism test must use n_dyads > 12 to actually exercise the
Monte-Carlo sampling path where the seed matters.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multisync.validation.l2_between_condition import (
    _modality_seed_offset,
    between_condition_by_modality,
)

REPO = Path(__file__).resolve().parents[1]  # .../syncpipe
PY = sys.executable


def _df(n_dyads_per_modality=14, data_seed=123):
    """Two modalities x n_dyads x 2 conditions, >12 dyads per modality
    so the Monte-Carlo path (seed-dependent) is taken."""
    rng = np.random.default_rng(data_seed)
    rows = []
    for mod in ("EDA", "ECG", "RESP"):
        for i in range(n_dyads_per_modality):
            a = rng.normal(1.0, 0.3)
            b = rng.normal(0.6, 0.3)
            rows.append(
                {
                    "dyad_label": f"{mod}_d{i}",
                    "modality": mod,
                    "condition": "A",
                    "peak_amplitude": a,
                }
            )
            rows.append(
                {
                    "dyad_label": f"{mod}_d{i}",
                    "modality": mod,
                    "condition": "B",
                    "peak_amplitude": b,
                }
            )
    return pd.DataFrame(rows)


def _fingerprint(res: dict) -> dict:
    fp = {}
    for mod, r in res.items():
        if "error" in r:
            fp[mod] = "error"
            continue
        fp[mod] = {
            f.feature: round(float(f.p_fdr), 12) for f in r["per_feature"]
        }
    return fp


# ── unit: helper is process-stable and not the broken hash ────────────────
def test_modality_seed_offset_structure_and_stable():
    # Structural correctness: equals md5-derived offset.
    assert _modality_seed_offset("EDA") == int(
        hashlib.md5("EDA".encode("utf-8")).hexdigest(), 16
    ) % 10000
    # Same input -> same output (trivially true, documents intent).
    assert _modality_seed_offset("EDA") == _modality_seed_offset("EDA")
    # Distinct modalities get distinct offsets.
    assert _modality_seed_offset("EDA") != _modality_seed_offset("ECG")


# ── in-process: no accidental RNG leakage between calls ───────────────────
def test_by_modality_deterministic_in_process():
    df = _df()
    a = between_condition_by_modality(
        df, modality_col="modality", feature_cols=["peak_amplitude"],
        n_permutations=500, seed=42,
    )
    b = between_condition_by_modality(
        df, modality_col="modality", feature_cols=["peak_amplitude"],
        n_permutations=500, seed=42,
    )
    assert _fingerprint(a) == _fingerprint(b)


# ── cross-process: the real catch-the-bug test ────────────────────────────
_SUBPROCESS_SCRIPT = """
import json, sys
sys.path.insert(0, r"{repo}")
import numpy as np, pandas as pd
from multisync.validation.l2_between_condition import between_condition_by_modality

rng = np.random.default_rng({data_seed})
rows = []
for mod in ("EDA","ECG","RESP"):
    for i in range({n}):
        a = rng.normal(1.0,0.3); b = rng.normal(0.6,0.3)
        rows.append({{"dyad_label": f"{{mod}}_d{{i}}", "modality": mod,
                      "condition": "A", "peak_amplitude": a}})
        rows.append({{"dyad_label": f"{{mod}}_d{{i}}", "modality": mod,
                      "condition": "B", "peak_amplitude": b}})
df = pd.DataFrame(rows)
res = between_condition_by_modality(df, modality_col="modality",
        feature_cols=["peak_amplitude"], n_permutations=500, seed=42)
fp = {{}}
for mod, r in res.items():
    if "error" in r:
        fp[mod] = "error"
    else:
        fp[mod] = {{f.feature: round(float(f.p_fdr), 12) for f in r["per_feature"]}}
print(json.dumps(fp))
"""


def _run_subprocess() -> dict:
    script = _SUBPROCESS_SCRIPT.format(
        repo=str(REPO), data_seed=123, n=14
    )
    out = subprocess.run(
        [PY, "-c", script], cwd=str(REPO),
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_by_modality_seed_stable_across_processes():
    """Two independent Python processes must produce identical per-modality
    p-values. Before the fix this failed because hash() is per-process."""
    fp_a = _run_subprocess()
    fp_b = _run_subprocess()
    assert fp_a == fp_b

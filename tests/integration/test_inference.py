from __future__ import annotations

# === source: test_audited_inference_pipeline.py ===
"""Tests for the v1 audited evidence-chain inference API."""

import numpy as np
import pandas as pd

from multisync.dynamic_features import extract_dynamic_features, sliding_window_wcc
from multisync.inference_pipeline import InferencePipeline


def _signals(seed: int, n: int = 180, coupling: float = 0.7):
    rng = np.random.default_rng(seed)
    shared = np.sin(np.linspace(0, 8 * np.pi, n)) + 0.3 * rng.normal(size=n)
    a = coupling * shared + rng.normal(scale=0.6, size=n)
    b = coupling * shared + rng.normal(scale=0.6, size=n)
    return a, b


def _feature_df():
    rows = []
    hz = 1.0
    window = 20
    for i in range(4):
        for cond, coup in (("rest", 0.2), ("task", 0.8)):
            a, b = _signals(100 + i * 10 + (cond == "task"), coupling=coup)
            wcc = sliding_window_wcc(a, b, window_size=window, hz=hz)
            feats = extract_dynamic_features(wcc, hz=hz, wcc_window_sec=window / hz)
            row = {"dyad_id": f"dyad_{i}", "condition": cond}
            row.update(feats.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def test_audited_inference_api_runs_all_steps():
    df = _feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=10, seed=1)

    raw = {"dyad_0": _signals(1)}
    existence = pipe.run_synchrony_existence_audit(raw, wcc_window_size=20)
    assert existence["step"] == "synchrony_existence_audit"
    assert existence["n_pairs"] == 1

    cohort = {f"dyad_{i}": _signals(i) for i in range(3)}
    design = pipe.run_design_control_audit(
        cohort,
        wcc_window_size=20,
        n_pseudo_per_dyad=1,
        shift_lags_sec=(-40.0, 40.0),
    )
    assert design["audit"] == "design_controls"
    assert "feature_summary" in design

    segments = [
        ("seg1", _signals(10, n=60)[0], _signals(10, n=60)[1]),
        ("seg2", _signals(11, n=60)[0], _signals(11, n=60)[1]),
        ("seg3", _signals(12, n=60)[0], _signals(12, n=60)[1]),
    ]
    across = pipe.run_across_stimulus_shuffle_audit(
        segments,
        wcc_window_size=10,
        n_shuffles=8,
    )
    assert across["step"] == "across_stimulus_shuffle_audit"
    assert across["n_segments"] == 3

    group = pipe.run_group_condition_inference(n_permutations=20, contrast=("rest", "task"))
    assert group is not None


def test_run_audited_evidence_chain_returns_summary():
    df = _feature_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=20.0, surrogate_n=8, seed=2)
    raw = {"dyad_0": _signals(20)}
    cohort = {f"dyad_{i}": _signals(30 + i) for i in range(3)}
    result = pipe.run_audited_evidence_chain(
        raw,
        wcc_window_size=20,
        design_signal_pairs=cohort,
        n_permutations=20,
    )
    assert result["evidence_chain_version"] == "v1"
    assert "Synchrony-existence" in result["summary"]
    assert result["synchrony_existence"]["n_pairs"] == 1
    assert result["design_controls"] is not None
    assert result["group_condition_inference"] is not None

# === source: test_cascade_summary_l2_path.py ===
"""Regression test for the _build_cascade_summary() NameError on the
n_l2_sig > 0 path.

Background: _build_cascade_summary references FDR_FEATURES, but FDR_FEATURES was
only imported locally inside two methods, never at module top level. The
reference lives ONLY in the `n_l2_sig > 0` branch, so the crash was invisible
unless at least one feature passed BH-FDR — i.e. exactly the "we have a finding"
case. These tests exercise BOTH branches so the bug cannot silently return.
"""
import pytest

from multisync.inference_pipeline import _build_cascade_summary


def test_cascade_summary_no_l2_significant():
    """n_l2_sig == 0 branch (was always safe)."""
    s = _build_cascade_summary(l0_pass=3, l0_total=5, l1_pass=2, l1_total=5,
                               l2_results={"n_significant": 0})
    assert isinstance(s, str) and "L2" in s


def test_cascade_summary_some_l2_significant():
    """n_l2_sig > 0 branch — used to raise NameError: name 'FDR_FEATURES'."""
    s = _build_cascade_summary(l0_pass=3, l0_total=5, l1_pass=2, l1_total=5,
                               l2_results={"n_significant": 2})
    assert isinstance(s, str) and "L2" in s


def test_cascade_summary_strong_l2_significant():
    """n_l2_sig >= 4 branch (also references FDR_FEATURES)."""
    s = _build_cascade_summary(l0_pass=4, l0_total=5, l1_pass=3, l1_total=5,
                               l2_results={"n_significant": 4})
    assert isinstance(s, str) and "L2" in s


def test_fdr_features_importable_at_module_top():
    """FDR_FEATURES must be bound at module global scope (the actual fix)."""
    import multisync.inference_pipeline as ip
    assert hasattr(ip, "FDR_FEATURES")
    assert len(ip.FDR_FEATURES) >= 1

# === source: test_full_cascade_integration.py ===
"""END-TO-END integration test for run_full_cascade().

Motivation: three separate bugs (missing surrogate key, _build_cascade_summary
NameError, and between_condition_fdr kwarg-name drift) all passed the unit suite
because NO test ever drove run_full_cascade() all the way through L2. Each bug
only surfaced once the previous blocker was removed. This test exercises the
WHOLE cascade on a synthetic multi-dyad dataset and asserts a complete result,
so any future signature drift in between_condition_fdr / test_l2_condition is
caught immediately instead of by hand.
"""
import numpy as np
import pandas as pd
import pytest

from multisync.inference_pipeline import InferencePipeline
from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features
from multisync.feature_definitions import FDR_FEATURES


def _make_dyad_signals(rng, coupling, n=600):
    """Two signals sharing a latent driver at strength `coupling`."""
    shared = np.cumsum(rng.normal(0, 1, n))
    a = coupling * shared + rng.normal(0, 1.5, n)
    b = coupling * shared + rng.normal(0, 1.5, n)
    return a, b


@pytest.fixture
def cascade_inputs():
    """Build a paired multi-dyad dataset: each dyad has rest (low coupling) and
    task (high coupling) conditions, exactly one row each (L2 needs pairing)."""
    rng = np.random.default_rng(7)
    hz = 1.0
    window = 30
    rows, wcc_dict, raw_dict = [], {}, {}
    n_dyads = 8
    for d in range(n_dyads):
        for cond, coup in (("rest", 0.1), ("task", 0.9)):
            a, b = _make_dyad_signals(rng, coup)
            wcc = sliding_window_wcc(a, b, window_size=window)
            feats = extract_dynamic_features(wcc, hz=hz, wcc_window_sec=window / hz)
            label = f"dyad{d}__{cond}"
            wcc_dict[label] = wcc
            raw_dict[label] = (a, b)
            row = {"dyad_id": f"dyad{d}", "condition": cond, "label": label}
            row.update(feats.to_dict())
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, wcc_dict, raw_dict, window, hz


@pytest.mark.slow
def test_run_full_cascade_returns_complete_summary(cascade_inputs):
    df, wcc_dict, raw_dict, window, hz = cascade_inputs
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=50, seed=1)

    # This call exercises L0 + L1 surrogate tests AND L2 between_condition_fdr.
    # It would have crashed on: missing surrogate key / NameError / kwarg drift.
    result = pipe.run_full_cascade(
        raw_signals_dict=raw_dict,
        wcc_dict=wcc_dict,
        wcc_window_size=window,
        condition_col="condition",
        dyad_col="dyad_id",
        n_permutations=200,
    )

    # ---- structural assertions on the full cascade output ----
    for key in ("l0_summary", "l1_summary", "l2_results", "cascade_summary"):
        assert key in result, f"missing top-level key: {key}"

    # cascade_summary must be a non-empty string mentioning all three levels
    cs = result["cascade_summary"]
    assert isinstance(cs, str) and "L0" in cs and "L1" in cs and "L2" in cs

    # per-feature pass dicts (no OR-aggregate) present and symmetric L0/L1
    assert "per_feature_pass" in result["l0_summary"]
    assert "per_feature_pass" in result["l1_summary"]
    assert result["l0_summary"]["primary_feature"] == "peak_amplitude"
    assert result["l1_summary"]["primary_feature"] == "switching_rate"

    # L2 ran and produced per-feature FDR output
    l2 = result["l2_results"]
    assert l2 is not None


@pytest.mark.slow
def test_run_full_cascade_l2_param_names_are_correct(cascade_inputs):
    """Directly guards against between_condition_fdr kwarg-name drift:
    a wrong kwarg (e.g. fdr_alpha=/contrast=) would raise TypeError here."""
    df, wcc_dict, raw_dict, window, hz = cascade_inputs
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=20, seed=1)
    # test_l2_condition with an explicit contrast -> must map to condition_values
    res = pipe.test_l2_condition(
        condition_col="condition", dyad_col="dyad_id",
        fdr_alpha=0.05, n_permutations=100, contrast=("rest", "task"),
    )
    assert res is not None

# === source: test_l1_applicable_guard.py ===
"""Regression tests for gstack Finding 6 (corrected): L1 silent miscount.

A WCC too short for a valid L1 surrogate test must be reported as
"test not applicable" and EXCLUDED from the L1 denominator (l1_total /
n_l1), not silently counted as "L1 not significant".  Counting it as not
significant deflates the reported L1 pass rate in datasets that contain a
non-trivial fraction of short traces.
"""

import numpy as np
import pandas as pd
import pytest

from multisync.inference_pipeline import InferencePipeline
from multisync.dynamic_features import wcc_surrogate_test, sliding_window_wcc
from multisync.feature_definitions import extract_features


def _make_signals(rng, coupling, n=300):
    t = np.arange(n)
    a = np.sin(2 * np.pi * 0.05 * t) + coupling * np.sin(2 * np.pi * 0.05 * t + 0.3)
    b = np.sin(2 * np.pi * 0.05 * t + 0.3) + coupling * np.sin(2 * np.pi * 0.05 * t)
    a = a + 0.1 * rng.standard_normal(n)
    b = b + 0.1 * rng.standard_normal(n)
    return a, b


def test_wcc_level_early_return_is_not_applicable():
    rng = np.random.default_rng(0)
    # A 10-point WCC is below min_wcc_points (30) -> early return.
    short_wcc = rng.standard_normal(10)
    res = wcc_surrogate_test(
        short_wcc, hz=4.0, surrogate_n=5, seed=0, raw_signals=None,
        null_model="state_shuffle",
    )
    assert res.get("applicable", True) is False
    assert "too short" in res.get("notes", "").lower()


def test_wcc_level_normal_return_is_applicable():
    rng = np.random.default_rng(1)
    a, b = _make_signals(rng, 0.5, n=300)
    wcc = sliding_window_wcc(a, b, window_size=40, hz=4.0)
    res = wcc_surrogate_test(
        wcc, hz=4.0, surrogate_n=5, seed=0, raw_signals=None,
        null_model="state_shuffle",
    )
    assert res.get("applicable", False) is True


def _build_split_dataset():
    """Four dyads x two conditions. Odd-indexed dyads (d1, d3) use short
    signals so their WCC is below min_wcc_points -> L1 'not applicable'.
    Even dyads (d0, d2) use long signals."""
    rng = np.random.default_rng(3)
    window = 40
    hz = 4.0

    def sig(coupling, n):
        return _make_signals(rng, coupling, n=n)

    wcc_dict, raw_dict, rows = {}, {}, []
    for di in range(4):
        short = (di % 2 == 1)
        n = 50 if short else 400
        for cond, coup in (("rest", 0.6), ("task", 0.7)):
            a, b = sig(coup, n)
            wcc = sliding_window_wcc(a, b, window_size=window, hz=hz)
            feat = extract_features(wcc, hz=hz, wcc_window_sec=window / hz)
            label = f"d{di}__{cond}"
            wcc_dict[label] = wcc
            raw_dict[label] = (a, b)
            rows.append(
                {"dyad_id": f"d{di}", "condition": cond, "label": label,
                 **feat.to_dict()}
            )
    df = pd.DataFrame(rows)
    return df, wcc_dict, raw_dict, window, hz


@pytest.mark.slow
def test_run_full_cascade_excludes_inapplicable_l1_from_denominator():
    df, wcc_dict, raw_dict, window, hz = _build_split_dataset()
    pipe = InferencePipeline(features_df=df, hz=hz, surrogate_n=5, seed=1)

    result = pipe.run_full_cascade(
        raw_signals_dict=raw_dict,
        wcc_dict=wcc_dict,
        wcc_window_size=window,
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=20,
    )

    # L1: only d0/d2's four labels are applicable; d1/d3's four short-WCC
    # labels are excluded from the denominator (not counted as
    # L1-not-significant).
    assert result["l1_summary"]["total"] == 4, (
        "short-WCC dyads must be excluded from l1_total, not counted as "
        "L1-not-significant"
    )
    # L0 is unaffected: all eight labels have raw signals and run L0.
    assert result["l0_summary"]["total"] == 8


def test_summarize_excludes_inapplicable_l1():
    wcc_n = sliding_window_wcc(*_make_signals(np.random.default_rng(5), 0.6, n=400),
                               window_size=40, hz=4.0)
    short_wcc = np.random.default_rng(5).standard_normal(10)

    pipe = InferencePipeline(features_df=pd.DataFrame(), hz=4.0, surrogate_n=5, seed=1)
    pipe.test_l1_structure(wcc_n, label="normal")
    pipe.test_l1_structure(short_wcc, label="short")

    summary = pipe.summarize()
    # denominator must be 1 (only the applicable dyad), not 2.
    assert "L1 (WCC-level IAAFT): 0/1 significant" in summary, (
        f"summarize must exclude inapplicable L1 dyad from denominator; got:\n{summary}"
    )

# === source: test_l2_fixes.py ===
"""Regression tests for L2 between-condition fixes:

- `cohens_d` field renamed to `perm_effect_size` (it is observed_diff /
  SD(null), NOT classical Cohen's d).  The old `cohens_d` attribute is a
  deprecated property that warns.
- Small-n (n_dyads <= 12) now uses *exact* enumeration of all 2^n sign
  flips, giving an honest discrete p-value resolution (1/(2^n + 1)),
  not the spurious 1/(n_permutations + 1).
"""

import numpy as np
import pandas as pd
import pytest

from multisync.validation.l2_between_condition import between_condition_fdr


def _small_df(n_dyads=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dyads):
        a, b = rng.normal(1.0, 0.3), rng.normal(0.5, 0.3)
        rows.append({"dyad_label": f"d{i}", "condition": "A", "peak_amplitude": a})
        rows.append({"dyad_label": f"d{i}", "condition": "B", "peak_amplitude": b})
    return pd.DataFrame(rows)


def test_perm_effect_size_field_and_summary_column():
    df = _small_df(n_dyads=6)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=1
    )
    r = res["per_feature"][0]
    # New canonical name exists and is a float.
    assert hasattr(r, "perm_effect_size")
    assert isinstance(r.perm_effect_size, float)
    # summary_df column renamed (no longer "cohens_d").
    assert "perm_effect_size" in res["summary_df"].columns
    assert "cohens_d" not in res["summary_df"].columns


def test_cohens_d_attribute_deprecated():
    df = _small_df(n_dyads=6)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=1
    )
    r = res["per_feature"][0]
    with pytest.warns(DeprecationWarning):
        val = r.cohens_d
    # The deprecated alias returns the same value as the canonical field.
    assert val == r.perm_effect_size


def test_small_n_exact_discrete_p_resolution():
    """With n_dyads=4, the exact null has 2^4 = 16 points, so p_raw must be
    a multiple of 1/17 (honest resolution), NOT 1/10001."""
    df = _small_df(n_dyads=4)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=10000, seed=2
    )
    p_raw = res["per_feature"][0].p_raw
    # p_raw = (n_ge + 1) / (16 + 1); check it lands on the 1/17 grid.
    scaled = p_raw * 17.0
    assert abs(scaled - round(scaled)) < 1e-9, f"p_raw={p_raw} not on 1/17 grid"
    assert 0.0 <= p_raw <= 1.0


def test_large_n_still_runs():
    """n_dyads > 12 takes the Monte-Carlo path and returns finite results."""
    df = _small_df(n_dyads=20, seed=5)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude"], n_permutations=2000, seed=3
    )
    assert res["n_dyads"] == 20
    assert np.isfinite(res["per_feature"][0].p_raw)

# === source: test_l2_modality_seed_determinism.py ===
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

# === source: test_l2_vif_gate.py ===
"""
VIF gate integration test (gstack #14).

The L2 between-condition test now attaches a ``vif_gate`` diagnostic to its
result: when FDR-family features are severely collinear the gate must FLAG
it (warn + report) rather than silently drop frozen features.
"""
import numpy as np
import pandas as pd

from multisync.validation.l2_between_condition import between_condition_fdr


def _make_df(rng, n, extra=None):
    base = {
        "dyad_label": [f"d{i}" for i in range(n)] * 2,
        "condition": ["A"] * n + ["B"] * n,
        "peak_amplitude": rng.normal(0.5, 0.1, n * 2),
        "dwell_time": rng.normal(10.0, 2.0, n * 2),
        "switching_rate": rng.normal(3.0, 1.0, n * 2),
    }
    if extra:
        base.update(extra)
    return pd.DataFrame(base)


def test_vif_gate_present_and_passes_for_independent_features():
    rng = np.random.default_rng(0)
    df = _make_df(rng, 14)
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude", "dwell_time", "switching_rate"],
        n_permutations=100, seed=1,
    )
    assert "vif_gate" in res
    assert res["vif_gate"]["passed"] is True
    assert res["vif_gate"]["skipped"] is False


def test_vif_gate_flags_severe_collinearity():
    rng = np.random.default_rng(1)
    # build a peak_amplitude column, then inject an EXACT copy of it so the
    # two FDR-family features are perfectly collinear (VIF -> inf -> severe)
    pa = rng.normal(0.5, 0.1, 28)
    df = _make_df(rng, 14, extra={"pa_dup": pa})
    df["pa_dup"] = df["peak_amplitude"]  # force exact collinearity
    res = between_condition_fdr(
        df, feature_cols=["peak_amplitude", "pa_dup"],
        n_permutations=100, seed=1,
    )
    assert "vif_gate" in res
    assert res["vif_gate"]["passed"] is False
    assert "pa_dup" in res["vif_gate"]["vif_severe"]

# === source: test_p0_technical_fixes.py ===
"""Regression tests for P0/P1 technical fixes (2026-07-22).

Copy into SyncPipe/tests/ and run:
    pytest tests/test_p0_technical_fixes.py -q
"""

import warnings

import numpy as np
import pandas as pd
import pytest


def test_p0_1_l0_peak_matches_ssot_smoothed_peak():
    """Existence-null obs peak must equal SSoT peak_amplitude (smoothed)."""
    from multisync.feature_definitions import extract_features, smoothed_wcc, compute_peak_amplitude
    from multisync.dynamic_features import _signal_level_surrogate_test

    rng = np.random.default_rng(0)
    # Spiky WCC: raw max >> smoothed peak
    wcc = np.zeros(120)
    wcc[50] = 0.95
    wcc[51:56] = 0.45
    wcc += rng.normal(0, 0.02, size=120)

    feat = extract_features(wcc, hz=1.0, wcc_window_sec=20.0, threshold=0.5)
    ssot_peak = feat.peak_amplitude
    raw_max = float(np.max(wcc[np.isfinite(wcc)]))
    assert abs(raw_max - ssot_peak) > 0.1, "fixture must create raw vs smoothed gap"

    # Build trivial raw signals long enough for signal-level path
    n = 200
    sig_a = rng.normal(size=n)
    sig_b = 0.6 * sig_a + rng.normal(scale=0.3, size=n)
    # Use the spiky series as "observed WCC" with matching window size heuristic off
    result = _signal_level_surrogate_test(
        sig_A=sig_a,
        sig_B=sig_b,
        wcc=wcc,
        hz=1.0,
        surrogate_n=20,
        seed=1,
        wcc_window_size=30,
    )
    obs_peak = result["obs_peak_amplitude"]
    assert np.isfinite(obs_peak)
    assert abs(obs_peak - ssot_peak) < 1e-12, (
        f"L0 obs peak {obs_peak} != SSoT peak {ssot_peak} (raw would be {raw_max})"
    )


def test_p0_2_l2_refuses_multimodal_pool():
    from multisync.validation.l2_between_condition import between_condition_fdr

    rows = []
    for d in range(8):
        for cond in ("rest", "task"):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=0.2 if cond == "rest" else 0.8,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="ECG",
                    peak_amplitude=0.8 if cond == "rest" else 0.2,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="modalit"):
        between_condition_fdr(
            df,
            condition_col="condition",
            dyad_col="dyad_id",
            feature_cols=["peak_amplitude"],
            n_permutations=50,
            seed=0,
            condition_values=("rest", "task"),
        )


def test_p0_2_single_modality_still_works():
    from multisync.validation.l2_between_condition import between_condition_fdr

    rows = []
    for d in range(8):
        for cond, val in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=val,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    df = pd.DataFrame(rows)
    out = between_condition_fdr(
        df,
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=100,
        seed=0,
        condition_values=("rest", "task"),
    )
    assert out["n_dyads"] == 8
    assert abs(out["per_feature"][0].observed_diff - (0.2 - 0.8)) < 1e-9


def test_p0_3_finite_pair_warns_on_length_mismatch():
    from multisync.design_controls import _finite_pair

    a = np.arange(100.0)
    b = np.arange(70.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aa, bb = _finite_pair(a, b)
        assert any("unequal lengths" in str(x.message) for x in w)
    assert len(aa) == len(bb) == 70


def test_p0_3_finite_pair_raise_mode():
    from multisync.design_controls import _finite_pair

    with pytest.raises(ValueError, match="unequal lengths"):
        _finite_pair(np.arange(10.0), np.arange(7.0), on_length_mismatch="raise")


def test_p1_1_dwell_splits_across_nan_seam():
    from multisync.feature_definitions import compute_dwell_time

    # two elevated runs of length 3 separated by NaN seam
    w = np.array([0.1, 0.1, 0.8, 0.8, 0.8, np.nan, 0.8, 0.8, 0.8, 0.1, 0.1])
    d = compute_dwell_time(w, hz=1.0, threshold=0.5, hysteresis_delta=0.0, gap_policy="segment")
    # explicit gap_policy="segment": mean of two length-3 runs = 3.0
    assert d == pytest.approx(3.0)
    # opt-in merge_valid restores 2026-07-13 glue-across-gap behaviour
    d_merge = compute_dwell_time(
        w, hz=1.0, threshold=0.5, hysteresis_delta=0.0, gap_policy="merge_valid"
    )
    assert d_merge == pytest.approx(6.0)


def test_p1_4_nan_fraction_set_in_ssot():
    from multisync.feature_definitions import extract_features

    w = np.array([0.1, 0.2, np.nan, 0.8, 0.9, 0.7, np.nan, 0.1])
    f = extract_features(w, hz=1.0, wcc_window_sec=5.0)
    assert f.nan_fraction == pytest.approx(2.0 / 8.0)

# === source: test_p1_residual_fixes.py ===
"""P1 residual fixes (contrast forward, gap_policy on mask, session masks)."""

import numpy as np
import pandas as pd
import pytest

from multisync.computation_pipeline import ComputationPipeline
from multisync.feature_definitions import compute_dwell_time, extract_features
from multisync.inference_pipeline import InferencePipeline
from multisync.session_threshold import (
    compute_session_pooled_threshold,
    _generate_surrogate_coupling_matrix,
)
from multisync.dynamic_features import sliding_window_wcc, _apply_discontinuity_mask


def _multimodal_df(n_dyads=8):
    rows = []
    for d in range(n_dyads):
        for cond, val_eda in (("rest", 0.2), ("task", 0.8)):
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="EDA",
                    peak_amplitude=val_eda,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
            # opposite pattern on ECG
            rows.append(
                dict(
                    dyad_id=f"d{d}",
                    condition=cond,
                    modality="ECG",
                    peak_amplitude=1.0 - val_eda,
                    dwell_time=1.0,
                    switching_rate=1.0,
                )
            )
    return pd.DataFrame(rows)


def test_p1_r1_multimodal_forwards_contrast():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=7)
    out = pipe.run_group_condition_inference(
        condition_col="condition",
        dyad_col="dyad_id",
        feature_cols=["peak_amplitude"],
        n_permutations=50,
        contrast=("rest", "task"),
        modality_col="modality",
    )
    assert set(out) >= {"EDA", "ECG"}
    for mod in ("EDA", "ECG"):
        assert "error" not in out[mod], out[mod]
        assert out[mod]["condition_a"] == "rest"
        assert out[mod]["condition_b"] == "task"


def test_p1_r1_by_modality_accepts_contrast_and_seed():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=99)
    out = pipe.test_l2_by_modality(
        feature_cols=["peak_amplitude"],
        n_permutations=40,
        contrast=("task", "rest"),  # reversed on purpose
        seed=99,
    )
    for mod, sub in out.items():
        assert sub["condition_a"] == "task"
        assert sub["condition_b"] == "rest"


def test_p1_r1_chain_summary_counts_multimodal():
    df = _multimodal_df()
    pipe = InferencePipeline(df, hz=1.0, wcc_window_sec=10.0, surrogate_n=5, seed=0)
    # Build dummy existence-compatible raw signals (short chain call heavy);
    # unit-test the summary helper directly.
    group = {
        "EDA": {"n_significant": 2},
        "ECG": {"n_significant": 1},
    }
    s = InferencePipeline._build_audited_chain_summary(
        {"n_pairs": 0}, None, None, group
    )
    assert "per-modality" in s
    assert "3 significant" in s


def test_p1_r2_masked_pipeline_uses_segment_gap_policy():
    """Discontinuity mask mid-run: segment policy must not merge dwells."""
    n = 100
    a = np.sin(np.linspace(0, 8, n))
    b = a.copy()
    # Strong coupling whole series; mask creates a seam in the middle
    mask = np.ones(n, dtype=bool)
    mask[48:53] = False  # seam
    pipe = ComputationPipeline(hz=1.0, window_size=10, onset_threshold=0.3)
    pipe.load_signals(a, b, discontinuity_mask=mask)
    pipe.compute_wcc()
    feats = pipe.extract_features()
    assert feats.params.get("gap_policy") == "segment"
    # Reference: same WCC with explicit policies
    wcc = pipe._wcc
    d_seg = compute_dwell_time(wcc, hz=1.0, threshold=0.3, gap_policy="segment")
    d_merge = compute_dwell_time(wcc, hz=1.0, threshold=0.3, gap_policy="merge_valid")
    assert feats.dwell_time == pytest.approx(d_seg, nan_ok=True)
    # When both defined, segment dwell should be <= merge (no glue)
    if np.isfinite(d_seg) and np.isfinite(d_merge):
        assert d_seg <= d_merge + 1e-9


def test_p1_r2_unmasked_keeps_merge_valid_default():
    n = 80
    a = np.sin(np.linspace(0, 6, n))
    b = a + 0.01
    pipe = ComputationPipeline(hz=1.0, window_size=10, onset_threshold=0.3)
    pipe.load_signals(a, b)  # no mask
    pipe.compute_wcc()
    feats = pipe.extract_features()
    assert feats.params.get("gap_policy") == "merge_valid"


def test_p1_r3_session_threshold_applies_mask():
    rng = np.random.default_rng(0)
    n = 120
    a = rng.normal(size=n)
    b = 0.7 * a + rng.normal(scale=0.2, size=n)
    mask = np.ones(n, dtype=bool)
    mask[40:80] = False  # large gated region
    window = 15
    # Finite counts with vs without mask on a single surrogate matrix draw
    m_plain = _generate_surrogate_coupling_matrix(
        a, b, hz=1.0, window_size=window, surrogate_n=5, seed=1
    )
    m_mask = _generate_surrogate_coupling_matrix(
        a, b, hz=1.0, window_size=window, surrogate_n=5, seed=1,
        discontinuity_mask=mask,
    )
    assert np.isfinite(m_mask).sum() < np.isfinite(m_plain).sum()

    thr_plain, meta_p = compute_session_pooled_threshold(
        [(a, b)], hz=1.0, wcc_window_size=window, surrogate_n=20, seed=2
    )
    thr_mask, meta_m = compute_session_pooled_threshold(
        [(a, b)],
        hz=1.0,
        wcc_window_size=window,
        surrogate_n=20,
        seed=2,
        discontinuity_masks=[mask],
    )
    assert meta_m.get("n_discontinuity_masks_applied") == 1
    assert meta_m["n_finite_coupling_values"] < meta_p["n_finite_coupling_values"]


def test_p1_r3_mask_length_mismatch_raises():
    a = np.zeros(50)
    b = np.zeros(50)
    with pytest.raises(ValueError, match="discontinuity_masks"):
        compute_session_pooled_threshold(
            [(a, b), (a, b)],
            hz=1.0,
            wcc_window_size=10,
            surrogate_n=5,
            discontinuity_masks=[None],  # wrong length
        )


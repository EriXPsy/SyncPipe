"""Shared pytest fixtures for the SyncPipe test suite -- Phase 0 scaffold.

This is a deliberately *minimal* fixture set (plan constraint: build from the
smallest useful set, do not over-engineer). It is the landing pad for the
later consolidation phases (unit/integration/contracts directories). No
existing test is rewired to use these fixtures yet -- Phase 0 only stands up
the scaffolding. See ``tests/README.md`` for the run/regression contract.

Design constraints (plan §4.3 / §4.4):
  * Fixture names are ``snake_case`` and never prefixed with ``test_``.
  * No network / OSF I/O -- everything is synthesized locally.
  * Nothing is written outside ``tmp_path``; these fixtures are read-only.
  * No global warning suppression, no ``numpy`` print-option changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Function-scoped reproducible RNG.

    Defaults to ``seed=0``. Override by redefining this fixture in a test
    module, or by passing an explicit seed to the consuming code path.
    """
    return np.random.default_rng(seed=0)


@pytest.fixture
def toy_signals(rng):
    """A pair of *moderately coupled* 1-D signals for prediction-style tests.

    Returns ``(sig_a, sig_b, hz)``. ``sig_b`` tracks ``sig_a`` with a small
    lag plus a shared oscillatory drive, so the two are correlated but not
    identical -- enough structure for a real (non-degenerate) synchrony
    signal without any external data.
    """
    hz = 4.0
    n = 1200
    t = np.arange(n) / hz
    sig_a = np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.1, n)
    sig_b = (
        0.6 * np.roll(sig_a, 5)
        + 0.3 * np.sin(2 * np.pi * 0.5 * t + 0.4)
        + rng.normal(0, 0.1, n)
    )
    return sig_a, sig_b, hz


@pytest.fixture
def features_df_uni(rng):
    """``dyad`` x ``condition`` DataFrame with the real FDR feature columns.

    Values are synthetic but the column names track the actual frozen
    FDR family (``syncpipe.feature_definitions.FDR_FEATURES``), so the
    fixture is drop-in compatible with significance / FDR consumers. Task
    condition is slightly elevated vs rest to give tests a real effect to
    detect.
    """
    from syncpipe.feature_definitions import FDR_FEATURES

    dyads = [f"d{i}" for i in range(6)]
    conditions = ["rest", "task"]
    rows = []
    for dyad in dyads:
        for cond in conditions:
            row = {"dyad_id": dyad, "condition": cond}
            base = 0.3 if cond == "rest" else 0.7
            for feat in FDR_FEATURES:
                row[feat] = float(np.clip(base + rng.normal(0, 0.1), 0.0, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def features_df_multi(features_df_uni):
    """``features_df_uni`` augmented with a ``modality`` column (EDA/ECG).

    Each ``(dyad, condition)`` row is duplicated across the two modalities
    with modality-appropriate scaling so the two are not identical (keeps
    multimodal tests meaningful).
    """
    frames = []
    for modality, scale in (("EDA", 1.0), ("ECG", 0.8)):
        df = features_df_uni.copy()
        df["modality"] = modality
        num_cols = [
            c for c in df.columns if c not in ("dyad_id", "condition", "modality")
        ]
        df[num_cols] = df[num_cols] * scale + 0.05
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

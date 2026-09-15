"""Facade re-export parity + pairing single-source-of-truth guards.

``syncpipe.dynamic_features`` is a compatibility *facade*: the WCC kernels,
null models, feature extraction, pairing helpers and surrogates now live in
their own modules and are merely re-exported here.  This file pins the
**identity** (``is``, not value equality) of every re-exported name to its home
module, so that:

* a facade-local *re-definition* fails loudly instead of silently shadowing the
   SSoT objects every downstream caller imports (same defect class as the
   unreachable ``PairResult`` / ``compute_pair_pipeline`` duplicate removed from
   ``computation_pipeline.py`` on 2026-09-15), and
* ``pairing.py`` stays the single source of truth that feature extraction and
   WCC caching must reuse (``docs/REPOSITORY_SCOPE.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import syncpipe.core as core_module
import syncpipe.dynamic_feature_extraction as extraction_module
import syncpipe.dynamic_features as facade
import syncpipe.null_models as null_models_module
import syncpipe.pairing as pairing_module
import syncpipe.surrogate as surrogate_module
import syncpipe.wcc as wcc_module
from syncpipe.core import Dyad, DynamicAnalyzer


def _assert_reexport(name: str, home, home_name: str) -> None:
    """``facade.<name>`` must be the very object defined in ``home``."""
    assert hasattr(facade, name), f"facade lost its re-export of {name!r}"
    assert getattr(facade, name) is getattr(home, name), (
        f"syncpipe.dynamic_features.{name} is not the object from {home_name} "
        f"— a facade-local definition shadows the single source of truth."
    )


def test_facade_wcc_reexports_are_identical_to_wcc_module():
    for name in (
        "sliding_window_wcc",
        "sliding_window_wcc_masked",
        "_apply_discontinuity_mask",
        "_make_window_kernel",
        "_sliding_window_wcc_cumsum",
        "_sliding_window_wcc_stride",
    ):
        _assert_reexport(name, wcc_module, "syncpipe.wcc")


def test_facade_null_models_reexports_are_identical_to_null_models_module():
    for name in (
        "wcc_surrogate_test",
        "_signal_level_surrogate_test",
        "_wcc_level_surrogate_test",
        "_prepare_iaaft_segments",
        "_segmentwise_wcc",
        "_empty_result",
        "compute_surrogate_threshold_from_signals",
    ):
        _assert_reexport(name, null_models_module, "syncpipe.null_models")


def test_facade_extraction_pairing_surrogate_reexports_are_identical():
    _assert_reexport(
        "extract_dynamic_features", extraction_module,
        "syncpipe.dynamic_feature_extraction",
    )
    for name in ("pairing_policy", "iter_dyad_pairs"):
        _assert_reexport(name, pairing_module, "syncpipe.pairing")
    for name in ("iaaft_surrogate", "ft_surrogate", "prtf_surrogate"):
        _assert_reexport(name, surrogate_module, "syncpipe.surrogate")


def test_pairing_is_single_source_of_truth_for_facade_and_core():
    """Both the facade and ``core`` must reuse ``pairing``'s objects, never a
    copy — ``pair.keys`` (WCC-cache keys, manifest ``pairing_policy``) are only
    consistent if every path binds the same functions."""
    assert facade.iter_dyad_pairs is pairing_module.iter_dyad_pairs
    assert facade.pairing_policy is pairing_module.pairing_policy
    assert core_module.iter_dyad_pairs is pairing_module.iter_dyad_pairs
    assert core_module.pairing_policy is pairing_module.pairing_policy


def _facet_dataset() -> Dyad:
    """Small 3-signal single-modality dataset -> 3 same-modality pairs."""
    n = 40
    t = np.arange(n, dtype=float)
    eda = pd.DataFrame({
        "time": t,
        "person_a": np.sin(t),
        "person_b": np.cos(t),
        "person_c": np.sin(t + 0.5),
    })
    ds = Dyad(dyad_id="facade_parity", hz=1.0, eda=eda)
    ds.align(target_hz=1.0).zscore()
    return ds


def test_core_pairing_path_matches_direct_pairing_iteration():
    """``DynamicAnalyzer``'s pair path must yield the exact same pair keys as a
    direct ``pairing.iter_dyad_pairs`` call (proves reuse, not reimplementation)."""
    ds = _facet_dataset()
    direct = [pair[0] for pair in pairing_module.iter_dyad_pairs(ds)]
    analyzer = DynamicAnalyzer(window_size=5, surrogate_n=2, run_qc=False)
    via_core = [pair[0] for pair in analyzer._iter_pairs(ds)]
    assert direct, "fixture produced no pairs; the guard would be vacuous"
    assert via_core == direct

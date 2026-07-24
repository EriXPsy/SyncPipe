# SyncPipe Test Suite

This directory holds the SyncPipe test suite, organized as a **layered +
themed** structure. The 55 former flat `test_*.py` files were consolidated
into 8 themed suites under `unit/`, `integration/`, `contracts/` (the slow
`validation/` layer was already separate and is untouched). `conftest.py`
stays at the repo root and is auto-discovered by every sub-directory suite.

## Layout

```
tests/
├── conftest.py            # root fixtures (rng, toy_signals, features_df_uni/multi)
├── __init__.py
├── README.md
├── test_feature_definitions.py   # PINNED SSoT guard — stays standalone (never merged)
├── unit/
│   ├── test_features.py          # feature_definitions / dynamic_features / wcc / core / morphology
│   ├── test_significance.py      # surrogate / null models / significance / fdr / existence audit
│   ├── test_prediction.py        # prediction / cross_modal / failed_fold / seed / finding17 / leakage
│   ├── test_pipeline_io.py       # computation_pipeline / pipeline_bridge / io / dataset / loader
│   └── test_api_core.py          # cli / demo / public API entry tests
├── integration/
│   └── test_inference.py         # inference_pipeline / l0 / l1 / l2 / design_control / group inference
├── contracts/
│   └── test_release_contracts.py # p2 release hygiene + parity / audit-interface contract tests
└── validation/                   # SLOW layer — 4 files, all marked @pytest.mark.slow
    └── test_*.py
```

### Pinned independent guard

`tests/test_feature_definitions.py` is an SSoT (single source of truth)
methodology-isolation guard. It is intentionally **not** merged into any other
suite and lives as its own standalone file at the repo root:

```bash
pytest tests/test_feature_definitions.py -v
```

### Merge notes

- Module-level helper name collisions were resolved by suffixing (e.g.
  `_make_signals` → `_make_signals_fold` / `_make_signals_find17` /
  `_make_signals_seed`; `_make_wcc` → `_make_wcc_hyg` / `_make_wcc_parity`).
- `from __future__ import annotations` appears once, at the top of each suite.
- `test_feature_table_consistency.py` carried a module-level
  `pytestmark = skipif(...)` that would have skipped the whole merged suite;
  it was moved to a `_requires_build` decorator on the three tests that need
  `scripts/build_feature_table.py` (the CSV test also keeps its own skip).
- `conftest.py` (4 fixtures) is unchanged and discovered by sub-dir tests.

## How to run

```bash
# Daily / pre-commit / PR gate — everything EXCEPT the slow validation layer:
pytest tests/ -m "not slow" -q

# Full suite (CI main / nightly only — includes the slow validation layer):
pytest tests/ -q

# One themed suite:
pytest tests/unit/test_prediction.py -q
pytest tests/integration/test_inference.py -q
pytest tests/contracts/test_release_contracts.py -q

# Validation layer only (the explicitly slow tests):
pytest tests/validation -m slow -q
```

## Marker convention

Markers are declared in `pyproject.toml` (`[tool.pytest.ini_options]`).

| Marker  | Meaning | Included in PR run (`-m "not slow"`)? |
|---------|---------|----------------------------------------|
| `slow`  | Long-running validation / integration. | **No** — excluded. |
| (none)  | Default: fast unit / smoke. | **Yes** — this is the default. |

We do **not** use a `fast` marker: the convention is *unmarked = fast*. CI runs
`not slow` on every PR/push and the full suite only on `main` push / nightly.

## Shared fixtures (`conftest.py`)

A deliberately minimal set — extend only when a real duplication appears:

- `rng` — function-scoped `np.random.default_rng(seed=0)`.
- `toy_signals` — `(sig_a, sig_b, hz)`: a pair of *moderately coupled* 1-D signals.
- `features_df_uni` — `dyad × condition(rest/task)` DataFrame over the real
  FDR feature family (`multisync.feature_definitions.FDR_FEATURES`).
- `features_df_multi` — `features_df_uni` augmented with a `modality` column
  (`EDA` / `ECG`).

Constraints: snake_case names (no `test_` prefix), no network/OSF I/O, no writes
outside `tmp_path`, no global warning suppression, no `numpy` print-option
changes.

## Regression gate (collect-only count)

Any future consolidation or rename MUST preserve the number of collected tests.
After the change, the collect-only count must equal the recorded baseline:

```bash
python -m pytest tests/ --collect-only -q | tail -1
```

### Recorded baseline

| Metric                              | Value |
|------------------------------------|-------|
| Collected tests (`--collect-only`) | **388** |
| `slow` subset (`-m slow`)          | 59    |
| `not slow` subset (`-m "not slow"`)| 329   |

> The consolidation moved the 55 flat files into the 8 themed suites above
> (plus the pinned `test_feature_definitions.py`). The collect-only count is
> unchanged at **388** — no test was dropped or duplicated.

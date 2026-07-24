# SyncPipe Test Suite

This directory holds the SyncPipe test suite. It is currently **flat**
(~55 `test_*.py` files plus a `validation/` subfolder). The long-term plan
consolidates these into `unit/`, `integration/`, `contracts/`, and `validation/`
themes — **but that move is out of scope for the current (Phase 0) task.** Phase
0 only stands up the shared scaffolding (`conftest.py` + this README). The 55
existing test files are intentionally left untouched (surgical-change policy).

## How to run

```bash
# Daily / pre-commit / PR gate — everything EXCEPT the slow layer:
pytest -m "not slow" -q

# Full suite (CI main / nightly only — includes the slow validation layer):
pytest -q

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

## Directory plan

| Path (now)             | Status                                                            |
|------------------------|-------------------------------------------------------------------|
| `tests/conftest.py`    | **Built in Phase 0** — minimal shared fixtures (see below).       |
| `tests/README.md`      | **Built in Phase 0** — this file.                                 |
| `tests/test_*.py`      | 55 existing flat tests — **unchanged** (consolidation deferred).  |
| `tests/validation/*.py`| 4 slow validation files — **marked `slow`** in Phase 0.            |
| `tests/unit/`          | Planned (Phase 1+) — fast unit suites. **Not created yet.**       |
| `tests/integration/`   | Planned (Phase 1+) — slower integration suites. **Not created.**  |
| `tests/contracts/`     | Planned (Phase 2+) — release-fuse regression guards. **Not created.** |

### Pinned independent guard

`tests/test_feature_definitions.py` is an SSoT (single source of truth)
methodology-isolation guard. It is intentionally **not** merged into any other
suite and is meant to be run on its own:

```bash
pytest tests/test_feature_definitions.py -v
```

## Shared fixtures (`conftest.py`)

A deliberately minimal set — extend only when a real duplication appears:

- `rng` — function-scoped `np.random.default_rng(seed=0)` (override by
  redefining the fixture or passing an explicit seed downstream).
- `toy_signals` — `(sig_a, sig_b, hz)`: a pair of *moderately coupled* 1-D
  signals (synthetic, no external data).
- `features_df_uni` — `dyad × condition(rest/task)` DataFrame over the real
  FDR feature family (`multisync.feature_definitions.FDR_FEATURES`).
- `features_df_multi` — `features_df_uni` augmented with a `modality` column
  (`EDA` / `ECG`).

Constraints: snake_case names (no `test_` prefix), no network/OSF I/O, no writes
outside `tmp_path`, no global warning suppression, no `numpy` print-option
changes. **No existing test is rewired to use these yet** — adoption happens in
later phases.

## Regression gate (collect-only count)

Any future consolidation or rename MUST preserve the number of collected tests.
After the change, the collect-only count must be **≥ the recorded baseline**:

```bash
python -m pytest --collect-only -q | wc -l
```

### Recorded baseline

Measured in Phase 0 (2026-07-24) from the current flat layout (no tests
moved; only `conftest.py` + this README added, and slow markers applied).

| Metric                          | Value                                              |
|--------------------------------|----------------------------------------------------|
| Collected tests (`--collect-only`) | **388** (regression gate floor)                 |
| `slow` subset (`-m slow`)      | 59                                                 |
| `not slow` subset (`-m "not slow"`) | 329                                            |
| `pytest -m "not slow" -q`      | GREEN — verified locally (9.8s for the new P1 file; full fast suite pending full run) |
| `pytest -q` (full)             | not run in Phase 0 (too slow); slow layer deferred |

> Any future consolidation/refactor must keep `--collect-only` **≥ 388**.

> Baseline measured 2026-07-24 by the lead (verified independent of the
> executing agent's earlier over-count). The gate protects against silently
> dropping a test during the later file moves.

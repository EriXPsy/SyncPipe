# SyncPipe Test Suite

This directory holds the SyncPipe test suite, organized as a **layered +
themed** structure. The 55 former flat `test_*.py` files were consolidated into
themed suites under `unit/`, `integration/`, `contracts/` (the slow
`validation/` layer was already separate and is untouched). `conftest.py`
stays at the repo root and is auto-discovered by every sub-directory suite.

Every merged file keeps a `# === source: <original>.py ===` banner above the
block it absorbed, so a document that cites a pre-consolidation filename can
still be traced by grepping for that banner.

## Layout

```
tests/
├── conftest.py            # root fixtures (rng, toy_signals, features_df_uni/multi)
├── __init__.py
├── README.md
├── test_feature_definitions.py   # PINNED SSoT guard — stays standalone (never merged)
├── test_latest_hardening.py      # architecture-review hardening regression gate
├── test_suite_health.py          # guards the suite itself (see below)
├── unit/
│   ├── test_features.py          # feature_definitions / dynamic_features / wcc / core / morphology / eligibility
│   ├── test_significance.py      # surrogate / null models / significance / fdr / existence audit / kuramoto
│   ├── test_prediction.py        # prediction / cross_modal / failed_fold / seed / finding17 / leakage
│   ├── test_pipeline_io.py       # computation_pipeline / pipeline_bridge / io / dataset / real loaders
│   ├── test_api_core.py          # cli / demo / public API + namespace entry tests
│   ├── test_canonical.py         # canonical_runner / config resolution
│   ├── test_importer.py          # importer / delimiter sniffing
│   ├── test_realtest.py          # realtest loaders (lerique / gordon) unit level
│   └── test_session_threshold.py # session-pooled onset threshold
├── integration/
│   ├── test_inference.py         # inference_pipeline / l0 / l1 / l2 / design_control / group inference
│   └── test_canonical_parity.py  # cross-process canonical parity
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
`pyproject.toml` therefore declares `slow` and nothing else — a declared-but-
never-applied `fast` marker used to sit there, which would let `-m fast` select
zero tests and look like a clean run.

### Promotions to the PR gate

`slow` means "too expensive for every PR", never "less important". When a test
guards a cross-pipeline contract that a nightly-only failure would let reach
`main`, it belongs in the gate if its measured cost is acceptable. Four
`tests/integration/test_inference.py` tests were promoted on 2026-08-02
(measured total ≈ 19 s):

| Test | Why it cannot wait for nightly | Cost |
|---|---|---|
| `test_run_full_cascade_returns_complete_summary` | Only test exercising L0→L1→L2 in one call; catches key/kwarg drift between the three pipelines. | ~11 s |
| `test_by_modality_seed_stable_across_processes` | Cross-process reproducibility is a v1 release claim; per-process `hash()` seeding is invisible in-process. | ~6 s |
| `test_run_full_cascade_excludes_inapplicable_l1_from_denominator` | A silent L1 denominator inflation is a wrong *reported statistic*. | ~0.3 s |
| `test_run_full_cascade_l2_param_names_are_correct` | `between_condition_fdr` kwarg-name contract. | ~0.01 s |

Each promotion carries an inline comment at the test recording the measured cost
and the reason, and the `slow` / `not slow` baseline above must be updated in the
same change (`test_suite_health.py` fails otherwise).

## Shared fixtures (`conftest.py`)

A deliberately minimal set — extend only when a real duplication appears:

- `rng` — function-scoped `np.random.default_rng(seed=0)`.
- `toy_signals` — `(sig_a, sig_b, hz)`: a pair of *moderately coupled* 1-D signals.
- `features_df_uni` — `dyad × condition(rest/task)` DataFrame over the real
  FDR feature family (`syncpipe.feature_definitions.FDR_FEATURES`).
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

This is **enforced automatically** by `tests/test_suite_health.py`, so the table
below is not documentation-only: a drifting count fails the suite. When you
intentionally add or remove tests, update both places in the same commit.

### Recorded baseline

| Metric                              | Value |
|------------------------------------|-------|
| Collected tests (`--collect-only`) | **509** |
| `slow` subset (`-m slow`)          | 67    |
| `not slow` subset (`-m "not slow"`)| 442   |

> Recorded 2026-08-19. Collected total **509** = 442 fast (`not slow`) + 67
> `slow`. Enforced automatically by `tests/test_suite_health.py`; changing these
> numbers is a reviewed act, not a side effect.
>
> History: an earlier baseline (430 = 371 + 59) had gone stale. The split then
> moved from 59/447 to 55/451 when four integration tests were promoted from the
> nightly slow layer to the PR gate (see "Promotions to the PR gate" below).
> On 2026-08-03 nine tests were added to pin the IAAFT `argsort`+scatter
> rewrite and the `n_workers` parallel-existence-audit parity: six in
> test_significance and three in test_inference (506 → 515, 451 → 460).
> On 2026-08-17 the `multisync` → `syncpipe` clean-cut rename removed the
> dual-namespace/`sys.meta_path` alias tests in `test_api_core.py` (515 → 505,
> 460 → 450), and the `prediction.py` move-out removed its unit/parity/hardening
> tests (505 → 469, 450 → 414). The existence-gate rewrite to a second-order
> group surrogate test added one net test (469 → 470, 414 → 415). The
> autocorrelation-robustness validation added 5 tests (470 → 475, 415 → 420).
> The envelope exporter added 6 tests (475 → 481, 420 → 426).

## Suite self-health guard (`test_suite_health.py`)

Product-code guards all assume the suite is really executing what it claims to.
Three historical `parents[1]` / `parent.parent` bugs broke that assumption: a
miscomputed repo root produced a path that silently did not exist, and the
`skipif(not path.exists())` guarding on it then skipped forever while reporting
success. `test_suite_health.py` closes that hole by asserting

- the collection baseline and slow / not-slow split above, and
- that **every module-level `Path` in every test module actually exists**.

The second check is the static equivalent of a skip-count gate, and it is
deliberately not implemented by re-running the suite in a subprocess: this guard
lives under `tests/`, so such a run would re-invoke itself and recurse.

If a test legitimately references a path that does not exist yet (an output
written during the run), add it to `_ALLOWED_MISSING_PATHS` with a reason so the
exemption is explicit and reviewable.

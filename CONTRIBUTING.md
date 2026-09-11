# Contributing to SyncPipe

Thanks for considering a contribution. SyncPipe is measurement
infrastructure for dyadic synchrony studies: every public number must be
auditable and reproducible, so contributions are held to a few explicit
rules rather than taste.

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
python -m pip install -e ".[dev]"
pytest tests/ -m "not slow" -q   # fast layer
pytest tests/ -q                 # full layer (nightly / pre-release)
```

## The two-layer test rule

- Anything that runs Monte-Carlo replicates, long surrogate loops, or
  real-data end-to-end batteries must be marked `@pytest.mark.slow`.
- `tests/test_suite_health.py` freezes the collected-test count and the
  slow / not-slow split. Adding or removing tests requires updating the
  expected numbers **with a reason in the comment** — drift must always be
  an intentional act.
- New statistical endpoints need an H0-calibration row in
  `tests/test_h0_calibration_endpoints.py` (the endpoint must reject at
  ~nominal alpha under its own null) and a defaults guard in
  `tests/test_v1_defaults_guard.py` if it introduces a default.

## Local verification before pushing

Pull requests that touch `syncpipe/` inference or measurement code require a
green full-suite run locally before pushing — CI is a backstop, not a
trial-and-error target.

The fastest way to run the same checks CI performs:

```
python scripts/prepush_check.py          # editable install + fast test layer
python scripts/prepush_check.py --full   # editable install + full suite
```

The install step matters: CI builds the package from scratch on every push,
so packaging metadata errors (invalid `pyproject.toml` fields, license
classifier conflicts) only surface there — a green test suite alone does not
cover it.

Untracking files matters too: CI checks out only tracked files, so a test
that references an untracked path fails on CI with `FileNotFoundError` even
while passing locally (where the file still exists on disk). Before
untracking anything, grep `tests/` for references to it — and if a test
legitimately targets an untracked path, guard it with `skipif` and add the
path to `_ALLOWED_MISSING_PATHS` in `tests/test_suite_health.py`.

## Protocol and documentation conventions

- Public files stay launch-level: no internal review jargon, no
  competitor/model hard-binding, neutral positioning.
- Claim language follows the claim ceiling: the demo and README show what the
  measures do; they never assert that synchrony features indicate
  relationship quality or health.
- `docs/METHOD_LOG.md` records why a statistical choice was made;
  `docs/LIMITATIONS.md` records what the current design cannot support.
  Add an entry when your change alters a protocol decision.
- `docs/API_REFERENCE.md` is generated — re-run
  `python scripts/build_api_reference.py` after any public-API change; do
  not edit it by hand.

## Versions

Release version, config-schema version, and manifest-schema version are
intentionally independent contracts (see `pyproject.toml` and
`syncpipe/__about__.py`). Bump only the one your change actually breaks.

## License

By contributing you agree that your contributions are licensed under the
repository license (MIT).

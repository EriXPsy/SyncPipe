# Experimental / out-of-scope code

These files are **not** part of the SyncPipe v1 package and are **not** required
to reproduce v1 results. They are retained as a development staging area only.

## `prediction.py`

Removed from the v1 package on 2026-08-17 (rolling-origin CV, cross-modal
prediction, AR baseline). v1's scientific scope is same-modality measurement +
audited inference; the cross-modal prediction regression family contradicted
that scope, so it was moved here out of the package.

## `scripts/`

One-off diagnostics, superseded validation runners, falsified experiments, and
dataset-local analyses from the v0.x development history. Most were archived to
`../archive/experimental_scripts/` on 2026-08-17; only the negative-evidence
scripts below remain here.

### Negative-evidence scripts — KEPT TEMPORARILY, DEPRECATION WATCH

The following scripts are referenced by `docs/METHOD_LOG.md` /
`docs/SCRIPT_MAP.md` as the record of *negative* or *retired* results. They are
kept so those citations stay resolvable, but they are **candidates for removal
once the referenced method sections are rewritten** — do not rely on them as
maintained code:

- `circular_shift_timing_null_FALSIFIED.py` — the falsified circular-shift
  timing null (retired; see METHOD_LOG §7d).
- `analyze_pgt2_fixed.py` — one-off "fixed" analysis reading `pgt2_grid_results.csv`.
- `diagnose_pgt2_drift.py` — one-off drift diagnosis.
- `diagnose_h2_switching_entropy.py` — one-off switching/entropy noise diagnosis.
- `run_lerique_shuffle.py` — Lerique-local shuffle/robustness analysis.

> **Deprecation watch:** if any of the above is deleted, first update the
> METHOD_LOG / SCRIPT_MAP citations to either drop the reference or relocate the
> evidence into `docs/`.

# Design Principles

> Status: v1.0 review follow-up (Blind Spot C).
> Motivation: several v1.0 bugs shared one *shape* — the default path
> preferred "run if you can" over "fail loudly if something is wrong".
> This document names that tendency as a principle so future bugs announce
> themselves instead of being found one-by-one.

## Principle 1 — Fail loud at pipeline boundaries

Every boundary where external/raw data enters the pipeline, or where one
stage hands off to the next, **must validate its inputs** and **raise or
warn** on anything suspicious. A `ValueError` at ingestion is cheap; a
silent wrong number in a paper figure is not.

Concretely:

- **Input validation, not silent coercion.** If a required field is missing
  or malformed, raise. Do not silently substitute a default that changes the
  scientific meaning (e.g. do not re-anchor two independently-recorded
  timelines to a shared `0.0` and pretend the offset was corrected).
- **Suspicious defaults trigger a warning.** A default that *changes
  interpretation* (e.g. `force_zero_start=True` collapsing a between-file
  clock offset) must emit a `UserWarning` explaining the consequence, not
  proceed silently.
- **"Looks fine" is not "is correct".** Passing NaN through a computation is
  acceptable *only* when the downstream explicitly skips NaN (e.g. WCC
  windows gated by `discontinuity_mask`, feature extraction via
  `np.isfinite`). NaN propagation that is *unintended* is a bug and should be
  caught by a contract test, not by hoping.

## Principle 2 — Every number-citing path has a test

Any code path that produces a figure, table, or reported statistic in the
paper must have at least one **contract test** that fails if the path is
silently broken. The loader → bridge → inference chain and the simulation
ground-truth generators are the highest-value fuses (see
`tests/test_realtest_and_bridge_contracts.py`,
`tests/test_simulation_kuramoto.py`).

Rationale (Blind Spot B): `multisync/realtest/*` and `pipeline_bridge` had
**zero** dedicated tests at v1.0, despite being exactly where the
`discontinuity_mask` single-sided-drop bug had lived. A future refactor of
WCC defaults would corrupt the paper's numbers with no test catching it.

## Principle 3 — Docstrings state actual behavior, and say so when they don't

A docstring is a contract. If the implementation diverges from the docstring,
**fix the implementation** (preferred) or **fix the docstring to match** —
never leave a docstring that promises behavior the code does not deliver
(the old `split_half_icc` / "ICC(2,1)" case, now deprecated and labeled).
When a function is intentionally approximate or a known simplification, the
docstring must *say so* in one line.

## Principle 4 — Don't over-claim scope in user-facing docs

README / docs promises are read by reviewers as claims. "Multimodal" means
the validated, continuous, low-frequency modalities (ECG/IBI, EDA, RESP,
motion energy) — **not** fNIRS/EEG hyperscanning or discrete event streams
(facial AUs, turn-taking). State the validated scope explicitly
(`README.md`, `docs/LIMITATIONS.md`). Discrete-event synchrony is v2 scope.

## Known silent-failure audit checklist

Items to verify (or already fixed) when touching the relevant code:

| Area | Risk | Status |
|------|------|--------|
| `importer.merge_person_files` | `force_zero_start=True` silently eats `offset_b_sec` | Fixed in `92da615` (offset applied after re-anchor) |
| `realtest/lerique_2024` default branch | kept only one person's `discontinuity_mask` | Fixed (AND-combined + carried into Dyad) |
| `prediction.check_cv_feasibility` | false "feasible" on tiny samples | Guarded by `min_window` auto-grow + FSR warning (`92da615`) |
| `validation/recovery.split_half_icc` | docstring claimed ICC(2,1), computed Pearson | Deprecated + labeled (pre-v1.0) |
| `feature_definitions._binarize_with_hysteresis` | NaN bridged across seams (docstring said False) | Fixed in `4efc72c` (NaN → hard False) |

New code should not add to this list. When reviewing a diff, ask: *"if this
breaks, will anyone hear it?"*

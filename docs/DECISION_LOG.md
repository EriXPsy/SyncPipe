# SyncPipe Decision Log

This file records **current v1 decisions and changes**. Older exploratory or superseded decision history should be kept in an archive if needed, but the active decision log should remain short enough for users and reviewers to audit.

---

## Current v1 feature-family stance

**Decision.** SyncPipe v1 uses a narrow primary FDR family:

- `peak_amplitude`
- `dwell_time`
- `switching_rate`

`mean_synchrony` is a reference comparator. `fraction_above_threshold`, `bimodality_coefficient`, `synchrony_entropy`, `onset_latency`, `rise_time`, `recovery_time`, `first_peak_time`, and `inter_peak_cv` are reported as exploratory / secondary descriptors with paradigm restrictions and definedness reporting where applicable.

**Rationale.** The v1 contribution is audited measurement infrastructure, not a claim that every WCC-derived descriptor is a validated psychological construct. A narrow primary family reduces multiplicity and keeps interpretation defensible.

**Source of truth.** `multisync/feature_definitions.py`, `multisync/feature_status.py`, and generated `docs/FEATURE_TABLE.md`.

---

## Current v1 threshold stance

**Decision.** SyncPipe separates threshold scope:

- `within_dyad` / per-pair signal-level surrogate threshold: for single-dyad descriptive and synchrony-existence workflows.
- `session_pooled` threshold: for between-dyad or group-comparable episode descriptors, implemented in `BatchComputationPipeline` / `session_threshold.py`.
- `fixed` threshold: for sensitivity analysis and explicit user-specified comparisons.

**Rationale.** Per-dyad thresholds adapt to each dyad's null distribution but make group comparisons of episode features harder to interpret. Pooled thresholds preserve a shared episode definition across dyads/conditions.

---

## Current v1 null-model stance

**Signal-level IAAFT.** Used as a synchrony-existence audit for distributional WCC descriptors. It tests whether observed WCC-derived descriptors exceed an independent autocorrelated-signal null. It does **not** prove dyad-specific interpersonal coupling.

**WCC-level IAAFT / order nulls.** Used cautiously as trace-level structure audits for descriptors such as `dwell_time` and `switching_rate`. Because WCC traces inherit autocorrelation from overlapping windows, WCC-level nulls are not presented as mature confirmatory tests of psychological temporal structure in v1.

**Timing / morphology nulls.** `onset_latency`, `rise_time`, `recovery_time`, `first_peak_time`, and `inter_peak_cv` remain exploratory. A validated existence null for these descriptors is deferred to v2.

---

## 2026-07-01 — Safety-fix sprint

**Implemented.**

1. IAAFT implementation now returns the final rank-adjusted sequence, preserving the empirical amplitude distribution exactly while approximating the power spectrum / autocorrelation. Documentation was corrected accordingly.
2. `DynamicFeatures.from_dict()` now round-trips all public dataclass fields exported by `to_dict()`.
3. Timestamp alignment now correctly allows all-absolute timestamp inputs and fails only true absolute/relative/unknown mixtures.
4. `zscore()` no longer turns all-NaN channels into zeros; all-NaN channels remain NaN and are reported in stats.
5. `DynamicAnalyzer.fit_transform()` now runs QC by default. QC FAIL raises `DataQualityError` unless `qc_raise_on_fail=False` is set for exploratory inspection.
6. `DynamicAnalyzer` now passes `surrogate_n` and `seed` into surrogate threshold computation.
7. Threshold mode is made explicit: `DynamicAnalyzer` supports `within_dyad` and `fixed`; session-pooled thresholds are routed to `BatchComputationPipeline`.
8. Top-level public API was narrowed to the v1 stable surface. Advanced modules remain importable from submodules.
9. Broken low-level computation paths were repaired (`ComputationPipeline.compute_wcc(method="stride")`, `DataImporter.load_signal`).
10. `syncpipe` was added as the preferred import/CLI namespace while `multisync` remains a compatibility alias.
11. Timing fields now use raw undefined semantics (`NaN` when undefined) with explicit `*_imputed` companion fields for ML-only imputation.
12. CI workflow with pytest and demo smoke test was added.
13. QC now has a user-facing PASS/WARN/FAIL formatter and CLI `analyze` prints actionable QC messages before WCC computation.
14. Warning cleanup reduced test warnings by removing BC near-constant precision warnings, sklearn `l1_ratio` warnings, and prediction window-size warnings.
15. README/User Manual/SKILL examples now prefer the `syncpipe` command/import namespace.
16. PGT-2, PGT-2 surrogate, PGT-3, and EGT-4 self-contained validation artifacts were rerun under the updated timing semantics.

**Verification.** Full test suite after the sprint:

```text
178 passed, 1 xfailed
```

---

## 2026-07-13 — Episode-feature robustness to missing data

**Decision.** `compute_dwell_time` and `compute_switching_rate` now compute
episode statistics over *valid (finite) WCC samples only*:

- A missing point (NaN, i.e. a `discontinuity_mask` gap) is excluded from the
  binary sequence, **not** treated as a low-state sample. Consequently an
  artifact gap (a) does not split one elevated run into two shorter runs
  (which would deflate `dwell_time`) and (b) does not inject spurious
  False->True / True->False transitions (which would inflate `switching_rate`).
- `switching_rate` duration is **valid time only** (`finite.sum() / hz / 60`),
  not the full array length including gaps.

The Schmitt-trigger binarizer (`_binarize_with_hysteresis`) is intentionally
*unchanged*: it still hard-breaks at NaN (the 2026-07-08 fix), so surrogate /
state-shuffle logic is unaffected. The gap-robustness lives only in the two
summary functions.

**Why this matters.** `dwell_time` and `switching_rate` are the two features
carrying SyncPipe's "quality of synchrony" argument in the Kuramoto validation.
If missing rates differ across experimental conditions, the old behaviour
would have created a confound with the condition (dual direction: one inflated,
one deflated) — exactly the kind of artifact a reviewer would flag.

**Source of truth.** `multisync/feature_definitions.py` (`compute_dwell_time`,
`compute_switching_rate`); guarded by `tests/test_feature_definitions.py`
(gap-robustness tests).

---

## Pending v1 cleanup items

1. Decide whether to suppress or re-route expected relative-timestamp warnings in synthetic tests/demos.
2. Monitor the external sklearn/scipy L-BFGS-B deprecation warning.
3. Archive older exploratory decision history outside this active decision log.
4. Continue validating WCC-level order nulls before making stronger structure claims.

---

## Dual namespace (`multisync` vs `syncpipe`): intentional, with a sunset plan

**Decision.** The `syncpipe` namespace is the preferred import/CLI surface
(`syncpipe` command and `import syncpipe`). The older `multisync` namespace
remains **only as a compatibility alias** during the transition away from the
original project name *MultiSync* (renamed to avoid collision with the
published `multiSyncPy`).

**Sunset plan (added 2026-07-09).**

- **v1.0 (current):** both namespaces work; new code and docs use `syncpipe`.
- **v1.1:** importing `multisync` emits a `DeprecationWarning` pointing users
  to `syncpipe`.
- **v2.0:** the `multisync` alias is removed; `syncpipe` is the only namespace.

A compatibility alias with no removal date tends to become permanent technical
debt. Pinning it to a version schedule keeps the migration honest and gives
downstream users a clear migration window.

---

## Related tools (positioning, not competition)

SyncPipe sits in a small field of interpersonal-synchrony tooling. Naming
proximity is real and worth stating explicitly so reviewers are not surprised:

- **multiSyncPy** (Hudson, Wiltshire & Atzmueller, 2023, *Behavior Research
  Methods*): group/multi-person synchrony beyond dyads, with Kuramoto
  simulations demonstrating sensitivity to coupling strength and convergent
  validity across multivariate synchrony indices. SyncPipe's dyad-focused,
  WCC-episode framing is narrower; the rename away from *MultiSync* was
  precisely to avoid implying overlap that does not exist.
- **SyncPy** (Varni & Avril, 2015): a unified open-source library for
  interpersonal/machine synchrony on dyadic and multiparty time series.
  Name is visually/phonetically close to *SyncPipe*; SyncPipe is a distinct,
  independently developed tool and should be cited separately. No code is
  shared.
- **SUSY** (Ramseyer & Tschacher lineage; R package, v0.1.1): synchrony
  measurement via segment-reshuffling surrogates. Its pseudo-pair design
  control is spiritually similar to SyncPipe's design-control audit; the
  difference is language (R vs Python) and the WCC-episode feature layer.
- **rMEA** (Kleinbub & Ramseyer, 2020): motion-energy analysis in R, a close sibling for
  the motion-energy modality specifically.

SyncPipe's differentiating claim is **audited measurement infrastructure**: a
transparent WCC trace, a governed feature table, signal- and design-level null
models, and exported artifacts — rather than a single synchrony score or a
broad metric zoo.


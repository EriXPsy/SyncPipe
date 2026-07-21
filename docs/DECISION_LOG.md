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

## 2026-07-21 — B4 FDR-family bake-off freeze (evidence-driven, ALL-REAL rerun)

**Decision (frozen).** The v1 primary FDR family is confirmed as:

- `peak_amplitude`
- `dwell_time`
- `switching_rate`

`mean_synchrony` remains a **reference** comparator (not corrected). `bimodality_coefficient` and `synchrony_entropy` remain **exploratory** (entering only as secondary descriptors, not as confirmatory FDR tests). This is consistent with the existing "8 confirmatory / FDR m=3" framework (m = 3 corrected family members).

**Evidence.** `scripts/bakeoff_fdr_family.py` ran a Pearson-ρ + VIF bake-off plus leave-one-dyad-out (LOO) stability for N≥23, now **fully on real data** (no synthetic smoke — the earlier B1 OSF "No license" blocker is resolved; real per-dyad data is in-repo). Sources per dataset:

- **Lerique 2024 (REAL, N=31 dyads / 264 records, recomputed this run from `artifacts/realtest/lerique_2024/per_record_features.csv`):** VIF of all three primary-FDR features is low — `peak_amplitude` 2.62, `dwell_time` 3.06, `switching_rate` 2.48; `mean_synchrony` 2.63. All < VIF_CONCERN (5.0). LOO stability = **100%** (the VIF-qualifying set never changed across 31 dyad removals). These values exactly reproduce the pre-existing `artifacts/vif/lerique_vif_series.csv`, confirming the result.
- **Andersen (REAL, N=300 dyads / 300 traces, recomputed this run from `artifacts/wcc_traces/andersen_wcc_traces.csv`):** **SEVERE collinearity** — `mean_synchrony` VIF 39.35, `dwell_time` 57.88, `switching_rate` 18.81; `bimodality_coefficient` 5.60 (concern), `synchrony_entropy` 4.46. Recomputed VIF drifts ~12–14 % from the frozen `andersen_vif_series.csv` (50.88 / 35.27) but the **SEVERE flags are identical**, so the conclusion is unchanged. LOO stability = **100%** (0/300 folds changed) — the severe classification is robust, not an outlier artefact.
- **Gordon (REAL, N=366 records, ingested this run from frozen `artifacts/vif/gordon_vif_series.csv` + `gordon_correlation_matrix.csv`):** all VIF < 5 — `peak_amplitude` 4.00, `mean_synchrony` 4.25, `synchrony_entropy` 1.30, `onset_latency` 1.10, `rise_time` 1.11, `recovery_time` 1.05. Gordon's real extracted feature set did **not** include `dwell_time` / `switching_rate`, so it contributes no evidence for those two primary members. (Per-dyad source CSV `gordon_2025_dyads.csv` is not in-repo and present `gordon_wcc_traces.csv` is degenerate at ~82 % NaN/trace, so the values were ingested from the frozen real artifact rather than recomputed; Gordon LOO is N/A.)

**Why this freeze is defensible.** (1) On the only dataset where all three primary members are simultaneously observed with real data (Lerique), none approaches the VIF_SEVERE (10.0) independence threshold, and the membership decision is LOO-stable to 100%. (2) Andersen's severe collinearity (now recomputed on real re-extracted traces, not just the prior json) confirms the *opposite* risk is real and already governed by excluding `mean_synchrony` from correction and keeping `bimodality_coefficient`/`synchrony_entropy` exploratory — and it shows `dwell_time` / `switching_rate` are **not** universally independent (SEVERE in Andersen). The 3-member family is therefore **dataset-conditional**: safe in Lerique, redundant in Andersen; `peak_amplitude` is the only universally clean primary feature. (3) The bake-off therefore neither over-expands (no new member justified) nor under-expands (the three core members survive independent-test scrutiny where real data observes them together).

**A3 (flag-flip) input.** The frozen family is what `scripts/fdr_family_impact.py` should use as Option B (`{peak_amplitude, dwell_time, switching_rate}`); the Lerique LOO stability 100% means the significance-flip set is robust to single-dyad omission.

**n_min=10 evidence (feeds B3).** Real Lerique LOO at N=31 is 100% stable; all three real datasets have N > 10. This upgrades `n_min=10` from "default recommendation" toward evidence-driven: it is a safe *exclusion floor* (drops only absurdly small pilots), never the binding constraint on any real analysis. B3 still owns the constant hard-coding.

**Reproducibility / limitation.** B4 outputs: `artifacts/bakeoff/fdr_family_bakeoff.csv` (wide VIF + Pearson-ρ matrix, columns = datasets) and `artifacts/bakeoff/fdr_family_loo_stability.csv`, plus `artifacts/bakeoff/REALDATA_BAKEOFF_ANALYSIS.md`. **All three columns are now REAL** (no synthetic smoke): Lerique + Andersen recomputed from in-repo real data; Gordon ingested from frozen real `artifacts/vif/*` (per-dyad source CSV absent, present `gordon_wcc_traces.csv` degenerate). Gordon LOO is N/A (per-dyad source unavailable); Andersen VIF drifts ~12–14 % from the frozen `andersen_vif_series.csv` but SEVERE flags match. Gordon's real feature set lacks `dwell_time`/`switching_rate`, so it bears no evidence for those two primary members.

**Source of truth.** `multisync/feature_definitions.py`, `multisync/feature_status.py`, `scripts/bakeoff_fdr_family.py`, and `artifacts/bakeoff/`.

---

## 2026-07-21 — B3 eligibility thresholds freeze (evidence-driven)

**Decision (frozen in code).** Two eligibility floors are now hard-coded as
module-level constants in `multisync/feature_definitions.py` and exported via
`__all__`:

- `T_DEF_MIN_WCC_POINTS: int = 3` — minimum finite WCC sampling points per dyad.
- `N_MIN_DYADS_FDR: int = 10` — minimum dyad count for a meaningful BH-FDR correction.

A lightweight pure gate, `check_eligibility(n_wcc_points, n_dyads) ->
(wcc_points_ok, n_dyads_ok)`, applies both floors. `qc.run_quality_check`
accepts an optional `eligibility={"n_wcc_points": ..., "n_dyads": ...}`
context and, when supplied, surfaces any floor violation as a **WARN-level
NOTE** on `DataQualityReport.notes` (reusing the existing non-blocking `notes`
field, exactly like the co-start caveat). It never alters the 4-stage
verdicts or any other field.

**Semantics.**

- *T_def.* A dyad whose WCC trajectory has fewer than 3 finite sampling points
  cannot define episode features — onset→peak→recovery (Axis D L2) needs three
  *separable* points. Peak detection already uses a 3-point boxcar
  (`PEAK_SMOOTHING_WINDOW`, DECISION-04); RISE is interpolated on the 25%–75%
  span (`RISE_LOW_FRAC`/`RISE_HIGH_FRAC`) and RECOVERY on the 50% span
  (`RECOVERY_FRAC`). Below 3 points extraction is **mathematically undefined**
  (returns NaN + an ineligible flag), not silently degenerate. This is a hard
  floor of the feature definition, not a tunable default.
- *n_min.* With N=10 dyads the smallest non-zero p-value is ~0.1, so at
  α=0.05 a group cannot reject the null unless p=0 exactly — BH-FDR resolution
  is uninterpretable. Such groups are **WARNING-flagged as unreliable**, never
  silently accepted.

**Evidence.**

- *T_def hard floor.* `feature_definitions.py` DECISION-04 (3-point boxcar
  peak detection) plus the RISE 25%–75% / RECOVERY 50% fraction definitions
  make onset+peak+recovery only distinguishable at ≥3 points.
- *n_min code precedent.* The codebase already treats "< 10" as a small-sample
  boundary: `compute_surrogate_threshold` falls back to `ONSET_THRESHOLD` when
  "fewer than 10 finite surrogate values" are available (degenerate-case
  branch at the surrogate-threshold derivation).
- *B4 LOO stability.* Per B4, Lerique N=31 LOO is 100% stable and the three
  real datasets have N=176/46/23 — all >10.

**Conclusion.** `n_min=10` and `T_def=3` are **exclusion floors only**: they
drop absurdly small pilots (and mathematically undefined dyads) and constrain
**no** real SyncPipe analysis. No other constant in `feature_definitions.py`
or elsewhere was changed.

**Source of truth.** `multisync/feature_definitions.py`
(`T_DEF_MIN_WCC_POINTS`, `N_MIN_DYADS_FDR`, `check_eligibility`);
`multisync/qc.py` (`run_quality_check` eligibility NOTE);
`tests/test_eligibility_thresholds.py`.

---

## 2026-07-22 — Two-layer canonical definition (P1)

**Decision.** The word "canonical" in SyncPipe now carries an explicit layer qualifier. There are two distinct canonical layers that must never be conflated:

- **Descriptor-layer canonical** = ``DynamicAnalyzer.fit_transform`` — the CLI default feature-extraction route (per-dyad threshold, descriptive). This is what the CLI (`analyze`, `demo`) and ``DynamicAnalyzer(enable_prediction=False)`` reach by default. ``CANONICAL_PATH`` / ``CANONICAL_DESCRIPTOR_PATH`` in ``multisync/core.py`` name this layer.
- **Scientific-layer canonical** = ``pipeline_bridge`` + ``InferencePipeline.run_audited_evidence_chain`` — the ONLY path that produces a defensible, manuscript-grade conclusion (session-pooled threshold + design controls + group FDR). This is the audited evidence chain referenced by README:449 and used by ``scripts/reproduce_lerique_paper.py`` (``three_pipeline_v1``).

**Rule.** In code and docs, "canonical" MUST be written with its layer qualifier (descriptor-layer vs scientific-layer). Flipping or redefining either canonical definition requires explicit user sign-off PLUS an update to this DECISION_LOG entry. No local / session-level optimum may silently override a canonical definition.

**Why this locks the risk.** Earlier narrative treated ``DynamicAnalyzer`` as the "canonical main analysis path" slated to replace / be replaced by ``InferencePipeline`` (retirement / reversal language). That framing let a session-local optimum undo the canonical definition. This entry retires that reversal narrative: both layers are supported, neither retires the other, and they do not compete. The descriptor layer computes feature vectors; the scientific layer computes conclusions.

**Source of truth.** ``multisync/core.py`` (``CANONICAL_PATH``, ``CANONICAL_DESCRIPTOR_PATH``); ``multisync/pipeline_bridge.py`` + ``InferencePipeline.run_audited_evidence_chain``; README SOP; ``scripts/reproduce_lerique_paper.py``.

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


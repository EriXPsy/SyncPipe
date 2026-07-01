# SyncPipe Method Log

> Records the current measurement architecture used by the README, demo
> outputs, and manuscript Table 1.

---

## 1. Core repositioning

SyncPipe v1 is positioned as **measurement infrastructure for interpersonal synchrony**, not merely as a synchrony feature profile or a collection of three pipelines.

The unit of contribution is the standardized measurement procedure:

1. aligned dyadic signals;
2. WCC trace construction;
3. WCC-derived descriptor table;
4. synchrony-existence audit;
5. design-control audit;
6. group condition inference;
7. reproducible exports and governance.

This repositioning is intended to make SyncPipe useful even when a feature result is negative or confounded.  A tool that shows a result is likely shared-stimulus-driven is doing scientific work.

---

## 2. SSoT boundary update

SyncPipe now uses two explicit SSoT layers.

### 2.1 Mathematical SSoT

File:

```text
multisync/feature_definitions.py
```

Purpose:

- implement WCC-derived feature mathematics;
- provide dataclass serialization;
- define internal constants and helper functions;
- prevent duplicate feature math across modules.

This layer may retain internal mathematical invariance labels where useful for implementation and null-model selection.

### 2.2 Communication SSoT

File:

```text
multisync/feature_status.py
```

Purpose:

- provide the external-facing v1 feature status table;
- support README, demo artifacts, and manuscript Table 1;
- state source level, incremental information, paradigm restrictions, recommended tests, status, and risks.

This layer intentionally avoids making external readers interpret the older Core/Conditional/L0/L1/L2 labeling scheme.

---

## 3. Evidence chain replaces feature-label hierarchy as external inference story

Older internal language used L0/L1/L2 labels for mathematical invariance and/or inferential levels.  This created ambiguity because "L2" was used both for event-locked morphology and between-condition inference.

The external v1 inference story is therefore:

### Step 1 — Synchrony-existence audit

Question:

> Do aligned signals show WCC features exceeding independently autocorrelated surrogate signals?

Default implementation:

```python
InferencePipeline.run_synchrony_existence_audit
```

Default null:

- signal-level IAAFT.

Interpretation:

- necessary but not sufficient evidence for a synchrony-like phenomenon;
- does not rule out shared stimulus, co-presence, task rhythm, or other common-driver confounds.

### Step 2 — Design-control audit

Question:

> Does the result survive controls targeting partner identity, temporal alignment, and shared stimulus structure?

Default implementation:

```python
InferencePipeline.run_design_control_audit
InferencePipeline.run_across_stimulus_shuffle_audit
```

Controls:

- pseudo-pair: real partner vs mismatched partner;
- time-shift: original alignment vs within-dyad shifted alignment;
- across-stimulus shuffle: original segment order vs independently permuted stimulus segments.

Interpretation:

- these controls do not solve all common-driver problems;
- they make nuisance explanations empirically visible and reportable.

### Step 3 — Group condition inference

Question:

> Do descriptors differentiate conditions, groups, or theoretically meaningful predictors?

Default implementation:

```python
InferencePipeline.run_group_condition_inference
```

Default test:

- dyad-paired permutation + BH-FDR.

---

## 4. Feature status stance

The v1 feature table is a measurement map, not a promotion list.

Current stance:

- `peak_amplitude` is the primary workhorse for synchrony-existence detection.
- `mean_synchrony` remains a reference comparator, not a sufficient construct definition.
- `dwell_time` and `switching_rate` are exploratory-secondary structure descriptors because thresholding, WCC overlap, and jitter affect their interpretation.
- `onset_latency`, `rise_time`, and `recovery_time` are event-mode exploratory descriptors, not general synchrony descriptors.
- `bimodality_coefficient` and `synchrony_entropy` are distribution-shape diagnostics with construct-validity caveats.
- `fraction_above_threshold` is implemented in the mathematical SSoT as an exploratory-secondary occupancy descriptor, but is not included in the primary FDR family in v1.
- `first_peak_time` and `inter_peak_cv` are morphology-agnostic exploratory descriptors and require definedness-rate reporting.

---

## 5. Null-model stance

### Signal-level IAAFT

Use for synchrony-existence auditing of distributional WCC descriptors.  It preserves single-signal spectra and amplitude distributions while disrupting cross-signal alignment.

Claim it supports:

> WCC-derived synchrony descriptors exceed an independent autocorrelated-signal null.

Claim it does not support alone:

> The observed synchrony is dyad-specific interpersonal coupling.

### WCC-level order nulls

Use cautiously for exploratory structure descriptors.  WCC traces are overlapping-window summaries; their autocorrelation partly reflects measurement construction.  Therefore WCC-level IAAFT or block permutation should not be oversold as a mature confirmatory test of psychological temporal structure in v1.

### Pseudo-pair and time-shift

Use as design-control audits.  They are not generic replacements for experimental design, but they directly address partner-identity and temporal-alignment alternatives.

### Across-stimulus shuffle

Use only when meaningful stimulus/trial segments exist.  It is inappropriate for unsegmented free interaction.

---

## 6. Demo and artifacts

The v1 demo now exports:

```text
viewer_results.json
feature_table.csv
feature_status_table.csv
synchrony_existence_audit.json
design_control_audit.json
DEMO_REPORT.md
```

These artifacts are intended to support reproducible inspection rather than black-box analysis.

---

## 7. 2026-06-29 update: fraction_above_threshold SSoT integration

Decision:

- Implement `fraction_above_threshold` in the mathematical SSoT.
- Define it as the fraction of finite WCC samples satisfying $\mathrm{WCC}[t] \geq \theta_{sync}$.
- Classify it externally as an exploratory-secondary occupancy descriptor.
- Exclude it from the primary FDR family in v1.

Rationale:

- It is transparent and easily interpretable as above-threshold synchrony coverage.
- It adds an occupancy dimension distinct from episode duration (`dwell_time`) and transition frequency (`switching_rate`).
- It is permutation-invariant and therefore does not measure temporal organization by itself.
- Initial artifact-level audits showed likely redundancy with mean/peak synchrony in some datasets, so confirmatory promotion is premature.

Implementation:

- `compute_fraction_above_threshold()` in `feature_definitions.py`.
- `DynamicFeatures.fraction_above_threshold` field and `to_dict()` export.
- Included in `feature_status.py`, `feature_pipeline.py`, `scripts/build_feature_table.py`, demo `feature_table.csv`, and design-control default audit features.

## 7b. 2026-06-29 update: timing-descriptor wiring + BRM full-text migration

Two changes were made in one pass at the author's request.

### 7b.1 Wiring of two morphology-agnostic timing descriptors

`compute_inter_peak_cv()` and `compute_first_peak_time()` already existed
in `feature_definitions.py` but were never called by `extract_features`,
had no `DynamicFeatures` fields, and were absent from `to_dict()` and the
status/tier tables — i.e. they were isolated (un-wired) code.

Before wiring, an artifact-level redundancy audit was run
(`scripts/audit_timing_features.py`, output
`artifacts/timing_feature_audit.csv`) on the existing Andersen, Gordon,
and Lerique WCC traces. Findings:

- Pooled maximum |Pearson r| against mean/peak/fraction-above-threshold:
  - `first_peak_time`: 0.24 (lowest redundancy)
  - `inter_peak_cv`: 0.42
  - (for comparison, `fraction_above_threshold` reached |r| up to 0.97
    against mean synchrony in the earlier audit — strong redundancy.)
- Definedness (fraction of traces with a finite value):
  - `inter_peak_cv`: Andersen 1.00, Gordon 0.05, Lerique 0.53
  - `first_peak_time`: Andersen 1.00, Gordon 0.39, Lerique 0.66

Conclusion: both descriptors carry information beyond the magnitude
descriptors (they are NOT redundant with mean/peak — unlike
fraction_above_threshold), but their definedness is strongly
paradigm/length-dependent. `baseline_fraction` (also un-wired) was NOT
wired in, because it overlaps in semantics with `first_peak_time` and is
more redundant (pooled |r| = 0.51).

Implementation:

- `extract_features` now computes both and passes them to
  `DynamicFeatures`; new fields `inter_peak_cv`, `first_peak_time`
  (default NaN) added; both exported in `to_dict()` and accepted in
  `from_dict()`.
- Registered as `conditional` in `FEATURE_TIER`, `L2` in
  `MATHEMATICAL_TIER` (they depend on the ordering/spacing of
  threshold-crossing peaks and are NOT permutation-invariant), and added
  to `TEMPORAL_FEATURES`.
- Status set to `exploratory-secondary` in `feature_status.py` (was
  `exploratory-proposed`); both have `enters_primary_fdr = False`.
- Annotated in `scripts/build_feature_table.py`; `FEATURE_TABLE.{md,csv}`
  now list 12 features, still 5 in the FDR family.
- Tests added (`test_timing_descriptors_are_wired_and_non_fdr`,
  `test_timing_descriptors_are_nan_when_undefined`).

These descriptors require definedness-rate reporting and are NOT in the
primary FDR family. They are NOT claimed as validated psychological
constructs; the BRM Future Directions section states the upgrade
standard (real-data incremental value under design controls).

### 7b.2 BRM_draft.md full-text narrative migration

The legacy feature-tier vocabulary that remained in Methods §2.3–§2.4 and
the Discussion (Core / Conditional / Reference features, prose L0/L1/L2
labels, "two-level inference", "Intensity-governed (L1) contrast",
"Intensity-layer prediction", "GT-1 through GT-5", "promoted to the Core
tier", "feature profile/framework") was migrated to the
measurement-infrastructure / descriptor-table / audited-evidence-chain
vocabulary already used in the Introduction and Methods §2.1–§2.2. All
empirical numbers, dataset descriptions, citations, and effect sizes were
left unchanged; only framing language was rewritten. Conceptual dimension
terms (synchrony intensity / structure / temporal dynamics) were retained
where they denote dimensions of synchrony rather than feature-tier codes.

## 7c. 2026-06-29 update: primary-FDR SSoT consolidation + timing-descriptor validation

### 7c.1 Primary group-condition FDR family consolidated to 3 features (Option B)

Three sources of truth previously disagreed about the confirmatory FDR
family. Code `FDR_FEATURES` listed 5 members
({mean_synchrony, peak_amplitude, bimodality_coefficient, dwell_time,
switching_rate}); the external `feature_status.py` marked only
`peak_amplitude` with `enters_primary_fdr=True`; and
`build_feature_table.py` annotations partially disagreed. Because the FDR
family size directly determines the Benjamini-Hochberg threshold, this is
not a labelling nicety — it changes which results are reported as
significant.

An impact analysis (`scripts/fdr_family_impact.py`) re-applied BH-FDR to
the existing Lerique 2024 main-contrast p-values under the candidate
definitions. `peak_amplitude` was significant under every definition;
`dwell_time`/`switching_rate` were significant in ECG/EDA only when kept
in the family; `mean_synchrony` was the only feature whose inclusion was
internally contradictory (tiered "reference" yet consuming family budget).

DECISION (Option B): the primary confirmatory FDR family is now exactly
**{peak_amplitude (L0), dwell_time (L1), switching_rate (L1)}**.
- `mean_synchrony` is a reported reference comparator (not corrected). It
  remains an L0 feature for the synchrony-existence audit and is still
  reported with a surrogate p-value (REFERENCE_TAILS in pgt1_intensity).
- `bimodality_coefficient` is removed from the confirmatory family
  (its membership was explicitly "provisional"). It remains a permutation-
  invariant L0 distribution-shape descriptor for the existence audit and
  is still computed and serialized.

The synchrony-existence audit null grouping (`_NULL_MODEL_L0`/
`_NULL_MODEL_L1` in `dynamic_features.py`) was deliberately NOT changed:
confirmatory FDR membership (Axis C) and existence-null grouping (Axis D)
are separate axes. Files updated for consistency: `feature_definitions.py`
(FDR_FAMILIES/FDR_FEATURES), `feature_status.py` (dwell/switching ->
enters_primary_fdr=True, status primary-structure), `feature_pipeline.py`
(mean/BC fdr_member=False), `scripts/build_feature_table.py` (BC
annotation), `validation/pgt1_intensity.py` (FEATURE_TAILS=3,
REFERENCE_TAILS={mean_synchrony}), `validation/recovery.py`
(FEATURE_COLUMNS = FDR + reference), plus the dependent tests.
`to_dict()`/`from_dict()` were updated so `bimodality_coefficient` is
still serialized despite leaving FDR (it was previously emitted only via
FDR_KEYS).

### 7c.2 Validation of inter_peak_cv and first_peak_time

`scripts/validate_timing_descriptors.py` ran two analyses.

**Part 1 — Peak-timing existence null.**

*Round-3 attempt (FALSIFIED, archived).* The first attempt used an L2
circular time-shift null on the Kuramoto EGT synchrony traces. It was
found to have essentially no power: circular shifting preserves the
trace's amplitude distribution and autocorrelation, which largely
*determine* first-peak and inter-peak statistics, so the null is close to
trivially satisfied (sustained quasi-null rejected at 0.08-0.12, i.e.
above alpha, while true-peak conditions rejected at ~0.00). It is NOT a
valid existence test and was retired. The script is archived for
reproducibility under
`experimental/scripts/circular_shift_timing_null_FALSIFIED.py`.

*Round-4 attempt (current): cyclic block-bootstrap null.* The trace is cut
into equal-length blocks (5 s = 25 samples at 5 Hz on EGT; `round(5 s × hz)`
on real traces) and the BLOCK ORDER is randomly permuted before
re-concatenation. This preserves within-block short-range autocorrelation /
local peak shape AND the global marginal distribution, while destroying the
long-range temporal anchoring of peaks. A two-tailed Phipson-Smyth p-value
is computed per trace (499 permutations). Success criteria fixed in advance:
true-peak conditions should reject WELL above alpha; the no-localized-peak
`sustained` quasi-null should reject NEAR alpha (=0.05).

*1a — Kuramoto EGT (faithful substrate-level null), rejection rates at α=0.05:*

| condition | first_peak_time | inter_peak_cv |
|---|---|---|
| single_peak (true localized peak) | 0.000 | 0.000 |
| delayed_peak (true localized peak) | 0.000 | 0.017 |
| sustained (no localized peak; quasi-null) | 0.083 | 0.050 |

The null is **methodologically clean** — unlike the circular-shift null, the
`sustained` quasi-null rejects at ~alpha (no spurious power). But the
true-peak conditions also reject at ~0, i.e. the null has **no power on the
EGT morphologies**. The reason is substantive, not a null defect: Kuramoto
EGT traces are "a single Gaussian peak on a flat noisy baseline." Permuting
block order merely relocates that single peak; the trace is still "one peak +
baseline," so a single peak's *position* (first_peak_time) and the
near-degenerate inter-peak statistic carry no falsifiable *temporal structure*
to detect. A single peak has no temporal organisation to scramble. The EGT
test bed is therefore the wrong substrate for these descriptors' existence
test — the negative result is correct and interpretable, not a null failure.

*1b — Real WCC traces (TRACE-LEVEL null; weaker interpretation).* Applied to
the synchrony traces themselves (no raw signals are available), this asks
whether observed peak timing exceeds what block-reordering of the *trace*
would produce — not a raw-signal surrogate. Rejection rates (on the testable,
i.e. defined, subset):

| dataset / condition | first_peak_time | inter_peak_cv | definedness (fpt / cv) |
|---|---|---|---|
| lerique / trials_concat | 0.342 | 0.890 | 0.90 / 0.83 |
| lerique / rest1 | 0.189 | 0.429 | 0.42 / 0.24 |
| gordon / exp1 | 0.000 | 0.000 | 0.39 / 0.04 |
| gordon / exp4 | 0.000 | (no testable) | 0.36 / 0.00 |

On real data the null DOES show power, and in the expected direction (task
> rest for Lerique inter_peak_cv: 0.89 vs 0.43), indicating that real
synchrony traces contain multi-peak temporal structure that block-reordering
cannot reproduce. Gordon traces are largely undefined (cv definedness ~0) and
uninterpretable.

DECISION (conservative): this is treated as a **negative / inconclusive**
result for existence-test purposes, and existence-test status is **deferred
to v2**. Rationale: (i) the EGT substrate cannot test these descriptors
(no structure to scramble); (ii) the real-data signal is confounded by
strong definedness selection (rest1 cv definedness only 0.24, so the 0.43
rejection is computed on a self-selected ~quarter of traces) and is a
trace-level rather than signal-level null. The cyclic block-bootstrap null
is methodologically sound (sustained does not over-reject) and the real-data
result is promising, but it is not sufficient to claim an existence test in
v1. A v2 signal-level validation (block/IAAFT surrogates on raw signals,
reported alongside definedness) is required before any promotion.

**Part 2 — Incremental AUC (baseline = mean_synchrony, then +timing).**

| dataset | +inter_peak_cv ΔAUC | +first_peak_time ΔAUC |
|---|---|---|
| EGT-2 temporal (early vs late peak, mean-matched) | +0.125 | +0.137 |
| Lerique pooled (rest vs task) | +0.142 | +0.083 |
| Lerique EDA | +0.206 | +0.033 |
| Lerique RESP | +0.194 | +0.176 |
| EGT-1 structure (N=14 matched; high variance) | +0.000 | +0.200 |
| Gordon exp1 vs exp4 (illustrative; labels unmapped) | -0.001 | +0.030 |

The strongest evidence is EGT-2: under exact mean-matching the baseline
mean_synchrony AUC was 0.47 (at chance, confirming the match), and adding
the two timing descriptors raised AUC to 0.74. This demonstrates
incremental information beyond mean synchrony when mean is uninformative —
i.e. these descriptors are NOT magnitude proxies. Lerique EDA/RESP agree.
Caveats retained: EGT-1 had only 14 matched pairs with large fold variance
(its +0.200 is unreliable), and Gordon showed near-zero increment (and its
condition semantics were not mapped, so it is illustrative only).

DECISION: both descriptors remain **exploratory-secondary, not in the
primary FDR family**. The incremental-AUC evidence is positive and
strengthens their descriptive value, but they lack a *validated* existence
null (the block-bootstrap null is methodologically sound but its
existence-test status is deferred to v2; see Part 1), so they are not
promoted to confirmatory status in v1.

Artifacts: `artifacts/timing_validation/timing_validation_summary.json`,
`artifacts/timing_validation/block_permute_null_egt.csv`,
`artifacts/timing_validation/block_permute_null_real.csv`. The falsified
round-3 circular-shift script is archived at
`experimental/scripts/circular_shift_timing_null_FALSIFIED.py`.

## 7d. 2026-06-29 update: block-bootstrap null lineage + v1.0 final code audit

### 7d.1 Theoretical lineage of the cyclic block-bootstrap peak-timing null

The cyclic block-bootstrap null introduced in §7c.2 (round 4) is not an ad-hoc
invention; it sits on two documented lineages, mirroring how the existing WCC
nulls are positioned relative to SUSY / multiSyncPy.

- **Statistical-methodology lineage.** Künsch (1989) introduced the Moving Block
  Bootstrap (resample fixed-length blocks to preserve within-block dependence);
  Politis & Romano (1992) introduced the Circular Block Bootstrap, which wraps
  the series end-to-start so that tail observations left over when the length is
  not divisible by the block length are still resampled — exactly the tail
  handling used in `scripts/validate_timing_descriptors.py`. These are the
  standard `arch.CircularBlockBootstrap` / `MovingBlockBootstrap` methods.

- **Synchrony / physiological-time-series lineage.** Ramseyer & Tschacher (2011, J. Consulting and Clinical Psychology, 79, 284–295)
  used *segment shuffling* (cut a series into segments and re-append them in
  random order) as a WCC/motion-energy surrogate in the rMEA tradition; it is
  catalogued as one of the standard WCC surrogate methods alongside data
  shuffling and participant shuffling. Moriano et al. (2024, PLOS Biology) list
  the stationary block bootstrap alongside IAAFT, twin, and cyclic-permutation
  surrogates for physiological-time-series correlation testing, using 499
  surrogates as we do.

Positioning, consistent with the existing null vocabulary:
- WCC + surrogate (data/participant) shuffling → SUSY / multiSyncPy lineage;
- WCC + IAAFT → a small refinement on that lineage;
- **WCC + cyclic block permutation → a cross of the Künsch / Politis–Romano
  block-bootstrap family with Ramseyer–Tschacher segment shuffling.**

**Documented weakness (why existence status is still deferred to v2).** Schwartz
et al. (eLife, 2025) show that block-bootstrap nulls scramble data in time and
therefore distort autocorrelation, producing unacceptable false-positive rates
for *some* correlation statistics — i.e. the null's validity is statistic-
dependent. Our own EGT results are consistent with this caution: the null is
clean on a structureless quasi-null (no over-rejection) but underpowered on
single-peak morphologies. We therefore do NOT claim it is a validated existence
test for `inter_peak_cv` / `first_peak_time` in v1; a signal-level validation is
required in v2 before any promotion.

### 7d.2 v1.0 final code audit (read-then-fix, all changes traceable)

A full read-only audit was run before the v1.0 release cleanup.
Resolved this round:

- **Version SSoT.** Package version is `1.0.0` in `__about__.py`, `pyproject.toml`,
  and `__init__.py`. Added `multisync --version` (reads `__about__.__version__`).
  `core.py`'s `schema_version "0.2.0"` is annotated as the JSON *output-structure*
  version, deliberately independent of the package version. (BRM_draft.md keeps
  its own manuscript revision number; it is not a software version.)
- **Docstring↔decision consistency.** Removed stale "circular time-shift null
  pending" / "EXCLUDED pending null-model implementation" language from
  `feature_definitions.py` (FDR_FEATURES, FDR_FAMILIES) and the three event-only
  rows in `feature_status.py`, replacing it with the block-bootstrap / v2-deferral
  wording. The `bimodality_coefficient` docstring no longer says "provisional
  pending a dated DECISION_LOG entry" (that decision was made on 2026-06-29,
  Option B); it now states BC is removed from the confirmatory FDR family but
  retained as an L0 existence-audit descriptor. Its tier remains CONDITIONAL,
  which matches `FEATURE_TIER` (verified).
- **Script trunk separation.** `scripts/` now holds only main-trunk result
  generators (see `docs/SCRIPT_MAP.md`, which maps each script to the BRM trunk
  result it supports). Three one-off diagnostic/"fixed" scripts
  (`analyze_pgt2_fixed.py`, `diagnose_pgt2_drift.py`,
  `diagnose_h2_switching_entropy.py`) and the falsified circular-shift null were
  moved to `experimental/scripts/` (v2 staging).

## 8. Open methodological limitations

1. ISC/shared-stimulus/co-presence confounds are audited, not solved.
2. WCC remains the default substrate for transparency, not because it is universally optimal.
3. Several descriptors remain exploratory and should not be described as independently validated psychological constructs.
4. The feature status table should evolve through explicit method-log entries and tests, not silent edits.
5. Future work should benchmark alternative substrates such as WCLC, PLV, CRQA, MI, and recurrence methods with substrate-specific null models.

---

## 9. Minimal reporting language

Recommended manuscript wording:

> We treated synchrony measurement as an audited evidence chain.  First, signal-level IAAFT tested whether WCC-derived descriptors exceeded independent autocorrelated-signal nulls.  Second, pseudo-pair, time-shift, and where applicable across-stimulus shuffle controls evaluated partner-identity, temporal-alignment, and shared-stimulus alternatives.  Third, dyad-paired permutation tests evaluated whether audited descriptors differentiated experimental conditions.  Feature descriptors were reported with explicit source level, incremental information, paradigm restrictions, and risk notes.

Avoid:

- "IAAFT proves interpersonal coupling";
- "L1/L2 features are confirmatory";
- "onset/rise/recovery are general synchrony features";
- "more synchrony is always better".

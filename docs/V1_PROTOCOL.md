# SyncPipe v1.0 — Frozen Scientific Protocol

> **Status:** Frozen Draft v1.0 (for maintainer ratification)  
> **Baseline:** GitHub HEAD `e89f461` + observation-opportunity audit hardening  
> **Test baseline:** see `tests/README.md` (enforced by `tests/test_suite_health.py`)  
> **Companion doc:** `V1_CLAIM_CEILING.md` (what v1 does and does NOT claim)  
>
> This document freezes the scientific object, statistical design, feature
> hierarchy, null models, and governance of SyncPipe v1. It is the contract
> that every code path (descriptor stack, canonical scientific path, CLI,
> Python API) must obey. No feature, threshold, or null model may enter the
> v1 confirmatory claim set without answering the four pass criteria in
> §10.

---

## 0. One-line scientific object

> SyncPipe v1 provides an **auditable measurement and inference procedure**
> for **continuous low-frequency same-modality dyadic synchrony** — not a
> validated measure of the psychological construct "synchrony quality".
>
> The **validated confirmatory claim** of v1 is a pre-specified two-condition
> dyad-paired contrast. The **measurement core, the L0/L1 existence & structure
> audits, and the predictive mode are design-agnostic** and provide first-class
> support for single-condition, free-play, and continuous synchrony designs
> (see §1).

---

## 1. Unit of analysis

- **Atomic observation:** one zero-lag WCC trace (length `n_wcc_points`) for a
  single dyad within a single condition and modality.
- **Primary confirmatory estimand:** a **contrast** between two pre-specified
  conditions for the *same* dyad (dyad-paired). The dyad is the pairing unit;
  condition is the within-dyad factor. This is the *validated confirmatory
  claim* of v1 — not the only capability of the tool.
- **Design-agnostic core (no condition required):** for any aligned same-modality
  dyad — including a single continuous free-play session — SyncPipe produces the
  WCC trace, interpretable descriptors, and the L0 (signal-level existence) and
  L1 (WCC-level structure) audits. These are legitimate first-class results for
  free-play / continuous designs that do not posit a between-condition contrast.
- **Predictive mode:** `prediction.py` (cross-modal prediction, AR baseline,
  ΔAUC) supports "does dyadic synchrony predict an outcome?" — a regression
  family, distinct from the contrast family, and not part of the confirmatory
  FDR claim.
- **Modality family:** EDA / ECG / RESP / motion-energy are analyzed as
  *separate* families. v1 does **not** build a unified cross-modal psychological
  construct.
- **Time scale:** continuous, low-frequency (preprocessed envelope) signals.
  v1 measures co-fluctuation of aligned low-frequency envelopes, which is a
  *measurement* object, not an inferred cognitive state.

---

## 2. Same-modality dyad (formal definition)

A **same-modality dyad** is:

- two signals from the **same** modality (e.g., both EDA), one per participant
  (A, B);
- already preprocessed to a low-frequency envelope;
- at the **same** `hz`;
- of **equal length** (or, when discontinuities exist, explicitly masked /
  segmented via the discontinuity mask, not silently re-merged);
- analyzed with `cross_modal=False`.

The effective policy is resolved by `pairing_policy(dataset, cross_modal)` and
recorded in the result manifest:

| Policy | Meaning |
|---|---|
| `same_modality` | ≥2 feature columns per modality; canonical same-modality dyad |
| `two_file_dyad_fallback` | exactly 2 modalities, 1 column each, treated as one dyad |
| `cross_modal` | `cross_modal=True`; out of v1 confirmatory scope |
| `same_modality_no_pair` | same modality but no dyad pairing available; diagnostic only |

---

## 3. WCC window and overlap

- **Window:** integer `window_size` samples, **leading window** — `output[i]`
  covers samples `[i, i + window_size - 1]`.
- **Overlap:** controlled by `step` (default `1` → maximal overlap). Window type
  default `"rect"` (rectangular, unweighted).
- **`window_size` is a free parameter, not a claim of optimality.** Its choice
  must be pre-specified and is subject to sensitivity analysis (Gate 4). Reports
  must state the value used.
- **Observation opportunity** (`n_wcc_points`, `n_valid_wcc_points`,
  `valid_wcc_fraction`, `wcc_observation_sec`) is recorded per dyad-condition and
  governs comparability (see §7).

---

## 4. Zero-lag boundary

- v1 computes **zero-lag WCC only**.
- `max_lag_sec` **must be `0`**; any non-zero value raises `ValueError`
  (fail-loud). There is no working lagged estimator in v1.
- "Zero-lag" means the two signals are aligned on the sample grid. Any temporal
  alignment, jitter, or lead-lag correction must be resolved in the
  **preprocessing / alignment** step *before* v1, never inside v1.
- v1's morphology descriptors (`onset_latency`, `rise_time`, `recovery_time`)
  describe the **shape of the WCC trace**, *not* participant lead-lag (see §8).

---

## 5. `peak_amplitude` estimand (primary endpoint)

- **Definition:** the maximum value of the zero-lag WCC trace across valid WCC
  points within the analyzed segment/episode, under the leading-window
  definition.
- **As confirmatory primary endpoint:** estimates the **condition contrast** of
  this maximum (e.g., `task − rest`), conditional on observation opportunity.
- **What it is NOT:** an estimate of *interpersonal coupling strength* per se.
  It is the magnitude of co-fluctuation under the specified, auditable
  procedure. Positive `peak_amplitude` under a significant L2 contrast is
  evidence of *differential co-fluctuation between conditions*, not of causal
  coupling.

---

## 6. Comparable conditions

v1 treats **only** this as the confirmatory design:

```
same dyad
+ two PRE-SPECIFIED conditions
+ dyad-paired permutation
+ explicit contrast = (condition_a, condition_b)
```

**Out of the v1 confirmatory FDR claim** (never promoted to confirmatory without a
new Gate-0 freeze), but several are **supported as exploratory / diagnostic
modes** and may appear labeled as such:

- unpaired group comparison — out of scope (exploratory only);
- mixed-effects models — out of scope (exploratory only);
- arbitrary covariate formulas — out of scope (exploratory only);
- **continuous predictors / outcomes** — supported via `prediction.py`
  (regression family; exploratory, not in the confirmatory FDR set);
- **multi-condition / phased designs** (e.g., early-vs-late, baseline-vs-task) —
  expressible by labeling the phase as a condition and reusing the L2 machinery
  (exploratory-to-contrasting, subject to pre-specification);
- unified cross-modal models — out of scope (diagnostic only).

---

## 7. Length / missingness stop conditions

These conditions **fail loud** (no silent imputation, no silent positive):

| Condition | Behavior |
|---|---|
| signal not 1-D or unequal length per dyad | raise |
| `hz` mismatch between paired records | raise |
| mask length mismatch | raise |
| ambiguous signal column in bridge input | raise |
| `n_valid_wcc_points` below defined floor | governed by `observation_policy` / `eligibility_policy` |
| unequal observation opportunity **across conditions** | `raise` (or `warn`) per `observation_policy` |
| unequal observation opportunity **within a dyad-condition cell** (trial-length variation) | detected and `raise` (or `warn`) per `observation_policy` |
| informative undefinedness (e.g., threshold cannot be derived) | `claimable=False`; no confirmatory claim |

The observation-opportunity aggregation uses `mean()` (not `first()`) and
explicitly detects within-cell trial-length variation, so silent masking of
unequal trials is impossible.

---

## 8. Results that are EXPLORATORY only

| Feature | v1 status | Allowed claim |
|---|---|---|
| `peak_amplitude` | **confirmatory primary endpoint** | condition contrast of max co-fluctuation, controlling observation opportunity |
| `mean_synchrony` | reference comparator | average zero-lag WCC level (descriptive) |
| `dwell_time` | conditional exploratory structure descriptor | episode duration *when* definedness sufficient & threshold policy explicit |
| `switching_rate` | conditional exploratory structure descriptor | WCC state-transition frequency *when* definedness sufficient & threshold policy explicit |
| `onset_latency` / `rise_time` / `recovery_time` | event/morphology exploratory | WCC trace morphology only; **not** participant lead-lag |
| `bimodality_coefficient` / entropy | diagnostic explorer | existence audit; **not** in FDR family |
| prediction (via `prediction.py`) | exploratory regression family | implemented (cross-modal prediction, AR baseline, ΔAUC); **not** in the confirmatory FDR claim |
| morphology clustering | **not in v1** | v2 or appendix only |

> **Core principle:** v1 *computes* structural features, but does **not** wrap
> them into a universal psychological "synchrony quality" construct.

---

## 9. Results that must NOT be interpreted as interpersonal coupling

A positive / significant WCC result is **not by itself** evidence of dyad-specific
interpersonal coupling. The following are explicit null scenarios (Gate 3) that
can produce WCC co-fluctuation without coupling:

- independent white noise (baseline calibration);
- independent autocorrelated signals;
- **shared stimulus** without dyad-specific coupling;
- slow drift / common external driver;
- periodic signals;
- unequal duration (must be rejected or explicitly flagged);
- missingness (must not silently create positives);
- segment seams (must not be re-merged into a false episode).

v1 separates signal from noise via the L0/L1 existence audits, and controls
within-dyad confounds via the L2 paired design — but neither establishes
causality. Causal coupling is **out of scope** for every v1 claim.

---

## 10. Null models and governance

### Three-level evidence chain (locked)

| Level | Question | Method |
|---|---|---|
| **L0** signal-level | "Does a synchrony signal exist beyond noise?" | IAAFT / PRTF surrogate on raw signals |
| **L1** WCC-level | "Does the WCC trace show structured episodes?" | IAAFT on WCC trace |
| **L2** between-condition | "Do features differ across conditions?" | dyad-paired permutation + BH-FDR |

### Pre-registered existence gate (locked)

The L0 existence stage is decided by ONE frozen endpoint on a pre-registered
set of PRIMARY modalities — not by an OR across features or across all
channels present in the dataset. Deciding "synchrony exists" from whichever
feature/modality happened to reach p < .05 is an undeclared multiple
comparison; freezing the endpoint and the modality set removes that freedom.

| Parameter | Code SSoT (`feature_definitions.py`) | v1 value | Rationale |
|---|---|---|---|
| Endpoint | `PRIMARY_EXISTENCE_ENDPOINT` | `peak_amplitude` | Same feature as `PRIMARY_FDR_FAMILY` (n=1), so the existence gate and the confirmatory claim cannot diverge |
| Primary modalities | `PRIMARY_EXISTENCE_MODALITIES` | `("ECG", "EDA")` | Autonomic primary set; ECG and EDA are two readouts of the same autonomic-synchrony construct, so ONE confirming channel suffices (requiring both over-tightens the gate) |
| Dyad-majority threshold | `EXISTENCE_GATE_MIN_PASS_RATE` | `0.5`, strict `>` | "Synchrony exists in this modality" is indefensible below half the dyads |

Gate logic (`_existence_gate_by_modality` in `inference_pipeline.py`): pass rate
is computed **per modality** as the fraction of that modality's dyads
significant on the frozen endpoint; the gate is satisfied when **at least one
primary modality** has pass rate strictly `> 0.5`. Non-primary modalities
(e.g. RESP, which is largely paced/entrained by task structure) are **reported
but excluded** from the gate — they are sensitivity/comparator channels.

`PRIMARY_EXISTENCE_MODALITIES` is **dataset-specific** (the default is the
Lerique ECG/EDA/RESP composition). Datasets with a different channel
composition MUST declare their own primary set via
`SyncPipeConfig.primary_modalities` **before** looking at results; leaving it
`None` inherits the Lerique default. Both parameters are recorded in the run
config and report, so the declared set is auditable after the fact.

### FDR and governance parameters (must be explicit in every run)

- `fdr_scope`: `global` or `within_modality` (PRDS preservation depends on scope).
- `undefined_policy`: `gate` (informative undefinedness blocks confirmatory claim).
- `observation_policy`: `raise` / `warn` / `ignore`.
- `eligibility_policy`: `raise` / `warn`.
- `n_min_dyads`: minimum dyads for a claimable inference (default 10).
- `threshold` / `discontinuity_masks`: per-pair / per-modality mapping must be recorded.

### Claim ceiling (stage_status)

Every L2 result carries `definedness_status`, `claimable`, `eligibility_status`,
and `stage_status`. A result is confirmatory **only if** all gates pass.

---

## 11. Pass criteria (freeze gate)

Any feature, threshold, or null model added to v1 must answer:

```
1. Which estimand does it serve?
2. What claim type is it (confirmatory / reference / conditional-exploratory / morphology / diagnostic)?
3. What is its null?
4. Under what condition does it fail (and fail loud)?
```

If any answer is missing, the element is **not** frozen into v1.

---

## 12. Relationship to code paths

- **Canonical scientific path** (preferred for paper-level results):
  `manifest + config → preflight QC → canonical records → ComputationPipeline →
  WCC + features → observation-opportunity audit → existence audit (L0/L1) →
  design-control audit → L2 paired inference → FDR/definedness/eligibility/
  claimability governance → reproducible report bundle`.
- **`DynamicAnalyzer`**: retained as a low-level descriptor API, **downgraded**
  from the paper-level default path. CLI/API paper-level results must come from
  the canonical path, not a differently-defaulted descriptor path.
- **Unified output bundle** (same manifest+config via Python API and CLI must
  produce identical feature values, thresholds, p-values, exclusions,
  claimability, and manifest metadata).

---

## 13. Known limitations not papered over in v1

- **Peak-duration bias:** handled by reporting observation metadata + strict
  rejection of unequal length. A duration-aware peak correction is **deferred**
  pending independent simulation validation (no maximum-statistic correction
  silently added).
- **NaN / dropout policy:** hard NaN-ratio guard exists; the methodologic
  decision (dropout vs segment-seam distinction, minimum finite WCC points,
  per-finite-segment computation, whether dwell may cross short dropouts) is
  **not yet finalized** and must be stated as a limitation.
- **Real-data status:** datasets are classified as `fully rerunnable` /
  `artifact-backed` / `diagnostic only`. Gordon / Andersen loaders are **not**
  complete; they must not be claimed as fully reproducible validation.

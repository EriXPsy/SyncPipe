# SyncPipe v1 — Limitations

This document states, up front, what SyncPipe v1 is **not yet** and what a
reader should keep in mind when interpreting its results. It is written to be
citable from a methods/limitations section, not to be buried.

---

## 1. Maturity and validation status

**Methodology iterated intensively over a short period.** The v1 design,
feature tiering, decision log, and validation strategy were developed in a
high-intensity sprint spanning **2026-07-01 to 2026-07-08** (36 commits,
single author). The result is internally consistent and well-tested at the
unit level, but it has **not yet been validated by an independent dataset or
by third-party long-term use**.

What this means for interpretation:

- The infrastructure is robust *as a procedure*, but its psychological claims
  (e.g. that a given WCC descriptor indexes a specific coordination mechanism)
  rest on simulation and on a small number of published datasets, not on a
  broad replication base.
- Reported effect sizes from the bundled validation scripts should be read as
  *demonstrations of measurement behavior*, not as established population
  effects.
- A future change to a default (WCC window, onset threshold, surrogate count)
  can shift validation numbers. Re-running the validation suite after any such
  change is mandatory, not optional.

---

## 2. Self-review ceiling

All of the methodological rigor in this repository — the decision log, the
feature-tiering scheme, the null-model audits — was developed through a
dialogue between the author and an AI assistant. This process is high quality,
but it has a structural ceiling that no amount of internal care removes:

> A person (or an AI paired with that person) cannot be *surprised* by their
> own blind spots.

Concrete consequences:

- No independent researcher or external user has, to date, run SyncPipe on
  their own data and publicly agreed or disagreed with a design decision.
- Design choices that feel obviously correct to the author may feel arbitrary
  to a first external reader. The decision log mitigates this by making
  reasoning explicit, but does not replace external scrutiny.
- **Action in progress:** external validation is being sought actively
  (outreach to authors of related tools, sharing with labs, and soliciting
  independent test runs). This is a "find people" task that runs in parallel
  with code work; it is started, not finished.

---

## 3. Modality scope

SyncPipe is multimodal only in the narrow sense defined in `README.md`
("Scope and modalities"):

- In scope: continuous, low-frequency envelopes (ECG/IBI, EDA, respiration,
  motion energy, neural envelopes).
- Out of scope for v1: discrete-event modalities (facial AU sequences,
  turn-taking, speech onsets) and raw high-density EEG/fNIRS. These need
  event-based synchrony measures and are planned for v2.

The package's "modality independence" assumes every input has already been
flattened into the same statistical object (an aligned low-rate trace). Whether
that flattening is equally fidelity-preserving across all modalities is **not
independently verified** — it is an assumption of the input contract, not a
tested guarantee.

---

## 4. Validation-path coverage gaps

The code that *produces* the validation numbers historically had weak
test coverage:

- real-data loaders (`multisync/realtest/lerique_2024.py`,
  `multisync/realtest/gordon_2025.py`) and the pipeline bridge
  (`multisync/pipeline_bridge.py`) carried little or no dedicated unit testing.
- simulation ground-truth generators (e.g. the Kuramoto scripts under
  `scripts/`) were, until recently, inline rather than imported modules.

This matters because a silent bug in a loader or a simulator (a mis-wired
coupling parameter, a dropped segment-boundary mask) would produce validation
output that *looks* completely normal while the floor is wrong. Contract tests
now exercise these paths end-to-end:

- `tests/unit/test_pipeline_io.py` — the pipeline-bridge mask propagation,
  **and** `load_lerique_dataset` / `load_gordon_dataset` called on synthetic
  on-disk datasets (including the P1/P2 discontinuity-mask AND logic — the
  original single-sided mask-drop bug site).
- `tests/unit/test_significance.py` (§ "source: test_simulation_kuramoto.py") —
  the Kuramoto coupling → synchrony generator pulled into
  `multisync/simulation/kuramoto.py`.

> **Honesty note on loader coverage.** The loader *bodies* are only
> partially covered. The `preprocess=False` (raw passthrough) and
> `preprocess=True` (EDA/RESP via scipy) AND-mask branches in
> `lerique_2024.py` are now exercised; the ECG/IBI path (neurokit2) and the
> Gordon CSV resampling edge cases remain lightly covered. Treat loader
> outputs as validated at the *contract* level, not exhaustively.

---

## 5. Statistical caveats carried by specific features

- **Small-sample prediction.** The cross-modal prediction model can be
  over-parameterized when the effective window count is small relative to the
  feature count. `PredictionResult` now reports `n_samples` and
  `feature_to_sample_ratio`; a warning is raised when the ratio drops below 3.
- **Surrogate thresholds are method anchors, not clinical cutoffs.** They rule
  out "no structured synchrony" under specific null models; they do not certify
  psychological meaning.
- **`switching_rate` / `dwell_time` depend on hysteresis binarization.** The
  binarizer now hard-disconnects NaN (segment-boundary) positions; changing the
  hysteresis delta changes these descriptors. Report the delta used.

---

## 6. Naming and related tools

SyncPipe was renamed from *MultiSync* to avoid collision with the published
`multiSyncPy` (Hudson, Wiltshire & Atzmueller, 2023, *Behavior Research
Methods*). A separate, older library named **SyncPy** (Varni & Avril, 2015)
also exists in the same domain and is visually/phonetically close; SyncPipe is
deliberately distinct and is positioned against it in
`docs/DECISION_LOG.md` ("Related tools"). The legacy `multisync` import/CLI
namespace remains only as a compatibility alias and has a published removal
timeline (deprecation warning in v1.1, removal in v2.0).

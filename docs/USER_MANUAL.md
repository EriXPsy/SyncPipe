# SyncPipe v1.0 — User Manual

> Measurement infrastructure for multimodal interpersonal synchrony.
> This manual is for human users. For an agent-oriented capability sheet see
> [`SKILL.md`](SKILL.md). For the intellectual lineage see
> `SYNCPIPE_FAMILY_TREE.md` (when present) and `METHOD_LOG.md`.

---

## 1. What SyncPipe is (and is not)

SyncPipe is a **narrow, validated toolkit** that turns a windowed
cross-correlation (WCC) synchrony trace into a set of **auditable descriptors**
and runs them through a **three-step audited evidence chain**. It is *not* a
black-box "synchrony score" generator: every descriptor carries an explicit
source level, paradigm restriction, and risk note (the *feature status table*),
and every inference claim is gated behind a null model.

The guiding stance: **synchrony measurement is an audited evidence chain, not a
single number.** What SyncPipe gives you is the audit, not a verdict.

---

## 2. Installation

```bash
cd SyncPipe
python -m pip install -e .          # core
python -m pip install -e ".[dev]"   # + test tools
```
Requires Python ≥ 3.10. Check the install:
```bash
syncpipe --version          # -> syncpipe 1.0.0
```
`syncpipe` is the preferred v1 command/import namespace. The older `multisync`
namespace is retained as a compatibility alias during the transition.

---

## 3. Reproduction smoke check

From the repository root:
```bash
python -m pytest
python -m syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 --no-prediction -o artifacts/demo_v1
python scripts/build_feature_table.py
```
This runs the test suite, the synthetic demo with the audited evidence chain,
and the authoritative feature table from the single source of truth.

> The full Gordon / Lerique / Andersen real-data pipelines need raw datasets
> that are not shipped in the repo; `docs/SCRIPT_MAP.md` lists the per-dataset
> runner scripts and the trunk result each supports.

---

## 4. Data input & QC gate

### Input format
The CLI `analyze` command takes comma-separated CSV paths, one per modality.
Each CSV holds the two partners' aligned signals for one modality. Provide the
sampling rate with `--hz` and (optionally) modality names with `-n`.

### The QC gate (`qc.run_quality_check`)
Before any analysis, data passes a **3-stage quality gate** (`multisync/qc.py`).
Each stage returns PASS / WARN / FAIL; a FAIL raises `DataQualityError`.

| Stage | Checks | Why it matters |
|---|---|---|
| **1. Temporal alignment** | whether modalities share a time base / co-start | misaligned start times create a **false CCF lag** equal to the offset — the single most dangerous silent error in lag-based synchrony |
| **2. NaN integrity** | location and fraction of missing values | NaN runs distort WCC windows and episode definitions |
| **3. Sampling uniformity** | constant sampling interval | non-uniform sampling invalidates the fixed-window WCC |

The demo deliberately surfaces the alignment warning so you can see the gate
working. Treat WARN as "confirm this is expected", FAIL as "fix before trusting
results".

---

## 5. The two CLI commands

### `syncpipe demo`
Runs the complete methods demonstration on a synthetic ground-truth dyad and
writes all audit reports. Fast smoke run:
```bash
syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo
```
Outputs: `viewer_results.json`, `feature_table.csv`, `feature_status_table.csv`,
`TABLE1_FEATURE_STATUS.tex`, `DEMO_REPORT.md`.

### `syncpipe analyze`
Runs the pipeline on your own data:
```bash
syncpipe analyze -i behavior.csv,neural.csv -n behavior,neural \
    --hz 4 --window-size 40 --surrogates 500 -o results.json
```

---

## 6. The three-step audited evidence chain (the spine)

This is the conceptual core. Do not skip steps or reorder them.

1. **Synchrony-existence audit** — `synchrony_existence_audit(...)`.
   Null = **signal-level IAAFT** (randomise each signal while preserving its
   amplitude distribution and autocorrelation). Question: *do the observed
   WCC-derived descriptors exceed what independent autocorrelated signals
   produce?* A significant result is **necessary but not sufficient** for
   interpersonal coupling — it does not rule out shared-stimulus or co-presence
   explanations.

2. **Design-control audit** — `design_control_audit(...)`.
   Pseudo-pair (shuffle which two people are paired), time-shift (break temporal
   alignment), and — where applicable — across-stimulus shuffle. Question: *is
   the effect partner-specific and time-locked, or an artifact of shared input?*

3. **Group condition inference** — `InferencePipeline`.
   Dyad-paired permutation tests with **Benjamini–Hochberg FDR** across the
   **3-feature primary family**: `peak_amplitude`, `dwell_time`, `switching_rate`.
   `mean_synchrony` is reported as a **reference comparator** but is *not* in the
   multiplicity correction. Question: *do the audited descriptors differ across
   conditions/groups?*

---

## 7. The feature / descriptor table

The single source of truth is `multisync/feature_definitions.py` (math) and
`multisync/feature_status.py` (communication). `scripts/build_feature_table.py`
emits `docs/FEATURE_TABLE.csv` / `.md`.

Key facts (v1.0):
- **Primary FDR family (confirmatory):** `peak_amplitude`, `dwell_time`,
  `switching_rate`.
- **Reference comparator:** `mean_synchrony` (reported, not FDR-corrected).
- **Exploratory / secondary** (reported with definedness, never confirmatory):
  `bimodality_coefficient`, `synchrony_entropy`, `fraction_above_threshold`,
  `first_peak_time`, `inter_peak_cv`, and the event-only morphology descriptors.
- **Onset threshold** (`ONSET_THRESHOLD`) = 0.5 by default; episodes are
  segments above threshold.

Always read a descriptor's row in the status table before reporting it: it tells
you the paradigm restriction (e.g. event-only), the main risk, and whether it
enters the primary FDR family.

Timing / morphology descriptors use raw missing-value semantics: if an event is
not scientifically defined in the WCC trace, the main timing field is `NaN`
(JSON `null`) and the corresponding `*_defined` flag is 0.  Separate
`*_imputed` companion fields exist only for downstream machine-learning workflows
that explicitly need filled duration-like predictors; do not report imputed
values as measured latencies.

---

## 8. Surrogate thresholds: two grounded cut-offs

SyncPipe does not use an arbitrary r-value anchor for "what counts as
synchrony". It derives the threshold from a null distribution (lineage: Lykken &
Venables 1971; Ben-Shakhar 1985):

- **Per-dyad surrogate threshold** (`compute_surrogate_threshold`,
  `SURROGATE_THRESHOLD_PERCENTILE` = 95). The 95th percentile of *that dyad's
  own* IAAFT-surrogate WCC values — "the WCC level this dyad would reach by
  chance". **Use for within-dyad existence**; it adapts to each dyad's null.
- **Session-/condition-pooled threshold** (`compute_session_pooled_threshold`,
  `compute_condition_pooled_thresholds`, `multisync/session_threshold.py`). One
  threshold pooled across all dyads (or per condition). **Use for between-dyad
  comparability** of episode features (dwell_time, switching_rate): if every
  dyad used its own threshold, the very definition of "episode" would differ per
  dyad and group comparisons would be confounded.

Rule of thumb: per-dyad for "does this dyad show synchrony?"; pooled for "do
these groups/conditions differ in synchrony structure?".

---

## 9. Validation status of the descriptors

- **Ground-truth recovery**: PGT-2 (structure), PGT-3 (temporal), EGT-4
  (emergent), GT-5 (Gordon-conditions) batteries recover the intended
  descriptors on known-answer simulations.
- **Real-data incremental value**: on Lerique 2024 (rest vs task) the timing
  descriptors add cross-validated AUC over a `mean_synchrony` baseline,
  demonstrating information *beyond* intensity (not magnitude proxies).
- **Honest limitation (v2)**: the timing descriptors lack a *validated existence
  null*. The circular time-shift null was falsified; the cyclic block-bootstrap
  null is methodologically sound but underpowered on single-peak morphologies
  and confounded on trace-level real data. Existence-test status is deferred to
  v2. Report these descriptors as exploratory, with definedness rates.

---

## 10. Reporting language

Recommended:
> We treated synchrony measurement as an audited evidence chain: signal-level
> IAAFT existence testing, then pseudo-pair / time-shift / across-stimulus
> design controls, then dyad-paired permutation tests with BH-FDR over a
> pre-specified 3-feature family. Descriptors were reported with source level,
> incremental information, paradigm restrictions, and risk notes.

Avoid: "IAAFT proves interpersonal coupling"; "exploratory descriptors are
confirmatory"; "more synchrony is always better".

---

## 11. Where things live

| Path | What |
|---|---|
| `multisync/feature_definitions.py` | math single source of truth |
| `multisync/feature_status.py` | communication single source of truth |
| `multisync/{computation,feature,inference}_pipeline.py` | the three pipelines |
| `multisync/design_controls.py` | existence + design-control audits |
| `multisync/qc.py` | 3-stage data quality gate |
| `multisync/session_threshold.py` | pooled surrogate thresholds |
| `scripts/` | main-trunk result generators (see `docs/SCRIPT_MAP.md`) |
| `experimental/` | v2 staging: unintegrated / falsified / one-off code |
| `docs/METHOD_LOG.md` | dated methodological decisions |

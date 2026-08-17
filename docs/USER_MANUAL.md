# SyncPipe v1.0 — User Manual

> Measurement infrastructure for same-modality dyadic synchrony, analyzed across multiple modality families (EDA / ECG / RESP).
> This manual is for human users. For an agent-oriented capability sheet see
> [`SKILL.md`](SKILL.md). For the methodological lineage see [`METHOD_LOG.md`](METHOD_LOG.md).

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
`syncpipe` is the command/import namespace.

---

## 3. Quick start (5 minutes)

Three escalating checks, all runnable without any third-party data.

**Step 1 — 30 seconds: is it installed?**
```bash
cd SyncPipe
python -m pip install -e ".[dev]"
syncpipe --version            # -> syncpipe 1.0.0
syncpipe demo -o artifacts/demo   # synthetic dyad, full audit report
```

**Step 2 — 2 minutes: does the scientific (canonical) path run?**
```bash
python scripts/reproduce_lerique_paper.py --fast
```
`--fast` builds a synthetic toy-dyad proxy and runs the whole audited evidence
chain (existence → design controls → group inference) with **no OSF download**.
It is a wiring check, not a scientific result.

**Step 3 — 2 minutes: does it hold on real data?**
```bash
python scripts/verify_realdata_consistency.py
```
Re-runs the group inference on the committed Lerique 2024 derived feature
table and checks that (a) the primary endpoint `peak_amplitude` is significant
in every modality and (b) `dwell_time` is correctly gated by the definedness
eligibility rule. Prints `PASS` on success.

Then run the full test suite to confirm the install:
```bash
python -m pytest
python scripts/build_feature_table.py   # regenerate the authoritative feature table
```

> The full Gordon / Lerique / Andersen pipelines from **raw** data need the
> OSF mirrors (see `docs/DATA_ACCESS.md` for the Lerique/ECSU-PCE download and
> layout); the derived Lerique tables needed for Step 3 are committed under
> `artifacts/realtest/lerique_2024/`.

---

## 4. Data input & QC gate

### Two input paths

SyncPipe has **two** data-entry commands, for two different jobs:

- **`syncpipe analyze`** — the paper-level scientific path. It reads a strict
  **manifest CSV** (one row per dyad × modality × condition, pointing at two
  per-person signal files) plus a **TOML config** (with the pre-specified
  contrast), and runs the full audited evidence chain. This is the
  confirmatory path.
- **`syncpipe describe`** — the exploratory descriptor path. It reads plain
  CSVs (a same-modality dyad as `person_a`/`person_b` columns, or two
  single-column files) and emits the viewer JSON. No manifest/config required.

Both accept **preprocessed, aligned, low-frequency envelope signals** (e.g.
ECG→IBI, EDA→SCL, respiration, motion energy). Raw high-frequency data must be
reduced to a second-level envelope before entering SyncPipe.

### The QC gate (`qc.run_quality_check`)
Before any analysis, data passes a **4-stage quality gate** (`syncpipe/qc.py`).
Each stage returns PASS / WARN / FAIL; a FAIL raises `DataQualityError`.

| Stage | Checks | Why it matters |
|---|---|---|
| **1. Temporal alignment** | whether modalities share a time base / co-start | misaligned start times create a **false CCF lag** equal to the offset — the single most dangerous silent error in lag-based synchrony |
| **2. NaN integrity** | location and fraction of missing values | NaN runs distort WCC windows and episode definitions |
| **3. Sampling uniformity** | constant sampling interval | non-uniform sampling invalidates the fixed-window WCC |
| **4. Signal integrity** | zero/near-zero variance, flatline runs, optional physiological range | a flatline passes the NaN and interval checks trivially, yet zeroes the WCC denominator (→ NaN) or yields a degenerate coupling estimate; declared marker channels are exempt from the zero-variance FAIL |

The demo deliberately surfaces the alignment warning so you can see the gate
working. Treat WARN as "confirm this is expected", FAIL as "fix before trusting
results".

---

## 5. The three CLI commands

### `syncpipe demo`
Runs the complete methods demonstration on a synthetic ground-truth dyad and
writes all audit reports. Fast smoke run:
```bash
syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo
```
Outputs: `viewer_results.json`, `feature_table.csv`, `feature_status_table.csv`,
`TABLE1_FEATURE_STATUS.tex`, `DEMO_REPORT.md`.

### `syncpipe analyze` (confirmatory path)
Runs the audited evidence chain from a manifest + config:
```bash
syncpipe analyze -m manifest.csv -c config.toml -o results/
```
- `manifest.csv` columns: `dyad_id,modality,condition,person_a_path,person_b_path,hz[,mask_path]`
- `config.toml`: `[analysis]` section; `contrast` is required (two
  pre-specified condition labels).

Minimal `manifest.csv` (two dyads, one modality, two conditions):
```csv
dyad_id,modality,condition,person_a_path,person_b_path,hz
d01,EDA,rest,data/d01_rest_a.csv,data/d01_rest_b.csv,1
d01,EDA,task,data/d01_task_a.csv,data/d01_task_b.csv,1
d02,EDA,rest,data/d02_rest_a.csv,data/d02_rest_b.csv,1
d02,EDA,task,data/d02_task_a.csv,data/d02_task_b.csv,1
```
Minimal `config.toml`:
```toml
[analysis]
contrast = ["rest", "task"]
window_size = 10
surrogate_n = 1000
n_permutations = 10000
```
Each `person_*_path` file is a CSV with a `time` column plus one signal column,
and person A/B time axes must be aligned (same grid, same length).

### `syncpipe describe` (exploratory path)
Runs the descriptor path on ad-hoc CSVs (no manifest/config):
```bash
syncpipe describe -i dyad.csv -n eda --hz 1 --window-size 10 --surrogates 500 -o out.json
```
`dyad.csv` has `time`, `person_a`, `person_b` columns (one same-modality dyad).
Alternatively pass two single-column files via `-i a.csv,b.csv -n x,y`.

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
   Dyad-paired permutation tests with **Benjamini–Hochberg FDR**, applied within
   pre-registered families rather than one flat pool:
   - **primary** (`PRIMARY_FDR_FAMILY`): `peak_amplitude` alone, so the primary
     BH denominator is **m = 1**;
   - **secondary** (`SECONDARY_FDR_FAMILY`): `dwell_time`, `switching_rate`,
     corrected within their own family (**m = 2**) and reported in parallel.

   The two are kept apart because they rest on different null models (L0 vs L1),
   so a shared denominator would not be valid. `mean_synchrony` is reported as a
   **reference comparator** and is *not* corrected at all. Question: *do the
   audited descriptors differ across conditions/groups?*

---

## 7. The feature / descriptor table

The single source of truth is `syncpipe/feature_definitions.py` (math) and
`syncpipe/feature_status.py` (communication). `scripts/build_feature_table.py`
emits `docs/FEATURE_TABLE.csv` / `.md`.

Key facts (v1.0):
- **Primary FDR family (confirmatory, m = 1):** `peak_amplitude`.
- **Secondary FDR family (parallel, m = 2):** `dwell_time`, `switching_rate` —
  BH-corrected within their own family, never pooled into the primary denominator.
- `FDR_FEATURES` is primary + secondary (3 names) and is the descriptive export
  surface, not the primary multiplicity denominator.
- **Reference comparator:** `mean_synchrony` (reported, not FDR-corrected).
- **Exploratory / secondary** (reported with definedness, never confirmatory):
  `bimodality_coefficient`, `synchrony_entropy`, `fraction_above_threshold`,
  `first_peak_time`, `inter_peak_cv`, and the event-only morphology descriptors.
- **Onset threshold.** The scientific pipeline derives the onset cut-off
  **per modality** from a pooled IAAFT surrogate null (§8,
  `records_to_inference_inputs(onset_threshold="session_pooled")`): EDA and ECG
  therefore get *different*, modality-calibrated thresholds while every dyad of a
  modality still shares one cut-off for cross-dyad comparability. `ONSET_THRESHOLD
  = 0.5` is the **fallback / sensitivity constant only** (used when a modality's
  pooled null is degenerate, and as the fixed baseline in sensitivity sweeps) —
  it is **not** the scientific default. Surrogate-derived thresholds are
  hard-capped at `SURROGATE_THRESHOLD_MAX = 0.9` and fall back to 0.5 above that
  ceiling.

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

## 8. Surrogate thresholds: grounded cut-offs by granularity

SyncPipe does not use an arbitrary r-value anchor for "what counts as
synchrony". It derives every onset threshold from an IAAFT surrogate null
distribution (lineage: Lykken & Venables 1971; Ben-Shakhar 1985), at one of
three granularities.

**Canonical scientific default — per-modality pooled.**
`compute_session_pooled_thresholds_by_modality` (`syncpipe/session_threshold.py`)
derives *one* threshold **per modality** by pooling IAAFT surrogates across all
dyads of that modality. This is the default used by the audited evidence chain
(`records_to_inference_inputs(onset_threshold="session_pooled")`,
`BatchComputationPipeline`): it preserves **cross-modal comparability** (every
dyad of a modality shares one threshold) *and* **within-modality calibration** —
slow/smooth signals (e.g. EDA, low WCC amplitude) and fast/spiky signals (e.g.
ECG, high WCC amplitude) get *different*, modality-appropriate cut-offs instead
of being forced onto a single global value that fits neither. If a modality's
pooled null is degenerate (too few dyads), that modality falls back to
`ONSET_THRESHOLD = 0.5` with a fail-loud warning.

**Optional — per-dyad surrogate threshold.** `compute_surrogate_threshold`
(`SURROGATE_THRESHOLD_PERCENTILE` = 95) returns the 95th percentile of *that
dyad's own* IAAFT-surrogate WCC values — "the WCC level this dyad would reach
by chance". Use for within-dyad existence; it adapts to each dyad's null.

**Optional — session-/condition-pooled threshold.**
`compute_session_pooled_threshold` / `compute_condition_pooled_thresholds` pool
*all* dyads (or per condition) into a single global null. Use when
modality-specific calibration is not needed.

**Sensitivity / fallback constant — `ONSET_THRESHOLD = 0.5`.** A fixed value
forwarded unchanged for sensitivity sweeps and paper reproductions; also the
fallback when a pooled null is degenerate. It is **not** the scientific default.
All surrogate-derived thresholds are hard-capped at `SURROGATE_THRESHOLD_MAX =
0.9` (periodicity / strong-autocorrelation protection): above 0.9 a derived
cut-off is treated as an artifact and falls back to 0.5.

Rule of thumb: per-dyad for "does this dyad show synchrony?"; **per-modality
pooled** for the canonical group/condition pipeline; coarser session/condition
pooling when modality calibration is unwanted; fixed 0.5 for sensitivity
analysis.

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
| `syncpipe/feature_definitions.py` | math single source of truth |
| `syncpipe/feature_status.py` | communication single source of truth |
| `syncpipe/{computation,feature,inference}_pipeline.py` | the three pipelines |
| `syncpipe/design_controls.py` | existence + design-control audits |
| `syncpipe/qc.py` | 4-stage data quality gate |
| `syncpipe/session_threshold.py` | pooled surrogate thresholds |
| `scripts/` | main-trunk result generators (see `docs/SCRIPT_MAP.md`) |
| `experimental/` | v2 staging: unintegrated / falsified / one-off code |
| `docs/METHOD_LOG.md` | dated methodological decisions |

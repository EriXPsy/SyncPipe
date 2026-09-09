# SyncPipe — User Manual

This manual explains the normal user path. Technical method notes are linked at
the end rather than mixed into every instruction.
Package version: see `syncpipe --version` (release notes in
[`CHANGELOG.md`](../CHANGELOG.md)).

## 1. What question does SyncPipe answer?

SyncPipe analyzes two aligned time series from two people.

It does not simply ask:

> Are the two signals correlated?

It asks:

1. Is the apparent co-movement larger than expected from two independent but
   naturally slow or rhythmic signals?
2. Is it stronger for the real partners than for mismatched partners?
3. Does it depend on the original timing?
4. Could a shared event or stimulus explain it?
5. Does the chosen measure differ between the study's conditions?

The result is a set of separate answers and a short statement of the strongest
conclusion supported.

## 2. What data can I use?

Use SyncPipe when you have:

- exactly two people;
- the same type of signal for both people;
- aligned time points;
- a constant sampling rate;
- a continuous, already preprocessed, low-frequency trace.

Examples:

- EDA/SCL envelopes;
- ECG-derived IBI or other justified cardiac traces;
- respiration envelopes;
- continuous motion-energy traces.

Do not feed raw ECG, raw EEG, raw fNIRS, facial events, speech turns, or other
high-dimensional/event data into the current study-analysis path.

## 3. Install and check

```bash
python -m pip install -e .
syncpipe --version
```

Optional extras — install only what your data needs:

```bash
python -m pip install -e ".[ecg]"   # raw ECG -> IBI preprocessing (neurokit2)
python -m pip install -e ".[rqa]"   # recurrence-quantification convergence channel
python -m pip install -e ".[xls]"   # legacy .xls OSF workbooks (e.g. Han-bzkdy)
```

Create a safe example project:

```bash
syncpipe external-kit -o example
cd example
syncpipe analyze -m manifest.csv -c config.toml -o results
```

Open this file first:

```text
results/REPORT.md
```

## 4. Two ways to use SyncPipe

### A. Describe one pair

Use this when exploring one aligned dataset:

```bash
syncpipe describe \
  -i dyad.csv \
  -n EDA \
  --hz 1 \
  --window-size 20 \
  -o description.json
```

This calculates descriptive values. It does not run the complete study-level
checks.

### B. Analyze a study

Use this for multiple dyads and/or a planned condition comparison:

```bash
syncpipe analyze \
  -m manifest.csv \
  -c config.toml \
  -o results
```

This is the recommended path for research reporting.

### C. Call SyncPipe from Python (notebooks and scripts)

The CLI is a thin wrapper. The study-level canonical entry from Python is the
audited evidence chain:

```python
from syncpipe.pipeline_bridge import records_to_inference_inputs
from syncpipe.inference_pipeline import InferencePipeline

# records: objects exposing dyad_label, modality, condition,
#          person_a / person_b (DataFrame or 1-D array), target_hz
inputs = records_to_inference_inputs(
    records, hz=1.0, window_size=20, onset_threshold="session_pooled",
)
pipe = InferencePipeline(
    inputs.features_df, hz=1.0, surrogate_n=100, seed=42, n_workers=4,
)
chain = pipe.run_audited_evidence_chain(
    raw_signals=inputs.raw_signals,
    wcc_window_size=20,
    condition_col="condition", dyad_col="dyad_id",
)
print(chain["legacy_fields"]["permitted_claim"])  # strongest supported claim
```

Rules the API enforces (also visible in error messages):

- observation labels are built as `<dyad>__<modality>__<condition>`;
  a dyad id containing `__` is rejected (it would corrupt label parsing);
- all records of one analysis share one sampling rate — resample first;
- every public class and function is listed in
  [`API_REFERENCE.md`](API_REFERENCE.md).

## 5. Prepare the three input files

### 5.1 Signal CSV files

Each person needs one file per signal type and condition:

```csv
time,value
0,0.12
1,0.18
2,0.10
```

Person A and Person B must have the same time points and length.

### 5.2 Manifest CSV

The manifest tells SyncPipe which two files form a pair:

```csv
dyad_id,modality,condition,person_a_path,person_b_path,hz,signal_type,unit,preprocessing_path
d01,EDA,rest,data/d01_rest_a.csv,data/d01_rest_b.csv,1,EDA_envelope,z_score,preprocessing/eda.json
d01,EDA,task,data/d01_task_a.csv,data/d01_task_b.csv,1,EDA_envelope,z_score,preprocessing/eda.json
```

Column meanings:

| Column | Meaning |
|---|---|
| `dyad_id` | pair identifier |
| `modality` | short signal label used in results |
| `condition` | study condition |
| `person_a_path` / `person_b_path` | aligned signal files |
| `hz` | samples per second |
| `signal_type` | what the values represent |
| `unit` | output unit after preprocessing |
| `preprocessing_path` | JSON record of processing steps |
| `mask_path` | optional file marking unusable samples or boundaries |

### 5.3 Analysis settings

```toml
[analysis]
contrast = ["rest", "task"]
main_measure = "peak_amplitude"
main_modalities = ["EDA"]
window_size = 20
surrogate_n = 1000
n_permutations = 10000
```

| Setting | Plain meaning |
|---|---|
| `contrast` | conditions to compare |
| `main_measure` | value selected before viewing results |
| `main_modalities` | signal types used for the main conclusion |
| `window_size` | samples per correlation window |
| `surrogate_n` | randomized independent-signal comparisons |
| `n_permutations` | condition-label swaps |

The current complete analysis supports `peak_amplitude` as the main measure.
Other values are supplementary and may have additional limitations.

### 5.4 Processing record

Example:

```json
{
  "schema_version": "1.0.0",
  "signal_type": "EDA_envelope",
  "output_unit": "z_score",
  "software": {"name": "my-preprocessor", "version": "0.1"},
  "steps": [
    {"name": "lowpass", "parameters": {"cutoff_hz": 0.05}},
    {"name": "resample", "parameters": {"target_hz": 1.0}},
    {"name": "zscore", "parameters": {"scope": "person_session"}}
  ]
}
```

The manifest and processing record must agree on signal type and unit.

## 6. What happens during analysis?

### Step 1 — Check the data

SyncPipe checks:

- matching time points;
- matching lengths;
- constant sampling;
- missing and infinite values;
- flat or near-flat signals;
- usable continuous segments.

Invalid settings stop the analysis. Individual unusable observations are listed
in `exclusion_report.csv` with a reason.

### Step 2 — Build a co-movement trace

SyncPipe slides a window across both signals and computes Pearson correlation in
each complete window. Missing values and declared boundaries are never removed
to join non-adjacent samples.

### Step 3 — Describe the trace

The current main value, `peak_amplitude`, is the largest locally smoothed
positive correlation in the trace. Secondary values describe average level,
time above a threshold, episode duration, and switching.

These values are descriptions of the correlation trace—not direct measurements
of relationship quality or causal influence.

### Step 4 — Try to explain the result away

SyncPipe runs separate checks:

| Check | Simple question |
|---|---|
| independent-signal comparison | Could two unrelated rhythmic signals reach this value? |
| mismatched-partner comparison | Are real partners stronger than shuffled partners? |
| shifted-time comparison | Does the result depend on the original timing? |
| shared-stimulus comparison | Can the measured event order explain it? |

A check can return yes, no clear support, not enough information, or could not
test. “Not enough information” is not treated as a negative finding.

### Step 5 — Compare conditions

For each dyad, SyncPipe calculates the difference between the two conditions and
uses label swapping to test whether the median difference is larger than
expected by chance. It reports:

- median difference in the original units;
- the middle 50% of dyad differences;
- a 95% median interval when sample size permits;
- raw and adjusted p-values;
- number of usable dyads;
- whether the result is suitable for the main report.

## 7. Read the output in this order

### 1. `REPORT.md`

Short plain-language summary, layered so that confirmatory results lead:

- strongest supported conclusion;
- condition comparison (pre-registered primary measure);
- pre-specified secondary measures (multiplicity-corrected within their family);
- reference and exploratory measures, folded at the bottom of the report —
  visible for transparency, not confirmatory;
- checks that passed, failed, or lacked information;
- explanations still possible;
- excluded data;
- important limits.

In `evidence_graph.json`, the existence stage status is one of `supported`,
`not_supported`, or `inconclusive`; the raw existence gate additionally
carries `gate_status` (`pass` / `fail` / `not_evaluable`) — `not_evaluable`
means the signal type was not among the pre-registered primary modalities,
i.e. the gate did not adjudicate. That is a scope statement, not a negative
finding.

### 2. `evidence_graph.json`

Machine-readable details for every check. Intended for software integrations and
advanced review.

### 3. `qc_report.json`

How much data was usable, segment lengths, sampling information, and thresholds.

### 4. `exclusion_report.csv`

Every excluded observation and the reason.

### 5. `features.csv`

Calculated values per dyad, signal type, and condition.

Other files contain detailed randomized comparisons and WCC traces.

## 8. How to write the result

Prefer:

> The two signals showed co-movement beyond the independent-signal comparison,
> but real partners did not clearly exceed mismatched partners. The result does
> not support a partner-specific coupling interpretation.

Avoid:

> SyncPipe proved that the participants were physiologically coupled.

A significant condition difference only means that the selected trace summary
differed between conditions. It does not establish causality, direction,
relationship quality, or clinical importance.

## 9. Missing data and short recordings

SyncPipe keeps the original time axis. Missing samples and recording boundaries
split the signal into continuous pieces. Randomized comparisons are generated
inside eligible pieces rather than across gaps.

A result may be unavailable because:

- no continuous piece is long enough;
- too few WCC windows remain;
- too few dyads have valid values in both conditions;
- the number of randomizations cannot produce a sufficiently small p-value.

These cases are reported as insufficient information, not silently converted to
zero or “no effect.”

## 9b. Troubleshooting and FAQ

| Symptom | Most likely cause | What to do |
|---|---|---|
| `ValueError: dyad_label ... contains '__'` | A dyad id embeds the label separator (e.g. `p1__p2`) | Rename the dyad with a different separator (`p1-p2`) |
| `ModuleNotFoundError: neurokit2` | Raw-ECG loading without the `[ecg]` extra | `pip install ".[ecg]"` |
| `Missing optional dependency 'xlrd'` | Legacy `.xls` workbooks without the `[xls]` extra | `pip install ".[xls]"` |
| `target_hz ... does not match bridge hz` | Records mixed sampling rates | Resample all records to one rate before pooling |
| `No usable records after quality gates` | High NaN share, flat signals, or traces shorter than the window | Check `exclusion_report.csv`; lower `window_size` only if justified |
| Existence gate `not_evaluable` | Signal type not among the pre-registered primary modalities | Scope statement, not a failure; see §7 |
| p-values never below the reported minimum | Randomization count too low for the attainable p floor (two-sided floor ≈ 2/(n+1)) | Increase `surrogate_n` / `n_permutations` for publication runs |
| Results differ from a pre-1.0.1 run | Known statistical-kernel fixes in 1.0.1 (see `CHANGELOG.md`) | Regenerate p-values from the current package; old p-values are void |

**Can I use behavioral (non-physiological) data?** Yes — the pipeline is
modality-agnostic (it has been fully audited on 10 Hz motion-energy data).
What is fixed is the list of *pre-registered primary modalities* that
adjudicate the existence gate; other signal types still get the complete
per-pair audits and condition comparison.

**Where do I get a runnable example?** `syncpipe external-kit -o example`
creates a self-contained project; `syncpipe demo -o demo_results` runs the
synthetic single-dyad demo.

## 10. Old project migration

```bash
syncpipe migrate \
  -m manifest.v1.csv \
  -c config.v1.toml \
  -o migrated \
  --signal-type EDA_envelope \
  --unit z_score \
  --preprocessing-path preprocessing/eda.json \
  --main-modalities EDA
```

Review `MIGRATION_REPORT.json`; user-supplied assumptions are not scientific
facts inferred by the software.

## 11. Current scientific status

SyncPipe has:

- internal known-answer simulations;
- adversarial negative controls;
- public-data development analyses;
- automated software and method-contract tests;
- an external-validation kit.

SyncPipe does not yet have an unaffiliated published own-data replication.
`peak_amplitude` is sensitive to recording length and artifacts, so it should
not be described as a universal measure of interpersonal synchrony.

## 12. Advanced reference

| Topic | Document |
|---|---|
| every public class and function | [`API_REFERENCE.md`](API_REFERENCE.md) |
| limitations | [`LIMITATIONS.md`](LIMITATIONS.md) |
| construct and rival explanations | [`CONSTRUCT_VALIDITY.md`](CONSTRUCT_VALIDITY.md) |
| external validation | [`EXTERNAL_VALIDATION.md`](EXTERNAL_VALIDATION.md) |
| architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| feature definitions | [`FEATURE_TABLE.md`](FEATURE_TABLE.md) |
| historical decisions | [`METHOD_LOG.md`](METHOD_LOG.md) |
| contributing | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| agent-facing skill sheet | [`SKILL.md`](SKILL.md) |

## Technical terms

| Term | Plain meaning |
|---|---|
| WCC | sliding-window Pearson correlation |
| surrogate / null | randomized comparison under a stated alternative explanation |
| endpoint | main value selected before analysis |
| evidence profile | separate results of the different checks |
| claim ceiling | strongest conclusion supported |
| provenance | record of data processing |
| FDR | correction for testing several values |

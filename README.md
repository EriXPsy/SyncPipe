# SyncPipe

**SyncPipe checks whether two aligned time series move together, tests common alternative explanations, compares conditions, and writes a report that states what the result does—and does not—support.**

It is designed for dyadic, continuous, low-frequency signals such as processed
EDA, ECG/IBI, respiration, and motion-energy traces.

## The problem

A high correlation between two people does not necessarily mean they influenced
each other. It can also come from:

- the same video, task, or event;
- slow trends in both signals;
- movement or measurement artifacts;
- choosing a favorable time window;
- treating overlapping windows as independent data;
- mismatched preprocessing or missing-data rules.

SyncPipe turns these concerns into repeatable checks instead of leaving them to
one-off analysis scripts.

## What it does

For each pair of aligned signals, SyncPipe:

1. checks timing, missing data, sampling, and flat signals;
2. computes a sliding-window Pearson correlation trace;
3. summarizes that trace with clearly defined measures;
4. compares the observed result with independently randomized signals;
5. tests whether real partners exceed mismatched partners;
6. tests whether the result depends on the original timing;
7. optionally tests whether a shared stimulus can explain the result;
8. compares pre-selected conditions across dyads;
9. writes a short report, detailed tables, and a machine-readable audit trail.

SyncPipe does **not** prove causality, relationship quality, clinical meaning,
or who leads whom.

## Current scope

Supported scientific path:

- exactly two people;
- two aligned signals from the same signal type;
- continuous, already preprocessed, low-frequency traces;
- zero-lag sliding-window correlation;
- a pre-selected two-condition comparison for the main statistical test.

Not currently supported as a validated path:

- raw ECG, raw EEG, or raw fNIRS preprocessing;
- native high-dimensional hyperscanning;
- turn-taking or other discrete-event synchrony;
- cross-signal-type fusion such as EDA from one person versus ECG from another;
- causal or directional coupling.

## How SyncPipe relates to other tools

Several open-source packages cover parts of this space, and SyncPipe is
designed to complement rather than replace them:

- **multiSyncPy** (Hudson, Wiltshire & Atzmueller, 2023, *Behavior Research
  Methods*) offers a broad collection of multivariate, group-level
  coordination measures with two surrogation techniques.
- **SUSY / mv-SUSY** (Tschacher & Meier) provides lightweight surrogate-based
  synchrony estimates for physiological and behavioral recordings; the
  research community has documented which kinds of shared signal structure
  this family of methods can and cannot detect (see Upham's `susy_limits`
  repository).
- **pyspi** compiles a large library of pairwise interaction measures.

SyncPipe's contribution is not more measures — it is a **governed inference
standard for dyadic continuous synchrony**: one pre-specified estimator
(zero-lag sliding-window correlation), a three-tier null ladder
(signal-level existence → structure-level → dyad-paired condition
comparison), design controls for shared-stimulus and partner-mismatch
explanations, empirically calibrated false-positive rates shipped as CI
tests, and a descriptor-status table that states, for every measure, exactly
which conclusions the evidence supports. If you need group-level
multivariate measures, use multiSyncPy; if you need auditable two-person
inference with stated claim limits, that is SyncPipe's niche.

## Install

For normal use, install the released package from PyPI:

```bash
python -m pip install syncpipe
```

For source development and tests, install the checked-out repository with its
 development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Check the installation:

```bash
syncpipe --version
```

## Try it without your own data

Two ways to see SyncPipe work:

```bash
# 1. A full self-contained example project (manifest -> report):
syncpipe external-kit -o example
cd example
syncpipe analyze -m manifest.csv -c config.toml -o results

# 2. A synthetic single-dyad demo with audits and a viewer file:
syncpipe demo -o demo_results
```

Start with:

```text
results/REPORT.md
```

Then inspect:

```text
results/evidence_graph.json   # every check and its result
results/qc_report.json        # data-quality and usable-data details
results/exclusion_report.csv  # what was excluded and why
results/features.csv          # calculated values
```

The example checks software usability only. It is not scientific validation.

## Analyze your own study

```bash
syncpipe analyze \
  -m manifest.csv \
  -c config.toml \
  -o results
```

### 1. Signal files

Each person's file must contain:

```csv
time,value
0,0.12
1,0.18
2,0.10
```

The two files must use the same time points and sampling rate.

### 2. Manifest

One row describes one dyad, signal type, and condition:

```csv
dyad_id,modality,condition,person_a_path,person_b_path,hz,signal_type,unit,preprocessing_path
d01,EDA,rest,data/d01_rest_a.csv,data/d01_rest_b.csv,1,EDA_envelope,z_score,preprocessing/eda.json
d01,EDA,task,data/d01_task_a.csv,data/d01_task_b.csv,1,EDA_envelope,z_score,preprocessing/eda.json
```

### 3. Analysis settings

```toml
[analysis]
contrast = ["rest", "task"]
main_measure = "peak_amplitude"
main_modalities = ["EDA"]
window_size = 20
surrogate_n = 1000
n_permutations = 10000
```

Plain meaning:

- `contrast`: the two conditions to compare;
- `main_measure`: the value chosen before looking at the result;
- `main_modalities`: the signal types used for the main conclusion;
- `window_size`: samples in each correlation window;
- `surrogate_n`: randomized comparisons for the independent-signal check;
- `n_permutations`: label swaps for the condition comparison.

The locked v1.0 protocol currently supports `peak_amplitude` as the only main
measure with a complete signal-level check. Other measures are reported as
secondary or exploratory results.

### 4. Processing record

`preprocessing_path` points to JSON that records how the signal was produced:

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

SyncPipe stores this record and its file hash with the results.

## How to read the report

The report answers seven plain questions:

1. Is the observed co-movement larger than expected from independent signals?
2. Was time structure tested?
3. Are real partners stronger than mismatched partners?
4. Does the result depend on the original timing?
5. Can the measured shared stimulus explain it?
6. Was reciprocal interaction itself tested?
7. Is the chosen measure different between conditions?

Possible answers are:

- **YES** — this check supports the stated conclusion;
- **NO CLEAR SUPPORT** — the check ran but did not support it;
- **NOT ENOUGH INFORMATION** — the study or simulation count could not answer it;
- **COULD NOT TEST** — required values were invalid or unavailable.

The report also lists explanations that remain possible. A result can be
specific to timing but not specific to the real partners; SyncPipe keeps both
facts instead of compressing them into one score.

## Quick exploratory description

For one aligned CSV without a full study manifest:

```bash
syncpipe describe \
  -i dyad.csv \
  -n EDA \
  --hz 1 \
  --window-size 20 \
  -o description.json
```

This calculates descriptive values. It is not the full audited study analysis.

## Migrating old inputs

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

Review `MIGRATION_REPORT.json`; migration cannot infer scientific choices for
you.

## Validation status

SyncPipe currently has internal simulation checks, public-data development
analyses, automated software/method tests, and an external-validation kit. It
does not yet have an independent published own-data replication.

`peak_amplitude`, the current main measure, is sensitive to recording length and
artifacts. Equal observation opportunity, data-quality checks, randomized
comparisons, and design controls reduce risk but do not make it a universal
measure of interpersonal synchrony.

See:

- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`docs/CONSTRUCT_VALIDITY.md`](docs/CONSTRUCT_VALIDITY.md)
- [`docs/EXTERNAL_VALIDATION.md`](docs/EXTERNAL_VALIDATION.md)
- [`docs/PLAIN_LANGUAGE.md`](docs/PLAIN_LANGUAGE.md)
- [`CHANGELOG.md`](CHANGELOG.md)

## Technical glossary

These terms appear in machine-readable files and advanced APIs; ordinary users
can rely on `REPORT.md`.

| Technical term | Plain meaning |
|---|---|
| WCC | sliding-window Pearson correlation |
| surrogate / null | randomized comparison showing what can happen without the tested link |
| endpoint | main measure chosen before analysis |
| evidence profile | results of the separate checks |
| claim ceiling | strongest conclusion supported after considering failed or missing checks |
| provenance | record of how input data were processed |
| FDR | correction for testing several measures |

## Documentation

| Document | What it covers |
|---|---|
| [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) | End-to-end walkthrough: preparing signals, manifest, settings, reading the report |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Generated reference for every public class and function |
| [`docs/DESIGN_PRINCIPLES.md`](docs/DESIGN_PRINCIPLES.md) | Why the pipeline is shaped the way it is |
| [`docs/V1_PROTOCOL.md`](docs/V1_PROTOCOL.md) | Locked statistical protocol (windows, nulls, gates, FDR families) |
| [`docs/V1_CLAIM_CEILING.md`](docs/V1_CLAIM_CEILING.md) | What conclusions the current evidence supports — and what it does not |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Known limitations and boundary conditions |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute: test layers, verification rules, conventions |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

## Development

```bash
pytest tests/ -m "not slow" -q
pytest tests/ -m slow -q
python -m build
```

## License

MIT

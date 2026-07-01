# SyncPipe

> **Measurement infrastructure for multimodal interpersonal synchrony.**  
> SyncPipe is not intended to be merely a feature-profile generator. Its v1 goal is to provide a standardized, auditable measurement procedure for dyadic synchrony: from aligned signals, to WCC traces, to interpretable descriptors, to null-model audits, to design-specific confound checks, to group-level inference.

SyncPipe is an open-source Python package for analyzing **dyadic peripheral physiological and behavioral synchrony**. It is built for psychology, social neuroscience, psychophysiology, developmental science, psychotherapy, teamwork, and related fields where researchers need more than a single mean-correlation score but also need stronger statistical guardrails than one-off scripts usually provide.

The central claim is deliberately narrow:

> SyncPipe helps researchers measure, audit, and report synchrony evidence in a standardized way. It does **not** automatically prove interpersonal coupling, causality, clinical meaning, or psychological mechanism.

---

## The niche: synchrony measurement infrastructure

Most synchrony tools provide one or more synchrony metrics. SyncPipe aims to provide the **measurement infrastructure around the metric**:

1. **Trace construction** — a transparent default synchrony trace based on Windowed Cross-Correlation (WCC), with clear limitations.
2. **Feature descriptor table** — a structured set of WCC-derived descriptors with explicit source, incremental information, paradigm restrictions, and risk notes.
3. **Standardized procedure** — a reproducible sequence from quality control to feature extraction to inference.
4. **Null-model audits** — signal-level and design-level tests that clarify what a positive synchrony result does and does not rule out.
5. **Governance** — a single source of feature math, method logs, tests, and exported artifacts so that definitions do not silently drift across papers.

In this sense, SyncPipe is closer in spirit to DPABI-like scientific infrastructure than to a single synchrony score. The ambition is not to declare one universal synchrony measure, but to make synchrony measurement **transparent, auditable, comparable, and falsifiable**.

---

## What SyncPipe does and does not do

### It does

- Accept aligned dyadic time series, typically preprocessed physiological/behavioral envelopes at a common low rate, e.g. ECG/IBI, EDA, respiration, motion energy, or neural envelopes.
- Compute WCC traces as the default measurement substrate.
- Extract WCC-derived synchrony descriptors, including intensity, occupancy, structure, distribution-shape, and event-timing descriptors.
- Provide a simple feature status table via `multisync.feature_status_table()` and `artifacts/demo_v1/feature_status_table.csv`.
- Run a three-step audited evidence chain:
  1. synchrony-existence audit;
  2. design-control audit;
  3. group condition inference.
- Export reproducible JSON/CSV/Markdown artifacts for inspection and viewer integration.

### It does not

- Prove causality. Lead-lag estimates are temporal-precedence descriptions, not evidence of psychological driving.
- Prove dyad-specific interpersonal coupling from WCC+IAAFT alone. Signal-level IAAFT does not remove shared-stimulus or co-presence confounds.
- Replace raw physiological preprocessing. High-frequency raw signals should be converted into scientifically justified second-level time series before entering SyncPipe.
- Provide clinically calibrated thresholds. Current thresholds are methodological anchors, not diagnostic cutoffs.
- Claim that every descriptor is confirmatory. Several descriptors are intentionally exploratory or event-mode-only.

---

## Conceptual architecture

SyncPipe has five infrastructure layers.

| Layer | Question | Main object | Output |
|---|---|---|---|
| 1. Data/QC layer | Are the signals aligned, finite, and sampled consistently? | aligned dyadic time series | quality report / diagnostics |
| 2. Trace layer | What is the moment-to-moment synchrony substrate? | WCC trace | WCC arrays per dyad/modality/condition |
| 3. Descriptor layer | What aspects of the WCC trace are being summarized? | WCC-derived features | feature table |
| 4. Audit/inference layer | What nuisance explanations have been ruled out? | null and design-control tests | evidence-chain report |
| 5. Governance/export layer | Can the analysis be reproduced and inspected? | SSoT, method log, artifacts | JSON/CSV/Markdown outputs |

The descriptor layer and the inference layer are deliberately separated. A feature can be useful descriptively without being a primary confirmatory endpoint.

---

## Recommended v1 evidence chain

The recommended v1 inference logic is no longer presented externally as a feature-label hierarchy. It is presented as a **measurement evidence chain**.

### Step 1 — Synchrony-existence audit

**Question:** Do the aligned signals show WCC features that exceed what independent autocorrelated signals could produce?

**Default test:** signal-level IAAFT surrogate audit.

**Interpretation:** passing this step is evidence for a synchrony-like phenomenon above a conservative independent-signal null. It is **necessary but not sufficient** evidence for dyad-specific interpersonal coupling.

Python API:

```python
pipe.run_synchrony_existence_audit(raw_signals, wcc_window_size=20)
```

---

### Step 2 — Design-control audit

**Question:** Could the result be explained by shared stimulus timing, co-presence, task structure, slow drift, or partner-identity mismatch?

**Default controls:**

| Control | What it tests | Main interpretation |
|---|---|---|
| pseudo-pair | real partners vs mismatched partners | if real ≈ pseudo, dyad-specificity is weak |
| time-shift | original alignment vs shifted within-dyad alignment | if shifted remains high, slow drift/block structure remains plausible |
| across-stimulus shuffle | real stimulus order vs independently permuted stimulus segments | for segmented shared-stimulus designs; audits stimulus-locked ISC-like effects |

Python API:

```python
pipe.run_design_control_audit(signal_pairs, wcc_window_size=20)
pipe.run_across_stimulus_shuffle_audit(segments, wcc_window_size=20)
```

This layer is where SyncPipe tries to be most useful to the field: not by pretending shared-stimulus and co-presence problems are solved, but by making them empirically visible and reportable.

---

### Step 3 — Group condition inference

**Question:** Do audited synchrony descriptors differ across experimental conditions or groups?

**Default test:** dyad-paired permutation test with BH-FDR correction.

Python API:

```python
pipe.run_group_condition_inference(
    condition_col="condition",
    dyad_col="dyad_id",
)
```

---

### End-to-end API

```python
from multisync.inference_pipeline import InferencePipeline

pipe = InferencePipeline(features_df, hz=1.0, wcc_window_sec=20.0, surrogate_n=99)

result = pipe.run_audited_evidence_chain(
    raw_signals,
    wcc_window_size=20,
    design_signal_pairs=signal_pairs,
    across_stim_segments=None,  # optional; use for segmented shared-stimulus designs
)
print(result["summary"])
```

---

## Feature descriptors: Table 1 philosophy

SyncPipe's feature table is not a claim that every descriptor is equally validated. It is a measurement map.

Each descriptor is characterized by:

- **source level:** raw signal, WCC trace, threshold-state sequence, distribution shape, or event morphology;
- **incremental information:** what it adds beyond mean synchrony;
- **order sensitivity:** whether temporal order matters;
- **paradigm restrictions:** all, continuous, event-only, or long multi-episode traces;
- **default audit/test:** signal-level IAAFT, design controls, group permutation, or descriptive-only;
- **status:** primary, reference, exploratory-secondary, exploratory-event-only, or proposed.

Programmatic access:

```python
import multisync as ms

table = ms.feature_status_table()
print(table)
```

The current Table 1 candidate is exported by the demo as both CSV and LaTeX:

```text
artifacts/demo_v1/feature_status_table.csv
artifacts/demo_v1/TABLE1_FEATURE_STATUS.tex
```

It can also be generated programmatically:

```python
ms.feature_status_latex("TABLE1_FEATURE_STATUS.tex")
```

---

## Why WCC remains the default substrate

WCC is not assumed to be the universally correct synchrony metric. It is used as the default because it is:

- interpretable;
- widely used in dyadic synchrony work;
- compatible with time-local feature extraction;
- compatible with signal-level and design-level surrogate audits;
- easy to inspect visually and export.

SyncPipe's claim is not “WCC solves synchrony.” The claim is:

> WCC is a transparent measurement substrate around which a standardized audit infrastructure can be built.

Alternative metrics such as WCLC, PLV, CRQA, mutual information, or recurrence methods may be added as optional substrates, but each requires its own null model and bias audit.

---

## Demo

Install in editable mode:

```bash
cd SyncPipe
pip install -e ".[dev]"
```

Run the complete synthetic demo:

```bash
python -m multisync demo \
  --surrogates 20 \
  --audit-surrogates 20 \
  --demo-dyads 4 \
  --no-prediction \
  -o artifacts/demo_v1
```

Outputs:

```text
artifacts/demo_v1/
├── DEMO_REPORT.md
├── TABLE1_FEATURE_STATUS.tex
├── design_control_audit.json
├── feature_status_table.csv
├── feature_table.csv
├── synchrony_existence_audit.json
└── viewer_results.json
```

---

## Minimal usage

```python
import multisync as ms

# dyad should contain aligned/preprocessed person_a/person_b columns per modality
dyad = ms.Dyad(hz=1.0, eda=df_eda, resp=df_resp)
dyad.align(target_hz=1.0).zscore()

analyzer = ms.DynamicAnalyzer(
    window_size=10,
    surrogate_n=500,
    enable_prediction=False,
)
results = analyzer.fit_transform(dyad)
results.export_viewer_json("results.json")
```

Group-level audited inference:

```python
from multisync.inference_pipeline import InferencePipeline

pipe = InferencePipeline(features_df, hz=1.0, wcc_window_sec=10.0, surrogate_n=99)
existence = pipe.run_synchrony_existence_audit(raw_signals, wcc_window_size=10)
design = pipe.run_design_control_audit(signal_pairs, wcc_window_size=10)
group = pipe.run_group_condition_inference(condition_col="condition", dyad_col="dyad_id")
```

---

## Governance and SSoT

SyncPipe uses two governance layers:

1. **Mathematical SSoT:** `multisync/feature_definitions.py` contains the implementation-level definitions of WCC-derived features. Other modules should import feature math from here rather than reimplementing it.
2. **Communication SSoT:** `multisync/feature_status.py` contains the external-facing v1 feature status table used for README, demo exports, and manuscript Table 1 drafts.

This separation is intentional. Internal mathematical invariance labels are useful for implementation and null-model selection; external readers need a simpler measurement table: source level, incremental information, applicable paradigm, recommended use, risk, and evidence status.

The current v1 method log is maintained in:

```text
docs/METHOD_LOG.md
```

Negative results, abandoned feature promotions, and null-model limitations should be logged rather than silently removed.

---

## Current validation stance

SyncPipe's strongest current claim is not that all dynamic descriptors are independently validated psychological constructs. The strongest claim is methodological:

> A synchrony result should be treated as an audited evidence chain, not as a single significant feature.

Accordingly:

- `peak_amplitude` is currently the most robust workhorse descriptor for synchrony-existence detection.
- `mean_synchrony` remains a reference comparator, not the whole construct.
- `dwell_time` and `switching_rate` are useful structure descriptors but remain sensitive to thresholding and WCC overlap.
- `onset_latency`, `rise_time`, and `recovery_time` are event-mode exploratory descriptors, not general synchrony features.
- `fraction_above_threshold` is implemented as an exploratory-secondary occupancy descriptor, but is not part of the primary FDR family in v1.
- `first_peak_time` and `inter_peak_cv` are proposed exploratory descriptors pending further validation and reporting conventions.
- Passing signal-level IAAFT does not rule out shared stimulus or co-presence; design controls are required.

---

## Relationship to existing tools

SyncPipe is complementary to tools such as multiSyncPy, rMEA, and mv-SUSY. Those tools provide valuable synchrony metrics and surrogate workflows. SyncPipe's niche is the infrastructure around measurement:

- feature status table;
- standardized evidence chain;
- design-control audit layer;
- WCC trace export;
- reproducible JSON/CSV/Markdown artifacts;
- governance logs and tests.

The goal is to make synchrony measurement easier to inspect, criticize, reproduce, and improve.

---

## License

MIT

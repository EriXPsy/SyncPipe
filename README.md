# SyncPipe
<img width="1376" height="768" alt="image" src="file:///C:/Users/%E9%99%88%E6%80%9D%E4%B8%9E/WorkBuddy/20260413150513/syncpipe-logo-mark.svg" />

> **Measurement infrastructure for dyadic, continuous low-frequency multimodal interpersonal synchrony.**  
> SyncPipe is not intended to be merely a feature-profile generator. Its v1 goal is to provide a standardized, auditable measurement procedure for dyadic synchrony: from aligned signals, to WCC traces, to interpretable descriptors, to null-model audits, to design-specific confound checks, to group-level inference.

SyncPipe is an open-source Python package for analyzing **dyadic peripheral physiological and behavioral synchrony**. It is built for psychology, social neuroscience, psychophysiology, developmental science, psychotherapy, teamwork, and related fields where researchers need more than a single mean-correlation score but also need stronger statistical guardrails than one-off scripts usually provide. The preferred v1 command/import namespace is `syncpipe`; the older `multisync` namespace remains as a compatibility alias during transition.

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
- Provide a simple feature status table via `syncpipe.feature_status_table()` and `artifacts/demo_v1/feature_status_table.csv`.
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

## Scope and modalities

SyncPipe is **multimodal in a deliberately narrow sense**. All modalities it
validates are **continuous, low-frequency time series that have already been
preprocessed into a common second-level envelope** — e.g. ECG/IBI, EDA,
respiration, motion-energy, or (in principle) neural envelopes. The package's
"modality independence" rests on the fact that every supported signal has been
flattened into the *same statistical object* (an aligned low-rate trace) before
it reaches the WCC layer.

This scope does **not** currently cover:

- high-frequency raw signals that have not been reduced to an envelope;
- discrete-event modalities such as facial action-unit sequences, turn-taking
  events, or speech onsets, which require event-based (rather than continuous
  WCC) synchrony measures — planned for v2;
- EEG/fNIRS hyperscanning in its native high-dimensional form (the *envelope*
  of such signals is in scope; the raw high-density recording is not).

If your modality cannot be expressed as a continuous low-frequency envelope
without losing the effect you care about, SyncPipe is the wrong tool for v1.
See `docs/LIMITATIONS.md` for the full scope statement and `docs/DECISION_LOG.md`
for positioning relative to related tools (SyncPy, multiSyncPy, SUSY, rMEA).

---
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/17d3eaa0-215b-418d-88aa-5e87b8373c14" />
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/6feccbfe-c42c-4807-8155-f969d4c05fe4" />

## Conceptual architecture

SyncPipe has five infrastructure layers.

| Layer | Question | Main object | Output |
|---|---|---|---|
| 1. Data/QC layer | Are the signals aligned, finite, and sampled consistently? | aligned dyadic time series | quality report / diagnostics |
| 2. Trace layer | What is the moment-to-moment synchrony substrate? | WCC trace | WCC arrays per dyad/modality/condition |
| 3. Descriptor layer | What aspects of the WCC trace are being summarized? | WCC-derived features | feature table |
| 4. Evidence/inference layer | What nuisance explanations have been ruled out? | null and design-control tests | evidence-chain report |
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
import syncpipe as sp

pipe = sp.InferencePipeline(features_df, hz=1.0, wcc_window_sec=20.0, surrogate_n=100)

result = pipe.run_audited_evidence_chain(
    raw_signals,
    wcc_window_size=20,
    design_signal_pairs=signal_pairs,
    across_stim_segments=None,  # optional; use for segmented shared-stimulus designs
)
print(result["summary"])
```

---

## Standard Operating Procedure (SOP) for reviewers and new users

This is the canonical, copy-pasteable path from raw aligned signals to an
audited synchrony claim. A fresh reviewer with a real dataset should be able
to follow it end-to-end. It mirrors `artifacts/reviewer_audit/AUDIT_REPORT.md`
§3 and is the recommended methodology section for manuscripts.

### Stage 0 — Load real data

```python
from multisync.realtest.lerique_2024 import load_lerique_dataset

records = load_lerique_dataset(
    data_root="/path/to/Lerique-47n3p",
    preprocess=True,
    drop_incomplete=True, drop_misaligned=True, drop_short_duration=True,
)
# records: List[LeriqueDyadCondition]  (dyad_label, modality, condition,
#                                        person_a / person_b as time/value frames)
```

The loader returns dataset records; the three pipelines consume them through
the bridge `multisync.pipeline_bridge.records_to_inference_inputs`.

### Stage 1 — Feature consultation (Pipeline 1, select-only, no computation)

```python
from multisync.feature_pipeline import print_feature_table, recommend_features
print(print_feature_table())            # 12 descriptors: Tier / Axis / FDR / Unit
rec = recommend_features("general")
# rec["primary"]    == FDR family == ('peak_amplitude', 'dwell_time', 'switching_rate')
# rec["reference"]  == ('mean_synchrony',)   # comparator, NOT in FDR family
# rec["supplementary"] == exploratory descriptors
```

### Stage 2 — Compute (Pipeline 2)

```python
from multisync.pipeline_bridge import records_to_inference_inputs
inputs = records_to_inference_inputs(
    records, hz=1.0, window_size=30, onset_threshold="session_pooled",
    design_condition="trials_concat",
)
inputs.features_df     # one row per (dyad, modality, condition) + all descriptors
inputs.raw_signals    # "dyad__mod__cond" -> (sig_a, sig_b)  for existence audit
inputs.design_pairs   # "dyad__mod"        -> (sig_a, sig_b)  for design controls
```

Under the hood: `ComputationPipeline.load_signals → compute_wcc → extract_features
→ to_dataframe` per record. WCC uses an O(n) cumsum backend; thresholds can be
per-modality session-pooled for cross-dyad comparability.

### Stage 3 — Audited evidence chain (Pipeline 3)

```python
from multisync.inference_pipeline import InferencePipeline
from multisync.feature_definitions import FDR_FEATURES

pipe = InferencePipeline(
    features_df=inputs.features_df, hz=1.0,
    wcc_window_sec=30.0, surrogate_n=100, seed=42,
)
chain = pipe.run_audited_evidence_chain(
    raw_signals=inputs.raw_signals,
    wcc_window_size=30,
    design_signal_pairs=inputs.design_pairs,
    condition_col="condition", dyad_col="dyad_id",
    feature_cols=list(FDR_FEATURES),
    fdr_alpha=0.05, n_permutations=10000,
)
print(pipe.summarize())

# MANDATORY for multimodal data — see rule below
by_mod = pipe.test_l2_by_modality(
    modality_col="modality", condition_col="condition", dyad_col="dyad_id",
    feature_cols=list(FDR_FEATURES), n_permutations=10000,
)
```
The chain is three steps: (1) signal-level IAAFT **synchrony-existence** audit;
(2) **design-control** audit (pseudo-pair + time-shift); (3) dyad-paired
permutation + **BH-FDR** **group-condition** inference.

---

### Mandatory reporting rule — per-modality L2 for multimodal data

> **When more than one modality is present, never report only the pooled L2.**

Pooling modalities inside one FDR family dilutes per-modality effects. This is
empirically demonstrated: in the synthetic-proxy audit the pooled L2 was 1/3
significant while EDA and RESP each showed 2/3; on **real Lerique data** EDA
showed 8/8 descriptors significant (peak_amplitude p_fdr=0.0008, dwell_time
p_fdr=0.025, switching_rate p_fdr=0.036) while RESP showed only 1/8
(bimodality_coefficient). Always run `test_l2_by_modality` and report each
modality's L2 alongside the pooled result. `reviewer_end_to_end.py` and
`realdata_l2_audit.py` produce `per_modality_l2` automatically.

---

### Publication-grade statistical power

Before any result enters a manuscript, the analysis MUST use:

| Parameter | Minimum | Default in code |
|---|---|---|
| `surrogate_n` (signal-level IAAFT / existence) | ≥ 100 | 100 (`design_controls.py`, `inference_pipeline.py`) |
| `n_permutations` (dyad-paired L2) | ≥ 10000 | 10000 (`inference_pipeline.py`) |
| `n_pseudo_per_dyad` (design-control pseudo-pair) | ≥ 10 | 10 (`design_controls.py`) |

These are now the package defaults; lower values are acceptable only for smoke
tests and demos. Raise further for very small cohorts where the pseudo-pair
null needs more draws to stabilise.

---

### Handling undefined descriptors (`dwell_time` / `switching_rate`)

`dwell_time` and `switching_rate` are **conditional** descriptors: they are
defined only when the WCC trace contains at least one sustained above-threshold
run. Under weak coupling, rest conditions, or very short traces they return
`NaN` **by construction** — this is the null of "sustained synchrony episode",
not a bug. The L2 layer therefore (a) tests only dyads where the feature is
finite in *both* conditions, (b) reports definedness rates (`defined_a` /
`defined_b`), and (c) emits a `[WARN]` + `p_definedness` when definedness differs
across conditions (potential survivor bias). **Always report definedness
alongside every dwell/switching result; never treat the NaN as
missing-at-random.** Real-data definedness rates: Lerique dwell 59.7% defined,
Han 98.4%, Gordon only 24.3% (its WCC traces are 18–22 points long — too short
for sustained runs). This gradient is exactly what the construct predicts.

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
import syncpipe as sp

table = sp.feature_status_table()
print(table)
```

The current Table 1 candidate is exported by the demo as both CSV and LaTeX:

```text
artifacts/demo_v1/feature_status_table.csv
artifacts/demo_v1/TABLE1_FEATURE_STATUS.tex
```

It can also be generated programmatically:

```python
sp.feature_status_latex("TABLE1_FEATURE_STATUS.tex")
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
syncpipe demo \
  --surrogates 20 \
  --audit-surrogates 99 \
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
import syncpipe as sp

# dyad should contain aligned/preprocessed person_a/person_b columns per modality
dyad = sp.Dyad(hz=1.0, eda=df_eda, resp=df_resp)
dyad.align(target_hz=1.0).zscore()

analyzer = sp.DynamicAnalyzer(
    window_size=10,
    surrogate_n=500,
    enable_prediction=False,
)
results = analyzer.fit_transform(dyad)
results.export_viewer_json("results.json")
```

Group-level audited inference:

```python
import syncpipe as sp

pipe = sp.InferencePipeline(features_df, hz=1.0, wcc_window_sec=10.0, surrogate_n=100)
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

---
name: syncpipe
description: >
  Auditable dyadic synchrony measurement with SyncPipe. Use when a user has
  two-party time series (physiological, behavioral, or neural) and wants
  rigorous synchrony quantification: WCC descriptors, an existence audit
  against randomized independent-signal nulls, design-control audits
  (pseudo-pair, time-shift, shared stimulus), and dyad-paired condition
  inference with pre-registered FDR families. Also use when the user asks to
  run, interpret, troubleshoot, or report a SyncPipe analysis.
version: 1.0.1
license: MIT
---

# SyncPipe — agent skill

Human documentation: [`docs/USER_MANUAL.md`](USER_MANUAL.md) · generated API
mirror: [`docs/API_REFERENCE.md`](API_REFERENCE.md).

## What this does

Turn aligned two-person time series into **auditable descriptors** and run the
**three-step audited evidence chain** (existence audit → design-control audit →
group inference) on a sliding-window cross-correlation (WCC) substrate.
Measurement infrastructure for same-modality dyadic synchrony across modality
families (EDA, ECG-derived IBI, respiration, motion energy — the core is
modality-agnostic).

## When to use

- Dyadic / two-party continuous time series that must be quantified rigorously
  rather than with a single ad-hoc score.
- Testing whether observed synchrony exceeds chance, ruling out confounds
  (shared stimulus, misalignment, partner identity), or comparing conditions
  with multiplicity control.
- Interpreting or troubleshooting an existing SyncPipe run.

## When NOT to use (guardrails)

- Not for triads or groups (dyadic only in v1.x).
- Not for raw ECG/EEG/fNIRS waveforms or event-style data — feed a continuous,
  preprocessed, low-frequency trace (for raw ECG use the `[ecg]` extra's
  IBI path first).
- Never report exploratory descriptors (`bimodality_coefficient`,
  `synchrony_entropy`, timing descriptors) as confirmatory — they are outside
  the pre-registered FDR families; the report itself folds them under a
  labelled `<details>` section. Preserve that folding in any summary you write.
- A passed existence audit is **necessary but not sufficient** for coupling.
  Never claim partner-specific coupling, causality, relationship quality, or
  clinical meaning from SyncPipe output.

## Setup

```bash
python -m pip install -e .            # core
python -m pip install -e ".[ecg]"     # raw-ECG -> IBI preprocessing (neurokit2)
python -m pip install -e ".[rqa]"     # RQA convergence channel
python -m pip install -e ".[xls]"     # legacy .xls OSF workbooks
syncpipe --version                    # sanity check
```

## Agent workflow for a user request

1. **Clarify the data shape**: two people? same signal type? aligned,
   constant-rate, continuous? If any answer is no, stop and say so — do not
   force the tool onto event data or group designs.
2. **Choose the path**:
   - quick single-pair description → `syncpipe describe ...` (exploratory only);
   - study with multiple dyads / condition contrast → `syncpipe analyze` with
     manifest + config (the confirmatory path);
   - a hands-on tour → `syncpipe external-kit -o example`, then
     `syncpipe demo -o demo_results`.
3. **Prepare inputs** exactly per USER_MANUAL §5 (signal CSVs, manifest,
   settings, processing record). Manifest signal type and unit must match the
   processing record.
4. **Run** and read outputs in this order: `REPORT.md` →
   `evidence_graph.json` → `qc_report.json` → `exclusion_report.csv` →
   `features.csv`. Surface the strongest-supported-claim sentence verbatim;
   keep confirmatory results ahead of exploratory ones in anything you write.
5. **Report honestly**: include definedness rates, exclusion reasons, and the
   "still not ruled out" line from `REPORT.md`. Use the plain-language phrasing
   patterns in USER_MANUAL §8.

## CLI entry points

```bash
syncpipe analyze -m manifest.csv -c config.toml -o results/   # confirmatory
syncpipe describe -i dyad.csv -n eda --hz 1 --window-size 10 -o out.json
syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo
syncpipe external-kit -o example
```

## Python API (canonical study path)

```python
from syncpipe.pipeline_bridge import records_to_inference_inputs
from syncpipe.inference_pipeline import InferencePipeline

inputs = records_to_inference_inputs(
    records, hz=1.0, window_size=20, onset_threshold="session_pooled",
    design_condition="task",
)
pipe = InferencePipeline(
    inputs.features_df, hz=1.0, surrogate_n=100, seed=42, n_workers=4,
)
chain = pipe.run_audited_evidence_chain(
    raw_signals=inputs.raw_signals, wcc_window_size=20,
    design_signal_pairs=inputs.design_pairs,
    condition_col="condition", dyad_col="dyad_id",
)
chain["legacy_fields"]["permitted_claim"]   # strongest supported claim
```

Bridge rules (fail-loud): all records share one `hz`; observation labels are
`<dyad>__<modality>__<condition>` and a dyad id containing `__` is rejected;
duplicate `(dyad, modality, condition)` keys are rejected. Modality tokens may
embed `__` (cross-modal pairing convention, e.g. `neural__behavior`).

Lower-level steps (rarely needed directly):
`syncpipe.synchrony_existence_audit(sig_a, sig_b, hz, window_size)` →
`syncpipe.design_control_audit(...)` → `InferencePipeline.run_group_condition_inference`.

## Key contracts

- `PRIMARY_FDR_FAMILY = ('peak_amplitude',)` — the pre-registered primary
  endpoint (BH denominator m = 1).
- `SECONDARY_FDR_FAMILY = ('dwell_time', 'switching_rate')` — corrected within
  its own family (m = 2); `mean_synchrony` is a reference comparator, never
  confirmatory.
- The existence gate reports `primary_pass` plus an explicit `gate_status`
  (`pass` / `fail` / `not_evaluable`). `not_evaluable` = the signal type is not
  among the pre-registered primary modalities — a scope statement, **not** a
  negative finding; the typed evidence chain maps it to INCONCLUSIVE.
- Onset thresholds default to per-modality session-pooled IAAFT cut-offs
  (`session_pooled`), hard-capped at 0.9; the bare 0.5 constant is a fallback /
  sensitivity value only.
- Statistical defaults are frozen by `tests/test_v1_defaults_guard.py` and
  calibrated by `tests/test_h0_calibration_endpoints.py` (each endpoint must
  reject at ~alpha under its own H0).

## Interpreting results (say this, not more)

- Report each check's verdict separately (supported / not supported /
  not enough information / could not test). "Not enough information" is not a
  negative finding.
- A condition-comparison p-value refers to the selected trace summary only —
  never to "synchrony" as a general construct.
- If the existence gate is `not_evaluable` for the user's modality, explain the
  pre-registration logic instead of implying failure.

## Pointers

- Release history: `CHANGELOG.md` · decisions: `docs/METHOD_LOG.md` and
  `docs/DECISION_LOG.md` · script map: `docs/SCRIPT_MAP.md`.
- Descriptor status table: `docs/FEATURE_TABLE.{csv,md}` — check a descriptor's
  row before reporting it.
- Contributing / verification rules: `CONTRIBUTING.md`.

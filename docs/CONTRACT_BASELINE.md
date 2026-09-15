# SyncPipe Contract Baseline

Status: v1.2.1 baseline

This document defines the externally observable contracts protected during the first refactoring phase. Structural changes must preserve these contracts unless a new version explicitly changes them.

## 1. Supported scientific path

```text
manifest + analysis configuration
→ preparation and QC
→ aligned dyadic observations
→ WCC-derived measurement
→ existence and design-control audits
→ planned condition inference
→ evidence and claimability
→ report bundle
```

The supported v1 scope is two aligned, preprocessed, continuous signals from the same signal type, zero-lag sliding-window correlation, and a pre-specified two-condition comparison.

## 2. Public entry points

### Recommended

- `syncpipe.analyze(manifest, settings, output="results")`
- `syncpipe.run_canonical(manifest, settings, output="results")`
- `syncpipe.make_example(output="syncpipe_example")`

### Stable data contracts

- `AnalysisSpec`
- `ManifestRecord`
- `CanonicalResult`
- `EvidenceChain` and evidence model classes
- `PreparedObservation`, `PreparedCohort`, and `PreparationExclusion`

### Compatibility and advanced APIs

The following remain import-compatible during v1 restructuring:

- `Dyad`, `DynamicAnalyzer`, `AnalysisResults`
- `ComputationPipeline`, `BatchComputationPipeline`
- `compute_pair_pipeline`, `quick_compute`, `batch_compute`
- `InferencePipeline`
- QC, migration, threshold, and audit helpers

Internal helpers and underscore-prefixed functions are not public contracts.

## 3. Per-pair computation contract

The computation layer preserves:

- equal-length one-dimensional signal input;
- finite sampling rate and valid window size;
- NaN as governed missing data; Inf rejected;
- rect and tapered window behavior;
- cumsum and stride implementation behavior;
- WCC and WCLR backend behavior;
- discontinuity-mask seam handling;
- gap-policy behavior;
- feature names, values, definedness, metadata, and row ordering.

Both supported input modes remain valid:

```text
signals → coupling trace → features
precomputed coupling trace → features
```

## 4. Batch contract

Batch computation preserves:

- modality-specific pooled thresholds;
- threshold mode, value, fallback, and reason metadata;
- modality sentinel behavior for unspecified modalities;
- dyad labels, row order, feature columns, and empty-input behavior.

## 5. Canonical result contract

`CanonicalResult` continues to expose:

- `output_dir`
- `manifest`
- `config`
- `chain`
- `qc`
- `exclusions`
- `features_df`
- `wcc_traces`
- `environment`
- `claimability`
- `report_paths`

The standard bundle continues to contain:

```text
manifest_resolved.json
config_resolved.toml
environment.json
qc_report.json
exclusion_report.csv
features.csv
existence_audit.json
design_control_audit.json
group_inference.json
evidence_graph.json
claimability.json
REPORT.md
```

## 6. Statistical and evidence contracts

Refactoring must not change without an explicit methods decision:

- estimands;
- null hypotheses;
- randomization units;
- seed derivation;
- modality grouping;
- FDR families and denominators;
- reference-feature exclusion;
- evidence status and claimability propagation.

A passed existence or design audit is not a causal claim and does not establish interpersonal influence or psychological mechanism.

## 7. Error and exclusion contract

The canonical path preserves the distinction between:

- invalid scientific/configuration contracts that stop analysis;
- expected record-level loading failures recorded as exclusions;
- unexpected runtime failures that propagate.

Exception types, exclusion categories, and report representations are regression-protected.

## 8. Verification gates

Every structural change must pass, as applicable:

```text
pytest -m "not slow"
pytest tests/integration
pytest tests/contracts
pytest tests/validation
```

The canonical path additionally requires bundle completeness and CLI/API parity. Numerical changes require explicit scientific review rather than being hidden inside a structural refactor.

## 9. Versioning rule

Any intentional change to an observable contract requires:

1. a methods or release decision;
2. updated tests;
3. updated version/schema metadata when applicable;
4. updated user-facing documentation;
5. an explicit changelog entry.

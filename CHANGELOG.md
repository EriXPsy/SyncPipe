# Changelog

## 2.0.0 — 2026-08-19

### Changed

- Requires canonical endpoint and modality declarations.
- Requires signal type, unit, and preprocessing provenance in manifests.
- Uses immutable analysis, preparation, and evidence contracts.
- Uses shared finite/discontinuity geometry across computation and null stages.
- Adds segment-wise IAAFT and pooled-threshold handling.
- Adds typed E0–E5 claim propagation and `evidence_graph.json`.
- Reports surrogate and permutation precision plus L2 median uncertainty.
- Separates canonical orchestration, contracts, preparation, evidence, and export.

### Added

- Adds peak-duration and adversarial discriminant-validity validation.
- Adds frozen validation acceptance reports.
- Adds packaged JSON Schemas.
- Adds explicit v1-to-v2 manifest/config migration with hashed migration report.

### Breaking

- Legacy v1 canonical manifests/configs no longer run without migration.
- `SyncPipeConfig` is now an immutable compatibility alias for `AnalysisSpec`.
- Canonical exclusions and evidence stages use typed contracts.

## 1.0.0 — 2026-07-08

- Establishes the original WCC descriptor and audited-inference pipeline.

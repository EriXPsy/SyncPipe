# Changelog

## 1.0.1 — 2026-09-08

### Fixed

- **BUG-2 (NaN stride backend)**: `_sliding_window_wcc_stride` summed raw
  windows containing NaN, so `0 * NaN = NaN` poisoned every overlapping
  window and any NaN input returned an all-NaN WCC trace — silently voiding
  the documented pairwise-deletion + `min_valid_ratio` contract.  Invalid
  positions are now zeroed before windowing (algebraically exact; weight 0
  there).  Gappy signals now yield finite, near-null values on the
  design-control paths instead of a dead trace.
- **BUG-3 (shared surrogate seed)**: the existence audit passed one master
  seed to every pair, so each pair consumed an identical RNG stream —
  surrogate draws correlated across dyads (measured null-peak r ≈ +0.33),
  inflating the second-order group-null spread ~1.46×, a systematic
  conservative bias.  Per-pair seeds are now derived from the master seed
  and a stable label hash; runs stay bit-reproducible.
- **BUG-4 (L1 default null was self-invariant)**: `test_l1_structure`
  defaulted to `state_shuffle`, under which both reported L1 statistics
  (dwell_time, switching_rate) are invariant by construction (p ≡ 1.0 on
  every input).  The default is restored to the protocol null (WCC-level
  IAAFT, V1_PROTOCOL §10); `state_shuffle` remains selectable for
  order-sensitive diagnostics and now emits a degeneracy warning.

### Added

- Cross-validation regression tests for the WCC core and IAAFT surrogate
  against independent reference implementations.
- Protocol-default guard tests (FDR families, existence gate, L1/L2 nulls).
- Optional `[ecg]` (neurokit2) and `[rqa]` (PyRQA) extras; the WCC core
  installs with numpy/scipy/pandas/scikit-learn only.
- `scripts/run_crqa_convergence.py` (convergent validity vs PyRQA) and
  `scripts/rerun_l1_pvalues.py` (BUG-4 remediation for stored WCC traces).

### Migration notes

- Existence-audit and L1 p-values computed before 2026-09-08 must be
  regenerated (MC realization changed by BUG-3; L1 null corrected by
  BUG-4).  Point estimates (WCC, descriptors, feature tables) are
  unaffected — they are RNG-free.

## 1.0.0 — 2026-08-19

### Changed

- Rewrites the first-run documentation, CLI help, and main report in plain language.
- Adds `main_measure` and `main_modalities` as preferred settings; older names remain accepted.
- Adds `syncpipe.analyze()` and `syncpipe.make_example()` as simple starting points.
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

# Changelog

## 1.2.0 — 2026-09-13

### Changed (audit round-6: adversarial-review fixes)

- **Peak smoothing is now edge-exact and NaN-aware** (`smoothed_wcc`
  masked-convolution revision). Interior values are unchanged
  (bit-identical); edge peaks are no longer attenuated by zero-padding and
  a NaN seam no longer removes neighbouring positions from the peak
  search. `peak_amplitude` values on traces whose dominant episode touches
  an edge, or that carry NaN seams adjacent to the peak, may differ from
  v1.1.x output.
- `smoothed_wcc` accepts `window_sec` + `hz` for sampling-rate-constant
  smoothing bandwidth (recommended for cross-dataset work); the v1 default
  (`PEAK_SMOOTHING_WINDOW=3` samples) is retained and documented as
  hz-dependent.
- L0 existence audit now also tests `synchrony_entropy` (was declared but
  never tested). Result dicts gain `p_/null_/obs_/n_valid_synchrony_entropy`
  keys; `per_feature_significant` gains the `synchrony_entropy` entry.
- L1 test (`wcc_surrogate_test` / `test_l1_structure`) forwards
  `gap_policy`; observed dwell/switching honour the caller's convention
  (default unchanged = `merge_valid` semantics).
- `between_condition_fdr` gains `max_feature_aggregation` ({"mean","max"});
  mean-aggregating duplicate extremum-feature rows now warns about the
  estimand.
- `run_design_control_audit` warns when structure features use the fixed
  0.5 fallback threshold instead of the canonical pooled surrogate
  threshold.
- Design-control internals: a discontinuity mask shorter than its signal
  now raises instead of being silently dropped (public API behaviour
  unchanged for well-formed inputs).
- Terminology: "pre-registered" endpoint/modality language downgraded to
  "frozen a priori in the development log"; cascade narratives rephrased
  from per-dyad certainty claims to cohort-level audit descriptions.

### Added

- `tests/test_audit_round6_fixes.py`: 24 regression tests covering all of
  the above.

## 1.2.1 — 2026-09-14

### Changed (audit round-6b: minor adversarial-review fixes)

- Second-order existence gate consumes all valid surrogate draws
  (full-width NaN-padded aggregation instead of shortest-length chop).
- Exploratory AUC diagnostics impute inside the CV pipeline
  (`_FoldMedianImputer`); no more whole-data median leak.
- `Dyad(dyad_id=<non-string>)` coerces with a warning (was silently
  replaced by "dyad_01").
- `_binarize_with_hysteresis` vectorised; bit-identical to the previous
  loop (random parity test), ~2x faster on long traces.
- `SIGNFLIP_MAX_DRAWS` constant de-duplicated across design controls and
  the evidence builder.
- L1 WCC-level results record the effective `wcc_window_sec` and a
  heuristic-fallback flag.
- Documentation: cumsum tolerance, score_view pooling scope, hz default
  provenance, BC threshold caveat, five debug-logged exception branches,
  Axis-D comment drift, lag sign convention.

### Added

- `tests/test_audit_minor_fixes.py`: 10 regression tests (suite now 596
  collected / 525 fast / 71 slow).

## 1.1.0 — 2026-09-10

### Added

- `compute_synchrony_entropy(..., fixed_range=True)`: cross-dyad-comparable
  entropy variant histogrammed over the theoretical `[-1, 1]` range. The
  default adaptive-range behavior is unchanged; its cross-dyad caveat is
  now documented in the docstring and `docs/LIMITATIONS.md` §5.
- Community infrastructure: `CODE_OF_CONDUCT.md` (Contributor Covenant
  2.1), issue templates (bug / feature / usage question), a PR template
  encoding the two-layer test rule and governance checks, `[project.urls]`
  + classifiers + keywords in `pyproject.toml`, and a "How SyncPipe relates
  to other tools" positioning section in the README.

### Changed

- Design-control sign-flip p-values now salt each feature's seed with a
  stable CRC32 of the feature label. Marginal p-values remain valid and
  reproducible; on audits with >12 dyads, p-values differ from v1.0.x
  output (≤12 dyads use exhaustive enumeration and are unchanged).
- `CITATION.cff` re-stamped (version 1.1.0, date-released 2026-09-10);
  reference-paper PDFs are no longer redistributed (copyright) — source
  links live in `docs/REALDATA_PAPER_ALIGNMENT.md`.
- Methodology narrative: `V1_PROTOCOL.md` §3 peak-definition scope note and
  two-tailed equivalence note; `LIMITATIONS.md` §7 (null/functional
  compatibility, conservative engineering choices) and §8 (construct
  validity roadmap).

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

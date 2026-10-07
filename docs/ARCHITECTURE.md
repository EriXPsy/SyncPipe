# SyncPipe Architecture Convergence

## Direction

SyncPipe is converging from feature-oriented modules toward a checked analysis
pipeline. New descriptors are frozen while settings, prepared data, checks,
reports, and migration are consolidated. Internal typed objects preserve rigor;
public language follows [`PLAIN_LANGUAGE.md`](PLAIN_LANGUAGE.md) so users can
work from ordinary questions rather than architecture terms.

## Milestone A — unified analysis specification (implemented)

The single source of scientific analysis semantics is
`syncpipe.contracts.AnalysisSpec`, an immutable dataclass composed with:

- `EndpointSpec` — estimand, unit, multiplicity family, duration policy and
  permitted/forbidden claims;
- `NullSpec` — null level, sampling unit, tail, preserved/destroyed structure
  and missingness policy;
- `ModalitySpec` — study-declared primary/comparator role.

`SyncPipeConfig` is a compatibility alias, not a second implementation. TOML
parsing delegates to `analysis_spec_from_mapping`; unknown keys fail. The
canonical runner resolves the endpoint contract once and passes its endpoint to
the existence gate. AnalysisSpec is frozen, so orchestration derives effective
values such as `design_condition` without mutating user configuration.

The v1 registry currently contains only `peak_amplitude`, because adding an
endpoint requires a matching validated null contract rather than only feature
math.

## Milestone B — prepared observations (implemented core)

Immutable `PreparedObservation`, `SignalGeometry`, and `PreparedCohort` now own
the joint finite mask, source discontinuity mask, combined analysis mask,
contiguous segments, WCC opportunity and threshold eligibility. Pipeline bridge
construction hard-gates computation with the shared analysis mask; segment-wise
IAAFT delegates to the same geometry resolver; canonical design controls receive
the prepared design-condition masks; pooled surrogate thresholds generate within
the same eligible segments. Preparation diagnostics are exported in QC.

`PreparationExclusion` now carries typed loading/preparation/QC exclusion codes,
stages, details and claim effects inside `PreparedCohort`; canonical exclusion
CSV/Markdown are rendered from those objects. Milestone B is complete for the
v1 canonical path.

## Milestone C — typed evidence graph (implemented with compatibility layer)

`EvidenceStageResult`, `EvidenceStatus`, `EvidenceProfile`, `EvidenceChain`, and
`ClaimDecision` represent E0–E5 plus L2. The profile preserves evidence as a
vector; conservative claim propagation separately stops at the first
unsupported/inconclusive requirement. E2/E3 generic controls use Holm adjustment,
insufficient p-value resolution produces `inconclusive`, and E5 remains
inconclusive without a reciprocity-breaking design. Canonical output includes
`evidence_graph.json`; legacy `stage_status` and `claim_ceiling` are derived from
the graph, explicitly deprecated, and scheduled for removal in 3.0.

## Milestone D — orchestration/export split (implemented)

`canonical_runner.py` now contains config parsing, `CanonicalResult`, and the
parse → prepare → compute → audit → infer → export orchestration only. Manifest
and provenance contracts live in `syncpipe.contracts.manifest`; signal loading
lives in `syncpipe.preparation.loading`; claimability derivation lives in
`syncpipe.evidence.claimability`; strict JSON/runtime capture, Markdown
rendering, and bundle writing live in `syncpipe.export`. Compatibility re-exports
preserve the existing public API and CLI/API byte parity.

## Milestone E — versioning and migration (implemented)

Package releases are versioned as 1.2.1; the breaking canonical contract ships
as config/manifest schema v2.0.0 (schema versions track the data-contract
shape, not the release). Viewer analysis JSON remains independently versioned
at 0.3.0, preprocessing provenance at 1.0.0, and typed evidence at 1.1.0. `syncpipe migrate` and the
programmatic migration API convert legacy manifest/config inputs only when the
user explicitly supplies signal identity, unit, preprocessing provenance,
endpoint, and primary modality assumptions. Every migration writes source and
target hashes plus warnings in `MIGRATION_REPORT.json`.

The external validation infrastructure is not external validation: the milestone remains scientifically incomplete until an unaffiliated researcher publishes or archives an own-data run and methodological critique under `docs/EXTERNAL_VALIDATION.md`.

## Strategic position: v1 measurement infrastructure

SyncPipe v1 is intentionally a **narrow, deep, auditable measurement infrastructure** for aligned dyadic continuous signals. It is not a universal synchrony toolbox and does not claim to measure the complete interpersonal synchrony construct.

The v1 contribution is workflow standardisation: a declared input contract, shared preparation geometry, one validated primary estimator (zero-lag windowed cross-correlation, with WCLR retained as a separate backend), explicit null hypotheses, design controls, evidence states, claim ceilings, and reproducible bundles. WCC is an estimator, not the construct itself.

The measurement object is therefore represented as:

```text
construct → estimand → estimator → descriptor → null → admissible claim
```

The v1 primary path freezes the estimator and endpoint rather than exposing an unconstrained algorithm menu. This limits implementation and validation debt while making cross-pipeline comparisons defensible.

## Evidence architecture

L0, L1, and L2 are parallel evidence components, not a mechanical filter that removes observations. L0 audits whether the observed association exceeds an independent signal-level null; L1 audits temporal structure in the derived trace; L2 tests a pre-specified within-dyad condition contrast. L0/L1 results must not silently select the sample used by L2.

The public result is an evidence profile, not a single synchrony confidence score:

```text
existence status + trace-structure status + condition-contrast status
→ permitted claim + prohibited interpretation
```

Undefined, untestable, and unsupported states remain distinct from non-significance.

## Version boundary and roadmap

- **v1.x:** WCC/WCLR computation, aligned dyadic preparation, L0/L1/L2 evidence governance, design controls, reproducible export, calibration and external validation infrastructure. No confirmatory multi-estimator claim.
- **v2.x:** only after simulation and independent validation: a measurement-estimator contract, one additional estimator family (initially phase or wavelet coherence), estimator-specific null compatibility, and sensitivity reporting. New estimators remain descriptive until their null and calibration contracts are complete.
- **v3.x:** optional multiverse and additional nonlinear/warping estimators, structured observation identities, incremental caching, and broader modality adapters. These are not v1 commitments.

The governing rule is that an estimator enters the supported scientific path only when its estimand, preprocessing assumptions, definedness rules, null model, calibration evidence, performance envelope, and interpretation ceiling are documented and tested together.

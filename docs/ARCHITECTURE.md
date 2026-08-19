# SyncPipe Architecture Convergence

## Direction

SyncPipe is converging from feature-oriented modules toward a typed evidence
pipeline. New descriptors are frozen while contracts, prepared observations,
evidence stages, exports, and schema migration are consolidated.

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

`EvidenceStageResult`, `EvidenceStatus`, `EvidenceChain`, and `ClaimDecision`
now represent E0–E5 plus L2. Claim propagation is cumulative and stops at the
first unsupported/inconclusive lower stage: E3 cannot justify an
alignment-specific headline when E2 partner specificity failed, and E5 remains
inconclusive without a reciprocity-breaking design. Canonical output includes
`evidence_graph.json`; legacy `stage_status` and `claim_ceiling` keys are derived
from the graph during migration rather than authored independently.

## Milestone D — orchestration/export split (implemented)

`canonical_runner.py` now contains config parsing, `CanonicalResult`, and the
parse → prepare → compute → audit → infer → export orchestration only. Manifest
and provenance contracts live in `syncpipe.contracts.manifest`; signal loading
lives in `syncpipe.preparation.loading`; claimability derivation lives in
`syncpipe.evidence.claimability`; strict JSON/runtime capture, Markdown
rendering, and bundle writing live in `syncpipe.export`. Compatibility re-exports
preserve the existing public API and CLI/API byte parity.

## Milestone E — versioning and migration

Bump schema/package versions intentionally, provide migration from legacy
manifests/configs, and prevent scientific contract changes under an unchanged
release identifier.

## Milestone F — external validation

Freeze the architecture and obtain independent installation, data preparation,
execution and methodological criticism before expanding substrates.

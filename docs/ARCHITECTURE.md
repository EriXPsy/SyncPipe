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

## Milestone C — typed evidence graph

Replace stage dictionaries and `completed/not_run` strings with typed stage
results (`supported`, `not_supported`, `inconclusive`, `not_applicable`,
`invalid`) and explicit claim propagation from E0 through E5.

## Milestone D — orchestration/export split

Reduce `canonical_runner.py` to parse → prepare → compute → audit → infer →
export. Move contracts, preparation, claimability, serialization and report
rendering into dedicated packages.

## Milestone E — versioning and migration

Bump schema/package versions intentionally, provide migration from legacy
manifests/configs, and prevent scientific contract changes under an unchanged
release identifier.

## Milestone F — external validation

Freeze the architecture and obtain independent installation, data preparation,
execution and methodological criticism before expanding substrates.

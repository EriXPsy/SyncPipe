# External Validation Protocol

## What this can and cannot establish

An external-kit run can establish that an unaffiliated user can install the
software, understand the contracts, execute the canonical path, inspect the
evidence graph, and identify claim limits. It cannot establish construct
validity from the bundled synthetic data.

## Phase 1 — blinded usability run

The reviewer creates a kit with `syncpipe external-kit`, follows `RUNBOOK.md`,
and completes the feedback/report templates before contacting the maintainer.
No numerical result is designated as the expected answer. Structural output is
checked with `syncpipe external-check`.

## Phase 2 — independent own-data run

The reviewer prepares a dataset not used to develop SyncPipe and declares:

- study design and reciprocity manipulation;
- signal identity, unit, and preprocessing provenance;
- endpoint and primary modalities before inspecting results;
- expected E0–E5 level;
- exclusion rules and deviations.

They submit the complete canonical bundle, environment, migration report if
applicable, and signed methodological disagreements. A failed or conflicting
result remains part of the validation record.

## Independence declaration

The report must state prior collaboration, access to unpublished expected
results, maintainer assistance, and whether interpretation was formed before
consulting project explanations.

## Acceptance

External validation is not reduced to “the command ran.” Minimum evidence is:

1. fresh-environment installation;
2. complete result bundle and structural audit;
3. independent interpretation of permitted and forbidden claims;
4. disclosure of all maintainer assistance;
5. own-data replication or principled disagreement;
6. public report or archived immutable artifact.

SyncPipe must not label the project externally validated until at least one
unaffiliated own-data report satisfies these conditions.

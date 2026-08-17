# Archive — pre-v1.0 research history (NOT part of the v1.0 tree)

This directory holds scripts and data artifacts from the v0.x development
history that are **not** part of the clean SyncPipe v1.0 package. They are kept
for provenance only:

- `scripts/` — one-off trunk analyses superseded by other scripts (GT
  aggregation, threshold sweeps, figure generation).
- `experimental_scripts/` — superseded validation runners, one-off diagnostics,
  exploration scripts, and their generated data files.

None of these are imported by the package, the test suite, or the v1 trunk
`scripts/`. They may reference now-removed paths or moved modules, and are not
maintained. When rebuilding the clean release repository, exclude this
directory.

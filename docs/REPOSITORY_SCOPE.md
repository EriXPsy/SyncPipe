# Repository Scope

Status: v1.2.1

This repository separates supported software from research materials and generated output.

## Supported release surface

The supported release surface is limited to:

- `syncpipe/`: installable package and public APIs. `pairing.py` is the single source of truth for dyad pairing policy and pair iteration; feature extraction and WCC caching must reuse it;
- `tests/`: automated unit, integration, contract, and scientific validation tests;
- `docs/`: current user, methods, API, limitations, and validation documentation;
- `scripts/`: maintained reproducibility and release-support commands explicitly mapped in `docs/SCRIPT_MAP.md`;
- `.github/`, `pyproject.toml`, `README.md`, `LICENSE`, and `CITATION.cff`.

## Experimental and historical material

- `experimental/` contains exploratory or not-yet-supported analyses. It is not part of the supported v1 scientific path.
- `archive/` contains superseded historical scripts and outputs. It is retained for provenance, not as an active API.
- `artifacts/` contains generated research outputs and snapshots. An artifact is not a software contract unless a current test or document explicitly identifies it as a reproducibility fixture.
- Root-level `_tmp_*`, `_logs/`, caches, shortcuts, and generated outputs are local working material and must not be treated as release content.

## Rules for new files

1. New user-facing functionality belongs under `syncpipe/` and requires a contract test.
2. New reproducibility commands belong under `scripts/` only when they have a documented purpose and maintained entry point.
3. Exploratory, diagnostic, falsified, or superseded work belongs under `experimental/` or `archive/`.
4. Generated outputs belong outside the source tree or under an explicitly named artifact fixture directory.
5. Moving or deleting existing research material requires an explicit provenance decision; release cleanup must not silently destroy it.

## Release review

Before a release, review the tracked file list and verify that each non-package file is either a supported release surface item, a documented reproducibility fixture, or clearly classified as experimental/archive material. Run the repository-scope contract test together with the normal contract and integration suites.

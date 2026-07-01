# Experimental / V2.0 Modules

These modules and scripts were moved here on 2026-06-13 as part of the v1.0 BRM submission cleanup.
They are NOT part of the v1.0 SyncPipe public API and are NOT required to reproduce BRM paper results.

## Modules (`experimental/multisync/`)

| File | Status | Notes |
|------|--------|-------|
| `segmentation.py` | EXPERIMENTAL D4 | Gaussian HMM / HSMM-lite state segmenter — not integrated into main pipeline |
| `transition_detection.py` | Unintegrated | Treur-style transition detection — not connected to core API |
| `morphology.py` | Optional off-ramp | Epoch morphology descriptor — v0.1 (2026-06-08), not in public API |
| `arima_wcc.py` | New (2026-06-07) | ARIMA prewhitening for ISC confound control — never integrated |
| `metrics.py` | Non-default | Alternative synchrony metrics (WCLC, PLV, CRQA, MI) — WCC is default |
| `realtest/templeton_2022.py` | Unused dataset | Templeton 2022 loader — no data in artifacts, not used in BRM paper |

## Scripts (`experimental/scripts/`)

All scripts except the 6 BRM-paper core scripts:
- `run_lerique_pilot.py` (kept in `scripts/`)
- `run_lerique_incremental_auc.py` (kept)
- `run_lerique_shuffle.py` (kept)
- `run_kuramoto_l23_taxonomy.py` (kept)
- `run_gordon_case_study.py` (kept)
- `surrogate_controls.py` (kept)

The archived scripts include: validation runners (level1/2/3), GT-series, Kuramoto benchmarks, Treur validations, plotting scripts, exploration scripts, and data migration scripts. They may be useful for v2.0 development but are not needed for BRM v1.0.

## GitHub Note

Before pushing to GitHub for BRM submission, review these files and decide which to:
- Keep in repo (with a note they are experimental)
- Move to a separate `v2-dev` branch
- Delete entirely

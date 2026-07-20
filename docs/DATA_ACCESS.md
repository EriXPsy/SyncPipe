# Data Access & Reproducibility — Lerique et al. (2024) Reproduction

> Rebuttal template. This document explains how an independent reviewer can
> obtain the protected raw data and reproduce our analyses. **Placeholders
> marked `TODO` must be filled with the real values before submission —
> nothing here is fabricated.**

## 1. Source dataset

- **Paper:** Lerique et al. (2024) — *TODO: full citation / DOI*.
- **Repository:** Open Science Framework (OSF) component.
  - **Component URL:** `TODO: insert the OSF component URL for "Lerique-47n3p"`
  - **Access:** `TODO: public / embargoed / upon-request — state which`
  - **Version / snapshot:** `TODO: OSF version tag or snapshot date, if any`

## 2. License & terms of use

- **Data license:** `TODO: insert the license or access terms stated on OSF`
- **Derived-data license:** `TODO: license for any derived/processed tables we redistribute`
- **Attribution:** `TODO: required citation / acknowledgment text`

> We do **not** redistribute the raw protected signals in this repository.
> All raw traces remain on the OSF component above; this repo ships only
> (a) the analysis code and (b) derived tables produced from that data
> under the stated license.

## 3. Local mirror

The reproduction script expects a local mirror of the OSF component:

```text
<LOCAL_OSF_MIRROR>/
    ECG/
    EDA/
    RESP/
```

Point the reproduction script at it:

```bash
python scripts/reproduce_lerique_paper.py --pub \
    --data-root <LOCAL_OSF_MIRROR>
```

## 4. Pipeline (canonical, defensible)

Both `--pub` (real data) and `--fast` (synthetic proxy) run the same
**canonical three-pipeline** — the defensible v1 public workflow:

```text
records
  -> multisync.pipeline_bridge.records_to_inference_inputs
  -> multisync.inference_pipeline.InferencePipeline.run_audited_evidence_chain
       (synchrony-existence audit
        -> design controls
        -> group inference / BH-FDR)
```

- Uses a **single fixed/pooled onset threshold** (not the legacy per-dyad
  `DynamicAnalyzer` path). See `scripts/run_lerique_pilot.py` deprecation
  notice.
- Derived tables are written to `artifacts/paper_lerique/`.
- The manifest of data-source placeholders lives at
  `artifacts/paper_lerique/MANIFEST.json` (skeleton — no fabricated values).

## 5. Synthetic smoke run (no data dependency)

To verify the pipeline wiring without the protected dataset:

```bash
python scripts/reproduce_lerique_paper.py --fast
# writes artifacts/paper_lerique/reproduce_fast_features.csv
```

Covered by `tests/test_reproduce_smoke.py`.

## 6. Reviewer checklist

- [ ] OSF component URL filled in (§1).
- [ ] License / access terms filled in (§2).
- [ ] Local mirror layout matches §3.
- [ ] `MANIFEST.json` data-source fields populated (no `TODO` left as fact).
- [ ] `--pub` run reproduces the reported derived tables.
- [ ] `--fast` smoke run passes in CI.

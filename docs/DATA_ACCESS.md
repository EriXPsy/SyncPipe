# Data Access & Reproducibility — ECSU-PCE / Lerique (OSF 47n3p)

This document tells a reviewer how to obtain the **third-party** raw data and
regenerate SyncPipe analyses. SyncPipe does **not** ship raw physiological
time series.

## 1. Source dataset

- **Dataset project (raw + components):**  
  [OSF — Perceptual Crossing Dataset Paper](https://osf.io/47n3p/)  
  Project id: **47n3p** (public project).
- **Dataset paper / preprint:**  
  Lerique et al., *The ECSU-PCE Dataset: A comprehensive recording of embodied
  social interaction with EEG, peripheral physiology, and behavioral
  measurements in adults*  
  - Preprint: https://osf.io/preprints/osf/6hjfy  
  - DOI: https://doi.org/10.31219/osf.io/6hjfy
- **Modalities used by SyncPipe loaders:** peripheral physiology under the
  project tree (ECG / EDA / RESP layout expected by
  `multisync.realtest.lerique_2024`). EEG hyperscanning components linked from
  the preprint are **out of scope** for SyncPipe v1 continuous-envelope path.
- **Access:** **Public download** on OSF (no account wall observed at last
  check). Always re-verify before submission.
- **Version / snapshot:** Use the OSF project state at analysis time; record
  the date and, if available, OSF file version hashes in `MANIFEST.json`.

### Suggested citation (APA-style from OSF project page)

Unit, T. C. S., Estelle, S., Zapata-Fonseca, L., Hayashi, S. R., Morrissey,
B. R., Lerique, S., … Froese, T. (2024, September 17). *Perceptual Crossing
Dataset Paper*. OSF. https://osf.io/47n3p

Also cite the preprint when discussing the dataset paper itself.

## 2. License & terms of use (honest)

| Object | License field (verified) |
|---|---|
| OSF project **47n3p** | **No License** (public files ≠ Creative Commons grant) |
| Preprint **6hjfy** | **CC-BY 4.0** (covers the **manuscript**, not automatically every binary in OSF storage) |

**Implications for SyncPipe and users:**

1. You may **download and analyse** the public deposit for research, with
   citation.
2. You must **not** treat the raw `.mat` as CC-BY/CC0 unless the depositors
   change the OSF license field or grant written permission.
3. SyncPipe **does not redistribute** raw signals. This repository ships code,
   docs, and **derived** tables only.
4. Before journal submission, email the ECSU team if you need an explicit
   CC-BY (or written research-use) statement on the peripheral files.

**Derived tables** produced by SyncPipe (feature CSVs, audit JSON) are
outputs of your analysis under the package MIT license for the *code*; they
remain scientific products that must cite the OSF source data.

## 3. Local mirror layout

After download/unzip, point the loader at a root that looks like:

```text
$LERIQUE_ROOT/
  ECG/
    pce.../
  EDA/
    pce.../
  RESP/
    pce.../
```

Exact folder names must match `multisync.realtest.lerique_2024` expectations
(see that module’s docstring).

## 4. Reproduction commands

```bash
cd SyncPipe
pip install -e ".[dev]"

# Wiring smoke (no OSF) — CI-safe
python scripts/reproduce_lerique_paper.py --fast

# Publication-floor run (requires local mirror)
python scripts/reproduce_lerique_paper.py --pub   --data-root "$LERIQUE_ROOT"   -o artifacts/paper_lerique
# defaults: surrogate_n=1000 (canonical publication-grade; lower-level InferencePipeline default 100), n_permutations=10000
```

Outputs:

```text
artifacts/paper_lerique/
  MANIFEST.json                 # params + git + data-access facts
  reproduce_<mode>_features.csv
  reproduce_<mode>_summary.json
```

## 5. Canonical scientific path

```text
records
  -> multisync.pipeline_bridge.records_to_inference_inputs
  -> multisync.inference_pipeline.InferencePipeline.run_audited_evidence_chain
       (synchrony-existence
        -> design controls
        -> group inference / BH-FDR; multimodal auto per-modality)
```

- Fixed/pooled onset threshold on this path (`threshold_scope="fixed"`).
- Multimodal group tests must receive an explicit `contrast` when possible
  (script resolves `rest1`/`trials_concat` or `rest`/`trials` when present).

## 6. Reviewer checklist

- [ ] OSF project opens as public: https://osf.io/47n3p/
- [ ] License wording in the manuscript matches §2 (No License on data project)
- [ ] Local mirror layout matches §3
- [ ] `MANIFEST.json` written by the run (no leftover TODO as “facts”)
- [ ] `--pub` used publication-floor parameters (or deviations disclosed)
- [ ] `--fast` smoke passes in CI without OSF

## 7. Contact if OSF access breaks

- OSF project contributors listed on https://osf.io/47n3p/
- SyncPipe maintainer: repository GitHub issues on EriXPsy/SyncPipe

# SyncPipe v1 — Cleanup, Rename & Launch Plan

> Internal working plan. Status: draft for maintainer ratification.
> Maps the 2026-08 audit findings (A = engineering/product, B = psychometric/
> statistical, C = governance) onto executable items. Each item carries a
> `decision-needed` flag: **NO** = uncontroversial, do it; **YES** = blocks on
> a scientific/scope choice the maintainer must make first (see §6).

## Progress

- [x] Item 4 — dead external links removed (README).
- [x] Item 7 — CJK comments de-CJK'd in package code.
- [x] Item 1 — `multisync` → `syncpipe` clean-cut rename (Batches 1–4).
- [x] Item 6 — `prediction.py` removal (moved to `experimental/`; stripped from v1 package).
- [x] Item 3 — dangling commit-hash rewrite (decision: **rewrite to IDs**).
- [x] Item 2 — research-artifact cleanup: stale `experimental/multisync/` + `deliverables/` deleted; A-class regenerable outputs deleted (gt1/, gt5, incremental_auc/, morphology/, realdata_full/); B-class validation inputs kept (vif/, wcc_traces/, realtest/lerique_2024/, pgt2/pgt3/egt4 grids, realdata_audit/, reviewer_audit/, paper_lerique/).
- [x] B3 — feature status unified (peak=confirmatory-primary; dwell/switching=conditional-secondary; guard checks PRIMARY_FDR_FAMILY).
- [x] B4 — existence gate rewritten to a second-order group surrogate test (`EXISTENCE_GATE_ALPHA`).
- [x] B5 — definedness eligibility rule documented as pre-registered (`V1_CLAIM_CEILING.md` §4.1).
- [~] Item 5 — SSoT unification: feature_status + FEATURE_TABLE + guard now agree; C-class dangling doc refs (bakeoff/, figures/, incremental_value/, timing_validation/) still to clean.

---

## 0. Priority overview

| Priority | Scope | Items |
|---|---|---|
| P0 — internal consistency | fix contradictions & dangling refs before anything public | C1, C2, C3, B3 (decision) |
| P1 — package hygiene | make the repo look like a product, not a lab notebook | A1, A2, A3, A4, A5, A6 |
| P2 — branding | one name, one namespace, zero `multisync` user-facing | A7, rename plan (§1) |
| P3 — external validation | the "find people" work that makes it DPABI-like | launch plan (§8) |

---

## 1. Rename: multisync → syncpipe (item 1) — DONE

**Goal.** `syncpipe/` is the only package; `multisync` exists nowhere
user-facing (no directory, no console script, no `sys.meta_path` alias hack).

**What was done (2026-08-17).**

- **Batch 1** — physical merge: `multisync/*` moved into `syncpipe/`; the
  `_SyncpipeAliasFinder` meta_path hack replaced by the real `__init__`; the
  `multisync/` directory deleted.
- **Batch 2** — import rewrite: every `import multisync` / `from multisync ...`
  rewritten to `syncpipe` across the repo (scripts, tests, docs, deliverables,
  experimental, Dockerfile, pyproject, .gitignore, CITATION.cff).
- **Batch 3** — packaging: `multisync` console script and the `multisync*`
  `packages.find` entry removed; `syncpipe*` is the only include.
- **Batch 4** — verification: `pip install -e .` + full `pytest` green; residual
  `multisync` occurrences are only `multiSyncPy` (third-party) and *MultiSync*
  (historical project name). The transitional `scripts/audit_syncpipe_branding.py`
  was removed (its premise — a live migration — is complete). The
  `load_multisync_csv` method name and the `MULTISYNC_CORE` script constants were
  fixed.

**decision-needed: NO.**

---

## 2. Research-artifact policy (item 2)

**Principle.** The package repo ships: the package, its tests, its docs, and the
*minimal* reproducibility artifacts the docs/paper cite directly. Everything
else is a research archive — keep it, but **not in this repo**.

| Class | Example | Action |
|---|---|---|
| **Keep (tracked)** | `artifacts/paper_lerique/`, `artifacts/realdata_audit/`, `artifacts/reviewer_audit/`, all `docs/`, `tests/`, `scripts/` runners that produce a cited claim | keep in-repo |
| **Archive (move out)** | regenerable CSVs/figures, `experimental/scripts/`, `deliverables/*.md` session logs | move to a separate `SyncPipe-archive` repo, or delete |
| **Delete** | `experimental/multisync/` (stale duplicate of the package), dead one-off probes | delete |

**decision: hard delete** (maintainer, 2026-08-17) — keep only package + tests +
docs + cited minimal artifacts; `git rm` the rest.

---

## 3. Git history & dangling commit refs (item 3)

**Finding (2026-08-17):** `origin/main` has only **1 commit**. The history that
`docs/DECISION_LOG.md` and `docs/DESIGN_PRINCIPLES.md` cite (`92da615`,
`4efc72c`, …) is **permanently unrecoverable**.

**Root cause.** History was squashed; governance docs cite fragile git hashes.
Two fixes:

1. **Behavioral:** stop citing git hashes in governance docs — cite decision-IDs
   (`DECISION-04`, `BUG-3`) and dated entries, which survive squash/rebase. Going
   forward, do **not** squash/force-push history.
2. **Mechanical:** rewrite the dangling short-hash references to decision-IDs /
   behavioral descriptions, and add one note that pre-v1.0 history was squashed.

**decision: rewrite to IDs** (maintainer, 2026-08-17).

---

## 4. Dead external links (item 4) — DONE

Removed the two `github.com/user-attachments/...` `<img>` tags from README
(guaranteed-404 temporary asset URLs). Local `syncpipe-logo-mark.svg` retained.

---

## 5. Unify the SSoT (item 5)

**Root cause.** Three files each claim "single source of truth" and have
drifted: `feature_definitions.py` (math), `feature_status.py` (communication),
`docs/FEATURE_TABLE.md` ("Authoritative … SSoT").

**Fix.**
1. **One canonical source = `feature_definitions.py`.** Everything else is
   *generated* from it, never hand-maintained.
2. Make `feature_status.py` + `docs/FEATURE_TABLE.md` + the LaTeX/CSV exports
   generated artifacts, with a CI test that regenerates and diffs (fails on
   drift). Extend `_check_feature_status_consistency()` to cover **all** fields.
3. **Blocked on B3 (§6)** — the `dwell_time`/`switching_rate` status
   contradiction must be resolved *first*.

**decision-needed: YES** (B3 first).

---

## 6. prediction.py (item 6)

**Problem.** `syncpipe/prediction.py` (~1700 LOC: rolling-origin CV +
cross-modal prediction + LogisticRegression) is shipped but contradicts v1's
claim ceiling ("same-modality only"; prediction is an "opt-in side path"). It is
entangled into `core.DynamicAnalyzer` and `AnalysisResults.prediction`.

**decision: move out** (maintainer, 2026-08-17) — relocate to `experimental/` and
strip `enable_prediction` from `core.py`.

---

## 7. CJK comment cleanup (item 7) — DONE (package scope)

Translated the three stray CJK comments in package code (`core.py`, `importer.py`).
Remaining Chinese lives in files slated for archive/delete (`experimental/`,
`deliverables/`) and in two *intentional* places, each its own decision:
- `Intro.html` — deliberate Chinese landing page, still branded **MultiSync**
  (stale). Rebrand to SyncPipe or regenerate. **YES**
- `scripts/_gen_figs.py` / `_make_gallery.py` — emit **Chinese** figure labels.
  For a BRM methods paper the figures must be English. **YES**

---

## 8. Solo launch: the DPABI v1 playbook (item 8)

DPABI's influence came from a **published methods paper** (Yan & Zang 2010,
*Neuroinformatics*) plus a zero-friction tool — not from documentation alone.

1. **Write the methods paper** (target *Behavior Research Methods*). Lead with
   **one** crisp contribution: *a standardized, audited WCC-based measurement +
   evidence chain for continuous same-modality dyadic synchrony.*
2. **Pre-register the confirmatory claim** (the two-condition `peak_amplitude`
   contrast) so the primary endpoint is not retrofitted.
3. **Get 2–3 external labs to run it on their own data** before launch. Their
   friction reports *are* your validation. This is the "find people" task named
   in `docs/LIMITATIONS.md` — it is the actual bottleneck, not code.
4. **Cut scope to match the claim.** Ship the minimal auditable core; mark the
   rest experimental.
5. **Ship PyPI + CITATION.cff + a <5-minute worked tutorial on public data.**

---

## 9. Open scientific decisions — the grill (B3 / B4 / B5)

These block §5 (SSoT) and any manuscript:

- **B3.** Single confirmatory status of `dwell_time` / `switching_rate`? The
  docs disagree (feature_status.py "primary-structure" vs feature_definitions
  SECONDARY family vs V1_CLAIM_CEILING "forbidden to call confirmatory"). Pick
  one and align all four + code.
- **B4.** The existence gate is an OR over two primary modalities with a
  self-invented >50%-dyad majority rule. What is its family-wise error rate?
  Re-specify (single pre-registered modality, or a proper combined test)?
- **B5.** `dwell_time`/`switching_rate` are labeled core/primary yet undefined
  in 76% of Gordon dyads. Does "core" mean "defined nearly everywhere"?

---
*Maintained by the SyncPipe author. Update by editing this file; do not treat it
as a scientific claim document.*

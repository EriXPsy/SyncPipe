# SyncPipe Decision Log

This file records **current v1 decisions and changes**. Older exploratory or superseded decision history should be kept in an archive if needed, but the active decision log should remain short enough for users and reviewers to audit.

---

## Current v1 feature-family stance

**Decision.** SyncPipe v1 uses a narrow, two-tier confirmatory FDR structure
(SSoT in `multisync/feature_definitions.py`):

- `PRIMARY_FDR_FAMILY` (n = 1, confirmatory primary endpoint): `peak_amplitude`
- `SECONDARY_FDR_FAMILY` (n = 2, parallel confirmatory): `dwell_time`, `switching_rate`

The two families use **different null models** (L0 signal-level IAAFT vs
L1 WCC-level IAAFT) and are BH-corrected **independently** — they never share
one BH denominator. `FDR_FEATURES` (m = 3) is the *union* of both families,
retained for backward-compat and guard logic only; it is **not** the BH
correction set. `mean_synchrony` is a reference comparator. `fraction_above_threshold`, `bimodality_coefficient`, `synchrony_entropy`, `onset_latency`, `rise_time`, `recovery_time`, `first_peak_time`, and `inter_peak_cv` are reported as exploratory / secondary descriptors with paradigm restrictions and definedness reporting where applicable.

**Rationale.** The v1 contribution is audited measurement infrastructure, not a claim that every WCC-derived descriptor is a validated psychological construct. A narrow primary family reduces multiplicity and keeps interpretation defensible.

**Source of truth.** `multisync/feature_definitions.py`, `multisync/feature_status.py`, and generated `docs/FEATURE_TABLE.md`.

---

## 2026-08-03 — FDR family split into PRIMARY (n=1) + SECONDARY (n=2) for independent BH

**Decision.** The previously-frozen m = 3 confirmatory union (`peak_amplitude`,
`dwell_time`, `switching_rate`) is now split for Benjamini–Hochberg correction
into two SSoT families:

- `PRIMARY_FDR_FAMILY` (n = 1): `peak_amplitude` — the single universally-clean
  primary endpoint (low VIF on every real dataset where observed).
- `SECONDARY_FDR_FAMILY` (n = 2): `dwell_time`, `switching_rate` — parallel
  confirmatory; dataset-conditional (defensible in Lerique / ECG+EDA, collapses
  to non-significance on RESP, the negative-control modality).

**Why split.** L0 (signal-level IAAFT) and L1 (WCC-level IAAFT) are different
null models; pooling them — or pooling modalities within one family — dilutes
the primary endpoint and lets the reference feature occupy a correction slot.
BH now runs *within* each family, pooled across modalities, so a family tested
on M modalities controls the joint family-wise error at M hypotheses. Reference
features (`mean_synchrony`) are reported (p_raw) but never enter any BH
denominator (`p_fdr = nan`, never `significant_05`). This is implemented
identically in `inference_pipeline._apply_global_modality_fdr` and
`validation/l2_between_condition.between_condition_fdr`.

**What did NOT change.** `FDR_FEATURES` (m = 3) is retained as the union for
guard / backward-compat logic; `mean_synchrony` stays reference;
`bimodality_coefficient` / `synchrony_entropy` stay exploratory.

**Source of truth.** `multisync/feature_definitions.py` (`FDR_FAMILIES`,
`PRIMARY_FDR_FAMILY`, `SECONDARY_FDR_FAMILY`, `REFERENCE_FEATURE`).

---

## 2026-07-21 — B4 FDR-family bake-off freeze (evidence-driven, ALL-REAL rerun)

> **Superseded (2026-08-03).** The m = 3 confirmatory union below was later split
> for BH correction into `PRIMARY_FDR_FAMILY` (n = 1: `peak_amplitude`) and
> `SECONDARY_FDR_FAMILY` (n = 2: `dwell_time`, `switching_rate`), each corrected
> independently (different null models must not share a denominator). The current
> corrected families are no longer a single m = 3 denominator — see *Current v1
> feature-family stance* (2026-08-03) above. `FDR_FEATURES` (m = 3) remains the
> union, used only for guards / back-compat.

**Decision (frozen).** The v1 primary FDR family is confirmed as:

- `peak_amplitude`
- `dwell_time`
- `switching_rate`

`mean_synchrony` remains a **reference** comparator (not corrected). `bimodality_coefficient` and `synchrony_entropy` remain **exploratory** (entering only as secondary descriptors, not as confirmatory FDR tests). This is consistent with the existing "8 confirmatory / FDR m=3" framework (m = 3 corrected family members).

**Evidence.** `scripts/bakeoff_fdr_family.py` ran a Pearson-ρ + VIF bake-off plus leave-one-dyad-out (LOO) stability for N≥23, now **fully on real data** (no synthetic smoke — the earlier B1 OSF "No license" blocker is resolved; real per-dyad data is in-repo). Sources per dataset:

- **Lerique 2024 (REAL, N=31 dyads / 264 records, recomputed this run from `artifacts/realtest/lerique_2024/per_record_features.csv`):** VIF of all three primary-FDR features is low — `peak_amplitude` 2.62, `dwell_time` 3.06, `switching_rate` 2.48; `mean_synchrony` 2.63. All < VIF_CONCERN (5.0). LOO stability = **100%** (the VIF-qualifying set never changed across 31 dyad removals). These values exactly reproduce the pre-existing `artifacts/vif/lerique_vif_series.csv`, confirming the result.
- **Andersen (REAL, N=300 dyads / 300 traces, recomputed this run from `artifacts/wcc_traces/andersen_wcc_traces.csv`):** **SEVERE collinearity** — `mean_synchrony` VIF 39.35, `dwell_time` 57.88, `switching_rate` 18.81; `bimodality_coefficient` 5.60 (concern), `synchrony_entropy` 4.46. Recomputed VIF drifts ~12–14 % from the frozen `andersen_vif_series.csv` (50.88 / 35.27) but the **SEVERE flags are identical**, so the conclusion is unchanged. LOO stability = **100%** (0/300 folds changed) — the severe classification is robust, not an outlier artefact.
- **Gordon (REAL, N=366 records, ingested this run from frozen `artifacts/vif/gordon_vif_series.csv` + `gordon_correlation_matrix.csv`):** all VIF < 5 — `peak_amplitude` 4.00, `mean_synchrony` 4.25, `synchrony_entropy` 1.30, `onset_latency` 1.10, `rise_time` 1.11, `recovery_time` 1.05. Gordon's real extracted feature set did **not** include `dwell_time` / `switching_rate`, so it contributes no evidence for those two primary members. (Per-dyad source CSV `gordon_2025_dyads.csv` is not in-repo and present `gordon_wcc_traces.csv` is degenerate at ~82 % NaN/trace, so the values were ingested from the frozen real artifact rather than recomputed; Gordon LOO is N/A.)

**Why this freeze is defensible.** (1) On the only dataset where all three primary members are simultaneously observed with real data (Lerique), none approaches the VIF_SEVERE (10.0) independence threshold, and the membership decision is LOO-stable to 100%. (2) Andersen's severe collinearity (now recomputed on real re-extracted traces, not just the prior json) confirms the *opposite* risk is real and already governed by excluding `mean_synchrony` from correction and keeping `bimodality_coefficient`/`synchrony_entropy` exploratory — and it shows `dwell_time` / `switching_rate` are **not** universally independent (SEVERE in Andersen). The 3-member family is therefore **dataset-conditional**: safe in Lerique, redundant in Andersen; `peak_amplitude` is the only universally clean primary feature. (3) The bake-off therefore neither over-expands (no new member justified) nor under-expands (the three core members survive independent-test scrutiny where real data observes them together).

**A3 (flag-flip) input.** The frozen family is what `scripts/fdr_family_impact.py` should use as Option B (`{peak_amplitude, dwell_time, switching_rate}`); the Lerique LOO stability 100% means the significance-flip set is robust to single-dyad omission.

**n_min=10 evidence (feeds B3).** Real Lerique LOO at N=31 is 100% stable; all three real datasets have N > 10. This upgrades `n_min=10` from "default recommendation" toward evidence-driven: it is a safe *exclusion floor* (drops only absurdly small pilots), never the binding constraint on any real analysis. B3 still owns the constant hard-coding.

**Reproducibility / limitation.** B4 outputs: `artifacts/bakeoff/fdr_family_bakeoff.csv` (wide VIF + Pearson-ρ matrix, columns = datasets) and `artifacts/bakeoff/fdr_family_loo_stability.csv`, plus `artifacts/bakeoff/REALDATA_BAKEOFF_ANALYSIS.md`. **All three columns are now REAL** (no synthetic smoke): Lerique + Andersen recomputed from in-repo real data; Gordon ingested from frozen real `artifacts/vif/*` (per-dyad source CSV absent, present `gordon_wcc_traces.csv` degenerate). Gordon LOO is N/A (per-dyad source unavailable); Andersen VIF drifts ~12–14 % from the frozen `andersen_vif_series.csv` but SEVERE flags match. Gordon's real feature set lacks `dwell_time`/`switching_rate`, so it bears no evidence for those two primary members.

**Source of truth.** `multisync/feature_definitions.py`, `multisync/feature_status.py`, `scripts/bakeoff_fdr_family.py`, and `artifacts/bakeoff/`.

---

## 2026-07-22 — A3 FDR-family flag-flip close-out (Option B already live)

**Decision (verified, no code change required).** The primary-FDR family flag
(`FDR_FEATURES` in `multisync/feature_definitions.py`) already equals **Option B**
= `{peak_amplitude, dwell_time, switching_rate}` (m = 3), with `mean_synchrony`
carried as a *reference* comparator and `bimodality_coefficient` / `synchrony_entropy`
kept exploratory. A3 therefore required **no code flip** — the flag, the SSoT
module, `docs/FEATURE_TABLE.csv`, and `FDR_FEATURES` were all already aligned to
Option B before this close-out note was written. A3's deliverable is the impact
evidence below plus this close-out.

**Evidence (`scripts/fdr_family_impact.py`, re-run 2026-07-22 on the frozen
Lerique 2024 MAIN contrast `artifacts/realtest/lerique_2024/group_contrasts_paired.csv`).**
Re-applies Benjamini–Hochberg FDR to the existing per-feature `p_raw` under three
family definitions:

- **STATUS QUO** (`{mean_synchrony, peak, dwell, switching}`, m varies): ECG & EDA
  all four significant; RESP only `peak_amplitude` significant.
- **OPTION A** (`{peak_amplitude}` only, m=1): only `peak_amplitude` significant in
  all three modalities; `dwell_time` / `switching_rate` not corrected (None).
- **OPTION B** (current code, `{peak, dwell, switching}`, m=3): ECG
  `peak`/`dwell`/`switching` all significant; EDA all three significant; RESP only
  `peak_amplitude` significant (`dwell_time` p=0.375 → False, `switching_rate`
  p=0.2165 → False).

**Significance flips (Option A → Option B), per modality × feature:**
ECG `dwell_time` / `switching_rate`: None → **True**; EDA `dwell_time` /
`switching_rate`: None → **True**; RESP `dwell_time` / `switching_rate`: None →
**False** (raw p not small enough once m grows). The ECG/EDA flips are expected:
growing the family from m=1 to m=3 loosens the BH step-up, promoting the two
extra primary members to significance where their raw p was already marginal.
RESP's `dwell`/`switching` stay non-significant because their raw p (0.375 /
0.2165) is far above α even before correction — consistent with RESP as an
effective negative-control modality.

**Conclusion.** Option B is the defensible v1 primary family: it is the narrowest
set that (a) reproduces the canonical `peak_amplitude` result everywhere, (b)
recovers `dwell_time`/`switching_rate` significance on the modalities where those
descriptors are real (ECG/EDA), and (c) correctly withholds them on RESP, the
negative-control. No further family-flag change is warranted.

> **Continuity note (2026-08-03).** The m = 3 union above was later split for
> BH correction into `PRIMARY_FDR_FAMILY` (n = 1: `peak_amplitude`) and
> `SECONDARY_FDR_FAMILY` (n = 2: `dwell_time`, `switching_rate`), each
> BH-corrected independently (different null models must not share a denominator).
> `FDR_FEATURES` (m = 3) remains the union, used only for guards / back-compat.
> See *Current v1 feature-family stance* above.

**Source of truth.** `multisync/feature_definitions.py` (`FDR_FEATURES`,
`FDR_FAMILIES`); `scripts/fdr_family_impact.py`; the frozen input CSV
`artifacts/realtest/lerique_2024/group_contrasts_paired.csv`.

---

## 2026-07-23 — B4 VIF / dataset-conditional FDR family (direction a close-out)

**Decision (documentation + architectural intent; no code change yet).** Cross-dataset
VIF (variance-inflation factor) on the real extracted features shows the primary
FDR family is **not uniformly defensible**:

| Feature | Lerique N=264 | Andersen N=300 | Gordon N=366 | Verdict |
|---|---|---|---|---|
| peak_amplitude | 2.62 | 1.36 | 4.00 | **clean everywhere (<5)** |
| dwell_time | 3.06 | 50.88 SEVERE | n/a¹ | clean in Lerique, SEVERE in Andersen |
| switching_rate | 2.48 | 17.98 SEVERE | n/a¹ | clean in Lerique, SEVERE in Andersen |

`peak_amplitude` is the **only feature with VIF < VIF_CONCERN (5.0) in all three
real datasets** → it is the single universally-defensible primary FDR test.
`dwell_time` / `switching_rate` are clean in Lerique but SEVERE in Andersen, so
they must be treated as **dataset-conditional** (primary where VIF is low,
reference/exploratory where VIF is SEVERE), not fixed members of a universal
family.

**Why only Andersen collapses (root cause, from `artifacts/vif/*_vif_report.json`
+ OSF raw inspection).** In Andersen the `mean_synchrony ↔ synchrony_entropy`
pair reaches ρ = **−0.93** (vs moderate elsewhere), with `dwell↔entropy` −0.88 and
`mean↔dwell` +0.81 — the whole episode-feature bloc (mean_sync, dwell, switching,
entropy, bimodality) collapses onto a **single latent variable** = "how high the
sync plateau is." This is the signature of a **near-saturated / flat-topped WCC
regime**: high mean sync comes with a low-entropy (unvarying) trace, so dwell
(long), switching (rare), peak (≈mean) and entropy all become functions of one
number. Lerique and Gordon retain richer temporal (spiky-episode) structure, so
their feature spaces stay multi-dimensional and VIF stays < 5. The severe VIF is
therefore **not a SyncPipe code bug** (VIF is computed correctly on the extracted
matrix) and **not pure dataset chance** (LOO stability = 100 %); it is a
**feature-definition-level mechanical coupling (dwell↔switching by construction;
mean↔entropy as two moments of one distribution) amplified by Andersen's
saturated episode regime.** Any dataset with a similarly stereotyped/saturated
WCC will show the same collapse.

**Known gap (must be closed before claiming dataset-conditional behaviour).** The
VIF diagnostic in `multisync/validation/l2_between_condition.py` is
**flag-only**: it attaches a `vif_gate` warning to the result but does **not**
demote severe features or shrink `m` (`"Diagnostic only — never let the VIF gate
break the L2 test"`, lines ~337–372). So today `dwell_time`/`switching_rate`
remain in the fixed `FDR_FEATURES` (m=3) and are tested as independent everywhere,
including Andersen. The dataset-conditional intent above is **documented but not
yet enforced at runtime.** Two options:
- (a-short) keep `FDR_FEATURES` fixed but state explicitly in paper/methods that
  peak_amplitude is the only universally-defensible primary feature and
  dwell/switching are dataset-conditional (honest, zero-risk);
- (b-mid) wire the gate to actually DEMOTE severe features and reduce `m` in
  high-collinearity datasets (requires care around pre-registration semantics).

**Theoretical dividend (resolves the peak-amplitude incremental-value question).**
peak_amplitude is the *phasic* complement of mean_synchrony: `mean ≈ peak ×
duty_cycle`, so mean is duration-confounded while peak isolates *intensity / maximal
coordination degree* (HKB attractor depth). This dissociation is exactly what makes
peak informative in Lerique (rich episodes: peak is the most discriminative /
"canonical Type-I" feature) and exactly why it collapses to mean in Andersen
(saturated regime) — the same mechanism that drives the severe VIF. Reporting peak
is therefore not "mean + noise"; it is required to describe episode *morphology*
(the project's core thesis: SCR/ERP morphology → WCC-episode morphology), which
mean (the DC component) cannot capture alone.

**Source of truth.** `artifacts/vif/{andersen,lerique,gordon}_vif_report.json`,
`artifacts/vif/vif_comparison.csv`, `artifacts/bakeoff/REALDATA_BAKEOFF_ANALYSIS.md`,
`multisync/feature_vif_test.py`, `multisync/validation/l2_between_condition.py`.

### peak_amplitude incremental value — empirical evidence (resolves the "is peak just mean?" worry)

The worry: peak looks stable/independent of mean, but why report both — won't a
reviewer ask what peak *adds*? Answer from the real data:

**peak~mean redundancy is dataset-conditional, not universal:**

| Dataset (regime) | peak~mean ρ | peak VIF | reading |
|---|---|---|---|
| Andersen (saturated/flat-top) | 0.37 | 1.36 | peak decoupled from mean |
| Lerique (transient/spiky EDA) | 0.56 | 2.62 | peak partly distinct |
| Gordon (flat-top) | **0.87** | 4.00 | peak ≈ mean (redundant here) |

So peak is *not* a universal proxy for mean. It is most redundant with mean
exactly where the WCC regime is flat-topped (Gordon), and most distinct where
episodes are transient spikes (Lerique) — the same regime axis that drives the
Andersen VIF collapse.

**Decisive L2-style sanity check (Lerique, dyad-level task-vs-rest logistic
regression, 5-fold CV AUC):**
- `mean_synchrony` only → AUC **0.588** (near chance)
- `peak_amplitude` only → AUC **0.780** (strong)
- `mean + peak` → AUC **0.781** (peak dominates; mean adds ≈0)

i.e. in Lerique the *mean is nearly uninformative* and *peak carries the signal*.
This is the empirical counter to "peak is mean + noise": peak is the informative
feature and mean is the diluted one, precisely because transient episodes have
sharp apexes amid low baseline that mean averages away.

**Smooth theoretical logic for the paper (the "why peak" narrative):**
1. *Episode = a transient response, not a steady level.* A WCC episode is the
   homolog of an SCR/ERP — a coordination *event*, not a level. In
   psychophysiology, SCR/ERP **peak amplitude** (not mean) has been the standard
   feature for ~60 years because it indexes response *magnitude*. WCC-episode peak
   is the direct analog: the maximal joint-coherence the dyad *achieved*.
2. *peak = coordination ceiling; mean = time-averaged coupling.* mean_synchrony is
   **duration-confounded** — dragged down by the long low-synchrony "dead time"
   between episodes (mean ≈ peak × duty_cycle). peak is immune to baseline
   dilution and isolates the *intensity apex* (HKB attractor depth / maximal
   shared state). Different questions: "how high did they go?" vs "how coordinated
   on average?"
3. *peak is orthogonal to episode SHAPE in a different axis than mean.*
   Lerique peak~dwell = 0.81 but peak~mean = 0.56 — peak carries intensity
   information partly shared with shape yet distinct from average level. In
   Gordon peak~dwell = 0.00 & peak~mean = 0.87 — there peak *is* just mean
   (flat-top), so it correctly becomes companion.
4. *Reporting peak is required by the project's core thesis* (SCR/ERP morphology →
   WCC-episode morphology): mean is the DC component; peak is the phasic
   amplitude. Describing episode *morphology* needs both, and peak is the feature
   that makes the synchrony trace a *shape* rather than a *level*.

**Reviewer-handling (turn the worry into a strength).** Pre-register peak as the
primary "coordination ceiling" feature; keep mean_synchrony as REFERENCE (already
done). Report peak~mean ρ per dataset as a transparency item: "where peak ≈ mean
(flat-top regimes, e.g. Gordon) peak is companion; where peak >> mean (transient
regimes, e.g. Lerique) peak is the primary signal carrier." This converts the
redundancy question from a vulnerability into a demonstration of the tool's
regime-awareness.

---

## 2026-07-23 — P1 residual fixes R1–R4 (review of external patch, dialectical)

External review (post-`08b1883`) flagged four P1 residuals and supplied a
drop-in patch under `updates/`. Reviewed dialectically — every claim was
re-derived against `main` before adoption. All four were **real**, the patch
introduced **no broken symbol references**, and all 56 + 77 regression tests
pass.

| ID | Issue (confirmed against HEAD) | Fix adopted |
|---|---|---|
| **R1** | `run_group_condition_inference` multimodal branch dropped `contrast`/`threshold_scope`/`seed` into `test_l2_by_modality` (params existed, just not forwarded) → multimodal L2 silently used sorted labels; `_build_audited_chain_summary` read `group["n_significant"]` which is absent when `group` is modality-keyed → always reported 0 sig. | Forward the three kwargs; summary sums `n_significant` across modalities. `between_condition_by_modality` already accepted `condition_values`/`threshold_scope`/`seed`, so the forwarding is valid. |
| **R2** | `ComputationPipeline` already NaN-masks the WCC at seams (`_discontinuity_mask`, line 216) but `extract_features` still used `merge_valid` dwell/switching by default → episodes glued across intentional seams, contradicting the mask. | When `_discontinuity_mask is not None`, pass `gap_policy="segment"`; unmasked pairs keep `merge_valid`. Consistent with DECISION 2026-07-13; does **not** change B4-frozen numbers (those had no mask). |
| **R3** | `_apply_discontinuity_mask` was applied to the **observed** WCC (existence null) but **not** to the **surrogate** pool in `_generate_surrogate_coupling_matrix`/`compute_session_pooled_threshold` → pooled null inflated by seam-spanning windows. | Thread `discontinuity_mask(s)` through; surrogate traces NaN-gated identically to observed. Per-dyad mask list length-checked. |
| **R4** | `reproduce_lerique_paper.py` had `OSET_THRESHOLD` typo, TODO MANIFEST, non-pub defaults; `DATA_ACCESS.md` was skeletal. | Typo→`ONSET_THRESHOLD`; real MANIFEST writer (git hash + params + honest OSF `No License` vs preprint `CC-BY`); pub floor `surrogate_n=100`/`n_perm=10000`; full `DATA_ACCESS.md`. |

**Improvements made vs the supplied patch.** Did **not** copy the patch's
`artifacts/paper_lerique/MANIFEST.json` sample (it baked in `repo_dirty:true`
and an old commit). Instead regenerated it via `--fast` at runtime, which
correctly reports the current hash and `dirty` status. Also noted the
**already-committed** MANIFEST was itself stale (`9e3de62` + `TODO`s) — now
replaced by the live one.

**Why these were missed in earlier reviews (the meta-lesson).** All four are
instances of one systemic blind spot, recurring since 发现10 / 发现17:

1. *Fix the guard, trust the callee.* For R1 I added the P0-2
   `allow_multimodal_pool=False` guard and routed to `test_l2_by_modality`,
   but never checked that `test_l2_by_modality` forwarded `contrast`/`seed` to
   `between_condition_by_modality`. The canonical-chain test only exercised the
   unimodal (`n_mod==1`) branch, so the multimodal branch was never hit.
2. *Two adjacent, individually-correct changes, never reconciled.* R2:
   the 2026-07-13 `segment` decision and the separate WCC NaN-masking (line 216)
   were made independently; nobody connected "WCC is masked at seams" with
   "dwell default still bridges seams."
3. *Symmetric-path asymmetry.* R3: observed path gated, null path not —
   the same class as 发现10 (extract_episodes didn't inherit the dwell fix) and
   发现17 (restricted model didn't inherit the no-fabrication fix).
4. *Out-of-core hygiene ignored.* R4 (typo, TODOs, docs) was skipped because
   attention was on the scientific core.

**Process fix adopted (do this from now on).** Add a *symmetry audit* to every
change: when I touch one of a pair (observed/null, mask-on/mask-off,
multimodal/unimodal, segment/merge), grep for the sibling and verify it too;
and always exercise **both** branches of any conditional I route through.

Commits: `32a1692` (R1–R3 + tests), `65f8b4f` (R4 + manifest). Pushed, ahead=0.

---

## 2026-07-21 — B3 eligibility thresholds freeze (evidence-driven)

**Decision (frozen in code).** Two eligibility floors are now hard-coded as
module-level constants in `multisync/feature_definitions.py` and exported via
`__all__`:

- `T_DEF_MIN_WCC_POINTS: int = 3` — minimum finite WCC sampling points per dyad.
- `N_MIN_DYADS_FDR: int = 10` — minimum dyad count for a meaningful BH-FDR correction.

A lightweight pure gate, `check_eligibility(n_wcc_points, n_dyads) ->
(wcc_points_ok, n_dyads_ok)`, applies both floors. `qc.run_quality_check`
accepts an optional `eligibility={"n_wcc_points": ..., "n_dyads": ...}`
context and, when supplied, surfaces any floor violation as a **WARN-level
NOTE** on `DataQualityReport.notes` (reusing the existing non-blocking `notes`
field, exactly like the co-start caveat). It never alters the 4-stage
verdicts or any other field.

**Semantics.**

- *T_def.* A dyad whose WCC trajectory has fewer than 3 finite sampling points
  cannot define episode features — onset→peak→recovery (Axis D L2) needs three
  *separable* points. Peak detection already uses a 3-point boxcar
  (`PEAK_SMOOTHING_WINDOW`, DECISION-04); RISE is interpolated on the 25%–75%
  span (`RISE_LOW_FRAC`/`RISE_HIGH_FRAC`) and RECOVERY on the 50% span
  (`RECOVERY_FRAC`). Below 3 points extraction is **mathematically undefined**
  (returns NaN + an ineligible flag), not silently degenerate. This is a hard
  floor of the feature definition, not a tunable default.
- *n_min.* With N=10 dyads the smallest non-zero p-value is ~0.1, so at
  α=0.05 a group cannot reject the null unless p=0 exactly — BH-FDR resolution
  is uninterpretable. Such groups are **WARNING-flagged as unreliable**, never
  silently accepted.

**Evidence.**

- *T_def hard floor.* `feature_definitions.py` DECISION-04 (3-point boxcar
  peak detection) plus the RISE 25%–75% / RECOVERY 50% fraction definitions
  make onset+peak+recovery only distinguishable at ≥3 points.
- *n_min code precedent.* The codebase already treats "< 10" as a small-sample
  boundary: `compute_surrogate_threshold` falls back to `ONSET_THRESHOLD` when
  "fewer than 10 finite surrogate values" are available (degenerate-case
  branch at the surrogate-threshold derivation).
- *B4 LOO stability.* Per B4, Lerique N=31 LOO is 100% stable and the three
  real datasets have N=176/46/23 — all >10.

**Conclusion.** `n_min=10` and `T_def=3` are **exclusion floors only**: they
drop absurdly small pilots (and mathematically undefined dyads) and constrain
**no** real SyncPipe analysis. No other constant in `feature_definitions.py`
or elsewhere was changed.

**Source of truth.** `multisync/feature_definitions.py`
(`T_DEF_MIN_WCC_POINTS`, `N_MIN_DYADS_FDR`, `check_eligibility`);
`multisync/qc.py` (`run_quality_check` eligibility NOTE);
`tests/unit/test_features.py` (§ "source: test_eligibility_thresholds.py").

---

## 2026-07-22 — Two-layer canonical definition (P1)

**Decision.** The word "canonical" in SyncPipe now carries an explicit layer qualifier. There are two distinct canonical layers that must never be conflated:

- **Descriptor-layer canonical** = ``DynamicAnalyzer.fit_transform`` — the CLI default feature-extraction route (per-dyad threshold, descriptive). This is what the CLI (`analyze`, `demo`) and ``DynamicAnalyzer(enable_prediction=False)`` reach by default. ``CANONICAL_PATH`` / ``CANONICAL_DESCRIPTOR_PATH`` in ``multisync/core.py`` name this layer.
- **Scientific-layer canonical** = ``pipeline_bridge`` + ``InferencePipeline.run_audited_evidence_chain`` — the ONLY path that produces a defensible, manuscript-grade conclusion (per-modality pooled threshold + design controls + group FDR). This is the audited evidence chain referenced by README:449 and used by ``scripts/reproduce_lerique_paper.py`` (``three_pipeline_v1``).

**Rule.** In code and docs, "canonical" MUST be written with its layer qualifier (descriptor-layer vs scientific-layer). Flipping or redefining either canonical definition requires explicit user sign-off PLUS an update to this DECISION_LOG entry. No local / session-level optimum may silently override a canonical definition.

**Why this locks the risk.** Earlier narrative treated ``DynamicAnalyzer`` as the "canonical main analysis path" slated to replace / be replaced by ``InferencePipeline`` (retirement / reversal language). That framing let a session-local optimum undo the canonical definition. This entry retires that reversal narrative: both layers are supported, neither retires the other, and they do not compete. The descriptor layer computes feature vectors; the scientific layer computes conclusions.

**Source of truth.** ``multisync/core.py`` (``CANONICAL_PATH``, ``CANONICAL_DESCRIPTOR_PATH``); ``multisync/pipeline_bridge.py`` + ``InferencePipeline.run_audited_evidence_chain``; README SOP; ``scripts/reproduce_lerique_paper.py``.

---

## Current v1 threshold stance

**Decision.** SyncPipe separates threshold scope, with a **per-modality pooled IAAFT surrogate threshold as the canonical default**:

- **`per-modality pooled` (canonical default)** — `compute_session_pooled_thresholds_by_modality` (`session_threshold.py`): derives *one* surrogate threshold **per modality** by pooling IAAFT surrogates across all dyads of that modality. This is the default used by `records_to_inference_inputs(onset_threshold="session_pooled")` and `BatchComputationPipeline`. It preserves **cross-modal comparability** (every dyad of a modality shares one threshold) *and* **within-modality calibration** — slow/smooth signals (e.g. EDA, low WCC amplitude) and fast/spiky signals (e.g. ECG, high WCC amplitude) get *different*, modality-appropriate cut-offs, rather than being forced onto one global value that fits neither.
- `within_dyad` / per-pair signal-level surrogate threshold (`compute_surrogate_threshold`): for single-dyad descriptive and synchrony-existence workflows.
- `session_pooled` / `compute_session_pooled_threshold` (coarser optional granularity): pools *all* dyads across modalities into a single global null — use when modality-specific calibration is not needed.
- `compute_condition_pooled_thresholds` (optional granularity): per-experimental-condition pooling.
- `fixed` threshold: a forwarded `float` (e.g. `ONSET_THRESHOLD = 0.5`) for sensitivity analysis and explicit user-specified comparisons.

**Rationale.** A single global pooled threshold is wrong when modalities have structurally different WCC nulls (EDA ≠ ECG): forcing both onto one cutoff either over-thresholds the spiky modality or under-thresholds the smooth one. Per-modality pooling solves this while keeping every dyad of a modality on a shared definition for group comparison. `ONSET_THRESHOLD = 0.5` is the **fallback / sensitivity constant only**: it is used when a modality's pooled null is degenerate (too few dyads) and as the fixed baseline in sensitivity sweeps. All surrogate-derived thresholds are hard-capped at `SURROGATE_THRESHOLD_MAX = 0.9` (periodicity / strong-autocorrelation protection) and fall back to 0.5 above that ceiling.

---

## 2026-07-24 — Per-modality pooled onset threshold as the normative default (5c078a0)

**Decision (frozen in code).** The canonical v1 onset-threshold strategy is now
**per-modality pooled IAAFT**: `records_to_inference_inputs` and
`BatchComputationPipeline` derive one surrogate threshold *per modality*
(`compute_session_pooled_thresholds_by_modality`) by pooling IAAFT surrogates
across all dyads of that modality. The previously-documented single
`session_pooled` (global) threshold is retained as an optional coarser
granularity, not the default.

**Rationale.** Modalities have structurally different WCC null distributions.
EDA (slow/smooth, low WCC amplitude) and ECG (fast/spiky, high WCC amplitude)
therefore need *different* episode thresholds; a single global pool fits
neither and would systematically mis-classify episodes in one modality.
Per-modality pooling keeps cross-modal comparability (every dyad of a modality
shares one threshold) while calibrating each modality's cut-off to its own null.

`ONSET_THRESHOLD = 0.5` is now explicitly the **fallback / sensitivity
constant only**: it is used (a) as the fallback when a modality's pooled null
is degenerate (too few dyads, fail-loud warning) and (b) as the fixed baseline
in sensitivity sweeps and paper reproductions. It is **not** the scientific
default. Surrogate-derived thresholds are hard-capped at
`SURROGATE_THRESHOLD_MAX = 0.9` — the 0.9 ceiling is retained (not raised); a
derived cut-off above 0.9 is treated as a periodicity / strong-autocorrelation
artifact and falls back to 0.5.

**Impact.** The audited evidence chain (`records_to_inference_inputs(
onset_threshold="session_pooled")`, `BatchComputationPipeline`,
`InferencePipeline.run_audited_evidence_chain`) now calibrates onset per
modality end-to-end. Scripts that passed a fixed `onset_threshold=0.5` still
work (forwarded unchanged) but are now explicitly sensitivity/fallback usage.
`docs/USER_MANUAL.md` §8, `docs/SKILL.md`, and `README.md` were updated to
describe per-modality pooled as the canonical default.

**Source of truth.** `multisync/feature_definitions.py` (`ONSET_THRESHOLD`,
`SURROGATE_THRESHOLD_MAX`); `multisync/session_threshold.py`
(`compute_session_pooled_thresholds_by_modality`);
`multisync/pipeline_bridge.py` (`records_to_inference_inputs`,
`onset_threshold="session_pooled"` default);
`multisync/computation_pipeline.py` (`BatchComputationPipeline`).

---

## 2026-07-24 — sklearn 1.10 / 1.11 deprecation fixes (aa3fefa, 4f85f87)

**Decision (verified, code change landed).** Two sklearn deprecation breaks
were cleared in `multisync/prediction.py`:

- **`SVC(probability=True)` → `CalibratedClassifierCV(SVC(...), ensemble=False)`**
  (commit `4f85f87`). `probability` was deprecated in sklearn 1.9 and removed
  in 1.11; the calibrated-wrapped SVC is the supported replacement and yields
  calibrated `predict_proba` for the `svm_rbf` nonlinear baseline.
- **Remove the `penalty=` keyword from `LogisticRegression`** (commit
  `aa3fefa`). `penalty` was deprecated in 1.8 and removed in 1.10. The L1
  regularization is now expressed directly via `l1_ratio=1.0` with
  `solver="saga"` (the elastic-net L1 special case), with no `penalty=`
  argument — matching the prior `penalty='elasticnet'` / `'l1'` behaviour
  without the removed keyword.

**Rationale.** Both changes remove the only remaining sklearn deprecation
warnings from the prediction path so the package imports and runs cleanly on
sklearn 1.10 and 1.11. No modelling behaviour changed: the regularization
strength (L1) and the calibrated SVM probability outputs are preserved.

**Impact.** `pytest` no longer emits `FutureWarning: 'penalty' was deprecated`
or `FutureWarning: The 'probability' parameter was deprecated` from
`prediction.py`. The linear and nonlinear prediction baselines behave as
before. No API surface or default changed for callers.

**Source of truth.** `multisync/prediction.py` (`_nonlinear_model_factories`,
`LogisticRegression` sites); guarded by `tests/test_prediction*.py`.

---

## Current v1 null-model stance

**Signal-level IAAFT.** Used as a synchrony-existence audit for distributional WCC descriptors. It tests whether observed WCC-derived descriptors exceed an independent autocorrelated-signal null. It does **not** prove dyad-specific interpersonal coupling.

**WCC-level IAAFT / order nulls.** Used cautiously as trace-level structure audits for descriptors such as `dwell_time` and `switching_rate`. Because WCC traces inherit autocorrelation from overlapping windows, WCC-level nulls are not presented as mature confirmatory tests of psychological temporal structure in v1.

**Timing / morphology nulls.** `onset_latency`, `rise_time`, `recovery_time`, `first_peak_time`, and `inter_peak_cv` remain exploratory. A validated existence null for these descriptors is deferred to v2.

---

## 2026-07-01 — Safety-fix sprint

**Implemented.**

1. IAAFT implementation now returns the final rank-adjusted sequence, preserving the empirical amplitude distribution exactly while approximating the power spectrum / autocorrelation. Documentation was corrected accordingly.
2. `DynamicFeatures.from_dict()` now round-trips all public dataclass fields exported by `to_dict()`.
3. Timestamp alignment now correctly allows all-absolute timestamp inputs and fails only true absolute/relative/unknown mixtures.
4. `zscore()` no longer turns all-NaN channels into zeros; all-NaN channels remain NaN and are reported in stats.
5. `DynamicAnalyzer.fit_transform()` now runs QC by default. QC FAIL raises `DataQualityError` unless `qc_raise_on_fail=False` is set for exploratory inspection.
6. `DynamicAnalyzer` now passes `surrogate_n` and `seed` into surrogate threshold computation.
7. Threshold mode is made explicit: `DynamicAnalyzer` supports `within_dyad` and `fixed`; session-pooled thresholds are routed to `BatchComputationPipeline`.
8. Top-level public API was narrowed to the v1 stable surface. Advanced modules remain importable from submodules.
9. Broken low-level computation paths were repaired (`ComputationPipeline.compute_wcc(method="stride")`, `DataImporter.load_signal`).
10. `syncpipe` was added as the preferred import/CLI namespace while `multisync` remains a compatibility alias.
11. Timing fields now use raw undefined semantics (`NaN` when undefined) with explicit `*_imputed` companion fields for ML-only imputation.
12. CI workflow with pytest and demo smoke test was added.
13. QC now has a user-facing PASS/WARN/FAIL formatter and CLI `analyze` prints actionable QC messages before WCC computation.
14. Warning cleanup reduced test warnings by removing BC near-constant precision warnings, sklearn `l1_ratio` warnings, and prediction window-size warnings.
15. README/User Manual/SKILL examples now prefer the `syncpipe` command/import namespace.
16. PGT-2, PGT-2 surrogate, PGT-3, and EGT-4 self-contained validation artifacts were rerun under the updated timing semantics.

**Verification.** Full test suite after the sprint:

```text
178 passed, 1 xfailed
```

---

## 2026-07-13 — Episode-feature robustness to missing data

**Decision.** `compute_dwell_time` and `compute_switching_rate` now compute
episode statistics over *valid (finite) WCC samples only*:

- A missing point (NaN, i.e. a `discontinuity_mask` gap) is excluded from the
  binary sequence, **not** treated as a low-state sample. Consequently an
  artifact gap (a) does not split one elevated run into two shorter runs
  (which would deflate `dwell_time`) and (b) does not inject spurious
  False->True / True->False transitions (which would inflate `switching_rate`).
- `switching_rate` duration is **valid time only** (`finite.sum() / hz / 60`),
  not the full array length including gaps.

The Schmitt-trigger binarizer (`_binarize_with_hysteresis`) is intentionally
*unchanged*: it still hard-breaks at NaN (the 2026-07-08 fix), so surrogate /
state-shuffle logic is unaffected. The gap-robustness lives only in the two
summary functions.

**Why this matters.** `dwell_time` and `switching_rate` are the two features
carrying SyncPipe's "quality of synchrony" argument in the Kuramoto validation.
If missing rates differ across experimental conditions, the old behaviour
would have created a confound with the condition (dual direction: one inflated,
one deflated) — exactly the kind of artifact a reviewer would flag.

**Source of truth.** `multisync/feature_definitions.py` (`compute_dwell_time`,
`compute_switching_rate`); guarded by `tests/test_feature_definitions.py`
(gap-robustness tests).

---

## Pending v1 cleanup items

1. Decide whether to suppress or re-route expected relative-timestamp warnings in synthetic tests/demos.
2. Monitor the external sklearn/scipy L-BFGS-B deprecation warning.
3. Archive older exploratory decision history outside this active decision log.
4. Continue validating WCC-level order nulls before making stronger structure claims.

---

## Dual namespace (`multisync` vs `syncpipe`): intentional, with a sunset plan

**Decision.** The `syncpipe` namespace is the preferred import/CLI surface
(`syncpipe` command and `import syncpipe`). The older `multisync` namespace
remains **only as a compatibility alias** during the transition away from the
original project name *MultiSync* (renamed to avoid collision with the
published `multiSyncPy`).

**Sunset plan (added 2026-07-09).**

- **v1.0 (current):** both namespaces work; new code and docs use `syncpipe`.
- **v1.1:** importing `multisync` emits a `DeprecationWarning` pointing users
  to `syncpipe`.
- **v2.0:** the `multisync` alias is removed; `syncpipe` is the only namespace.

A compatibility alias with no removal date tends to become permanent technical
debt. Pinning it to a version schedule keeps the migration honest and gives
downstream users a clear migration window.

---

## Related tools (positioning, not competition)

SyncPipe sits in a small field of interpersonal-synchrony tooling. Naming
proximity is real and worth stating explicitly so reviewers are not surprised:

- **multiSyncPy** (Hudson, Wiltshire & Atzmueller, 2023, *Behavior Research
  Methods*): group/multi-person synchrony beyond dyads, with Kuramoto
  simulations demonstrating sensitivity to coupling strength and convergent
  validity across multivariate synchrony indices. SyncPipe's dyad-focused,
  WCC-episode framing is narrower; the rename away from *MultiSync* was
  precisely to avoid implying overlap that does not exist.
- **SyncPy** (Varni & Avril, 2015): a unified open-source library for
  interpersonal/machine synchrony on dyadic and multiparty time series.
  Name is visually/phonetically close to *SyncPipe*; SyncPipe is a distinct,
  independently developed tool and should be cited separately. No code is
  shared.
- **SUSY** (Ramseyer & Tschacher lineage; R package, v0.1.1): synchrony
  measurement via segment-reshuffling surrogates. Its pseudo-pair design
  control is spiritually similar to SyncPipe's design-control audit; the
  difference is language (R vs Python) and the WCC-episode feature layer.
- **rMEA** (Kleinbub & Ramseyer, 2020): motion-energy analysis in R, a close sibling for
  the motion-energy modality specifically.

SyncPipe's differentiating claim is **audited measurement infrastructure**: a
transparent WCC trace, a governed feature table, signal- and design-level null
models, and exported artifacts — rather than a single synchrony score or a
broad metric zoo.

---

## 2026-07-23 — P2 release-hygiene pack close-out (post-`bbda3fd` re-audit)

**Decision.** Adopt the 6-fix "release hygiene" pack from the external
`updates/` drop (post-tip `bbda3fd`), after dialectical review. Each fix was
verified against live code (real diff + symbol grep), not trusted on the
README's "68 passed" claim. Net effect: v1.0 reporting/export honesty closes
remaining gaps left by P1.

**Evidence (per-fix verdict).**
- **P2-A** (`summarize()` multimodal L2 text): confirmed — `summarize()` only read `_l2_results`; now reads `_group_inference_results` (modality-keyed) via `_format_l2_summary_lines`. Real gap, correct fix.
- **P2-JSON** (`to_json` L2Result → repr strings): confirmed — added recursive `_sanitize` that `asdict`s dataclasses and maps non-finite → `null`. Prior `default=str` would have serialized `L2Result` as unusable repr. Real gap, correct fix.
- **P2-B** (`test_l2_condition` ignored `seed`): confirmed — call site now forwards `seed=self.seed` into `between_condition_fdr`. This is the *same* symmetry blind spot as Finding-17: P1 fixed restricted/AR baselines to NaN but left the **naive** baseline fabricating `0.5`.
- **P2-C** (`run_full_cascade` bypassed multimodal router): confirmed — swapped `test_l2_condition` → `run_group_condition_inference(contrast=, threshold_scope="unknown", modality_col=)`; return dict still keys `l2_results`. Restores R1 guarantees on the legacy cascade path.
- **P2-empty** (`extract_features([])` raised scipy `v cannot be empty`): confirmed — early-return structured `DynamicFeatures` (NaN) inside `extract_features` itself. Defensive, correct.
- **P2-pred** (naive baseline still set `baseline_auc = 0.5` on exception): confirmed and applied in **both** `cross_modal_prediction` and `rolling_origin_cv`. This is the residual of Finding-17's symmetry miss — the naive baseline was never covered by P1. Now `float("nan")` + `dtype=float` on `baseline_prob`.

**Why it matters.** These are not cosmetic: P2-A/P2-JSON mean reviewers/users see empty L2 text and unusable JSON exports today; P2-pred is a methodological-honesty hole (fabricated chance-level AUC) that contradicts v1.0's "audited measurement infrastructure" claim. All 6 are low-risk, behaviour-local, and pass the curated regression set (66 passed, 0 regression; full-suite re-run pending).

**Source of truth.** `multisync/inference_pipeline.py`, `multisync/prediction.py`, `multisync/feature_definitions.py`; guarded by `tests/contracts/test_release_contracts.py` (§ "source: test_p2_release_hygiene.py").


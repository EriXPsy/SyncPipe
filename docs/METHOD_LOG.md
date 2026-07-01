# SyncPipe Decision Log
---

## Reversal Protocol

DECISION-xx reversal requires: (1) log proposal, (2) evidence (Tier 1: synthetic+2 real datasets), (3) impact assessment, (4) co-sign by ≥1 external reviewer, (5) archive old artifacts.

---

## 2026-05-31 · DECISION-14b — HSMM segmenter

**Decision**: Add HSMM epoch segmenter for paradigm-matched extraction (GT-3c/GT-3d). NOT universal WCC replacement.
**Rationale**: WCC = paradigm-free episodes; HSMM = paradigm-structured segments. Complementary.
**Evidence**: GT-3c (event): WCC ρ=0.83 vs HSMM ρ=0.56 → WCC better for event paradigms. GT-3d (continuous): HSMM ρ=0.76 vs WCC ρ=0.56 → HSMM better for continuous multi-regime.
**Impact**: Paradigm-matched routing: timing family → WCC/PLV; switching/entropy → HSMM.

---

## 2026-05-31 · DECISION-14 — Timing family validity

**Decision**: Timing family (onset/recovery) valid under parameterized GT (GT-3b: ρ=0.86/0.85). GT-2 "timing failure" reclassified as GT design ceiling (no onset/rise GT knobs).
**Rationale**: GT-2 generator hardcodes rise slope; no GT-parameterizable timing → ρ≈0 is mathematical necessity, not feature failure.
**Evidence**: GT-3b convolutional GT with onset_delay/tau_rise/tau_decay: onset_latency ρ=0.86, recovery_time ρ=0.85.
**Impact**: Timing family retains confirmatory status. GT-3b replaces GT-2 for timing validation.

---

## 2026-05-30 · DECISION-13 — synchrony_entropy diagnostic exclusion

**Decision**: synchrony_entropy stays diagnostic (not confirmatory). Exclusion rationale: cross-pair surrogate produces Type I+II error pattern (Bizzego: entropy H=44 false-positive, mean_synchrony NS false-negative).
**Rationale**: Entropy passed benchmarks (p=0.0008) but surrogate diagnosis reveals cross-pair alone insufficient for multi-feature frameworks.
**Impact**: Documents entropy exclusion rationale for reviewers. Methodological contribution: surrogate-diagnosis detects framework limitations.

---

## 2026-05-30 · DECISION-12 — Feature family taxonomy (4 orthogonal)

**Decision**: 8 features → 4 orthogonal families via PCA (Bizzego N=193, 75% variance).
**Families**: (1) Intensity (peak_amp, mean_sync, switching_rate); (2) Amplitude-variability (peak_amp, dwell_time, entropy); (3) Temporal-dynamics (onset, rise); (4) State-richness (entropy, switching, dwell).
**Implication**: Onset/rise load on PC3 (temporal) → low power on coupling-strength axis expected. Need σ/τ probe in GT to validate temporal family.
**Impact**: Paper narrative: 6 features → 4 operationalized dimensions.

---

## 2026-05-30 · DECISION-11 — WCLC alternative metric

**Decision**: Add WCLC synchrony() as alternative metric (metrics.py). WCC remains default.
**Rationale**: BM2 showed WCC=0% detection for switching_rate under linear coupling; WCLC=100%. WCLC captures cross-lagged leader-follower dynamics that 0-lag Pearson r misses.
**Evidence**: BM2: WCLC switching_rate detection=100% vs WCC=0% (linear coupling); WCLC also detects switching at 47% (lagged), 67% (nonlinear). WCC outperforms WCLC on intensity features (peak_amp: 100% vs 33%).
**Impact**: WCLC recommended for switching_rate in structured leader-follower paradigms. PLV recommended for phase-dominated signals.

---

## 2026-05-24 · DECISION-10 — Prediction feature baseline + leakage threshold

**Decision**: prediction.py main features = 6 epoch (CONFIRMATORY_FEATURES). mean_synchrony = AR baseline (external channel). LEAKAGE_DELTA_AUC_THRESHOLD = 0.30 (SSoT).
**Rationale**: Old 10-feature schema leaked (mean_synchrony in main matrix). New 6-epoch schema + AR baseline eliminates trivial leakage. 0.30 threshold calibrated vs sine-wave ceiling (0.366) and white-noise floor (≈0).
**Sub-decisions**:
- 10X1: Cross-modal ablation = Granger-style (drop source AR only, keep target AR).
- 10T1: Keep T1 (drop source AR only) — preserves Granger asymmetry.
**Impact**: prediction.py SSoT-compliant. Leakage audit: sine delta_AUC=0.366 > 0.30 threshold → flag.

---

## 2026-05-24 · DECISION-09 — Confirmatory vs Diagnostic Partition

**Decision**: FDR family = 6 confirmatory (onset_latency, rise_time, peak_amplitude, recovery_time, dwell_time, switching_rate). Diagnostics = 2 (mean_synchrony, synchrony_entropy), NOT in BH-FDR.
**Rationale**: DECISION-06 removed entropy/synchrony from main set; family partition was implicit in code. SSoT formalization prevents AI silently expanding family size.
**Impact**: SSoT constants: FEATURE_FAMILY, CONFIRMATORY_FEATURES, DIAGNOSTIC_FEATURES in feature_definitions.py. All FDR-aware code must import from SSoT.

---

## 2026-05-23 · v0.x.0 Methodology Lock-In

**Decisions**:
- DECISION-01: ONSET_THRESHOLD = 0.5 (Cohen's d large). Reject MAD-driven adaptive thresholding (violates measurement invariance).
- DECISION-02: onset_latency = first sustained crossing (baseline → elevated K samples). Reject "first ≥ threshold" version.
- DECISION-03: rise_time = 25%–75% quartile (Boucsein 2012). Reject "onset → peak" version (confounds with threshold).
- DECISION-04: peak_amplitude = 3-point boxcar smoothed max. Reject bare max (single-sample spike vulnerability).
- DECISION-05: recovery_time = half-recovery (Boucsein 2012). Reject "full recovery to baseline" (systematic NaN in high-coupling).
- DECISION-06: Replace {synchrony_entropy, mean_synchrony} with {dwell_time, switching_rate}. Dwell+switching > Shannon entropy (2D interpretability, no information loss).
- DECISION-07: Primary surrogate = IAAFT. FT surrogate = robustness check.
- DECISION-08: Dominant peak index shared across rise/recovery/peak. Onset decoupled.

**Impact**: 6 confirmatory + 2 diagnostic locked. SSoT in feature_definitions.py.

---

## 2026-05-27 · R4-LERIQUE-CLOSURE

**Decision**: Run Lerique 2024 (3 modalities × 2 conditions) with IAAFT surrogate.
**Rationale**: Complete 4-dataset cross-protocol validation (Han/Andersen/Gordon/Lerique).
**Key result**: 4/48 cells significant (FT surrogate ∩ IAAFT = 100% concordance). Cross-dataset FT-IAAFT agreement = 98/100 = 98.0%.
**Impact**: Lerique = Pattern B (stim-locked). Boundary-aware WCC masking implemented (NaN-ratio guard = 0.6).

---

## 2026-05-27 · FT-SURROGATE-COMPLETE

**Decision**: FT surrogate补跑 completed (Andersen 12' + Gordon 5' + Han 7').
**Key result**: Sig-agreement = 50/52 = 96.2%. FT surrogate stricter on switching_rate (2/52 cells).
**Impact**: DECISION-07 lock-in empirically supported. FT surrogate as robustness check validated.

---

## 2026-05-27 · R4-IAAFT-Andersen-Gordon

**Decision**: IAAFT surrogate for Andersen + Gordon completed.
**Key result**:
- Andersen: 5/6 confirmatory survive IAAFT (switching_rate reverse-direction finding: real > surrogate, opposite to Pattern A prediction).
- Gordon angular: anti-phase Pattern A' confirmed (real mean_sync=-0.21 vs surrogate≈0).
- Gordon radial: upgraded from Pattern B to "A-weak" (weak coupling + strong stim).
**Impact**: Pattern A (Andersen) survives 5-test battery. Pattern A' (Gordon angular) confirmed. Switching_rate reverse direction = publishable mechanistic refinement ("intermittent coupling").

---

## 2026-05-27 · R4-IAAFT-Han

**Decision**: Han IAAFT completed. Han reclassified from Pattern C to "C + A-hidden".
**Rationale**: Cross-pair surrogate preserves trace shape AND partner identity → cannot detect true coupling component. IAAFT preserves only cross-spectral coherence → exposes hidden true coupling.
**Key result**: 4/6 confirmatory + 2/2 diagnostic passed. rise_time/switching_rate fail due to 1Hz measurement resolution floor (not Pattern A refutation).
**Impact**: Han has true coupling component (magnitude ~0.15) invisible to cross-pair surrogate.

---

## 2026-05-26 · Andersen-dose-response + group-mixed-effects + event-locked

**Decision**: Three independent falsification tests for Andersen Pattern A.
**Key result**:
- Dose-response: 10/52 hits (close_count/arousal drive sync features, direction matches Pattern A).
- Group mixed-effects: 5/6 features survive Group random intercept (attenuation ≤11%,多数 negative = effect strengthens after group control).
- Event-locked proxy (HR-derived): 8/8 features significant (p < 1e-46).
**Impact**: Pattern A survives three independent falsification tests. Andersen = most validated cell in 4-pattern taxonomy.

---

## 2026-05-26 · Gordon-cross-protocol-diagnosis

**Decision**: Gordon angular = Pattern A' (anti-phase). Gordon radial = Pattern B (stim-locked).
**Rationale**: Angular velocity WCC shows anti-phase dyad-specific coupling (real mean_sync=-0.21 vs surrogate≈0). Radial distance WCC shows stim-locked (real ≈ cross_dyad).
**Key result**: Per-lag pattern: angular = sharp step at lag=0 (anti-phase), radial = smooth tent (stim-locked).
**Impact**: Pattern A upgraded to A ∪ A' (positive-sign + anti-phase). Taxonomy: A (Andersen + Gordon angular), B (Lerique + Gordon radial), C (Han), D (pending).

---

## 2026-05-26 · Andersen-cross-protocol-diagnosis

**Decision**: Andersen = Pattern A (true dyad-specific coupling).
**Key result**: cross_group_pseudo NS rejected (p<1e-17 for 4/6 features). Time-shift NS rejected (p<1e-26). Per-lag: smooth + flat.
**Impact**: 4-pattern taxonomy closed (A=Andersen, B=Lerique, C=Han, D=pending). Taxonomy has empirical exemplars for 3/4 cells.

---

## 2026-05-25 · Han-cross-protocol-diagnosis

**Decision**: Han = Pattern C (trace-shape autonomous similarity).
**Key result**: cross-dyad surrogate NS at all 3 levels. Time-shift: only peak_amplitude+dwell_time significant (partial). Per-lag: monotonic decay (no sawtooth).
**Impact**: Taxonomy雏形: A (Andersen), B (Lerique), C (Han), D (ARMA-noise theoretical position). Surrogate framework discriminates 4 synchrony sources.

---

## 2026-05-25 · Lerique-surrogate-controls

**Decision**: Cross-dyad pseudo-pair + within-dyad time-shift surrogate for Lerique.
**Key result**: Pseudo-pair: 0/4 significant. Time-shift: 4/4 significant. Per-lag: sawtooth dip at ±60s (trial period).
**Interpretation**: Lerique trial > rest elevation explained by stim-locked shared driver (trial-onset evokes shared physiological response), not dyad-specific coupling.
**Impact**: Pattern B (stim-locked) validated. Surrogate diagnosis framework established.

---

## 2026-05-25 · DynamicAnalyzer-enable-prediction-flag

**Decision**: Add enable_prediction flag (default=True, backward compatible).
**Rationale**: Surrogate/dose-response/trial-level scripts don't need prediction CV (wastes ~50-80% runtime).
**Impact**: Scripts opt-in enable_prediction=False for descriptive-only analysis.

---

## 2026-05-25 · Lerique-dose-response-followup

**Decision**: Rest2/3/4 pooling validated (slope p > 0.25, pairwise p > 0.05).
**Rationale**: Carry-over reaches steady state by first post-block rest (R2), does not accumulate further. Pooling strategy valid.
**Impact**: Pre-reg pooling strategy post-hoc validated. EDA peak_amplitude stable across session (possible "trait" interpretation, needs surrogate validation).

---

## 2026-05-25 · Hackathon-MiraclePlus-2026-submission

**Decision**: Scope lock for hackathon submission (external communication, NOT method change).
**Key constraints**: Partial results NOT confirmatory; no "Earth to Moon" deployment claims; no core SSoT modification.
**Impact**: External communication scope bounded. Lerique full run remains critical path.

---

## 2026-05-25 · Lerique-interim-N10-observation

**Decision**: Interim partial results (N=10) observed, NOT final. Full run (N≈28) mandatory before declaring Level A.
**Key result**: 3/18 main contrasts significant (direction: Trial > Rest). Consistent with pre-reg.
**Caveat**: Sample not random (dictionary-order prefix). N=10 small. Full run may regress.
**Impact**: Prioritize full run. Partial results鼓舞但不构成 confirmatory finding.

---

## 2026-05-25 · Lerique-batch-pipeline-ready

**Decision**: Batch pipeline + FDR scope locked. Main contrast = rest1 → trials_concat (18 tests, BH-FDR within 3 modalities). Sensitivity/reference = raw p only (NOT in FDR).
**Impact**: SSoT (CONFIRMATORY_FEATURES) → DyadResult → batch CSV column names aligned. FDR family = 18 tests (3 modalities × 6 features).

---

## 2026-05-25 · Lerique-preproc-smoke-pass

**Decision**: Lerique preprocessing smoke test 6/6 PASS. Three modalities implemented (ECG IBI, EDA, RESP).
**Rationale**: Preproc protocol locked (§1.4). ECG: Butterworth 5-20Hz + neurokit2 ecg_peaks. EDA: 0.05-5Hz. RESP: 0.1-1Hz.
**Known quirk**: EDA/RESP resample length = +1 sample vs ECG (scipy resample_poly ceiling). Does not affect current pipeline (each modality independent). Future cross-modal operation needs off-by-one fix.
**Impact**: Lerique batch analyze gate OPEN.

---

## 2026-05-25 · Lerique-rest1-length-heterogeneity

**Decision**: MIN_DURATION_SEC = 60s hard threshold (policy A+). Drop records with < 60s raw trace.
**Rationale**: WCC window=10s/step=5s → 60s segment produces ≥11 windows. 4 windows hard floor → 30s. 60s leaves ≥2× safety margin.
**Impact**: pce09 (159s) and pce26 (170s) both retained ( > 60s). Pre-reg §1.3 updated with three-policy comparison table.

---

## 2026-05-24 · Lerique-meta-correction

**Decision**: Lerique metadata correction (Fs / segment duration / Rest heterogeneity / three-condition-units).
**Cor corrections**: Fs=1000Hz (not 600Hz). Rest=180s (not 300s). Trial=60s (not 100s). Rest1 ≠ Rest2/3/4 (pre-task vs post-block).
**Rationale**: Previously inferred from sample counts without reading PDF. Violated "facts before inference" discipline.
**Impact**: Pre-reg §1.2/1.3/1.4 updated. Condition units: rest1 / rest_postblock / trials_concat (not binary Rest/Trial).

---

## 2026-05-24 · Synthetic-grid-no-extension

**Decision**: NOT extend synthetic grid (c=0.4/0.5, noise=1.5/2.0). Resources → Lerique real data pilot.
**Rationale**: c=0.3 → power=0%; c=0.6 → power=100% monotonic curve. Threshold bounded (0.3, 0.6). Adding tail noise (1.5/2.0) only paints floor. Lerique pilot higher information value.
**Impact**: GT-1/GT-2 finalized. Grid extension deferred until Lerique+P3 both show no signal.

---

## 2026-05-24 · P2-pilot-launch

**Decision**: Lerique-47n3p selected for P2 first. Bizzego deferred.
**Rationale**: Lerique: 3 modalities complete, 27/31 dyads full data, dyadic-task confirmed. Bizzego: IBI 2Hz + 0.04Hz LP mismatches SyncPipe 1Hz + 30s WCC. Methodologically incompatible.
**Impact**: P2 dataset = Lerique 2024. Bizzego retained as robustness dataset (post-Lerique).

---

## 2026-05-24 · GT-1 / GT-2 full run

**Decision**: GT-1 (power curve) + GT-2 (null FWER audit) completed. Results archived.
**Key result**:
- GT-2: 56 cells, all pass (Wilson 95% CI lower bound ≤ q=0.05).
- GT-1: peak_amplitude = strongest detector (c=0.6, noise=0.05 → 100%). dwell_time/recovery_time = secondary signal. onset/rise = no power (known caveat: sustained-elevated traces lack baseline phase).
**Impact**: METHODLOGY_LOCK_IN.md updated with GT-1/GT-2 chapters. FWER control verified.

---

## 2026-05-24 · R-C — Summary schema long-table

**Decision**: summarise_level1 output changed from wide to long format. Family partition explicit in schema.
**Rationale**: Wide format invisibly mixed confirmatory/diagnostic columns. Long format adds "family" column (categorical: confirmatory/diagnostic).
**Impact**: Old level1_summary.csv deprecated. New schema: (coupling, feature, family, mean, sd, n_seeds, onset_threshold). summarise_definedness() added.

---

## Pre-Lock-In Period (≤ 2026-05-22)

All commits/code/artifacts prior to 2026-05-23 classified as *pre-lock-in draft period*. NOT eligible for external citation (papers/grants/preprints). Methodological iterations viewed as internal learning process.

---

*Refactored by Ponytail principles: decisions + brief rationale only. Trigger narrative, detailed implementation, verbose results, and "next steps" removed.*

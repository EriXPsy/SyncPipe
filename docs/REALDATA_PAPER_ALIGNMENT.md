# SyncPipe vs. five published synchrony papers — method & conclusion alignment

> Purpose: answer the author's core worry — "is SyncPipe a wishful-thinking
> reimplementation, or does its method and conclusion direction align with the
> field?" This document extracts, **verbatim from the five PDFs in the repo
> root** (`Andersen.pdf`, `Bizzego.pdf`, `Gordon.pdf`, `Han.pdf`, `Lerique.pdf`),
> each paper's signal, analysis rate, synchrony estimator, null model, and
> headline conclusion, then maps each onto SyncPipe and states whether the
> direction agrees.
>
> It is a **methodology alignment audit**, not a set of unit tests. The
> reproducible "does the direction agree" checks live in `scripts/`
> (`verify_bizzego_convergence.py`, `verify_realdata_consistency.py`,
> `realdata_l2_audit.py`).

---

## 0. One-sentence synthesis

SyncPipe's three null models are **the field's three null models**, each with
direct precedent: the **IAAFT surrogate** (Bizzego 2020), the **circular/time
shift surrogate** (Andersen 2024), and the **segment shuffle** (SUSY /
Tschacher lineage). SyncPipe is not a new statistics; it is a **systematisation**
of the null models the field already trusts, layered with a descriptor table and
a group-inference layer the individual papers lack.

---

## 1. Bizzego et al. 2020 — *Behav. Sci.* 10(1):11

| | Paper | SyncPipe |
|---|---|---|
| Signal | ECG → **IBI** | ECG → IBI (`_preprocess_ecg`) |
| Analysis rate | **2 Hz** (low-pass 0.04 Hz, z-scored) | **1 Hz** (configurable `target_fs`) |
| Estimator | **max cross-correlation within ±10 s** (lag 1 s) | zero-lag WCC (`peak_amplitude`) |
| Null | **IAAFT surrogate** (+ 5 s moving avg) | **IAAFT surrogate** (signal-level existence audit) |
| Design controls | co-presence / stimulus / surrogate distributions | pseudo-pair / time-shift / existence |

**Headline conclusion.** Synchrony varies by relationship type and emotional
state; strangers showed *greater* synchrony than friends/romantic partners.

**Alignment.** The **null model is identical** (IAAFT — SyncPipe inherited it
from Bizzego's `pyphysio`/`physynch` lineage). Two scope differences: (a) 1 Hz
vs 2 Hz analysis rate — both Nyquist-sufficient for IBI/HRV envelopes, a
degree-of-conservatism choice (documented in `METHOD_LOG.md` §7e); (b)
zero-lag vs ±10 s max-CC — SyncPipe v1 declares zero-lag a scope boundary
(lead-lag is v2). These are **deliberate scope limits, not methodological
divergence**. The convergence check is `scripts/verify_bizzego_convergence.py`.

---

## 2. Andersen et al. 2024 — "Scared Together" (*Cognition and Emotion*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | ECG (1000 Hz) → **HR at 1 Hz** | ECG → IBI envelope |
| Estimator | **Pearson ISC** of HR (time-aligned) | WCC (zero-lag) |
| Null | **circular temporal shifts** (break time alignment) | **time-shift design control** |
| Closeness link | HR synchrony ↑ with social closeness | (not a relationship-type design in Lerique) |

**Headline conclusion.** HR synchrony emerged robustly above time-shifted
surrogates, and tracked social closeness / arousal.

**Alignment.** Andersen's **circular temporal shift null is SyncPipe's
`time_shift` design control**, applied per-dyad rather than cohort-level. The
"is the alignment doing the work?" question SyncPipe's design-control audit
asks is *exactly* Andersen's null. Same signal family (HR from 1000 Hz ECG),
same 1 Hz analysis rate.

---

## 3. Han et al. 2022 — skin-conductance synchrony (*Communication Methods and Measures*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | EDA, **2000 Hz** | EDA bandpass → 1 Hz envelope |
| Estimator | **Cross-recurrence quantification (CRQA)** — nonlinear | WCC (linear) |
| Null | (CRQA surrogate / determinism vs chance) | IAAFT / block-shuffle |

**Headline conclusion.** Calm > arousing, negative-arousing > positive-arousing,
and fast > slow change produced stronger, more deterministically structured,
more stable SCL synchrony.

**Alignment.** **Partial, and honestly flagged.** Han uses a *nonlinear*
recurrence-based estimator; SyncPipe v1's default substrate is *linear* WCC.
CRQA is on SyncPipe's "alternative substrates, each with its own null" roadmap
(METHOD_LOG §8.5), not yet implemented. The *conclusion direction* SyncPipe can
check is "does arousal/valence/change-rate structure SCL synchrony" — the
existing `REALDATA_COMPARISON.md` reports Han's switching/entropy features as
highly significant (emotion → synchrony), agreeing in direction even though the
estimator differs. This is the one paper where SyncPipe should be explicit
that it is **not yet** methodologically equivalent.

---

## 4. Mayo & Gordon 2025 — contextual pulls (*American Psychologist*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | IBI (spline **500 ms = 2 Hz**) + EDA | IBI/EDA envelopes |
| Estimator | IBI synchrony, EDA synchrony | WCC descriptors |
| Design | 2×2 within-subject (sync pull × seg pull) | condition contrast |

**Headline conclusion.** IBI synchrony was *positively* associated with
performance/cohesion in low-segregation contexts and *negatively* in
high-segregation contexts; EDA synchrony showed the converse pattern.

**Alignment.** Direction-dependent on context — this is exactly the
"contextual pull" idea SyncPipe's per-modality, per-condition reporting is
built for. SyncPipe has a **calibrated Gordon simulation**
(`simulation/gt5_gordon_conditions.py`, parameters taken from Gordon's own
`Full_Data.csv`) and `scripts/run_gordon_case_study.py`. The existing
`REALDATA_COMPARISON.md` flags a Gordon `peak_amplitude` direction discrepancy
(short 18–22-point traces) that is a **data-limitation finding to report**, not
a tool failure.

---

## 5. Lerique et al. 2024 — ECSU-PCE dataset (data descriptor)

| | Paper | SyncPipe |
|---|---|---|
| Role | **Data descriptor** (no synchrony method) | the analysis tool |
| Sampling | ECG/EDA/respiration **1000 Hz** (BrainVision), notch 60 Hz | `RAW_FS_HZ = 1000` |
| Analysis rate | not specified (left to downstream users) | 1 Hz (`TARGET_FS_HZ`) |

**Alignment.** Lerique is a *dataset*, not a method — there is no Lerique
synchrony estimator to converge with. SyncPipe's loader
(`realtest/lerique_2024.py`) consumes the 1000 Hz `.mat` and derives the 1 Hz
envelopes. The "does SyncPipe produce sane results on Lerique" check is
`scripts/verify_realdata_consistency.py` (peak_amplitude significant in all
three modalities).

---

## 6. Summary table

| Paper | Null model | SyncPipe equivalent | Direction agrees? |
|---|---|---|---|
| Bizzego 2020 | IAAFT surrogate | signal-level IAAFT existence audit | ✅ same null |
| Andersen 2024 | circular temporal shift | time-shift design control | ✅ same null |
| Han 2022 | CRQA (nonlinear) | *not yet implemented* (WCC is linear) | ⚠️ direction only |
| Gordon 2025 | (contextual × IBI/EDA sync) | condition contrast + calibrated sim | ✅ design-aligned |
| Lerique 2024 | (data descriptor) | loader consumes raw | ✅ input-aligned |

**Bottom line.** SyncPipe is **convergent with the field's null models**
(Bizzego's IAAFT, Andersen's time-shift, SUSY's segment-shuffle) and divergent
only where it *deliberately* narrows scope (zero-lag in v1) or *extends* the
toolkit (descriptor table, design-control layer, group inference). The one
honest gap is Han's nonlinear CRQA, which SyncPipe lists as future work rather
than pretending equivalence.

---

## 7. A divergence the convergence check surfaced: signed vs absolute intensity

`scripts/verify_bizzego_convergence.py` asked whether SyncPipe's
`peak_amplitude` is the zero-lag special case of Bizzego's max-CC. It is **not
exactly** — and the reason is substantive, not a bug:

- **SyncPipe `peak_amplitude`** = *signed* maximum of the smoothed WCC trace
  (`np.nanargmax`), i.e. the strongest **positive** correlation episode.
- **Bizzego max-CC** = maximum of the **absolute** cross-correlation over
  ±10 s, i.e. anti-phase (negative) correlation counts toward synchrony
  magnitude.

On the committed Lerique traces, **88% carry a strong anti-phase segment
(min WCC < −0.3)** and 69% carry min WCC < −0.5. In the extreme
(`pce09 EDA rest1`), Bizzego's `max|WCC|` = 0.93 while SyncPipe's
`peak_amplitude` = 0.227 — a −0.93 anti-phase episode is simply not counted by
the signed descriptor.

**This is a real, defensible scope choice, and it must be stated — not left
implicit.** The signed reading aligns with the Gordon synchrony/segregation
axis that motivates this project: a *negative* WCC is plausibly **segregation /
anti-phase coordination**, not "stronger synchrony", so excluding it from the
*synchrony* intensity descriptor is theoretically coherent. But a researcher
coming from Bizzego (or SUSY's `ES_abs`) expects anti-phase to count.

**Decision needed (open).** One of:

1. Keep `peak_amplitude` signed and **document** the choice (current state) —
   defensible against Gordon, but will read as "under-reporting synchrony" to a
   Bizzego/SUSY reviewer unless the rationale is explicit.
2. Add an **absolute** companion descriptor (`peak_abs_amplitude`, the true
   zero-lag special case of Bizzego's max-CC) as an exploratory intensity
   feature, so both readings are reported.
3. Switch the primary to absolute (would diverge from Gordon's
   synchrony/segregation framing).

Option 2 is the field-safest: it makes the Bizzego-convergence claim
*literally* true (a `peak_abs_amplitude` equals max-CC at lag 0) while keeping
the signed descriptor for the segregation-aware reading.


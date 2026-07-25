# SyncPipe v1.0 — Claim Ceiling

> **Status:** Frozen Draft v1.0 (for maintainer ratification)  
> **Companion:** `V1_PROTOCOL.md`  
>
> This document states, explicitly and defensibly, what SyncPipe v1 **does**
> and **does not** claim. It is the ceiling that every result report,
> manuscript sentence, and external-user instruction must respect. "Exploratory"
> never silently becomes "confirmatory."

---

## 1. The claim we ARE allowed to make

> SyncPipe v1 provides a **standardized and auditable measurement procedure**
> for continuous same-modality dyadic synchrony, combining explicit input
> contracts, WCC-based descriptor extraction, signal-level null audits,
> design controls, eligibility governance, and reproducible reporting.

Concretely, v1 may claim:

1. A **pre-specified, two-condition, dyad-paired** comparison of a WCC-derived
   descriptor (principally `peak_amplitude`), under explicit governance.
2. That the comparison was conducted under **auditable input contracts** (hz,
   length, mask, column resolution) that **fail loud** rather than silently
   mislead.
3. That signal-level existence (L0) and WCC-level structure (L1) audits were run
   to separate co-fluctuation from noise.
4. That **definedness / eligibility / claimability** gates were applied, and
   which results are excluded from confirmatory claims.
5. That the same manifest+config yields **identical** results through the Python
   API and the CLI (parity).

---

## 2. The claims we are NOT allowed to make

| # | Forbidden claim | Why |
|---|---|---|
| 1 | "SyncPipe measures the *quality* of synchrony as a psychological construct." | No construct validity demonstrated; v1 is a measurement procedure, not a psychometric instrument. |
| 2 | "SyncPipe establishes interpersonal *coupling* / *causality*." | Positive WCC can arise from shared stimulus, drift, periodicity, artifact (null scenarios). L2 controls within-dyad confounds but not causality. |
| 3 | "Cross-modal synchrony is a unified construct." | v1 analyzes modalities as separate families; no unified cross-modal model. |
| 4 | "`dwell_time` / `switching_rate` are confirmatory endpoints." | Conditional-exploratory only; claimable solely when definedness sufficient & threshold policy explicit. |
| 5 | "WCC morphology descriptors reveal participant lead-lag." | They describe WCC trace shape, not interpersonal timing direction. |
| 6 | "All listed real datasets are fully reproducible validations." | Gordon/Andersen loaders incomplete; only `fully rerunnable` datasets earn that label. |
| 7 | "A significant L2 result generalizes beyond the studied modality/design." | No universal generalizability claim; scope is the locked v1 protocol. |

---

## 3. Claim tiers (per feature)

| Tier | Features | May appear in the v1 primary result |
|---|---|---|
| **Confirmatory primary** | `peak_amplitude` | Yes — condition contrast |
| **Reference** | `mean_synchrony` | Yes — descriptive comparator |
| **Conditional exploratory** | `dwell_time`, `switching_rate` | Only with definedness + explicit threshold policy; labeled exploratory |
| **Morphology exploratory** | `onset_latency`, `rise_time`, `recovery_time` | Descriptive only |
| **Diagnostic / excluded from FDR** | `bimodality_coefficient`, entropy, etc. | Appendix / audit only |

---

## 4. Exploratory → confirmatory rule

- An exploratory result **never** becomes confirmatory by itself.
- Promotion requires: a pre-specified estimand, a declared null, a passing
  definedness/eligibility gate, and an explicit protocol change (new Gate 0
  freeze).
- `claimable=False` results are reported but **excluded** from the confirmatory
  claim set.

---

## 5. Dataset claim classification (real-data)

Every real dataset used in v1 reporting must be labeled:

| Class | Definition | Example use |
|---|---|---|
| `fully rerunnable` | raw access + loader + preprocessing + manifest complete | primary reproducibility showcase |
| `artifact-backed` | result artifact exists; raw loader / license chain incomplete | supporting, labeled |
| `diagnostic only` | exploratory / historical; not in reproducibility claim | excluded from validation battery |

Gordon / Andersen currently fall in `diagnostic only` until loaders are complete.

---

## 6. Sentence templates (safe for manuscript)

- ✅ "Under the locked v1 protocol, `peak_amplitude` differed between the two
  pre-specified conditions (dyad-paired permutation, BH-FDR), with signal-level
  existence audited at L0/L1."
- ✅ "`dwell_time` showed a conditional-exploratory effect, reported under
  explicit threshold policy and definedness gating."
- ❌ "SyncPipe demonstrates that dyads with higher synchrony quality couple more
  strongly." (violates §2.1 and §2.2)
- ❌ "The cross-modal synchrony construct generalizes across EDA, ECG, and RESP."
  (violates §2.3)

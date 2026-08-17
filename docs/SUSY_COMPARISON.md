# SyncPipe vs SUSY — methodology mapping and cross-validation plan

> Goal: establish that SyncPipe is not a "wishful-thinking" reimplementation,
> but a measurement infrastructure whose core null models are the same
> machinery the mature field-standard tools (SUSY, rMEA, SUCO) already use —
> with correct citations, and an honest statement of what can and cannot be
> verified inside this repository.

---

## 1. What SUSY does (with the correct citations)

SUSY (**Sur**rogate **Sy**nchrony; Tschacher & Meier 2020,
doi:10.1080/10503307.2019.1612114; Tschacher & Haken) computes synchrony as
**windowed cross-correlation** and establishes significance with a
**segment-shuffling null**. Concretely (from the SUSY reference manual and the
convergent-validity review of Meier & Tschacher 2021, *Entropy*):

1. Split each dyad member's time series into **non-overlapping segments**
   (e.g. 30 s).
2. Compute the **cross-correlation function** up to `maxlag` within each
   segment, transform to Fisher's *Z*, and aggregate (mean *Z* within segment,
   then across segments).
3. Build a null by **randomly shuffling the segment sequence of each dyad
   member independently**, then recomputing the same statistic.
4. Report an **effect size** `ES = (observed − null_mean) / null_SD`
   (`ES_abs` for absolute correlations, `ES_noabs` for signed).

The critical methodological point: SUSY shuffles the **raw signal segments**
and then **recomputes** the cross-correlation. This is why the null is
informative — after shuffling, the two members' segments no longer align, so
the recomputed correlation genuinely drops for truly coupled dyads.

---

## 2. Mapping: SUSY ↔ SyncPipe

| SUSY component | SyncPipe equivalent | Notes |
|---|---|---|
| windowed cross-correlation (WCC) | `sliding_window_wcc` | identical substrate (zero-lag in v1; SUSY scans a small `maxlag`) |
| Fisher-*Z* aggregation | `mean_synchrony` (mean over finite WCC) | *Z* ≈ r for small r, so comparable |
| **segment-shuffling null** | `block_permutation_surrogate` | **literally the same operation** — permute contiguous blocks |
| effect size `ES` | `perm_effect_size` / `real_minus_pseudo_mean` | standardized difference from the null |
| condition/group inference | dyad-paired permutation + BH-FDR (L2) | SUSY has no group layer; SyncPipe adds it |

SyncPipe additionally uses a **signal-level IAAFT** existence null (destroy
cross-signal coupling while preserving each signal's autocorrelation) and a
**state-transition shuffle** structure null. These are *complementary*, not
competitors, to SUSY's segment shuffle: they answer the same "is this real?"
question from different null angles.

---

## 3. What was verified, and what was NOT (honest)

**Verified in-repo (this is the key reassurance):**

The signal-level IAAFT existence audit was calibrated on **autocorrelated**
signals (the regime real EDA/HRV envelopes live in, lag-1 autocorrelation
≈ 0.9). Result (`scripts/run_autocorr_robustness.py`, 120 dyads × 199
surrogates per regime):

| regime | false-positive rate | power |
|---|---|---|
| white noise | 0.033 | 0.983 |
| **AR(1), φ=0.9 (realistic)** | **0.025** | **1.000** |

The existence audit **does not manufacture coupling from self-persistent
noise** (FPR ≈ 0.025 < 0.05, i.e. conservative) and **retains full power** on
coupled dyads. This is the direct answer to "will it produce absurd results on
real data" — no, on realistic statistical structure it is well calibrated.

**Not verified in-repo — a true head-to-head against the R SUSY package:**

A faithful SUSY reproduction requires (a) the **raw** two-person signals (to
shuffle segments and *recompute* the cross-correlation) and (b) an **R
environment**. This repository ships only **derived** WCC traces and feature
tables (the raw 1000 Hz `.mat` files live on OSF, project `47n3p`), so a true
SUSY run is out of scope here.

> A caution surfaced while designing this comparison: shuffling an
> **already-computed** correlation trace and taking `mean(|WCC|)` is **vacuous**,
> because the mean of absolute values is invariant to any permutation (the null
> SD is exactly 0). SUSY avoids this precisely because it shuffles raw segments
> and recomputes. This is a good example of why the null must be applied at the
> correct level — and why SyncPipe's own nulls (IAAFT on raw signals,
> block-permutation on WCC for order-dependent features) are applied where they
> are informative.

---

## 4. Running a true SUSY head-to-head (on your machine)

Prerequisites: R (`install.packages("SUSY")`), the OSF Lerique mirror
(`docs/DATA_ACCESS.md` §3), and a Python environment with SyncPipe installed.

### Step 1 — export Lerique raw dyad signals to a SUSY-readable CSV

```python
from syncpipe.realtest.lerique_2024 import load_lerique_dataset

records = load_lerique_dataset(
    data_root="/path/to/Lerique-47n3p",
    preprocess=False,           # keep raw (resampled) envelopes
    drop_incomplete=True, drop_misaligned=True, drop_short_duration=True,
)
# Pick one dyad + modality, write person_a/person_b columns
rec = next(r for r in records if r.modality == "EDA")
import pandas as pd
df = pd.DataFrame({
    "person_a": rec.person_a["value"].to_numpy(dtype=float),
    "person_b": rec.person_b["value"].to_numpy(dtype=float),
})
df.to_csv("lerique_eda_dyad.csv", index=False, header=True, sep=" ")
```

### Step 2 — run SUSY in R

```r
library(SUSY)
data <- read.csv("lerique_eda_dyad.csv", header = TRUE, sep = " ", na.strings = ".")
res <- susy(data[, c(1, 2)], segment = 30, Hz = 1, maxlag = 3)
res            # ES_abs / ES_noabs, %>Pseudo
```

`Hz = 1` matches SyncPipe's 1 Hz envelope; use the same value on both sides so
the comparison is apples-to-apples.

### Step 3 — compare

- SUSY `ES_abs > 0` (with `%>Pseudo` below your alpha) ⇒ "synchrony present".
- SyncPipe existence audit `peak_amplitude` significant ⇒ "synchrony present".
- If both agree per dyad (or per modality), the two approaches converge; a
  systematic disagreement is the thing to investigate.

---

## 5. Conclusion

SyncPipe's null-model machinery is the **same** segment-shuffling /
surrogate-testing approach SUSY popularised, re-expressed with additional
signal-level (IAAFT) and structure-level (state-shuffle) nulls and a
group-inference layer. Its existence test is empirically well-calibrated on
autocorrelated data. A definitive numeric head-to-head against the R SUSY
binary requires the raw OSF signals and an R environment, for which §4 provides
the exact steps.

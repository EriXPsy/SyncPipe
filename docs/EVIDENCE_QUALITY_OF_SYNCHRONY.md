# Evidence: "Quality" of Synchrony Is Distinguishable from Its "Quantity"

> ⚠️ MOTIVATIONAL SIMULATION — NOT CONFIRMATORY. The analyses here
> illustrate why audited synchrony descriptors matter; they are not
> confirmatory evidence for any psychological "quality of synchrony"
> construct.

> Status: v1.0 review follow-up (Blind Spot A).
> Source simulation: `scripts/run_kuramoto_l23_taxonomy.py`
> Raw numbers / figures: generated on demand by `scripts/run_kuramoto_l23_taxonomy.py`
> and `archive/scripts/plot_quality_of_synchrony_evidence.py` under the gitignored
> `artifacts/incremental_value/` and `artifacts/figures/` directories (not
> committed; reproducible from the scripts).

## The claim

SyncPipe's thesis is that *interpersonal synchrony* is not a single scalar
but decomposes into a **level** (how much, `mean_synchrony`) and a
**structure / temporal organization** (when and how it reorganizes —
`switching_rate`, `dwell_time`, `onset_latency`). The interesting scientific
claim is that *structure can be discriminated independently of level*.

The Kuramoto L2/L3 taxonomy is the cleanest available falsifier of that
claim, because it lets us **exact-match on `mean_synchrony`** and then ask
whether any other descriptor still separates two conditions whose *average*
coupling is statistically identical.

## L3_Temporal — the valid core evidence (PROMOTE)

Condition contrast: **early peak vs delayed peak** coupling profiles
(same total coupling, different *timing*). Exact 1:1 caliper matching on
`mean_synchrony` (caliper = 0.005).

| Step | Features | AUC | ΔAUC |
|------|----------|-----|------|
| L1 | `mean_synchrony` | **0.482** | — (chance ✓) |
| L1 | + `peak_amplitude` | 0.454 | −0.028 |
| L2 | + `dwell_time` | 0.397 | −0.057 |
| **L2** | **+ `switching_rate`** | **0.810** | **+0.413** |
| L2 | + `synchrony_entropy` | 0.826 | +0.017 |
| L3 | + `onset_latency` | 0.826 | 0.000 |
| L3 | + `rise_time` + `recovery_time` | 0.797 | −0.029 |

**Reading.** Matching drove `mean_synchrony` to chance (0.482). Yet adding
`switching_rate` lifted AUC to **0.810 (ΔAUC = +0.41)** — a large,
falsifiable separation that cannot be attributed to synchrony *quantity*,
because quantity was already pressed to coin-flip. Note `dwell_time` *alone*
drops *below* chance (0.397), so it is specifically `switching_rate` that
carries the temporal-structure signal here, not "more dynamics" in general.

This is the single result that should be promoted to a **core figure** in
the paper (`l3_temporal_core.png`). It is the most interpretation-independent
demonstration that SyncPipe descriptors capture *structure*, not just level.

> Wording caveat: call it **"temporal structure is discriminable independent
> of mean level"** rather than the looser "quality of synchrony". The latter
> invites a stronger psychological reading than the Kuramoto ground truth
> actually supports.

## L2_Structure — contaminated, DO NOT present as equal-strength (DEMOTE)

Condition contrast: **sustained vs single peak** coupling profiles. Same
caliper matching was requested, but the matcher only found **25 pairs** and
the script itself printed a WARNING that matching was ineffective.

| Step | Features | AUC | ΔAUC |
|------|----------|-----|------|
| L1 | `mean_synchrony` | **0.912** | — (NOT chance ✗) |
| L1 | + `peak_amplitude` | 0.912 | 0.000 |
| L2 | + `dwell_time` | 0.904 | −0.008 |
| L2 | + `switching_rate` | 0.904 | 0.000 |
| L2 | + `synchrony_entropy` | 0.936 | +0.032 |
| L3 | + `onset_latency` | 1.000 | +0.064 |
| L3 | + `rise` + `recovery` | 1.000 | 0.000 |

**Reading.** The matching **failed to suppress `mean_synchrony`** (0.912 at
step 1, far above chance). The subsequent small deltas (+entropy +0.032,
+onset +0.064) therefore most plausibly ride on *residual level signal*, not
on independent structure — and the ceiling is reached with only 25 pairs, so
sampling noise is large. **These deltas must not be reported as evidence that
structure explains variance beyond quantity.**

**Recommended handling:**
1. Re-run with a **tighter caliper** (or generate more candidates) until
   `mean_synchrony` AUC is at chance *before* adding structure descriptors;
   only then are the structure deltas interpretable.
2. Until that re-match exists, present L2 only as a **secondary / exploratory**
   result, explicitly flagged as "matching incomplete (N=25, mean_sync not
   suppressed)". Do **not** merge it with L3_Temporal into a single
   "quality vs quantity" headline.

## Why this matters for the paper

We now hold **one strong and one weak** "structure ≠ quantity" demonstration.
Treating them as a single undifferentiated result would over-claim. Keeping
them distinct — L3 as the flagship, L2 as a qualified/redo — is the honest
and stronger position.

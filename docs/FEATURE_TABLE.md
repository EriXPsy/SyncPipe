# SyncPipe — Authoritative Feature Table (Single Source of Truth)

> **Auto-generated** by `scripts/build_feature_table.py` directly from `feature_definitions.py`. Do not hand-edit. Total features computed: **12**; FDR-family (confirmatory multiplicity set): **3** (peak_amplitude, dwell_time, switching_rate); Reference (reported, not corrected): **mean_synchrony**.


**Four orthogonal axes** govern every feature:
- **Functional tier (Axis A)** — extraction robustness label: `reference` / `core` / `conditional`.
- **Informational class** — Results organisation: `intensity` / `structure` / `temporal`.
- **FDR family (Axis C)** — whether the feature is in the confirmatory multiplicity-corrected set.
- **Mathematical tier (Axis D)** — *sole* determinant of the null model: `L0` / `L1` / `L2`.


| Feature | Func. tier | Class | Primary? | FDR? | Fam | Math | Null model | Paradigm validity | Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| `mean_synchrony` | reference | intensity | Reference only (reported, NOT in FDR) | no | — | L0 | Signal-level IAAFT (shuffle raw signals, recompute WCC) | All paradigms; most robust, least specific | Average moment-to-moment coupling magnitude. |
| `peak_amplitude` | core | intensity | PRIMARY (intensity) | yes | L0 | L0 | Signal-level IAAFT (shuffle raw signals, recompute WCC) | All paradigms; cross-paradigm robust | Strongest sustained coupling reached during interaction. |
| `fraction_above_threshold` | conditional | structure | Exploratory-secondary (occupancy; NOT in FDR) | no | — | L0 | Signal-level IAAFT (shuffle raw signals, recompute WCC) | All paradigms with threshold justification; report threshold metadata | Fraction of finite WCC samples above the synchrony threshold (coverage). |
| `dwell_time` | core | structure | PRIMARY (structure) | yes | L1 | L1 | WCC-level IAAFT (shuffle WCC; preserves L0 moments) | Continuous & event paradigms; needs sufficient trace length | Mean duration of high-synchrony episodes (persistence). |
| `switching_rate` | core | structure | PRIMARY (structure) | yes | L1 | L1 | WCC-level IAAFT (shuffle WCC; preserves L0 moments) | Continuous & event paradigms; sensitive to window size | How often synchrony crosses in/out of high-coupling state. |
| `synchrony_entropy` | conditional | structure | Exploratory (distributional; NOT in FDR) | no | — | L0 | Signal-level IAAFT (shuffle raw signals, recompute WCC) | All paradigms; distribution shape, not temporal order | Dispersion/unpredictability of the synchrony distribution. |
| `bimodality_coefficient` | conditional | structure | Exploratory (distributional; not in the FDR family) | no | — | L0 | Signal-level IAAFT (shuffle raw signals, recompute WCC) | All paradigms; distribution shape, not temporal order | Degree to which synchrony is bistable (high vs low) rather than graded. |
| `onset_latency` | conditional | temporal | Exploratory — EVENT-LOCKED paradigms ONLY | no | — | L2 | Circular time-shift (preserves L0+L1; destroys phase anchor) | Event/stimulus-locked ONLY; undefined (NaN) in free interaction | Time from event onset to first sustained high-synchrony crossing. |
| `rise_time` | conditional | temporal | Exploratory — EVENT-LOCKED paradigms ONLY | no | — | L2 | Circular time-shift (preserves L0+L1; destroys phase anchor) | Event/stimulus-locked ONLY; estimator-shape confound (see Limitations) | Speed of synchrony build-up (WCC-derived; not a physiological waveform). |
| `recovery_time` | conditional | temporal | Exploratory — EVENT-LOCKED paradigms ONLY | no | — | L2 | Circular time-shift (preserves L0+L1; destroys phase anchor) | Event/stimulus-locked ONLY; estimator-shape confound (see Limitations) | Time for synchrony to return toward baseline after a peak. |
| `inter_peak_cv` | conditional | temporal | Exploratory-secondary (temporal regularity; NOT in FDR) | no | — | L2 | Circular time-shift (preserves L0+L1; destroys phase anchor) | Long, oscillatory traces with >= 3 prominent peaks; report definedness rate | CV of inter-peak intervals (regular vs irregular synchrony events). |
| `first_peak_time` | conditional | temporal | Exploratory-secondary (event timing; NOT in FDR) | no | — | L2 | Circular time-shift (preserves L0+L1; destroys phase anchor) | Any morphology with >= 1 prominent peak; report definedness rate | Time of the first prominent above-threshold synchrony peak. |

## Null-model legend

- **L0** — Signal-level IAAFT (shuffle raw signals, recompute WCC)
- **L1** — WCC-level IAAFT (shuffle WCC; preserves L0 moments)
- **L2** — Circular time-shift (preserves L0+L1; destroys phase anchor)

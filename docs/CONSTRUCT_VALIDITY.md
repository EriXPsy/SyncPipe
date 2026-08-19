# Construct and Validation Framework

> Status: v1 validation contract. This document defines what evidence is
> required before a SyncPipe output may be interpreted as interpersonal
> coordination rather than generic co-fluctuation.

## 1. Target construct

SyncPipe's target is **interaction-contingent interpersonal coordination**:
a statistical dependency between two people's physiological or behavioral
trajectories during reciprocal interaction that is stronger or more temporally
specific than expected from each person's dynamics, measured common inputs,
partner mismatch, and non-interactive co-presence.

This target is deliberately narrower than the everyday word *synchrony* and
broader than exact phase locking. It may include co-fluctuation, coordinated
state transitions, and recurrent temporal organization. A SyncPipe descriptor
is an indicator of this target under a specified design; it is not the construct
itself.

### v1 measurement boundary

v1 analyzes continuous, low-frequency, same-modality dyadic envelopes. It does
not yet establish neural hyperscanning validity, discrete event synchrony,
cross-modal coordination, directionality, or causality.

## 2. Rival explanations and required controls

The same positive zero-lag WCC can be produced by several data-generating
processes. Claims must be named by the strongest rival explanation ruled out.

| Evidence level | Rival explanation addressed | Minimum evidence | Permitted language |
|---|---|---|---|
| E0: co-fluctuation | independent autocorrelated signals | signal-level null | "synchrony-like co-fluctuation" |
| E1: temporal structure | unordered/local WCC states | WCC-level structure null | "temporally organized co-fluctuation" |
| E2: partner specificity | arbitrary partner pairing | pseudo-pair control | "partner-specific association" |
| E3: alignment specificity | slow drift/block timing | time-shift control | "alignment-specific association" |
| E4: shared-input specificity | common stimulus schedule | independently shuffled/yoked stimulus control | "not fully explained by measured shared stimulus" |
| E5: interaction contingency | co-presence and parallel responding | live reciprocal interaction versus replay/yoked/non-interactive co-presence | "interaction-contingent coordination" |
| E6: mechanism | unmeasured common causes and reverse direction | intervention or identified directional model | mechanism-specific/causal language |

Passing E0 does not imply E2-E6. SyncPipe's current computations can support
E0-E4 when the study design supplies the necessary data. **E5 cannot be
recovered from a single uncontrolled interaction recording by software.** It
requires an experimental contrast such as live interaction versus replay,
yoked partner, separated shared-stimulus exposure, or another design that
breaks reciprocity while preserving relevant sensory input.

`condition` labels are not evidence by themselves. Reports must describe what
was held constant and what rival explanation the contrast changes.

## 3. Reliability is feature- and timescale-specific

A dynamic interpersonal state need not show high rank-order stability across
sessions. Reliability must therefore not be reduced to one undifferentiated
ICC.

### 3.1 Technical reproducibility

Question: does the same input and configuration return the same output?

Required checks:

- CLI/API parity;
- fixed-seed reproducibility;
- environment and dependency recording;
- serialization round-trip;
- no mutation of source artifacts during tests.

This is software reliability, not psychological reliability.

For signal-level IAAFT with dropout or concatenated sessions, SyncPipe keeps the
original time axis and generates A/B surrogates independently inside each
eligible finite contiguous segment. The default minimum segment length is
`max(50, window_size + 19)` raw samples; shorter fragments are excluded and
reported. Observed and null WCC use the identical segment set and are pooled by
eligible WCC point for distributional summaries. No missing sample is deleted,
joined to a non-adjacent neighbor, or implicitly imputed.

### 3.2 Within-session dependability

Question: would the descriptor be similar under another representative sample
of moments from the same interaction context?

Recommended analysis:

- divide recordings into pre-specified, duration-matched contiguous blocks;
- never randomly split individual WCC points because overlapping windows are
  not independent;
- estimate block-resampling stability intervals using blocks longer than the
  WCC window and relevant autocorrelation scale; do not call these confidence
  intervals for a population maximum, because ordinary bootstrap cannot
  extrapolate beyond the observed endpoint;
- report the number and duration of usable blocks;
- evaluate duration curves rather than one arbitrary split point.

For `peak_amplitude`, dependability must be evaluated as a function of recording
duration because it is an extreme-value statistic.

### 3.3 Between-session stability

Question: are dyad differences stable across genuinely comparable sessions?

Use a paired-session ICC or multilevel variance decomposition only when the
construct is expected to be trait-like and sessions share the same design.
Report confidence intervals and the exact ICC form. Low stability may represent
real state variation rather than measurement error.

### 3.4 Generalizability

Where data permit, estimate variance attributable to:

- dyad;
- session;
- condition;
- modality;
- stimulus/block;
- dyad × condition and residual variation.

A descriptor should not be called generally reliable when its apparent
reliability depends on one window, threshold, modality, or session duration.

## 4. Validity evidence required per descriptor

### 4.1 Content validity

Every descriptor must have a declared estimand, unit, applicable paradigm,
failure mode, and forbidden interpretation. A mathematical summary does not
become psychologically meaningful merely because it has an intuitive name.

### 4.2 Structural validity

Test whether descriptors empirically form the proposed intensity, occupancy,
structure, and event-morphology axes. Use held-out datasets and avoid treating
the axes encoded by the developers as confirmed factors.

### 4.3 Convergent validity

Compare against independently implemented, theoretically related measures such
as mean correlation, CRQA/recurrence measures, coherence, or PLV only where the
signal type and assumptions make those measures appropriate. Moderate, not
perfect, convergence is expected when estimands differ.

### 4.4 Discriminant validity

A candidate descriptor must remain low or lose its effect under at least these
negative controls:

1. independent autocorrelated signals;
2. identical shared input without interpersonal coupling;
3. co-presence without reciprocal interaction;
4. mismatched partners;
5. temporally shifted partners;
6. common slow drift;
7. unequal duration and artifact spikes.

A feature that detects ground-truth coupling but also strongly detects these
negative controls lacks discriminant validity.

### 4.5 Known-groups and intervention validity

The strongest practical validation is a design that manipulates reciprocity or
coupling while preserving shared sensory input. Examples include live versus
replay interaction, contingent versus non-contingent feedback, true versus
yoked partner, or experimentally disrupted coordination.

### 4.6 Incremental validity

A descriptor has incremental validity only if it improves prediction or
condition discrimination beyond a pre-specified baseline containing, where
available:

- mean synchrony/reference correlation;
- individual signal variance and autocorrelation;
- recording duration and valid-window count;
- individual reactivity to the task/stimulus;
- shared-stimulus response predictors;
- artifact and motion covariates.

Feature selection, threshold tuning, and model tuning must occur inside nested
cross-validation grouped by dyad. Improvement on the same simulations or real
datasets used to select the descriptor is development evidence, not external
incremental validity. Code in `morphology.py` is therefore supportive and
exploratory until this design is met on held-out data.

## 5. Feature-specific validation priorities

### `peak_amplitude`

Primary risks: extreme-value opportunity, artifact sensitivity, trace
autocorrelation, and duration dependence.

Required next tests:

- null distribution across matched and mismatched duration;
- duration/dependability curves;
- robustness to isolated spikes and smooth shared drift;
- comparison with high quantiles and area-above-threshold summaries;
- preregistered external condition contrast.

### `dwell_time` and `switching_rate`

Primary risks: threshold dependence, overlapping-window smoothing, informative
undefinedness, and artificial episode merging.

Required next tests:

- threshold and hysteresis response surfaces;
- window/step sensitivity;
- episode recovery with independently generated state processes;
- definedness as an outcome, not missing-at-random data;
- replication under a pooled threshold fixed before condition testing.

### Event morphology descriptors

Primary risks: event-anchor ambiguity, multiple-episode ambiguity, smoothing
bias, and weak correspondence to interpersonal timing.

They remain descriptive until event-locked ground truth and held-out real data
show recovery, reliability, discriminant validity, and incremental information.
They must never be interpreted as participant lead-lag in v1.

## 6. Promotion rule

A descriptor may be promoted only after all of the following are recorded:

1. frozen estimand and null hypothesis;
2. code-level recovery on positive ground truth;
3. acceptable false-positive behavior on all relevant negative controls;
4. duration, missingness, window, threshold, and artifact sensitivity;
5. feature-appropriate reliability with uncertainty;
6. convergent and discriminant evidence;
7. incremental evidence against a pre-specified baseline where claimed;
8. preregistered replication on data not used to design or select the feature;
9. independent review or reproduction by someone outside the development loop.

For two-condition L2 inference, SyncPipe reports the paired median difference
in the descriptor's original units, its dyad-difference interquartile range,
and a distribution-free 95% sign-test interval for the population median. If
the cohort is too small to support a finite exact interval, the bounds are
reported as unbounded rather than replaced by a misleading bootstrap range.
Permutation output states whether enumeration was exact or Monte Carlo and
reports the null-draw count, minimum attainable p, and approximate Monte Carlo
standard error.

Until then, passing tests means the implementation is reproducible—not that the
psychological construct has been validated.

## 7. Reproducible validation commands

```bash
python scripts/run_peak_duration_validation.py \
  --n-replicates 200 --n-dyads 30 \
  -o artifacts/peak_duration_validation

python scripts/run_discriminant_validity.py \
  --n-replicates 20 --surrogate-n 99 \
  -o artifacts/discriminant_validity
```

The first command exports independent-AR(1) null-duration curves, contiguous
block dependability, and an explicitly labelled block-resampling stability
interval. The second exports per-replicate L0 results and cohort-level
pseudo-pair/time-shift controls for independent AR(1), shared stimulus, common
drift, aligned shared context, shared artifact, and reciprocal-VAR scenarios.

These are adversarial diagnostics. In particular, shared-input scenarios may
legitimately pass L0 because IAAFT does not remove common input; that outcome
records the claim boundary rather than being relabelled as successful construct
discrimination.

The discriminant script also writes `acceptance_report.csv/json` under frozen
default criteria: the exact independent-null FPR interval must contain alpha,
the lower confidence bound for reciprocal-control power must reach 0.80, the
upper confidence bound for each construct-negative L0 detection rate must not
exceed 0.10, and pseudo/time-shift specificity must have the pre-declared
scenario-appropriate direction. These strict criteria are allowed to fail;
failure is evidence about the method or insufficient benchmark precision, not a
reason to edit thresholds after seeing results.

# SKILL: SyncPipe v1.0

Agent-oriented capability sheet. Human docs: `docs/USER_MANUAL.md`.

## What this skill does
Turn a pre-computed or raw dyadic synchrony signal into **auditable descriptors**
and run a **three-step audited evidence chain** (existence audit → design-control
audit → group inference). It is measurement infrastructure for **same-modality dyadic synchrony across
multiple modality families**, built on a windowed cross-correlation (WCC) substrate.

## When to use
- User has dyadic / two-party time series (behavioral, physiological, neural) and
  wants to **quantify synchrony** rigorously rather than with a single ad-hoc score.
- User needs to **test whether observed synchrony exceeds chance** (existence),
  **rule out confounds** (shared stimulus, misalignment, partner identity), or
  **compare conditions/groups** with multiplicity control.
- User wants ground-truth-validated descriptors with explicit risk notes.

## When NOT to use / guardrails
- Not for triads or groups (v1.0 is dyadic only).
- Do NOT report exploratory descriptors (`bimodality_coefficient`,
  `synchrony_entropy`, `fraction_above_threshold`, `first_peak_time`,
  `inter_peak_cv`) as confirmatory; they are not in the FDR family and the timing
  descriptors lack a validated existence null (deferred to v2).
- A significant existence audit is **necessary but not sufficient** for coupling —
  always run the design-control audit before claiming interpersonal synchrony.
- Never describe `mean_synchrony` as confirmatory; it is a reference comparator.

## Environment
- Python ≥ 3.10. Install: `python -m pip install -e .` from the repository root.
- Version check: `syncpipe --version` → `syncpipe 2.0.0`.

## CLI entry points
```bash
# Methods demo + all audit reports on a synthetic ground-truth dyad
syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo

# Analyze user data (confirmatory path: manifest + config)
syncpipe analyze -m manifest.csv -c config.toml -o results/

# Exploratory descriptor path on ad-hoc CSVs
syncpipe describe -i dyad.csv -n eda --hz 1 --window-size 10 --surrogates 500 -o out.json

# Self-contained reproduction smoke check
python -m pytest
python -m syncpipe demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo_v1
python scripts/build_feature_table.py
```

## Python API (import syncpipe as sp)
```python
import syncpipe as sp

# Build a dyad and run dynamic analysis
dyad = sp.Dyad(...)                       # see core.py
analyzer = sp.DynamicAnalyzer(...)

# Pipelines
pipe  = sp.ComputationPipeline(hz=4.0, window_size=40)
infer = sp.InferencePipeline(features_df, hz=4.0, wcc_window_sec=10.0)

# Three-step evidence chain
sp.synchrony_existence_audit(sig_a, sig_b, hz=4.0, window_size=40)   # step 1
sp.design_control_audit(signal_pairs, hz=4.0, window_size=40)        # step 2
# step 3 = InferencePipeline (dyad-paired permutation + BH-FDR)

# Surrogate thresholds
sp.compute_session_pooled_thresholds_by_modality(...)  # CANONICAL default: per-modality IAAFT
sp.compute_session_pooled_threshold(...)        # optional coarser global pool
sp.compute_condition_pooled_thresholds(...)    # optional per-condition pool

# Feature surface
sp.FDR_FEATURES        # ('peak_amplitude','dwell_time','switching_rate')
sp.REFERENCE_FEATURE   # ('mean_synchrony',)
sp.feature_status_table()   # rows with source level / paradigm / risk
sp.explain_feature("dwell_time")
```

## Key constants
- `sp.ONSET_THRESHOLD` = 0.5 — **fallback / sensitivity constant only** (forwarded unchanged for sensitivity sweeps & paper reproduction; also the fallback when a modality's pooled null is degenerate). The scientific canonical default onset threshold is **per-modality pooled** (`compute_session_pooled_thresholds_by_modality`): one IAAFT-derived cut-off per modality, so EDA and ECG get different, calibrated thresholds while every dyad of a modality still shares one. Surrogate-derived thresholds are hard-capped at `SURROGATE_THRESHOLD_MAX = 0.9` (periodicity / strong-autocorrelation protection).
- `SURROGATE_THRESHOLD_PERCENTILE` = 95 (per-dyad surrogate cut-off).
- `sp.PRIMARY_FDR_FAMILY` = `('peak_amplitude',)` — the PRIMARY confirmatory
  claim rests on **one** pre-registered endpoint, so the primary BH denominator
  is **m = 1**. A single endpoint is what makes the existence gate and the group
  claim consistent; an OR across a family would reintroduce a hidden multiple
  comparison.
- `sp.SECONDARY_FDR_FAMILY` = `('dwell_time', 'switching_rate')` — reported in
  parallel, BH-corrected **within its own family** (m = 2). It does not enter the
  primary denominator.
- `sp.FDR_FEATURES` = primary + secondary (3 names). It is the descriptive export
  surface, **not** the primary multiplicity denominator.

## Mandatory workflow (do not reorder)
1. **QC gate**: `sp.run_quality_check(dataset)` → handle WARN/FAIL. A FAIL
   raises `DataQualityError`. Watch the temporal-alignment stage: misaligned
   start times create a false CCF lag.
2. **Existence audit** (signal-level IAAFT). Necessary, not sufficient.
3. **Design-control audit** (pseudo-pair + time-shift + across-stimulus).
4. **Group inference** (dyad-paired permutation + BH-FDR). The primary claim is
   BH over `PRIMARY_FDR_FAMILY` (m = 1); `SECONDARY_FDR_FAMILY` is corrected in
   parallel within its own family (m = 2); `mean_synchrony` is reported as
   reference only and is never corrected.
5. **Report** via the feature status table; include definedness rates for
   exploratory descriptors.

## Outputs to surface to the user
- `DEMO_REPORT.md` / `viewer_results.json` (demo).
- `docs/FEATURE_TABLE.{csv,md}` (authoritative descriptor table).
- `artifacts/incremental_auc/` (incremental AUC per modality) and
  `artifacts/prediction/` (prediction gap check).
- The status table row for any descriptor before reporting it.

## Pointers
- Decisions & lineage: `docs/METHOD_LOG.md` (esp. §3 evidence chain, §7d lineage).
- Script → trunk-result mapping: `docs/SCRIPT_MAP.md`.
- Visual overview: `SYNCPIPE_FAMILY_TREE.html` (repo root).
- v2 staging (do not treat as v1 API): `experimental/`.

# SKILL: SyncPipe v1.0

Agent-oriented capability sheet. Human docs: `docs/USER_MANUAL.md`.

## What this skill does
Turn a pre-computed or raw dyadic synchrony signal into **auditable descriptors**
and run a **three-step audited evidence chain** (existence audit → design-control
audit → group inference). It is measurement infrastructure for multimodal
interpersonal synchrony built on a windowed cross-correlation (WCC) substrate.

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
- Version check: `multisync --version` → `syncpipe 1.0.0`.

## CLI entry points
```bash
# Methods demo + all audit reports on a synthetic ground-truth dyad
multisync demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 -o artifacts/demo

# Analyze user data (one CSV per modality, comma-separated)
multisync analyze -i a.csv,b.csv -n behavior,neural --hz 4 --window-size 40 \
    --surrogates 500 -o results.json

# Self-contained reproduction smoke check
python -m pytest
python -m multisync demo --surrogates 100 --audit-surrogates 100 --demo-dyads 4 --no-prediction -o artifacts/demo_v1
python scripts/build_feature_table.py
```

## Python API (import multisync as ms)
```python
import multisync as ms

# Build a dyad and run dynamic analysis
dyad = ms.Dyad(...)                       # see core.py
analyzer = ms.DynamicAnalyzer(...)

# Pipelines
pipe  = ms.ComputationPipeline(hz=4.0, window_size=40)
infer = ms.InferencePipeline(features_df, hz=4.0, wcc_window_sec=10.0)

# Three-step evidence chain
ms.synchrony_existence_audit(sig_a, sig_b, hz=4.0, window_size=40)   # step 1
ms.design_control_audit(signal_pairs, hz=4.0, window_size=40)        # step 2
# step 3 = InferencePipeline (dyad-paired permutation + BH-FDR)

# Surrogate thresholds
ms.compute_session_pooled_threshold(...)        # between-dyad comparability
ms.compute_condition_pooled_thresholds(...)

# Feature surface
ms.FDR_FEATURES        # ('peak_amplitude','dwell_time','switching_rate')
ms.REFERENCE_FEATURE   # ('mean_synchrony',)
ms.feature_status_table()   # rows with source level / paradigm / risk
ms.explain_feature("dwell_time")
```

## Key constants
- `ms.ONSET_THRESHOLD` = 0.5 (episode = WCC above threshold).
- `SURROGATE_THRESHOLD_PERCENTILE` = 95 (per-dyad surrogate cut-off).
- Primary FDR family size = 3 → BH multiplicity denominator m = 3.

## Mandatory workflow (do not reorder)
1. **QC gate**: `ms.run_quality_check(dataset)` → handle WARN/FAIL. A FAIL
   raises `DataQualityError`. Watch the temporal-alignment stage: misaligned
   start times create a false CCF lag.
2. **Existence audit** (signal-level IAAFT). Necessary, not sufficient.
3. **Design-control audit** (pseudo-pair + time-shift + across-stimulus).
4. **Group inference** (dyad-paired permutation + BH-FDR over the 3 features;
   `mean_synchrony` reported as reference only).
5. **Report** via the feature status table; include definedness rates for
   exploratory descriptors.

## Outputs to surface to the user
- `DEMO_REPORT.md` / `viewer_results.json` (demo).
- `docs/FEATURE_TABLE.{csv,md}` (authoritative descriptor table).
- `artifacts/timing_validation/` (block-bootstrap null + incremental AUC).
- The status table row for any descriptor before reporting it.

## Pointers
- Decisions & lineage: `docs/METHOD_LOG.md` (esp. §3 evidence chain, §7d lineage).
- Script → trunk-result mapping: `docs/SCRIPT_MAP.md`.
- Visual overview: `docs/SYNCPIPE_FAMILY_TREE.md` (if present).
- v2 staging (do not treat as v1 API): `experimental/`.

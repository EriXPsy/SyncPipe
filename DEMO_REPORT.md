# SyncPipe demo report

This demo illustrates SyncPipe as single-modality synchrony measurement infrastructure: WCC trace construction, descriptor export, synchrony-existence audit, design-control audit, and viewer-ready output.

## Ground truth
- Synthetic dyadic coupling, noise ratio: 0.3.

## Outputs
- Viewer JSON: `demo_results.json`
- Feature table: `feature_table.csv`
- Feature status table: `feature_status_table.csv`
- Table 1 LaTeX: `TABLE1_FEATURE_STATUS.tex`
- Synchrony-existence audit: `synchrony_existence_audit.json`
- Design-control audit: `design_control_audit.json`

## Synchrony-existence audit
Signal-level IAAFT asks whether the observed WCC exceeds independent autocorrelated signals. It is necessary but not sufficient evidence for interpersonal coupling.

```json
{
  "mean_synchrony": 0.02,
  "peak_amplitude": 0.2,
  "bimodality_coefficient": 0.94
}
```

## Feature status table
`feature_status_table.csv` is the Table 1 draft: source level, incremental information, paradigm restriction, default audit/test, status, and risk. It separates descriptor usefulness from confirmatory status.

## Design controls
Pseudo-pair and time-shift controls are design-level audits for dyad-specificity and temporal-alignment dependence. They do not solve all ISC/co-presence problems, but they make those alternatives visible.

| feature | real median | pseudo median | time-shift median | p(real>pseudo) | p(real>shift) |
|---|---:|---:|---:|---:|---:|
| mean_synchrony | 0.204 | 0.043 | 0.041 | 0.0308 | 0.0308 |
| peak_amplitude | 0.829 | 0.708 | 0.791 | 0.0308 | 0.0615 |
| fraction_above_threshold | 0.277 | 0.105 | 0.120 | 0.0308 | 0.0308 |
| dwell_time | 5.659 | 3.983 | 4.440 | 0.0462 | 0.0308 |
| switching_rate | 5.065 | 3.117 | 3.115 | 0.0462 | 0.0308 |

## Caution
This demo is synthetic. Passing signal-level IAAFT does not prove dyad-specific coupling. For real event-locked or shared-stimulus designs, add pseudo-pair, time-shift, and when possible across-stimulus shuffle controls.
# API reference

Generated from the installed package by `scripts/build_api_reference.py` — do not edit by hand; re-run the script after public-API changes. Version: see `syncpipe.__version__`.

Top-level entry points (see README): `syncpipe.analyze()` for a full study from a manifest, `syncpipe.make_example()` for a runnable example project.

## Core objects

Module: `syncpipe.core`

### Classes

- **`AnalysisResults(dyad_id: 'str', dynamic_features: 'Dict[str, Dict[str, float]]' = <factory>, dynamic_features_segmented: 'Dict[str, Dict[str, Dict[str, float]]]' = <factory>, threshold_meta: 'Dict[str, Dict[str, Any]]' = <factory>, score_view: 'List[Dict[str, Any]]' = <factory>, diagnostics: 'List[Dict[str, Any]]' = <factory>, parameters: 'Dict[str, Any]' = <factory>) -> None`** — Complete analysis output — ready for Viewer JSON export.
- **`DataQualityError(...)`** — Raised when quality check fails at FAIL level.
- **`Dyad(hz: 'float' = 1.0, discontinuity_mask: 'Optional[np.ndarray]' = None, **modalities: 'pd.DataFrame') -> 'None'`** — User-friendly dyad container.
- **`DynamicAnalyzer(window_size: 'int' = 10, surrogate_n: 'int' = 5000, max_lag_sec: 'float' = 0.0, alpha: 'float' = 0.05, seed: 'int' = 42, onset_threshold: 'Optional[float]' = None, threshold_mode: 'str' = 'within_dyad', run_qc: 'bool' = True, qc_raise_on_fail: 'bool' = True, qc_config: 'Optional[Dict[str, Any]]' = None, window_type: 'str' = 'rect', cross_modal: 'bool' = False) -> 'None'`** — Default CLI / descriptor route through SyncPipe. For the study-level
- **`DynamicFeatures(onset_latency: 'float' = nan, rise_time: 'float' = nan, peak_amplitude: 'float' = nan, peak_abs_amplitude: 'float' = nan, recovery_time: 'float' = nan, dwell_time: 'float' = nan, switching_rate: 'float' = nan, synchrony_entropy: 'float' = nan, mean_synchrony: 'float' = nan, bimodality_coefficient: 'float' = nan, fraction_above_threshold: 'float' = nan, inter_peak_cv: 'float' = nan, first_peak_time: 'float' = nan, onset_latency_imputed: 'float' = nan, rise_time_imputed: 'float' = nan, recovery_time_imputed: 'float' = nan, onset_defined: 'int' = 0, rise_defined: 'int' = 0, recovery_defined: 'int' = 0, nan_fraction: 'float' = nan, notes: 'str' = '', params: 'Dict[str, Any]' = <factory>) -> None`** — Container for FDR-family features + reference + diagnostics + definedness flags.
- **`SynchronyDataset(dyad_id: 'str', modalities: 'Optional[Dict[str, pd.DataFrame]]' = None, discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'None'`** — Container for one dyad's multi-modal time-series.

### Functions

- **`extract_dynamic_features(wcc: 'np.ndarray', hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, max_nan_ratio: 'float' = 0.2, height: 'Optional[float]' = None, distance: 'Optional[int]' = None, prominence: 'Optional[float]' = None, aggregation: 'str' = 'mean', return_raw_profiles: 'bool' = False, wcc_window_sec: 'float' = 1.0, gap_policy: 'Optional[str]' = None) -> 'Any'`** — Extract features from a WCC series via the SSoT.
- **`extract_features_all_pairs(dataset: "'SynchronyDataset'", window_size: 'int' = 10, hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, wcc_window_sec: 'float' = 1.0, use_surrogate_threshold: 'bool' = True, surrogate_n: 'int' = 200, surrogate_seed: 'int' = 42, discontinuity_mask: 'Optional[np.ndarray]' = None, cross_modal: 'bool' = False) -> 'Tuple[Dict[str, DynamicFeatures], Dict[str, Dict[str, Any]]]'`** — Compute WCC + dynamic features for all modality pairs.
- **`extract_features_segmented(dataset: "'SynchronyDataset'", window_size: 'int' = 10, hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, max_nan_ratio: 'float' = 0.2, wcc_window_sec: 'float' = 1.0, use_surrogate_threshold: 'bool' = True, surrogate_n: 'int' = 200, surrogate_seed: 'int' = 42, discontinuity_mask: 'Optional[np.ndarray]' = None, cross_modal: 'bool' = False) -> 'Tuple[Dict[str, Dict[str, DynamicFeatures]], Dict[str, Dict[str, Any]]]'`** — Compute WCC + dynamic features per CONTEXT segment.
- **`iter_dyad_pairs(dataset, cross_modal: 'bool' = False)`** — Yield (src_key, name_a, name_b, col_a, col_b, x, y) for each valid pair.
- **`pairing_policy(dataset, cross_modal: 'bool' = False) -> 'str'`** — Return the effective dyad-pairing policy for the result manifest.
- **`run_quality_check(dataset: 'Any', config: 'Optional[Dict[str, Any]]' = None, raise_on_fail: 'bool' = False, eligibility: 'Optional[Dict[str, int]]' = None) -> 'DataQualityReport'`** — Run the 4-stage data quality check pipeline.
- **`sliding_window_wcc(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect') -> 'np.ndarray'`** — Compute sliding-window cross-correlation (WCC) between x and y.
- **`sliding_window_wcc_masked(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect', discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'np.ndarray'`** — sliding_window_wcc with discontinuity-boundary gating.

## Dataset assembly

Module: `syncpipe.dataset`

### Classes

- **`ContextLabel(start_sec: 'float', end_sec: 'float', label: 'str', score: 'float' = 0.0) -> None`** — A scored / labelled episode annotation (the psycho context layer).
- **`SynchronyDataset(dyad_id: 'str', modalities: 'Optional[Dict[str, pd.DataFrame]]' = None, discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'None'`** — Container for one dyad's multi-modal time-series.

## Data-to-pipeline bridge

Module: `syncpipe.pipeline_bridge`

### Classes

- **`ComputationPipeline(hz: float, window_size: int = 40, onset_threshold: Optional[float] = 0.5, backend: str = 'wcc', wclr_max_lag_samples: int = 2, wclr_metric: str = 'beta', window_type: str = 'rect')`** — End-to-end computation: load → WCC → features → DataFrame.
- **`InferenceInputs(features_df: 'pd.DataFrame', raw_signals: 'Dict[str, Tuple[np.ndarray, np.ndarray]]', design_pairs: 'Dict[str, Tuple[np.ndarray, np.ndarray]]', discontinuity_mask: 'Optional[Dict[str, np.ndarray]]' = None, wcc_traces: 'Optional[Dict[str, np.ndarray]]' = None, condition_col: 'str' = 'condition', dyad_col: 'str' = 'dyad_id', thresholds_by_modality: 'Optional[Dict[str, float]]' = None, prepared_cohort: 'Optional[PreparedCohort]' = None, preparation_diagnostics: 'Optional[Dict[str, Dict[str, object]]]' = None) -> None`** — Assembled inputs for :class:`syncpipe.inference_pipeline.InferencePipeline`.
- **`PreparationExclusion(key: 'str', dyad_id: 'str', modality: 'str', condition: 'str', code: 'str', stage: 'str', detail: 'str', claim_effect: 'str' = 'observation_excluded') -> None`** — Typed reason an observation did not enter the prepared cohort.
- **`PreparedCohort(observations: 'Tuple[PreparedObservation, ...]', exclusions: 'Tuple[PreparationExclusion, ...]' = ()) -> None`** — PreparedCohort(observations: 'Tuple[PreparedObservation, ...]', exclusions: 'Tuple[PreparationExclusion, ...]' = ())
- **`PreparedObservation(key: 'str', dyad_id: 'str', modality: 'str', condition: 'str', hz: 'float', signal_a: 'np.ndarray', signal_b: 'np.ndarray', geometry: 'SignalGeometry') -> None`** — Immutable data handed from preparation to computation and inference.

### Functions

- **`compute_session_pooled_thresholds_by_modality(dyad_signals: 'List[Tuple[np.ndarray, np.ndarray]]', modalities: 'List[str]', hz: 'float', wcc_window_size: 'int', surrogate_n: 'int' = 200, percentile: 'float' = 95.0, seed: 'int' = 42, surrogate_method: 'str' = 'iaaft', backend: 'str' = 'wcc', wclr_max_lag_samples: 'int' = 2, fallback_threshold: 'float' = 0.5, discontinuity_masks: 'Optional[List[Optional[np.ndarray]]]' = None, return_meta: 'bool' = False) -> 'Union[Dict[str, float], Dict[str, Tuple[float, Dict]]]'`** — Compute one surrogate threshold per modality (per-modality pooled null).
- **`records_to_inference_inputs(records: 'Sequence[Any]', *, hz: 'float', window_size: 'int', onset_threshold: 'Union[float, str]' = 'session_pooled', design_condition: 'Optional[str]' = None, condition_col: 'str' = 'condition', dyad_col: 'str' = 'dyad_id', feature_cols: 'Optional[Sequence[str]]' = None, window_type: 'str' = 'rect', initial_exclusions: 'Sequence[PreparationExclusion]' = ()) -> 'InferenceInputs'`** — Convert loader records into the three-pipeline-ready inputs.

## Batch computation

Module: `syncpipe.batch`

### Classes

- **`AnalysisResults(dyad_id: 'str', dynamic_features: 'Dict[str, Dict[str, float]]' = <factory>, dynamic_features_segmented: 'Dict[str, Dict[str, Dict[str, float]]]' = <factory>, threshold_meta: 'Dict[str, Dict[str, Any]]' = <factory>, score_view: 'List[Dict[str, Any]]' = <factory>, diagnostics: 'List[Dict[str, Any]]' = <factory>, parameters: 'Dict[str, Any]' = <factory>) -> None`** — Complete analysis output — ready for Viewer JSON export.
- **`BatchConfig(dyad_id: 'str', modalities: 'Dict[str, pd.DataFrame]' = <factory>, hz: 'float' = 1.0, preprocessing: 'Dict[str, Any]' = <factory>, context_labels: 'List[Dict[str, Any]]' = <factory>) -> None`** — Configuration for a single dyad in a batch analysis.
- **`Dyad(hz: 'float' = 1.0, discontinuity_mask: 'Optional[np.ndarray]' = None, **modalities: 'pd.DataFrame') -> 'None'`** — User-friendly dyad container.
- **`DyadResult(dyad_id: 'str', frac_significant_edges: 'float' = nan, mean_peak_lag_sec: 'float' = nan, driver_scores: 'Dict[str, float]' = <factory>, mean_onset_latency: 'float' = nan, mean_peak_sync: 'float' = nan, mean_build_up_rate: 'float' = nan, mean_breakdown_rate: 'float' = nan, mean_peak_amplitude: 'float' = nan, mean_rise_time: 'float' = nan, mean_recovery_time: 'float' = nan, mean_dwell_time: 'float' = nan, mean_switching_rate: 'float' = nan, mean_synchrony: 'float' = nan, mean_synchrony_entropy: 'float' = nan, onset_defined_rate: 'float' = nan, recovery_defined_rate: 'float' = nan, n_diagnostics: 'int' = 0, raw: 'Optional[AnalysisResults]' = None) -> None`** — Extracted scalar metrics from one dyad's AnalysisResults.
- **`DynamicAnalyzer(window_size: 'int' = 10, surrogate_n: 'int' = 5000, max_lag_sec: 'float' = 0.0, alpha: 'float' = 0.05, seed: 'int' = 42, onset_threshold: 'Optional[float]' = None, threshold_mode: 'str' = 'within_dyad', run_qc: 'bool' = True, qc_raise_on_fail: 'bool' = True, qc_config: 'Optional[Dict[str, Any]]' = None, window_type: 'str' = 'rect', cross_modal: 'bool' = False) -> 'None'`** — Default CLI / descriptor route through SyncPipe. For the study-level
- **`GroupComparisonReport(label_a: 'str', label_b: 'str', n_a: 'int', n_b: 'int', alpha: 'float', test_results: 'List[MetricTestResult]' = <factory>, dyad_results_a: 'List[DyadResult]' = <factory>, dyad_results_b: 'List[DyadResult]' = <factory>) -> None`** — Full group comparison report.
- **`MetricTestResult(metric: 'str', mean_a: 'float', mean_b: 'float', median_a: 'float', median_b: 'float', p_raw: 'float', p_fdr: 'float' = nan, effect_size: 'float' = nan, effect_size_type: 'str' = 'cohens_d', test_name: 'str' = 'mann_whitney_u', statistic: 'float' = nan, n_a: 'int' = 0, n_b: 'int' = 0, significant_fdr: 'bool' = False) -> None`** — Statistical test result for one metric.

### Functions

- **`apply_fdr(names: 'Sequence[str]', p_values: 'Sequence[float]', alpha: 'float' = 0.05, on_duplicate: 'str' = 'raise') -> 'Dict[str, Dict[str, float]]'`** — Single canonical FDR entry: dedupe -> BH-FDR -> keyed result.
- **`batch_analyze(configs: 'List[BatchConfig]', analyzer_kwargs: 'Optional[Dict[str, Any]]' = None, verbose: 'bool' = True) -> 'List[DyadResult]'`** — Run the full SyncPipe pipeline on a list of dyads.
- **`dedupe_fdr_input(names: 'Sequence[str]', values: 'Sequence[float]', on_duplicate: 'str' = 'raise') -> 'Tuple[List[str], List[float]]'`** — Guard an FDR input against duplicate keys (endpoints / features).
- **`group_comparison(group_a: 'List[DyadResult]', group_b: 'List[DyadResult]', label_a: 'str' = 'Group A', label_b: 'str' = 'Group B', alpha: 'float' = 0.05, test: 'str' = 'mann_whitney') -> 'GroupComparisonReport'`** — Compare two groups of dyads on all scalar synchrony metrics.
- **`residualize_features(df: 'pd.DataFrame', features: 'List[str]', baseline: 'str' = 'mean_synchrony', suffix: 'str' = '_residual', min_n: 'int' = 10) -> 'pd.DataFrame'`** — Remove linear contribution of baseline (mean_synchrony) from each feature.

## WCC computation and features

Module: `syncpipe.dynamic_features`

### Classes

- **`DynamicFeatures(onset_latency: 'float' = nan, rise_time: 'float' = nan, peak_amplitude: 'float' = nan, peak_abs_amplitude: 'float' = nan, recovery_time: 'float' = nan, dwell_time: 'float' = nan, switching_rate: 'float' = nan, synchrony_entropy: 'float' = nan, mean_synchrony: 'float' = nan, bimodality_coefficient: 'float' = nan, fraction_above_threshold: 'float' = nan, inter_peak_cv: 'float' = nan, first_peak_time: 'float' = nan, onset_latency_imputed: 'float' = nan, rise_time_imputed: 'float' = nan, recovery_time_imputed: 'float' = nan, onset_defined: 'int' = 0, rise_defined: 'int' = 0, recovery_defined: 'int' = 0, nan_fraction: 'float' = nan, notes: 'str' = '', params: 'Dict[str, Any]' = <factory>) -> None`** — Container for FDR-family features + reference + diagnostics + definedness flags.

### Functions

- **`compute_surrogate_threshold(wcc_surrogates: 'np.ndarray', percentile: 'float' = 95.0) -> 'Tuple[float, bool]'`** — Compute a per-dyad surrogate-derived onset threshold.
- **`compute_surrogate_threshold_from_signals(sig_a: 'np.ndarray', sig_b: 'np.ndarray', hz: 'float', wcc_window_size: 'int', surrogate_n: 'int' = 200, percentile: 'float' = 95.0, seed: 'int' = 42, discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'Tuple[float, bool]'`** — Compute a per-dyad surrogate-derived onset threshold from raw signals.
- **`extract_dynamic_features(wcc: 'np.ndarray', hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, max_nan_ratio: 'float' = 0.2, height: 'Optional[float]' = None, distance: 'Optional[int]' = None, prominence: 'Optional[float]' = None, aggregation: 'str' = 'mean', return_raw_profiles: 'bool' = False, wcc_window_sec: 'float' = 1.0, gap_policy: 'Optional[str]' = None) -> 'Any'`** — Extract features from a WCC series via the SSoT.
- **`extract_features_all_pairs(dataset: "'SynchronyDataset'", window_size: 'int' = 10, hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, wcc_window_sec: 'float' = 1.0, use_surrogate_threshold: 'bool' = True, surrogate_n: 'int' = 200, surrogate_seed: 'int' = 42, discontinuity_mask: 'Optional[np.ndarray]' = None, cross_modal: 'bool' = False) -> 'Tuple[Dict[str, DynamicFeatures], Dict[str, Dict[str, Any]]]'`** — Compute WCC + dynamic features for all modality pairs.
- **`extract_features_segmented(dataset: "'SynchronyDataset'", window_size: 'int' = 10, hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, max_nan_ratio: 'float' = 0.2, wcc_window_sec: 'float' = 1.0, use_surrogate_threshold: 'bool' = True, surrogate_n: 'int' = 200, surrogate_seed: 'int' = 42, discontinuity_mask: 'Optional[np.ndarray]' = None, cross_modal: 'bool' = False) -> 'Tuple[Dict[str, Dict[str, DynamicFeatures]], Dict[str, Dict[str, Any]]]'`** — Compute WCC + dynamic features per CONTEXT segment.
- **`ft_surrogate(x: numpy.ndarray, rng: numpy.random._generator.Generator) -> numpy.ndarray`** — Generate one FT surrogate (Fourier-phase randomization) of ``x``.
- **`iaaft_surrogate(x: numpy.ndarray, rng: numpy.random._generator.Generator, max_iter: int = 200, tol: float = 1e-08) -> numpy.ndarray`** — Generate one IAAFT surrogate of ``x``.
- **`iter_dyad_pairs(dataset, cross_modal: 'bool' = False)`** — Yield (src_key, name_a, name_b, col_a, col_b, x, y) for each valid pair.
- **`pairing_policy(dataset, cross_modal: 'bool' = False) -> 'str'`** — Return the effective dyad-pairing policy for the result manifest.
- **`prtf_surrogate(x: numpy.ndarray, rng: numpy.random._generator.Generator) -> numpy.ndarray`** — Generate one FT surrogate (Fourier-phase randomization) of ``x``.
- **`resolve_signal_geometry(signal_a: 'np.ndarray', signal_b: 'np.ndarray', discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'SignalGeometry'`** — Resolve the shared mask and segments without deleting any sample.
- **`sliding_window_wcc(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect') -> 'np.ndarray'`** — Compute sliding-window cross-correlation (WCC) between x and y.
- **`sliding_window_wcc_masked(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect', discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'np.ndarray'`** — sliding_window_wcc with discontinuity-boundary gating.
- **`wcc_surrogate_test(wcc: 'np.ndarray', hz: 'float' = 1.0, surrogate_n: 'int' = 5000, alpha: 'float' = 0.05, seed: 'int' = 42, method: 'str' = 'iaaft', raw_signals: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None, wcc_window_size: 'Optional[int]' = None, wcc_window_sec: 'Optional[float]' = None, window_type: 'str' = 'rect', min_wcc_points: 'int' = 30, null_model: 'str' = 'iaaft', block_size: 'Optional[int]' = None, threshold: 'float' = 0.5, discontinuity_mask: 'Optional[np.ndarray]' = None, min_segment_samples: 'Optional[int]' = None) -> 'Dict[str, Any]'`** — Test significance of WCC features using surrogate data.

## Feature contracts (registry)

Module: `syncpipe.feature_definitions`

### Classes

- **`DynamicFeatures(onset_latency: 'float' = nan, rise_time: 'float' = nan, peak_amplitude: 'float' = nan, peak_abs_amplitude: 'float' = nan, recovery_time: 'float' = nan, dwell_time: 'float' = nan, switching_rate: 'float' = nan, synchrony_entropy: 'float' = nan, mean_synchrony: 'float' = nan, bimodality_coefficient: 'float' = nan, fraction_above_threshold: 'float' = nan, inter_peak_cv: 'float' = nan, first_peak_time: 'float' = nan, onset_latency_imputed: 'float' = nan, rise_time_imputed: 'float' = nan, recovery_time_imputed: 'float' = nan, onset_defined: 'int' = 0, rise_defined: 'int' = 0, recovery_defined: 'int' = 0, nan_fraction: 'float' = nan, notes: 'str' = '', params: 'Dict[str, Any]' = <factory>) -> None`** — Container for FDR-family features + reference + diagnostics + definedness flags.

### Functions

- **`check_eligibility(n_wcc_points: 'int', n_dyads: 'int') -> 'Tuple[bool, bool]'`** — Lightweight eligibility gate for SyncPipe analysis (B3 freeze).
- **`compute_baseline_fraction(wcc: 'np.ndarray', hz: 'float', threshold: 'float' = 0.5, min_prominence: 'float' = 0.15, prominence_window_sec: 'float' = 50.0) -> 'float'`** — Fraction of samples below threshold *before* the first prominent peak.
- **`compute_bimodality_coefficient(wcc: 'np.ndarray') -> 'float'`** — Bimodality Coefficient (BC) of the WCC amplitude distribution.
- **`compute_dwell_time(wcc: 'np.ndarray', hz: 'float', threshold: 'float' = 0.5, hysteresis_delta: 'float' = 0.05, gap_policy: 'str' = 'merge_valid') -> 'float'`** — DECISION-06a · dwell_time = mean elevated run-length (seconds).
- **`compute_first_peak_time(wcc: 'np.ndarray', hz: 'float', threshold: 'float' = 0.5, min_prominence: 'float' = 0.15, prominence_window_sec: 'float' = 50.0) -> 'float'`** — Time of the first prominent peak above threshold (seconds).
- **`compute_fraction_above_threshold(wcc: 'np.ndarray', threshold: 'float' = 0.5) -> 'float'`** — Exploratory occupancy: fraction of finite WCC values >= threshold.
- **`compute_inter_peak_cv(wcc: 'np.ndarray', hz: 'float', threshold: 'float' = 0.5, min_prominence: 'float' = 0.15, min_peaks: 'int' = 3, prominence_window_sec: 'float' = 50.0) -> 'float'`** — Coefficient of variation of inter-peak intervals (CV = std / mean).
- **`compute_mean_synchrony(wcc: 'np.ndarray') -> 'float'`** — Reference: arithmetic mean over finite WCC values.
- **`compute_onset_latency(wcc: 'np.ndarray', hz: 'float', wcc_window_sec: 'float', threshold: 'float' = 0.5) -> 'Tuple[float, int]'`** — DECISION-02 · onset_latency.
- **`compute_peak_abs_amplitude(wcc_smoothed: 'np.ndarray') -> 'float'`** — DECISION-04b · peak_abs_amplitude = max |3-point smoothed WCC|.
- **`compute_peak_amplitude(wcc_smoothed: 'np.ndarray') -> 'Tuple[float, Optional[int]]'`** — DECISION-04 · peak_amplitude = max of 3-point smoothed WCC.
- **`compute_recovery_time(wcc: 'np.ndarray', peak_index: 'int', peak_value: 'float', hz: 'float', baseline: 'float' = 0.5) -> 'Tuple[float, int]'`** — DECISION-05 · half-recovery time.
- **`compute_rise_time(wcc: 'np.ndarray', peak_index: 'int', peak_value: 'float', hz: 'float', baseline: 'float' = 0.5) -> 'Tuple[float, int]'`** — DECISION-03 · rise_time.
- **`compute_surrogate_threshold(wcc_surrogates: 'np.ndarray', percentile: 'float' = 95.0) -> 'Tuple[float, bool]'`** — Compute a per-dyad surrogate-derived onset threshold.
- **`compute_switching_rate(wcc: 'np.ndarray', hz: 'float', threshold: 'float' = 0.5, hysteresis_delta: 'float' = 0.05, gap_policy: 'str' = 'merge_valid') -> 'float'`** — DECISION-06b · switching_rate = state transitions per minute.
- **`compute_synchrony_entropy(wcc: 'np.ndarray', n_bins: 'int' = 20) -> 'float'`** — Conditional: Shannon entropy of WCC amplitude distribution.
- **`extract_features(wcc: 'np.ndarray', hz: 'float', wcc_window_sec: 'float', threshold: 'float' = 0.5, paradigm: 'str' = 'auto', gap_policy: 'Optional[str]' = None) -> 'DynamicFeatures'`** — Compute features + diagnostics from a WCC series.
- **`find_dominant_peak(wcc_smoothed: 'np.ndarray') -> 'Optional[int]'`** — Return index of the dominant (= global argmax) smoothed peak,
- **`get_fdr_features(full_family_fdr: 'bool' = False) -> 'List[str]'`** — Return the feature set entered into the L2 between-condition BH-FDR.
- **`get_primary_fdr_features() -> 'List[str]'`** — Pre-registered PRIMARY confirmatory endpoints (single, peak_amplitude).
- **`get_secondary_fdr_features() -> 'List[str]'`** — SECONDARY confirmatory descriptors reported in parallel (dwell_time,
- **`smoothed_wcc(wcc: 'np.ndarray', window: 'int' = 3) -> 'np.ndarray'`** — 3-point boxcar smoothing with same-mode boundary (DECISION-04).

## Inference (L0/L1/L2 evidence chain)

Module: `syncpipe.inference_pipeline`

### Classes

- **`EvidenceChain(version: 'str', endpoint: 'str', stages: 'Tuple[EvidenceStageResult, ...]', profile: 'EvidenceProfile', decision: 'ClaimDecision') -> None`** — All checks and the conclusion used by REPORT.md and JSON output.
- **`InferencePipeline(features_df: pandas.core.frame.DataFrame, hz: float = 4.0, wcc_window_sec: Optional[float] = None, surrogate_n: int = 100, seed: int = 42, n_workers: int = 1)`** — Run the study-level checks and condition comparison.

### Functions

- **`across_stim_shuffle_test(segments: 'List[Tuple[str, np.ndarray, np.ndarray]]', wcc_func: 'Callable[[np.ndarray, np.ndarray], np.ndarray]', feature_func: 'Callable[[np.ndarray], Dict[str, float]]', n_surr: 'int' = 499, seed: 'int' = 2026, feature_names: 'Optional[List[str]]' = None) -> 'Dict[str, Dict]'`** — Full across-stimulus shuffle test.
- **`between_condition_by_modality(df: 'pd.DataFrame', modality_col: 'str' = 'modality', condition_col: 'str' = 'condition', dyad_col: 'str' = 'dyad_label', feature_cols: 'Optional[Sequence[str]]' = None, n_permutations: 'int' = 10000, seed: 'int' = 42, alpha: 'float' = 0.05, condition_values: 'Optional[Tuple[str, str]]' = None, threshold_scope: 'str' = 'unknown', observation_col: 'Optional[str]' = 'n_wcc_points', observation_policy: 'str' = 'warn', undefined_policy: 'str' = 'flag', min_defined_fraction: 'float' = 0.5, eligibility_policy: 'str' = 'warn', n_min_dyads: 'int' = 10) -> 'Dict[str, Dict]'`** — Run L2 between-condition test split by modality.
- **`between_condition_fdr(df: 'pd.DataFrame', condition_col: 'str' = 'condition', dyad_col: 'str' = 'dyad_label', feature_cols: 'Optional[Sequence[str]]' = None, n_permutations: 'int' = 10000, seed: 'int' = 42, alpha: 'float' = 0.05, condition_values: 'Optional[Tuple[str, str]]' = None, threshold_scope: 'str' = 'unknown', modality_col: 'Optional[str]' = 'modality', allow_multimodal_pool: 'bool' = False, observation_col: 'Optional[str]' = 'n_wcc_points', observation_policy: 'str' = 'warn', undefined_policy: 'str' = 'flag', min_defined_fraction: 'float' = 0.5, eligibility_policy: 'str' = 'warn', n_min_dyads: 'int' = 10) -> 'Dict[str, Union[List[L2Result], L2Result]]'`** — L2 between-condition permutation test with BH-FDR correction.
- **`build_evidence_chain(*, endpoint: 'str', existence_gate: 'Dict[str, Any]', design: 'Optional[Dict[str, Any]]', across_stimulus: 'Optional[Dict[str, Any]]', group: 'Optional[Dict[str, Any]]', alpha: 'float') -> 'EvidenceChain'`** — Create typed stages and propagate the strongest defensible claim.
- **`design_control_audit(signal_pairs: 'Mapping[str, SignalPair]', *, hz: 'float', window_size: 'int', threshold: 'Union[float, Mapping[str, float]]' = 0.5, feature_names: 'Sequence[str]' = ('mean_synchrony', 'peak_amplitude', 'fraction_above_threshold', 'dwell_time', 'switching_rate'), n_pseudo_per_dyad: 'int' = 10, shift_lags_sec: 'Sequence[float]' = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0), seed: 'int' = 42, window_type: 'str' = 'rect', discontinuity_masks: 'Optional[Mapping[str, np.ndarray]]' = None) -> 'Dict[str, Any]'`** — Run pseudo-pair and time-shift design controls for a cohort.
- **`extract_features(wcc: 'np.ndarray', hz: 'float', wcc_window_sec: 'float', threshold: 'float' = 0.5, paradigm: 'str' = 'auto', gap_policy: 'Optional[str]' = None) -> 'DynamicFeatures'`** — Compute features + diagnostics from a WCC series.
- **`get_fdr_features(full_family_fdr: 'bool' = False) -> 'List[str]'`** — Return the feature set entered into the L2 between-condition BH-FDR.
- **`sliding_window_wcc(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect') -> 'np.ndarray'`** — Compute sliding-window cross-correlation (WCC) between x and y.
- **`synchrony_existence_audit(sig_a: 'np.ndarray', sig_b: 'np.ndarray', *, hz: 'float', window_size: 'int', surrogate_n: 'int' = 100, seed: 'int' = 42, window_type: 'str' = 'rect', discontinuity_mask: 'Optional[np.ndarray]' = None, min_segment_samples: 'Optional[int]' = None) -> 'Dict[str, Any]'`** — Run signal-level IAAFT synchrony-existence audit for one pair.
- **`wcc_surrogate_test(wcc: 'np.ndarray', hz: 'float' = 1.0, surrogate_n: 'int' = 5000, alpha: 'float' = 0.05, seed: 'int' = 42, method: 'str' = 'iaaft', raw_signals: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None, wcc_window_size: 'Optional[int]' = None, wcc_window_sec: 'Optional[float]' = None, window_type: 'str' = 'rect', min_wcc_points: 'int' = 30, null_model: 'str' = 'iaaft', block_size: 'Optional[int]' = None, threshold: 'float' = 0.5, discontinuity_mask: 'Optional[np.ndarray]' = None, min_segment_samples: 'Optional[int]' = None) -> 'Dict[str, Any]'`** — Test significance of WCC features using surrogate data.

## Design controls

Module: `syncpipe.design_controls`

### Functions

- **`design_control_audit(signal_pairs: 'Mapping[str, SignalPair]', *, hz: 'float', window_size: 'int', threshold: 'Union[float, Mapping[str, float]]' = 0.5, feature_names: 'Sequence[str]' = ('mean_synchrony', 'peak_amplitude', 'fraction_above_threshold', 'dwell_time', 'switching_rate'), n_pseudo_per_dyad: 'int' = 10, shift_lags_sec: 'Sequence[float]' = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0), seed: 'int' = 42, window_type: 'str' = 'rect', discontinuity_masks: 'Optional[Mapping[str, np.ndarray]]' = None) -> 'Dict[str, Any]'`** — Run pseudo-pair and time-shift design controls for a cohort.
- **`extract_features(wcc: 'np.ndarray', hz: 'float', wcc_window_sec: 'float', threshold: 'float' = 0.5, paradigm: 'str' = 'auto', gap_policy: 'Optional[str]' = None) -> 'DynamicFeatures'`** — Compute features + diagnostics from a WCC series.
- **`extract_pair_features(sig_a: 'np.ndarray', sig_b: 'np.ndarray', *, hz: 'float', window_size: 'int', threshold: 'float' = 0.5, feature_names: 'Sequence[str]' = ('mean_synchrony', 'peak_amplitude', 'fraction_above_threshold', 'dwell_time', 'switching_rate'), window_type: 'str' = 'rect', discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'Dict[str, float]'`** — Compute WCC and selected SyncPipe features for one signal pair.
- **`sliding_window_wcc(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect') -> 'np.ndarray'`** — Compute sliding-window cross-correlation (WCC) between x and y.
- **`synchrony_existence_audit(sig_a: 'np.ndarray', sig_b: 'np.ndarray', *, hz: 'float', window_size: 'int', surrogate_n: 'int' = 100, seed: 'int' = 42, window_type: 'str' = 'rect', discontinuity_mask: 'Optional[np.ndarray]' = None, min_segment_samples: 'Optional[int]' = None) -> 'Dict[str, Any]'`** — Run signal-level IAAFT synchrony-existence audit for one pair.
- **`wcc_surrogate_test(wcc: 'np.ndarray', hz: 'float' = 1.0, surrogate_n: 'int' = 5000, alpha: 'float' = 0.05, seed: 'int' = 42, method: 'str' = 'iaaft', raw_signals: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None, wcc_window_size: 'Optional[int]' = None, wcc_window_sec: 'Optional[float]' = None, window_type: 'str' = 'rect', min_wcc_points: 'int' = 30, null_model: 'str' = 'iaaft', block_size: 'Optional[int]' = None, threshold: 'float' = 0.5, discontinuity_mask: 'Optional[np.ndarray]' = None, min_segment_samples: 'Optional[int]' = None) -> 'Dict[str, Any]'`** — Test significance of WCC features using surrogate data.

## Morphology analysis

Module: `syncpipe.morphology`

### Classes

- **`MorphologyAnalyzer(wcc_traces: 'List[np.ndarray]', hz: 'float' = 1.0)`** — High-level wrapper for the two morphology methods.

### Functions

- **`collinearity_report(feat_df: 'pd.DataFrame', features: 'List[str]') -> 'Tuple[pd.DataFrame, pd.Series]'`** — Return Spearman correlation matrix and VIF for a feature set.
- **`episode_archetype_cluster(wcc_traces: 'List[np.ndarray]', threshold: 'float' = 0.5, threshold_mode: 'str' = 'fixed', percentile: 'float' = 75.0, resample_len: 'int' = 20, min_len: 'int' = 4, k_range: 'Tuple[int, ...]' = (2, 3, 4, 5), seed: 'int' = 42) -> 'Dict[str, object]'`** — Cluster episodes into waveform archetypes (Method 2).
- **`episode_shape_features(ep: 'np.ndarray') -> 'Dict[str, float]'`** — Return shape features of a single episode.
- **`extract_dynamic_features(wcc: 'np.ndarray', hz: 'float' = 1.0, onset_threshold: 'Optional[float]' = None, onset_k: 'float' = 2.0, max_nan_ratio: 'float' = 0.2, height: 'Optional[float]' = None, distance: 'Optional[int]' = None, prominence: 'Optional[float]' = None, aggregation: 'str' = 'mean', return_raw_profiles: 'bool' = False, wcc_window_sec: 'float' = 1.0, gap_policy: 'Optional[str]' = None) -> 'Any'`** — Extract features from a WCC series via the SSoT.
- **`extract_episodes(wcc: 'np.ndarray', threshold: 'float' = 0.5, threshold_mode: 'str' = 'fixed', percentile: 'float' = 75.0, min_len: 'int' = 4) -> 'List[np.ndarray]'`** — Extract contiguous high-synchrony episodes from a WCC trace.
- **`incremental_value(feat_df: 'pd.DataFrame', y: 'np.ndarray', features: 'List[str]', baseline_drop_meansync: 'bool' = True, n_orders: 'int' = 20, seed: 'int' = 42) -> 'Tuple[pd.DataFrame, Dict]'`** — Random-order-averaged marginal AUC + LOFO.
- **`matched_mean_contrast(feat_df: 'pd.DataFrame', y: 'np.ndarray', features: 'List[str]', band: 'float' = 0.1) -> 'pd.DataFrame'`** — Within a narrow mean_synchrony band, which feature separates morphology?
- **`morphology_feature_table(wcc_traces: 'List[np.ndarray]', hz: 'float' = 1.0, wcc_window_sec: 'Optional[float]' = None, prominence: 'float' = 0.1, onset_threshold: 'Optional[float]' = None) -> 'pd.DataFrame'`** — Return a DataFrame with shape descriptors + SyncPipe features per trace.
- **`resample_waveform(ep: 'np.ndarray', L: 'int' = 20) -> 'np.ndarray'`** — Resample an episode to a fixed length L and amplitude-normalise.
- **`scalefree_descriptors(wcc: 'np.ndarray', prominence: 'float' = 0.1) -> 'Optional[Dict[str, float]]'`** — Return scale-free shape descriptors of a WCC trace.
- **`trace_shape_cluster(wcc_traces: 'List[np.ndarray]', max_k: 'int' = 5, seed: 'int' = 42, prominence: 'float' = 0.1) -> 'Dict[str, object]'`** — Cluster WCC traces by scale-free shape descriptors (Method 1).

## Session-pooled thresholds

Module: `syncpipe.session_threshold`

### Functions

- **`compute_condition_pooled_thresholds(condition_signals: 'Dict[str, List[Tuple[np.ndarray, np.ndarray]]]', hz: 'float', wcc_window_size: 'int', surrogate_n: 'int' = 200, percentile: 'float' = 95.0, seed: 'int' = 42, surrogate_method: 'str' = 'iaaft', backend: 'str' = 'wcc', wclr_max_lag_samples: 'int' = 2, fallback_threshold: 'float' = 0.5) -> 'Dict[str, Tuple[float, Dict]]'`** — Compute one pooled surrogate threshold per experimental condition.
- **`compute_session_pooled_threshold(dyad_signals: 'List[Tuple[np.ndarray, np.ndarray]]', hz: 'float', wcc_window_size: 'int', surrogate_n: 'int' = 200, percentile: 'float' = 95.0, seed: 'int' = 42, surrogate_method: 'str' = 'iaaft', backend: 'str' = 'wcc', wclr_max_lag_samples: 'int' = 2, fallback_threshold: 'float' = 0.5, discontinuity_masks: 'Optional[List[Optional[np.ndarray]]]' = None) -> 'Tuple[float, Dict]'`** — Compute a single surrogate threshold pooled across all dyads.
- **`compute_session_pooled_thresholds_by_modality(dyad_signals: 'List[Tuple[np.ndarray, np.ndarray]]', modalities: 'List[str]', hz: 'float', wcc_window_size: 'int', surrogate_n: 'int' = 200, percentile: 'float' = 95.0, seed: 'int' = 42, surrogate_method: 'str' = 'iaaft', backend: 'str' = 'wcc', wclr_max_lag_samples: 'int' = 2, fallback_threshold: 'float' = 0.5, discontinuity_masks: 'Optional[List[Optional[np.ndarray]]]' = None, return_meta: 'bool' = False) -> 'Union[Dict[str, float], Dict[str, Tuple[float, Dict]]]'`** — Compute one surrogate threshold per modality (per-modality pooled null).
- **`compute_surrogate_threshold(wcc_surrogates: 'np.ndarray', percentile: 'float' = 95.0) -> 'Tuple[float, bool]'`** — Compute a per-dyad surrogate-derived onset threshold.
- **`ft_surrogate(x: numpy.ndarray, rng: numpy.random._generator.Generator) -> numpy.ndarray`** — Generate one FT surrogate (Fourier-phase randomization) of ``x``.
- **`iaaft_surrogate(x: numpy.ndarray, rng: numpy.random._generator.Generator, max_iter: int = 200, tol: float = 1e-08) -> numpy.ndarray`** — Generate one IAAFT surrogate of ``x``.
- **`resolve_signal_geometry(signal_a: 'np.ndarray', signal_b: 'np.ndarray', discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'SignalGeometry'`** — Resolve the shared mask and segments without deleting any sample.
- **`sliding_window_wcc(x: 'np.ndarray', y: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, lag_samples: 'int' = 0, step_samples: 'int' = 0, min_valid_ratio: 'float' = 0.5, window_type: 'str' = 'rect') -> 'np.ndarray'`** — Compute sliding-window cross-correlation (WCC) between x and y.
- **`wclr_coupling_trace(sig_a: 'np.ndarray', sig_b: 'np.ndarray', window_size: 'int', hz: 'float' = 1.0, max_lag_samples: 'int' = 2, step_samples: 'int' = 1, min_valid_ratio: 'float' = 0.5, metric: 'str' = 'beta', absolute_beta: 'bool' = True, sign_stable: 'bool' = False) -> 'np.ndarray'`** — Convenience wrapper returning only the WCLR coupling trace.

## Feature status tables

Module: `syncpipe.feature_status`

### Functions

- **`feature_status_latex(path: 'str | None' = None, *, caption: 'str' = 'SyncPipe v1 feature-status table.', label: 'str' = 'tab:syncpipe_feature_status') -> 'str'`** — Return a manuscript-ready LaTeX longtable for Table 1.
- **`feature_status_table(as_dataframe: 'bool' = True)`** — Return the external-facing v1 feature status table.

## Quality control

Module: `syncpipe.qc`

### Classes

- **`DataQualityError(...)`** — Raised when quality check fails at FAIL level.
- **`DataQualityReport(dyad_id: 'str', stages: 'List[StageResult]' = <factory>, n_warnings: 'int' = 0, n_failures: 'int' = 0, warnings: 'List[str]' = <factory>, failures: 'List[str]' = <factory>, notes: 'List[str]' = <factory>, clock_offset_caveat: 'str' = '', co_start_verified: 'bool' = False) -> None`** — Aggregated data quality check report.
- **`StageResult(stage: 'str', verdict: 'str', details: 'List[Dict[str, Any]]' = <factory>, message: 'str' = '') -> None`** — Result for a single QC stage.
- **`StageVerdict()`** — Nominal verdict for a single QC stage.

### Functions

- **`check_eligibility(n_wcc_points: 'int', n_dyads: 'int') -> 'Tuple[bool, bool]'`** — Lightweight eligibility gate for SyncPipe analysis (B3 freeze).
- **`format_qc_report(report: 'DataQualityReport') -> 'str'`** — Return a concise user-facing QC message.
- **`run_quality_check(dataset: 'Any', config: 'Optional[Dict[str, Any]]' = None, raise_on_fail: 'bool' = False, eligibility: 'Optional[Dict[str, int]]' = None) -> 'DataQualityReport'`** — Run the 4-stage data quality check pipeline.

## Data loading

Module: `syncpipe.io`

### Classes

- **`AnalysisResults(dyad_id: 'str', dynamic_features: 'Dict[str, Dict[str, float]]' = <factory>, dynamic_features_segmented: 'Dict[str, Dict[str, Dict[str, float]]]' = <factory>, threshold_meta: 'Dict[str, Dict[str, Any]]' = <factory>, score_view: 'List[Dict[str, Any]]' = <factory>, diagnostics: 'List[Dict[str, Any]]' = <factory>, parameters: 'Dict[str, Any]' = <factory>) -> None`** — Complete analysis output — ready for Viewer JSON export.

### Functions

- **`export_viewer_json(results: 'AnalysisResults', filepath: 'str') -> 'str'`** — Export analysis results to Viewer-ready JSON.
- **`load_analysis_results(filepath: 'str') -> 'Dict[str, Any]'`** — Load a previously exported Viewer JSON (for inspection/testing).
- **`load_csv(filepath: 'str', time_col: 'str' = 'time', value_col: 'Optional[str]' = None) -> 'pd.DataFrame'`** — Load a single CSV file as a DataFrame.
- **`load_multimodal_csv(modality_files: 'Dict[str, str]', time_col: 'str' = 'time', value_col: 'Optional[str]' = None) -> 'Dict[str, pd.DataFrame]'`** — Load multiple CSV files, one per modality.

## Synthetic ground truth

Module: `syncpipe.synthetic`

### Classes

- **`SynchronyDataset(dyad_id: 'str', modalities: 'Optional[Dict[str, pd.DataFrame]]' = None, discontinuity_mask: 'Optional[np.ndarray]' = None) -> 'None'`** — Container for one dyad's multi-modal time-series.

### Functions

- **`generate_ground_truth_dyad(lead_modality: 'str' = 'behavior', lag_modality: 'str' = 'neural', true_lag_sec: 'float' = 12.0, noise_ratio: 'float' = 0.3, duration_sec: 'float' = 300.0, hz: 'float' = 1.0, seed: 'int' = 42, n_bursts: 'int' = 5, burst_sigma: 'float' = 3.0, gap_prob: 'float' = 0.02, morphology: "Literal['identical', 'divergent']" = 'identical', coupling: 'float' = 0.7) -> 'SynchronyDataset'`** — Generate synthetic dyad with controllable lead-lag + coupling.
- **`generate_multimodal_dyad(duration_sec: 'float' = 300.0, hz: 'float' = 1.0, seed: 'int' = 42, modalities: 'Optional[Dict[str, float]]' = None, noise_ratio: 'float' = 0.3) -> 'SynchronyDataset'`** — Generate a synthetic dyad with 3-4 modalities at different Hz.
- **`generate_shared_stim_null_dyad(modality: 'str' = 'behavior', shared_drive_strength: 'float' = 0.5, noise_ratio: 'float' = 0.5, duration_sec: 'float' = 180.0, hz: 'float' = 1.0, seed: 'int' = 42, n_bursts: 'int' = 5, burst_sigma: 'float' = 3.0) -> 'SynchronyDataset'`** — Generate shared-stimulus null dyad (zero inter-person coupling).

## External validation kit

Module: `syncpipe.external_validation`

### Functions

- **`audit_external_bundle(result_dir: 'str | Path') -> 'Dict[str, Any]'`** — Audit result structure without asserting expected scientific findings.
- **`create_external_validation_kit(output_dir: 'str | Path', *, seed: 'int' = 20260819, n_dyads: 'int' = 4) -> 'Dict[str, str]'`** — Create a deterministic, publication-ineligible external usability kit.

## Version migration

Module: `syncpipe.migration`

### Functions

- **`analysis_spec_from_mapping(mapping: 'Mapping[str, Any]', *, require_declarations: 'bool' = True) -> 'AnalysisSpec'`** — Parse a mapping through the one typed AnalysisSpec contract.
- **`detect_config_contract(path: 'str | Path') -> 'str'`** — 
- **`detect_manifest_contract(path: 'str | Path') -> 'str'`** — 
- **`migrate_config_v1_to_v2(source: 'str | Path', destination: 'str | Path', *, primary_endpoint: 'str', primary_modalities: 'Sequence[str]') -> 'Dict[str, Any]'`** — 
- **`migrate_manifest_v1_to_v2(source: 'str | Path', destination: 'str | Path', *, signal_type: 'str', unit: 'str', preprocessing_path: 'str') -> 'Dict[str, Any]'`** — 
- **`migrate_v1_project(*, manifest: 'str | Path', config: 'str | Path', output_dir: 'str | Path', signal_type: 'str', unit: 'str', preprocessing_path: 'str', primary_modalities: 'Sequence[str]', primary_endpoint: 'str' = 'peak_amplitude') -> 'Dict[str, Any]'`** — 


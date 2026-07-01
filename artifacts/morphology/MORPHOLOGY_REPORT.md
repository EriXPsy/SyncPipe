# SyncPipe — WCC Morphology / Collinearity / Incremental-Value Report

- **Data**: Lerique pilot traces  |  n traces = 176  |  **EXPLORATORY** (small-n; report effect sizes + CIs, not confirmatory).

## (A) Data-driven morphology clusters
- Best k = **3** (silhouette = 0.429, Agglomerative on standardised SHAPE descriptors).

Per-cluster mean of SyncPipe features:

|   cluster |   mean_synchrony |   peak_amplitude |   dwell_time |   switching_rate |   bimodality_coefficient |   synchrony_entropy |   onset_latency |   rise_time |   recovery_time |
|----------:|-----------------:|-----------------:|-------------:|-----------------:|-------------------------:|--------------------:|----------------:|------------:|----------------:|
|         0 |            0.11  |            0.88  |       21.15  |            1.109 |                    0.489 |               4.041 |         793.814 |     241.171 |          32.157 |
|         1 |           -0.006 |            0.466 |       10.286 |            0.244 |                    0.439 |               3.892 |         355.202 |     265.971 |         186.067 |
|         2 |            0.065 |            0.812 |       12.756 |            1.085 |                    0.441 |               4.074 |        1051     |     215     |           2     |

## (B) Collinearity

Spearman correlation matrix:

|                        |   mean_synchrony |   peak_amplitude |   dwell_time |   switching_rate |   bimodality_coefficient |   synchrony_entropy |   onset_latency |   rise_time |   recovery_time |
|:-----------------------|-----------------:|-----------------:|-------------:|-----------------:|-------------------------:|--------------------:|----------------:|------------:|----------------:|
| mean_synchrony         |             1    |             0.55 |         0.53 |             0.64 |                     0.18 |                0.09 |            0.04 |       -0.18 |           -0.19 |
| peak_amplitude         |             0.55 |             1    |         0.76 |             0.87 |                     0.22 |                0.33 |            0.41 |       -0.15 |           -0.45 |
| dwell_time             |             0.53 |             0.76 |         1    |             0.45 |                     0.49 |                0.3  |            0.04 |       -0.05 |            0.67 |
| switching_rate         |             0.64 |             0.87 |         0.45 |             1    |                     0.31 |                0.32 |            0.17 |       -0.26 |           -0.5  |
| bimodality_coefficient |             0.18 |             0.22 |         0.49 |             0.31 |                     1    |                0.4  |           -0.33 |       -0.28 |           -0.07 |
| synchrony_entropy      |             0.09 |             0.33 |         0.3  |             0.32 |                     0.4  |                1    |            0.12 |       -0.02 |           -0.13 |
| onset_latency          |             0.04 |             0.41 |         0.04 |             0.17 |                    -0.33 |                0.12 |            1    |        0.42 |           -0.03 |
| rise_time              |            -0.18 |            -0.15 |        -0.05 |            -0.26 |                    -0.28 |               -0.02 |            0.42 |        1    |            0.26 |
| recovery_time          |            -0.19 |            -0.45 |         0.67 |            -0.5  |                    -0.07 |               -0.13 |           -0.03 |        0.26 |            1    |

Variance Inflation Factor (VIF > 5 = concerning, > 10 = severe):

|                        |   VIF |
|:-----------------------|------:|
| mean_synchrony         |  2.61 |
| peak_amplitude         |  5.56 |
| dwell_time             |  2.02 |
| switching_rate         |  4.92 |
| bimodality_coefficient |  1.81 |
| synchrony_entropy      |  1.34 |
| onset_latency          |  2.87 |
| rise_time              |  2.07 |
| recovery_time          |  2.03 |

⚠️ High-VIF features (redundant): **peak_amplitude** — candidates to drop/merge before confirmatory FDR.

## (C) Order-unbiased incremental value (cluster as target)

> Marginal AUC averaged over random insertion orders (Shapley-style) removes the fixed-ordering bias of cumulative incremental steps. LOFO = AUC lost when the feature is removed from the full model.


**Baseline = drop mean_synchrony** (base AUC=0.500, full AUC=0.914):

| feature                |   shapley_marginal_auc |   lofo_auc_drop | baseline            |
|:-----------------------|-----------------------:|----------------:|:--------------------|
| peak_amplitude         |                 0.1382 |          0.0173 | drop_mean_synchrony |
| onset_latency          |                 0.0749 |          0.0058 | drop_mean_synchrony |
| dwell_time             |                 0.0593 |          0.0233 | drop_mean_synchrony |
| switching_rate         |                 0.0517 |         -0.0042 | drop_mean_synchrony |
| recovery_time          |                 0.0443 |         -0.0024 | drop_mean_synchrony |
| bimodality_coefficient |                 0.035  |         -0.0044 | drop_mean_synchrony |
| synchrony_entropy      |                 0.0311 |         -0.006  | drop_mean_synchrony |
| rise_time              |                -0.0205 |          0.0023 | drop_mean_synchrony |

**Baseline = keep mean_synchrony** (base AUC=0.575, full AUC=0.913):

| feature                |   shapley_marginal_auc |   lofo_auc_drop | baseline            |
|:-----------------------|-----------------------:|----------------:|:--------------------|
| peak_amplitude         |                 0.1192 |          0.0146 | keep_mean_synchrony |
| onset_latency          |                 0.0797 |          0.0142 | keep_mean_synchrony |
| switching_rate         |                 0.0412 |         -0.0032 | keep_mean_synchrony |
| dwell_time             |                 0.0381 |          0.026  | keep_mean_synchrony |
| recovery_time          |                 0.0285 |         -0.0005 | keep_mean_synchrony |
| synchrony_entropy      |                 0.0277 |         -0.0066 | keep_mean_synchrony |
| bimodality_coefficient |                 0.0112 |         -0.0039 | keep_mean_synchrony |
| rise_time              |                -0.0076 |          0.0036 | keep_mean_synchrony |

## (D) Matched-mean-synchrony contrast

> Among traces in a narrow mean_synchrony band, which single feature best separates morphology clusters? This is the core test of whether SHAPE carries information beyond synchrony MAGNITUDE.

| feature                |   auc_within_mean_band |   n_in_band |
|:-----------------------|-----------------------:|------------:|
| peak_amplitude         |                    nan |         109 |
| dwell_time             |                    nan |         109 |
| switching_rate         |                    nan |         109 |
| bimodality_coefficient |                    nan |         109 |
| synchrony_entropy      |                    nan |         109 |
| onset_latency          |                    nan |         109 |
| rise_time              |                    nan |         109 |
| recovery_time          |                    nan |         109 |

## Honest limitations
- **n is small**: all AUCs are exploratory with wide CIs; clusters may be unstable. Report bootstrap CIs and cluster-stability (e.g. ARI under resampling) before any claim.
- **Partial circularity caveat**: clusters come from SHAPE descriptors; predictors are the 9 SyncPipe features. These are different feature sets, but some MS features (dwell, switching) correlate with shape descriptors by construction, so 'predicting cluster' partly recovers the clustering geometry. Interpret incremental AUC as *descriptive structure*, not out-of-sample morphology classification.
- **Collinearity** directly threatens FDR validity on correlated features; see (B). High-VIF features should not be treated as independent tests.
# Real-data L2 audit (new SyncPipe inference layer)

n_permutations=10000, alpha=0.05, seed=42

## Lerique (paired design)
- contrast: ['rest1', 'trials_concat']

- note: paired key = dyad_label (31 dyads); per-modality L2 is the mandated report

### Per-modality L2 — ECG (n_dyads=27, significant=1)

| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |
|---|---|---|---|---|---|---|---|
| peak_amplitude | -0.2082 | 0.0001 | 0.0004 | True | -3.589 | 27/27 | 1.0 |
| dwell_time | -5.3333 | 0.3753 | 0.3753 | False | -1.231 | 10/26 | 0.0001 |
| switching_rate | -0.2284 | 0.2845 | 0.3753 | False | -1.354 | 27/27 | 1.0 |
| mean_synchrony | -0.0356 | 0.2821 | 0.3753 | False | -1.212 | 27/27 | 1.0 |

### Per-modality L2 — EDA (n_dyads=30, significant=4)

| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |
|---|---|---|---|---|---|---|---|
| peak_amplitude | -0.3913 | 0.0001 | 0.0004 | True | -2.66 | 30/30 | 1.0 |
| dwell_time | -10.9583 | 0.0138 | 0.0232 | True | -2.312 | 18/30 | 0.0006 |
| switching_rate | -0.5709 | 0.0282 | 0.0282 | True | -1.975 | 30/30 | 1.0 |
| mean_synchrony | -0.1396 | 0.0174 | 0.0232 | True | -1.987 | 30/30 | 1.0 |

### Per-modality L2 — RESP (n_dyads=31, significant=0)

| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |
|---|---|---|---|---|---|---|---|
| peak_amplitude | -0.1491 | 0.0269 | 0.1076 | False | -1.977 | 31/31 | 1.0 |
| dwell_time | None | 1.0 | 1.0 | False | None | 6/15 | 0.0349 |
| switching_rate | 0.0 | 1.0 | 1.0 | False | 0.0 | 31/31 | 1.0 |
| mean_synchrony | -0.0027 | 1.0 | 1.0 | False | -0.121 | 31/31 | 1.0 |

## Gordon (paired design)
- contrast: [1, 4]

- note: condition bookend 1 vs 4 (exploratory); paper's primary contrast is pull_sync vs pull_seg (see gordon_diagnosis_bhfdr.csv)

### Per-modality L2 — angular (n_dyads=46, significant=0)

| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |
|---|---|---|---|---|---|---|---|
| peak_amplitude | 0.0241 | 0.3967 | 1.0 | False | 0.72 | 46/46 | 1.0 |
| dwell_time | None | 1.0 | 1.0 | False | None | 4/6 | 0.6869 |
| switching_rate | 0.0 | 1.0 | 1.0 | False | None | 46/46 | 1.0 |
| mean_synchrony | 0.014 | 0.6201 | 1.0 | False | 0.556 | 46/46 | 1.0 |

### Per-modality L2 — radial (n_dyads=46, significant=0)

| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |
|---|---|---|---|---|---|---|---|
| peak_amplitude | None | 1.0 | 1.0 | False | None | 31/0 | 0.0001 |
| dwell_time | None | 1.0 | 1.0 | False | None | 28/0 | 0.0001 |
| switching_rate | None | 1.0 | 1.0 | False | None | 31/0 | 0.0001 |
| mean_synchrony | None | 1.0 | 1.0 | False | None | 31/0 | 0.0001 |

## Prior diagnosis_bhfdr (between-dyad / cross-pair designs)

### Andersen

| contrast | feature | n_hi | n_lo | median_hi | median_lo | delta_median | p_raw | p_fdr | replication_status |
|---|---|---|---|---|---|---|---|---|---|
| is_close | onset_latency | 168 | 132 | 35.0 | 38.0 | -3.0 | 0.34105904270829 | 0.5455021183675033 | NS_after_FDR |
| is_close | rise_time | 168 | 132 | 5.5 | 5.0 | 0.5 | 0.6531795696171702 | 0.6531795696171702 | NS_after_FDR |
| is_close | peak_amplitude | 168 | 132 | 0.9959414878708646 | 0.9955318247114138 | 0.0004096631594507 | 0.1134623049525059 | 0.3403869148575179 | NS_after_FDR |
| is_close | recovery_time | 158 | 125 | 21.5 | 20.0 | 1.5 | 0.6054011484946005 | 0.6531795696171702 | NS_after_FDR |
| is_close | dwell_time | 168 | 132 | 27.417989417989418 | 25.56882868089765 | 1.849160737091772 | 0.0087429371610952 | 0.0524576229665716 | NS_after_FDR |
| is_close | switching_rate | 168 | 132 | 2.050271393822696 | 2.0791087070634164 | -0.0288373132407206 | 0.3636680789116688 | 0.5455021183675033 | NS_after_FDR |
| is_close | mean_synchrony | 168 | 132 | 0.2811560713302055 | 0.2514845611637444 | 0.0296715101664611 | 0.0370127964181683 | None | None |
| is_close | synchrony_entropy | 168 | 132 | 4.038752178598196 | 4.097483110374561 | -0.0587309317763642 | 0.0169106373950694 | None | None |
| is_known | onset_latency | 220 | 80 | 35.0 | 40.0 | -5.0 | 0.2758125737375789 | 0.4137188606063684 | NS_after_FDR |
| is_known | rise_time | 220 | 80 | 5.0 | 7.0 | -2.0 | 0.099355030654095 | 0.19871006130819 | NS_after_FDR |
| is_known | peak_amplitude | 220 | 80 | 0.9959414878708646 | 0.9953088805001986 | 0.0006326073706659 | 0.0483961184899106 | 0.145188355469732 | NS_after_FDR |
| is_known | recovery_time | 207 | 76 | 21.0 | 20.0 | 1.0 | 0.7955570678891479 | 0.7955570678891478 | NS_after_FDR |
| is_known | dwell_time | 220 | 80 | 27.39861895794099 | 24.975 | 2.423618957940988 | 0.0008852735604948 | 0.005311641362969 | REPLICATED |
| is_known | switching_rate | 220 | 80 | 2.0618812366631176 | 2.059621048502642 | 0.0022601881604757 | 0.7162534324924463 | 0.7955570678891478 | NS_after_FDR |
| is_known | mean_synchrony | 220 | 80 | 0.2792012762952395 | 0.234148039092395 | 0.0450532372028445 | 0.0005006114283404 | None | None |
| is_known | synchrony_entropy | 220 | 80 | 4.038528622343427 | 4.12483114334154 | -0.0863025209981129 | 0.0005444978361203 | None | None |

### Han

| effect | feature | n_hi | n_lo | p_raw | p_fdr |
|---|---|---|---|---|---|
| Arousal | onset_latency | 1955 | 1838 | 0.0589807 | 0.176942 |
| Arousal | rise_time | 1970 | 1933 | 0.198477 | 0.279027 |
| Arousal | peak_amplitude | 2060 | 2060 | 0.970547 | 0.970547 |
| Arousal | recovery_time | 1834 | 1654 | 0.232522 | 0.279027 |
| Arousal | dwell_time | 2037 | 2035 | 0.152155 | 0.279027 |
| Arousal | switching_rate | 2060 | 2060 | 3.36069e-10 | 2.01641e-09 |
| Arousal | mean_synchrony | 2060 | 2060 | 0.000188493 | 0.000188493 |
| Arousal | synchrony_entropy | 2057 | 2038 | 4.79161e-16 | 4.79161e-16 |
| Arousal | wcc_mean | 2060 | 2060 | 0.000188493 | 0.000188493 |
| Valence | onset_latency | 1910 | 1883 | 0.332285 | 0.332285 |
| Valence | rise_time | 1945 | 1958 | 2.86763e-10 | 1.72058e-09 |
| Valence | peak_amplitude | 2060 | 2060 | 0.000324792 | 0.000487188 |
| Valence | recovery_time | 1726 | 1762 | 0.0387136 | 0.0464563 |
| Valence | dwell_time | 2032 | 2040 | 8.50361e-05 | 0.000170072 |
| Valence | switching_rate | 2060 | 2060 | 2.6301e-06 | 7.8903e-06 |
| Valence | mean_synchrony | 2060 | 2060 | 0.444978 | 0.444978 |
| Valence | synchrony_entropy | 2056 | 2039 | 0.000298182 | 0.000298182 |
| Valence | wcc_mean | 2060 | 2060 | 0.444978 | 0.444978 |
| ChangeRate | onset_latency | 1408 | 2385 | 8.958e-05 | 0.00026874 |
| ChangeRate | rise_time | 1463 | 2440 | 0.88383 | 0.88383 |
| ChangeRate | peak_amplitude | 1520 | 2600 | 0.0856064 | 0.12841 |
| ChangeRate | recovery_time | 1315 | 2173 | 0.0300909 | 0.0601818 |
| ChangeRate | dwell_time | 1512 | 2560 | 0.594266 | 0.71312 |
| ChangeRate | switching_rate | 1520 | 2600 | 8.11646e-05 | 0.00026874 |
| ChangeRate | mean_synchrony | 1520 | 2600 | 0.940744 | 0.940744 |
| ChangeRate | synchrony_entropy | 1513 | 2582 | 0.739411 | 0.739411 |
| ChangeRate | wcc_mean | 1520 | 2600 | 0.940744 | 0.940744 |

### Gordon

| contrast | feature | n_hi | n_lo | median_hi | median_lo | delta_median | p_raw | p_fdr | replication_status |
|---|---|---|---|---|---|---|---|---|---|
| pull_sync | onset_latency | 3 | 2 | None | None | None | None | None | None |
| pull_sync | rise_time | 3 | 2 | None | None | None | None | None | None |
| pull_sync | peak_amplitude | 92 | 91 | 0.1079722426219036 | 0.2015507804929514 | -0.0935785378710477 | 0.0006716593523204 | 0.0020149780569612 | REVERSED_SIG |
| pull_sync | recovery_time | 3 | 2 | None | None | None | None | None | None |
| pull_sync | dwell_time | 14 | 21 | 5.0 | 5.0 | 0.0 | 0.4062753471255901 | 0.4062753471255901 | NS_after_FDR |
| pull_sync | switching_rate | 92 | 91 | 0.0 | 0.0 | 0.0 | 0.1789244804639657 | 0.2683867206959485 | NS_after_FDR |
| pull_sync | mean_synchrony | 92 | 91 | -0.2398333542238545 | -0.1348238612404957 | -0.1050094929833588 | 9.847870939082135e-08 | None | None |
| pull_sync | synchrony_entropy | 92 | 91 | 2.9905863803852366 | 2.970573095811685 | 0.0200132845735514 | 0.8309170822254257 | None | None |
| pull_seg | onset_latency | 2 | 3 | None | None | None | None | None | None |
| pull_seg | rise_time | 2 | 3 | None | None | None | None | None | None |
| pull_seg | peak_amplitude | 91 | 92 | 0.2156737716542803 | 0.1027299642796896 | 0.1129438073745906 | 0.0002101319294308 | 0.0006303957882924 | REVERSED_SIG |
| pull_seg | recovery_time | 2 | 3 | None | None | None | None | None | None |
| pull_seg | dwell_time | 20 | 15 | 5.0 | 5.0 | 0.0 | 0.3955901120766626 | 0.3955901120766626 | NS_after_FDR |
| pull_seg | switching_rate | 91 | 92 | 0.0 | 0.0 | 0.0 | 0.3491904277493587 | 0.3955901120766626 | NS_after_FDR |
| pull_seg | mean_synchrony | 91 | 92 | -0.1391051940512206 | -0.2658619478767923 | 0.1267567538255717 | 2.023156957378392e-08 | None | None |
| pull_seg | synchrony_entropy | 91 | 92 | 3.027169118440618 | 2.941911165610775 | 0.0852579528298429 | 0.0820510848903507 | None | None |
| pull_sync | onset_latency | 17 | 25 | 45.0 | 50.0 | -5.0 | 0.6435839695668396 | 0.8710048286108152 | NS_after_FDR |
| pull_sync | rise_time | 15 | 26 | 5.0 | 5.0 | 0.0 | 0.187289142326396 | 0.3745782846527921 | NS_after_FDR |
| pull_sync | peak_amplitude | 31 | 42 | 0.4893305291007342 | 0.5423523731349629 | -0.0530218440342286 | 0.1563813541628499 | 0.3745782846527921 | NS_after_FDR |
| pull_sync | recovery_time | 15 | 25 | 10.0 | 10.0 | 0.0 | 0.8236735893520913 | 0.8710048286108152 | NS_after_FDR |
| pull_sync | dwell_time | 28 | 37 | 7.5 | 7.5 | 0.0 | 0.8710048286108152 | 0.8710048286108152 | NS_after_FDR |
| pull_sync | switching_rate | 31 | 42 | 1.090909090909091 | 2.181818181818182 | -1.090909090909091 | 0.0076874603415441 | 0.0461247620492651 | REPLICATED |
| pull_sync | mean_synchrony | 31 | 42 | 0.0581405406452975 | 0.15527151891838 | -0.0971309782730825 | 0.0069178661141535 | None | None |
| pull_sync | synchrony_entropy | 31 | 42 | 2.8110378683422788 | 3.1895610963434966 | -0.3785232280012178 | 0.0010966162906675 | None | None |
| pull_seg | onset_latency | 42 | 0 | None | None | None | None | None | None |
| pull_seg | rise_time | 41 | 0 | None | None | None | None | None | None |
| pull_seg | peak_amplitude | 73 | 0 | None | None | None | None | None | None |
| pull_seg | recovery_time | 40 | 0 | None | None | None | None | None | None |
| pull_seg | dwell_time | 65 | 0 | None | None | None | None | None | None |
| pull_seg | switching_rate | 73 | 0 | None | None | None | None | None | None |
| pull_seg | mean_synchrony | 73 | 0 | None | None | None | None | None | None |
| pull_seg | synchrony_entropy | 73 | 0 | None | None | None | None | None | None |

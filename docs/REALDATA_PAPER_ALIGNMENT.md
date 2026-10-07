# SyncPipe vs. five published synchrony papers — method & conclusion alignment

> Purpose: answer the author's core worry — "is SyncPipe a wishful-thinking
> reimplementation, or does its method and conclusion direction align with the
> field?" This is a provisional comparison of the Lerique, Gordon, Andersen,
> Bizzego, and Han studies, not a verbatim extraction from their PDFs.
> Original-author code checked locally is identified below; paper-specific
> methods and conclusion wording remain pending PDF verification unless an
> explicit source is given. Shared terminology does not establish equivalent
> estimators, null distributions, or inferential claims.
>
> It is a **methodology alignment audit**, not a set of unit tests. The
> reproducible "does the direction agree" checks live in `scripts/`
> (`verify_bizzego_convergence.py`, `verify_realdata_consistency.py`,
> `realdata_l2_audit.py`).

---

## 0. One-sentence synthesis

SyncPipe uses surrogate and design-control families that can be compared with
published synchrony work, but **overall null-model equivalence is not verified**.
Bizzego's checked code uses IAAFT; this establishes a shared algorithm family,
not identity of the complete test. The checked Andersen statistics script does
not establish a circular-shift null. Estimator, preprocessing, sampling unit
and reference-distribution construction must be checked separately.

### Evidence status (2026-09-18)

- **Original-author code checked:** `E:/OSF/Bizzego/Scripts/03_sync_ibi.py`
  (preprocessing, IAAFT calls and three output distributions),
  `04_test_cop_UFL.py` (pairwise relationship-group Kruskal tests), and
  `E:/OSF/Andersen-hj4k6/Code/stats_HR.R` (models consuming correlation CSVs).
- **Pending PDF verification:** paper wording, reported conclusion directions,
  and details not established by those scripts. These code checks are neither
  full paper replication nor full loader acceptance; external files were read
  only.

---

## 1. Bizzego et al. 2020 — *Behav. Sci.* 10(1):11

### Loader 核验补充（2026-09-18）

本次证据保存于 `artifacts/bizzego_adapter_20260918/`，没有修改 OSF 或旧 372 行 manifest。
64 个 raw dyad 的 MAT 目录完成格式清点，其中 29 个缺少 raw trigger 文件；
processed IBI 缺少 U07、L12 两对。F01 的 ECG 双方均有 3,899,392 点；
`tt.mat` 的差分中位数为 0，数值并非递增时间轴，不能据文件名猜作秒数。
F01 raw trigger 为模拟量而非视频编号 1–6，其解码依据尚缺。

作者 `02_compute_ibi.py` 设置 `MANUAL=True`，包含滤波、BeatFromECG、
BeatOutliers/FixIBI 和人工 Annotate；`03_sync_ibi.py` 才进行 2 Hz 重采样、
低通、视频起点加 10 秒及至少 470 点的检查。2 Hz 不是 IBI 幅度单位。
仅静态检查两个 gzip pickle 的 opcode/global，未反序列化；不能由此确认
IBI 单位、时间原点或逐文件人工校正历史，也不采用旧 converter 的默认采样率和
累积 IBI 时间轴替代真实元数据。raw ECG 若今后另行提取 IBI，必须标明新处理，非论文等价。

`run_bizzego_replication.py --from-manifest` 已与 raw 模块解耦，先检查真实 schema
和 provenance，再按 manifest 采样率运行当前逐 cell L0/L1 API；不会把六条件
重复当独立 dyad 做群体门控。旧 manifest 明确返回缺少 `signal_type`、`unit`、
`preprocessing_path`，不补造历史来源。`--data-root` 明确返回阻塞（退出码 3）。
输出目录必须新建或为空以保护历史产物。5 项回归通过，包含合成当前契约
manifest 的实际读入、features、L0/L1 全链路；这不是原始 Bizzego 验证。
**真实源数据加载→IBI 导出→回读→features 尚未完成；未实现或伪造 raw loader。**

| | Paper | SyncPipe |
|---|---|---|
| Signal | ECG → **IBI** | ECG → IBI (`_preprocess_ecg`) |
| Analysis rate | **2 Hz** (low-pass 0.04 Hz, z-scored) | **1 Hz** (configurable `target_fs`) |
| Estimator | code requests `cc` with `MAX_DELAY = 10`, `LAG = 20` samples; helper details pending verification | zero-lag WCC (`peak_amplitude`) |
| Null | **IAAFT surrogate** (+ 5 s moving avg) | **IAAFT surrogate** (signal-level existence audit) |
| Design controls | co-presence / stimulus / surrogate distributions | pseudo-pair / time-shift / existence |

**Headline conclusion.** Synchrony varies by relationship type and emotional
state; strangers showed *greater* synchrony than friends/romantic partners.

**Alignment.** Shared IAAFT terminology is supported by the original code,
but **identical null models and equivalent three-way decompositions are not
established**. `03_sync_ibi.py` resamples to 2 Hz, applies the low-pass filter,
standardizes video segments, generates IAAFT surrogates, smooths those
surrogates with a 5 s moving average, and writes `copresence`, `stimulus` and
`surrogate` distributions from `compute_distances_golland`. Those three outputs
are not a demonstrated one-to-one equivalent of SyncPipe's pseudo-pair,
time-shift and existence audits. The dependency helper and PDF methods still
need verification. `04_test_cop_UFL.py` compares relationship groups with
Kruskal tests; it does not establish equivalence to SyncPipe's paired-condition
inference. Sampling rate, estimator and test construction differences must be
reported rather than dismissed as scope limits.

---

## 2. Andersen et al. 2024 — "Scared Together" (*Cognition and Emotion*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | ECG (1000 Hz) → **HR at 1 Hz** | ECG → IBI envelope |
| Estimator | **Pearson ISC** of HR (time-aligned) | WCC (zero-lag) |
| Null | **Not established by the checked statistics script; pending PDF verification** | **time-shift design control** |
| Closeness link | HR synchrony ↑ with social closeness | (not a relationship-type design in Lerique) |

**Conclusion status.** The prior claim that HR synchrony exceeded circularly
shifted surrogates is withdrawn pending PDF and upstream correlation-code
verification. Closeness/arousal conclusion wording also requires PDF checking.

**Alignment.** The checked `stats_HR.R` reads `hr_correlations.csv` and fits
`Corr ~ Close/Know + Age + Gender + (1|Subject1) + (1|Subject2)` models; it also
models arousal using `Corr_tot`. It does not construct circular-shift
surrogates. This is not evidence that the paper did not use them elsewhere,
but it cannot support calling SyncPipe's `time_shift` the same null. HR/IBI
preprocessing, sampling rates and estimator equivalence remain separate checks.

---

## 3. Han et al. 2022 — skin-conductance synchrony (*Communication Methods and Measures*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | EDA, **2000 Hz** | EDA bandpass → 1 Hz envelope |
| Estimator | **Cross-recurrence quantification (CRQA)** — nonlinear | WCC (linear) |
| Null | (CRQA surrogate / determinism vs chance) | IAAFT / block-shuffle |

**Headline conclusion.** Calm > arousing, negative-arousing > positive-arousing,
and fast > slow change produced stronger, more deterministically structured,
more stable SCL synchrony.

**Alignment.** **Partial, and honestly flagged.** Han uses a *nonlinear*
recurrence-based estimator; SyncPipe v1's default substrate is *linear* WCC.
CRQA is on SyncPipe's "alternative substrates, each with its own null" roadmap
(METHOD_LOG §8.5), not yet implemented. The *conclusion direction* SyncPipe can
check is "does arousal/valence/change-rate structure SCL synchrony" — the
existing `REALDATA_COMPARISON.md` reports Han's switching/entropy features as
highly significant (emotion → synchrony), agreeing in direction even though the
estimator differs. This is the one paper where SyncPipe should be explicit
that it is **not yet** methodologically equivalent.

---

## 4. Mayo & Gordon 2025 — contextual pulls (*American Psychologist*)

| | Paper | SyncPipe |
|---|---|---|
| Signal | IBI (spline **500 ms = 2 Hz**) + EDA | IBI/EDA envelopes |
| Estimator | IBI synchrony, EDA synchrony | WCC descriptors |
| Design | 2×2 within-subject (sync pull × seg pull) | condition contrast |

**Headline conclusion.** IBI synchrony was *positively* associated with
performance/cohesion in low-segregation contexts and *negatively* in
high-segregation contexts; EDA synchrony showed the converse pattern.

**Alignment.** Direction-dependent on context — this is exactly the
"contextual pull" idea SyncPipe's per-modality, per-condition reporting is
built for. SyncPipe has a **calibrated Gordon simulation**
(`simulation/gt5_gordon_conditions.py`, parameters taken from Gordon's own
`Full_Data.csv`) and `scripts/run_gordon_case_study.py`. The existing
`REALDATA_COMPARISON.md` flags a Gordon `peak_amplitude` direction discrepancy
(short 18–22-point traces) that is a **data-limitation finding to report**, not
a tool failure. 本轮 v2 从 184 个行为文件完成 183 条记录、排除 1 条短文件；
`motion_intensity` 原物理单位 unknown，仅作描述性分析，不验证上述论文结论方向。

---

## 5. Lerique et al. 2024 — ECSU-PCE dataset (data descriptor)

| | Paper | SyncPipe |
|---|---|---|
| Role | **Data descriptor** (no synchrony method) | the analysis tool |
| Sampling | ECG/EDA/respiration **1000 Hz** (BrainVision), notch 60 Hz | `RAW_FS_HZ = 1000` |
| Analysis rate | not specified (left to downstream users) | 1 Hz (`TARGET_FS_HZ`) |

**Alignment.** Lerique is a *dataset*, not a method — there is no Lerique
synchrony estimator to converge with. 本轮 v2 已实际运行全量 ECG/EDA/RESP
预处理：279 条中 264 条完成、15 条缺失记录排除；ECG 使用 `neurokit2 0.2.13`
从 1000 Hz ECG 提取 1 Hz IBI，EDA/RESP 为 bandpassed signals，原物理单位 unknown。
88 条完成记录的科学特征全部 undefined（不含诊断量 `nan_fraction`）：拼接的 60 秒
trial 配合 30 秒 WCC 窗口，边界 mask 使约半数窗口不可用，现有 20% NaN gate
可导致 undefined，未放宽门槛。拼接轴不是 session wall-clock，拼接后滤波的边界
影响也仍需验证；这是重要未解决风险。历史 `verify_realdata_consistency.py`
显著性描述不作为本轮科学有效性或三模态显著性的证据。

**2026-09-18 v3 实证诊断**（`artifacts/realdata_standardized_20260918_v3`）：
全量重跑112.15秒，Lerique 264成功/15排除、Gordon 183成功/1排除，0失败。
6258个同键特征与v2逐值完全一致：defined 3615→3615（科学量3168→3168），
definedness变化0、数值变化0、全科学量undefined仍88条。88条trial的原始缺失率均0，
设计性无效510/1051=48.525214%，额外真实缺失无效窗口0。
runner已经传mask，pair已经传`gap_policy=segment`；20%门禁在SSoT之前作用，
不是漏传gap_policy。现有context分段入口要求3×window=90秒，59–60秒trial不满足；
没有已批准的trial→condition聚合契约，因此未改生产算法、未绕过门禁。
保留条件主表，另导出1848个独立段诊断（其中1584个trial段均不满足context长度门槛）；
均有mean_synchrony数值，但不替换条件估计、不证明效度提高。诊断仍沿用拼接后预处理，
不能排除滤波跨边界污染。`comparison_values.csv`保存全部前后值，`comparison.json`
保存排除清单与独立段示例。runner新增原始缺失率与设计无效窗口率，缺失即使落在mask内也不豁免。


---

## 6. Summary table

| Paper | Null model | SyncPipe equivalent | Direction agrees? |
|---|---|---|---|
| Bizzego 2020 | IAAFT appears in checked code; full paper null pending PDF verification | signal-level IAAFT existence audit | ⚠️ algorithm-family overlap only |
| Andersen 2024 | Circular-shift claim pending PDF/code verification | time-shift design control | ⚠️ not established as same null |
| Han 2022 | CRQA (nonlinear) | *not yet implemented* (WCC is linear) | ⚠️ direction only |
| Gordon 2025 | (contextual × IBI/EDA sync) | 本轮为描述性 motion_intensity WCC | 未验证论文等价或方向 |
| Lerique 2024 | (data descriptor) | 全量 ECG/EDA/RESP 预处理，88 条科学特征全 undefined | 输入处理完成不等于科学有效 |

**Bottom line.** This document supports **partial, evidence-scoped alignment**
only. The checked Bizzego code shares IAAFT terminology and the Andersen
statistics code consumes correlation outputs, but the PDF-level null,
estimator, and conclusion claims are not yet verified here. Do not describe
the three SyncPipe layers as one consistent field-wide null model or claim
Bizzego/Andersen full equivalence. The one explicit implementation gap remains
Han's nonlinear CRQA, which SyncPipe lists as future work rather than pretending
equivalence.

---

## 7. Historical signed-vs-absolute diagnostic (not paper-equivalence evidence)

The following local diagnostic and numerical results are retained as historical
claims, not revalidated in this update. The checked author script calls a
`pyphysio` helper; its absolute/max-lag semantics and the paper's wording still
require dependency/PDF verification. This section must not be used to infer
that a local absolute-peak descriptor reproduces Bizzego's estimator or null.

`scripts/verify_bizzego_convergence.py` asked whether SyncPipe's
`peak_amplitude` is the zero-lag special case of Bizzego's max-CC. It is **not
exactly** — and the reason is substantive, not a bug:

- **SyncPipe `peak_amplitude`** = *signed* maximum of the smoothed WCC trace
  (`np.nanargmax`), i.e. the strongest **positive** correlation episode.
- **Bizzego max-CC** = maximum of the **absolute** cross-correlation over
  ±10 s, i.e. anti-phase (negative) correlation counts toward synchrony
  magnitude.

On the committed Lerique traces, **88% carry a strong anti-phase segment
(min WCC < −0.3)** and 69% carry min WCC < −0.5. In the extreme
(`pce09 EDA rest1`), Bizzego's `max|WCC|` = 0.93 while SyncPipe's
`peak_amplitude` = 0.227 — a −0.93 anti-phase episode is simply not counted by
the signed descriptor.

**This is a real, defensible scope choice, and it must be stated — not left
implicit.** The signed reading aligns with the Gordon synchrony/segregation
axis that motivates this project: a *negative* WCC is plausibly **segregation /
anti-phase coordination**, not "stronger synchrony", so excluding it from the
*synchrony* intensity descriptor is theoretically coherent. But a researcher
coming from Bizzego (or SUSY's `ES_abs`) expects anti-phase to count.

**Decision needed (open).** One of:

1. Keep `peak_amplitude` signed and **document** the choice (current state) —
   defensible against Gordon, but will read as "under-reporting synchrony" to a
   Bizzego/SUSY reviewer unless the rationale is explicit.
2. Add an **absolute** companion descriptor (`peak_abs_amplitude`, the true
   zero-lag special case of Bizzego's max-CC) as an exploratory intensity
   feature, so both readings are reported.
3. Switch the primary to absolute (would diverge from Gordon's
   synchrony/segregation framing).

Option 2 is the field-safest: it makes the Bizzego-convergence claim
*literally* true (a `peak_abs_amplitude` equals max-CC at lag 0) while keeping
the signed descriptor for the segregation-aware reading.


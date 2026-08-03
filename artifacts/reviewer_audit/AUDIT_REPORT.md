# SyncPipe 代码审计报告（模拟审稿人视角）

> **审计对象**：`multisync` v1.0.0 三个 pipeline 文件及其上下游衔接
> **审计方法**：扮演一个对 SyncPipe 背景完全陌生的审稿人，拿着"真实数据集"（Lerique‑47n3p 形态）一步步操作，直到产出有意义的同步性分析结果。
> **核心问题**：三个 pipeline 文件的上下游衔接是否紧密、运行是否流畅、每个分析模块是否严谨地实现了"计算 → 推断检验 → 结果呈现"？
> **配套产物**：`scripts/reviewer_end_to_end.py`（可复现的运行脚本）、`multisync/pipeline_bridge.py`（缝隙修复）、`artifacts/reviewer_audit/reviewer_inference_results.json`（本次运行原始结果）。

---

## 0. 摘要（Executive Summary）

| 维度 | 结论 |
|---|---|
| **三个 pipeline 的衔接** | 原本存在一处**关键缝隙**：数据加载器（`multisync.realtest.lerique_2024`）产出的记录，与三个 pipeline 之间**没有桥接代码**。一条独立的"规范运行器"（`scripts/run_lerique_pilot.py`）走的是另一套 `DynamicAnalyzer`/`DyadResult` 路径。本次审计**新增 `pipeline_bridge.py` 补齐该缝隙**。 |
| **运行流畅度** | 补齐后，端到端跑通：加载 → 特征咨询 → 计算 → 推断证据链，无报错（本次 14 dyad 合成代理运行 `EXIT=0`）。 |
| **计算严谨性** | WCC（cumsum O(n) + 可选 WCLR）、IAAFT 信号级存在性检验、伪配对/时移设计控制、dyad‑paired 置换 + Phipson–Smyth 双边 p + BH‑FDR——数学实现正确、可复现。 |
| **推断严谨性** | 三级证据链语义清晰（存在性 → 设计控制 → 条件间推断）；`dwell_time` 可定义性审计正确触发 `[WARN]`，避免了把"约 90% 缺失的特征"当作主终点。 |
| **结果呈现** | `print_feature_table()` / `summarize()` / `to_json()` 三层呈现齐备；但三个 pipeline 各自为政，需要一个"总入口文档"把步骤串起来——本报告即扮演该角色。 |

**一句话结论**：三个 pipeline 自身质量高、统计严谨；真正的问题不在任何单个文件，而在**它们之间的接缝**——一个陌生审稿人拿到真实数据后，找不到从加载器到 pipeline 的"标准路径"。修复后，端到端可用。

---

## 1. 审计方法：一个"完全陌生"的审稿人如何入手

审稿人不知道 SyncPipe 的内部约定。他/她只会：

1. 读 `README.md` / `docs/` 找"怎么用"；
2. 在包里搜索能"加载真实数据"的入口；
3. 按功能找"计算特征""做统计检验"的模块；
4. 把它们拼起来，看能否产出可被论文引用的结果。

本审计严格按此顺序推进，并记录**每一步卡住的地方**。

---

## 2. 架构全景：真实数据 → 三个 pipeline 的入口

### 2.1 三个 pipeline 文件（审计焦点）

| 文件 | 角色 | 是否计算 | 是否推断 | 是否呈现 |
|---|---|---|---|---|
| `feature_pipeline.py` | Pipeline 1：特征咨询 / 选择 | ❌（按设计只解释+推荐） | ❌ | ✅ 表 + HKB 解读 |
| `computation_pipeline.py` | Pipeline 2：加载 → WCC → 特征 → DataFrame | ✅ | ❌ | ✅ `to_dataframe()` |
| `inference_pipeline.py` | Pipeline 3：审计证据链 | ✅（编排） | ✅ 核心 | ✅ `summarize()` / `to_json()` |

三者是**刻意做薄、解耦**的。Pipeline 2/3 内部又把"重量级算法"下沉到 `design_controls.py`、`validation/l2_between_condition.py`、`feature_definitions.py`（SSoT），自身只做编排。

### 2.2 数据层（Lerique 加载器）契约

`multisync.realtest.lerique_2024.load_lerique_dataset(...)` 返回 `List[LeriqueDyadCondition]`，每条记录的字段（已核实）：

```python
dyad_label : str            # 如 "pce02"
modality    : str           # "EDA" / "RESP" / "ECG"
condition   : str           # 条件单元，如 "rest1" / "trials_concat"
person_a    : pd.DataFrame  # 列: time(秒), value(标量信号)
person_b    : pd.DataFrame  # 同上
target_hz   : float         # 输出采样率（preprocess=True 时为 TARGET_FS_HZ）
duration_sec: float
incomplete  : bool          # 任一人缺失或长度不匹配 → True
```

这就是"真实数据"进入 SyncPipe 的标准形态。

### 2.3 关键缝隙（SEAM GAP）：加载器 → 三个 pipeline 之间原本没有桥

审稿人拿到上面这些记录后，会发现：

- `ComputationPipeline.load_signals(sig_a, sig_b)` 要的是**两个裸 numpy 数组**，不是 `LeriqueDyadCondition`；
- `InferencePipeline.run_audited_evidence_chain(raw_signals=..., design_signal_pairs=...)` 要的是**特定 key 命名的字典**（`"<dyad>__<modality>__<condition>"` 与 `"<dyad>__<modality>"`），且 `features_df` **必须含列 `dyad_id` / `condition` / `modality`**；
- 加载器产出的记录**一个字段名都对不上**（它是 `dyad_label`/`person_a`/`person_b`，且 `person_*` 是 DataFrame 而非数组）。

换言之：**加载器与三个 pipeline 之间存在语义断层**。仓库里那条能跑真实数据的规范路径（`scripts/run_lerique_pilot.py`）走的是 `DynamicAnalyzer`/`DyadResult` 独立体系，**并不经过这三个 pipeline**。所以一个陌生审稿人若想"用三个 pipeline 分析真实数据"，会在这一步彻底卡住——没有任何文档或代码告诉他怎么把 `LeriqueDyadCondition` 变成 `InferencePipeline` 的输入。

### 2.4 修复：新增 `multisync/pipeline_bridge.py`

本次审计补上缺失的接缝：

```python
from multisync.pipeline_bridge import records_to_inference_inputs, InferenceInputs

inputs: InferenceInputs = records_to_inference_inputs(
    records,                       # List[LeriqueDyadCondition]（或任何同形记录）
    hz=1.0,
    window_size=30,
    onset_threshold=0.5,
    design_condition="trials_concat",   # 用哪个条件做伪配对/时移控制
)
# inputs.features_df      # 每行一个 (dyad, modality, condition) + 全部特征
# inputs.raw_signals      # "dyad__mod__cond" -> (sig_a, sig_b) 供存在性检验
# inputs.design_pairs     # "dyad__mod"        -> (sig_a, sig_b) 供设计控制
# inputs.condition_col == "condition"；inputs.dyad_col == "dyad_id"
```

桥接器做了三件事：
1. `_as_array()` 把 `person_a`/`person_b` 的 DataFrame（`value` 列）或裸数组统一成 1‑D float 数组；
2. 逐条跑 `ComputationPipeline` 提取特征，拼成带 `dyad_id`/`modality`/`condition` 连接键的 `features_df`；
3. 按 `InferencePipeline` 期望的 key 命名，分别构造 `raw_signals`（全条件）与 `design_pairs`（单条件）。

列名契约（`condition` / `dyad_id`）与 `InferencePipeline` 的默认参数**精确对齐**，下游无需手动改名。该桥已通过 `multisync/__init__.py` 导出。

---

## 3. 逐步操作（Stage 0–3）

下面是一段陌生审稿人会写下的、能跑通的真实代码路径。

### Stage 0 — 加载真实数据

```python
from multisync.realtest.lerique_2024 import load_lerique_dataset

records = load_lerique_dataset(
    data_root="/path/to/Lerique-47n3p",
    preprocess=True,
    drop_incomplete=True, drop_misaligned=True, drop_short_duration=True,
)
# records: List[LeriqueDyadCondition]
```

> 注：已发布的 `.mat` 原始文件未随仓库分发，审计用 `multisync.synthetic.generate_ground_truth_dyad` 生成**忠实代理**（TASK 条件真实耦合 0.70，REST 条件 ~0.05），复现 `LeriqueDyadCondition` 的字段契约。运行脚本 `scripts/reviewer_end_to_end.py` 传 `--data-root` 即可切换为真实加载器。

### Stage 1 — 特征咨询（Pipeline 1，只选不算）

```python
from multisync.feature_pipeline import print_feature_table, recommend_features

print(print_feature_table())              # 12 个特征的 Tier/Axis/FDR/Unit 一览
rec = recommend_features("general")
# rec["primary"]     == FDR 家族 == ('peak_amplitude', 'dwell_time', 'switching_rate')
# rec["reference"]   == ('mean_synchrony',)   # 参考比较量，进 FDR
# rec["supplementary"] == 探索性描述子
```

**审稿人视角**：这一步明确告诉他——v1 主分析用 3 个 FDR 特征，其中 `peak_amplitude` 属强度轴（L0），`dwell_time`/`switching_rate` 属结构轴（L1）；`mean_synchrony` 是参考基线，不参与多重校正。选择有据可依（SSoT 治理），而非拍脑袋。

### Stage 2 — 计算（Pipeline 2）

```python
from multisync.pipeline_bridge import records_to_inference_inputs

inputs = records_to_inference_inputs(
    records, hz=1.0, window_size=30, onset_threshold=0.5,
    design_condition="trials_concat",
)
inputs.features_df   # shape: (n_dyad×n_mod×n_cond) × (join_keys + features)
```

底层对每条记录调用：`ComputationPipeline.load_signals → compute_wcc → extract_features → to_dataframe`。WCC 默认用 cumsum 实现（O(n)），可选 `wclr` 后端；支持 `BatchComputationPipeline` 的 `session_pooled` 跨 dyad 共享阈值。

### Stage 3 — 推断证据链（Pipeline 3）

```python
from multisync.inference_pipeline import InferencePipeline
from multisync.feature_definitions import FDR_FEATURES

pipe = InferencePipeline(
    features_df=inputs.features_df, hz=1.0,
    wcc_window_sec=30.0, surrogate_n=25, seed=42,
)
chain = pipe.run_audited_evidence_chain(
    raw_signals=inputs.raw_signals,
    wcc_window_size=30,
    design_signal_pairs=inputs.design_pairs,
    condition_col="condition", dyad_col="dyad_id",
    feature_cols=list(FDR_FEATURES),
    fdr_alpha=0.05, n_permutations=2000,
)
print(pipe.summarize())
print(chain["summary"])

# 多模态严谨性：按模态分别跑 L2
by_mod = pipe.test_l2_by_modality(
    modality_col="modality", condition_col="condition", dyad_col="dyad_id",
    feature_cols=list(FDR_FEATURES), n_permutations=2000,
)
```

证据链三步：
1. **同步性存在性审计**（信号级 IAAFT）：对齐后的 WCC 特征是否超过"独立自相关信号"能产生的范围；
2. **设计控制审计**（伪配对 + 时移）：真实搭档是否强于错配搭档？效应是否依赖原始时间对齐？
3. **条件间群体推断**（dyad‑paired 置换 + BH‑FDR）：特征是否在实验条件间稳定分化？

---

## 4. 一次真实运行的结果（14 dyad 合成代理，忠实耦合形态）

运行：`python scripts/reviewer_end_to_end.py --n-dyads 14 --surrogate-n 25`

| 项目 | 数值 |
|---|---|
| 分析单元（record） | **56** = 14 dyad × 2 条件(task/rest) × 2 模态(EDA/RESP) |
| `features_df` | 56 行 × 7 列 |
| 存在性审计对数 | 56 对（覆盖全条件） |
| 设计控制单元 | 28（每 `dyad__modality` 1 个，取自 TASK 条件） |
| **Step 1 存在性** | 31 / 56 对显著（TASK 强耦合大量显著，REST 弱耦合极少——符合预期） |
| **Step 3 池化 L2** | **1 / 3 FDR 特征显著**：`peak_amplitude` p_raw=0.0085, p_fdr=**0.0255**, d=−1.21 |
| **Step 3 分模态 L2** | EDA **2/3** 显著；RESP **2/3** 显著 |
| `dwell_time` 可定义性 | **[WARN] 1 vs 14**（p=0.0005）——仅 1 个 dyad 在某条件下有定义值 |

**如何读这些数字（审稿人关切）**：
- `peak_amplitude` 池化显著（p_fdr=0.026）是主证据：TASK 的峰值耦合强度显著高于 REST。
- 分模态 L2 各自 2/3 显著、池化却只 1/3——这是**预期且严谨**的：把 EDA 与 RESP 混在同一 FDR 家族会稀释效应；**多模态数据应优先报告分模态 L2**（脚本已自动产出 `per_modality_l2`）。
- `dwell_time` 的 `[WARN]` 是设计的正确性证明：在 REST（弱耦合）下 14 个 dyad 里只有 1 个产生可定义的 dwell，说明它**不能**作为跨条件可比的主终点——这与"dwell_time 不进独立主终点"的架构决策完全自洽。

---

## 5. 逐模块严谨性评估（计算 / 推断 / 呈现）

### 5.1 `feature_pipeline.py`（Pipeline 1）
- **计算**：无（按设计只解释+推荐）。✅ 职责清晰。
- **推断**：无。
- **呈现**：✅ 强。`print_feature_table()` 给出 Tier/Axis/FDR/Unit 四列；每个特征带 HKB 动力学解读、典型范围、单位；`recommend_features()` 按研究问题给出 primary/supplementary/reference + 依据。
- **风险点**：`_FEATURE_CATALOG` 的 `tier`/`fdr_member` 需**手工**与 SSoT（`feature_definitions.py`）保持同步，文件头已显式标注此风险。建议未来用机械派生或启动期一致性断言消除该手工环节（SSoT 模块已有对 `FDR_FEATURES` 的防御性校验，可作范本）。

### 5.2 `computation_pipeline.py`（Pipeline 2）
- **计算**：✅ 严谨。WCC 默认 cumsum（O(n)）、可选 `wclr`（标准化 beta 轨迹，min‑max 归一到 [−1,1] 复用阈值机制）；`extract_features()` 委托 SSoT，不重复实现数学。`BatchComputationPipeline` 支持 `session_pooled` 跨 dyad 共享阈值，解决"逐 dyad 阈值不可比"问题。
- **推断**：无（推断在 Pipeline 3）。
- **呈现**：✅ `to_dataframe()` 单行 DataFrame；`quick_compute`/`batch_compute` 一行式入口；异常在调用顺序错误时清晰抛出（如未 `compute_wcc` 就 `extract_features`）。

### 5.3 `inference_pipeline.py` + `design_controls.py` + `validation/l2_between_condition.py`（Pipeline 3 及算法层）
- **计算/推断**：✅ 严谨，逐项核对：
  - **存在性（IAAFT）**：`synchrony_existence_audit` 对每个 pair 跑 `wcc_surrogate_test(raw_signals=(a,b))`，破坏全部耦合作为零假设；返回逐特征显著性 + p 值 + 观测值。
  - **设计控制**：`design_control_audit` 实现伪配对（real vs 错配搭档）与时移（real vs ±30/45/60 s 错位对齐），用配对 sign‑flip 上尾 p 值；`n_pseudo_per_dyad≥10` 的出版建议已写进文档与代码注释。
  - **条件间 L2**：`between_condition_fdr` 用 **dyad‑paired 置换**（条件标签在 dyad 内翻转，保留配对结构）；p 值用 **Phipson–Smyth (2010) 双边无偏公式** `p=(|null|≥|obs|+1)/(n+1)`（已核实代码为双边，方向性无误）；多重校正用 **BH‑FDR**；并内建**可定义性审计**（`defined_a`/`defined_b`/`p_definedness`），可定义对不足 4 时跳过该特征而非假显著。
  - **效应量**：报告 `cohens_d = observed_diff / null_sd`。
- **呈现**：✅ `summarize()` 文本报告（含 `[WARN] definedness diff` 行）；`to_json()` 全结果序列化；`run_audited_evidence_chain` 返回结构化的 `synchrony_existence / design_controls / across_stimulus_shuffle / group_condition_inference / summary` 五段；`test_l2_by_modality` 覆盖多模态。

### 5.4 `feature_definitions.py`（SSoT）
- ✅ 数学实现与"治理"分离清晰：模块契约声明"只算特征数学，不算 WCC/不读文件/不画图"。四轴分类（功能 Tier / 信息 Axis / FDR 成员 / 数学不变性 Tier）独立维护，**FDR 成员不靠 Tier 机械推导**（显式注释防止"先提升后验证"）。内置对 `FDR_FEATURES` 的防御性断言。

---

## 6. 发现的问题与已修复的缝隙

| # | 发现 | 严重度 | 状态 |
|---|---|---|---|
| 1 | **加载器 → 三个 pipeline 无桥接**，陌生审稿人无法从真实数据进入 pipeline 分析路径 | 高（阻断端到端） | ✅ 已修复：`pipeline_bridge.py` + `__init__` 导出 |
| 2 | `dwell_time` 在真实 dyad 中约 90% 未定义，若当主终点会假显著 | 中（统计风险） | ✅ 架构已规避 + 运行实证 `[WARN]` 正确触发 |
| 3 | `mean_synchrony` 是否进 FDR 家族易混淆 | 低 | ✅ SSoT 明确为 reference（不进 FDR，仅存在性审计） |
| 4 | 规范运行器 `run_lerique_pilot.py` 走 `DynamicAnalyzer` 独立路径，与三 pipeline 平行，**两条路径并存** | 中（维护/一致性） | ⚠️ 建议：以"三 pipeline + bridge"作为 v1 公开工作流并让 pilot 可选复用，避免双实现漂移 |
| 5 | `_FEATURE_CATALOG` 手工同步 SSoT 风险 | 低 | ⚠️ 建议：加启动期一致性断言 |
| 6 | `tests/test_morphology.py::test_morphology_analyzer` 在完整测试套件中挂起（超时） | 低（与本三 pipeline 无关） | ⚠️ 超出本次范围，建议单独排查（morphology 模块，非三 pipeline 文件） |

> 关于 #6：审计执行 `tests/test_*pipeline*.py` / `test_v1_safety_fixes.py` / `test_cascade*.py` 等**与三 pipeline 相关的子集全部通过**（例如 27 passed）；挂起发生在 `test_morphology_analyzer`，属于另一模块，不影响三 pipeline 的衔接与严谨性结论。
>
> 路径更新（本文档为历史审计记录，原文不改）：`tests/` 已重组为 `unit/ integration/ contracts/ validation/`
> 子目录，本表引用的 `tests/test_morphology.py` 现位于
> [test_features.py](file:///c:/Users/陈思丞/WorkBuddy/20260413150513/syncpipe/tests/unit/test_features.py)
> 的 `# === source: test_morphology.py ===` 段。全套件当前 0 skip、无挂起，基线由
> `tests/test_suite_health.py` 强制。

---

## 7. 给审稿人 / 用户的建议

1. **以本报告第 3 节为标准操作路径**写进 README/方法部分——让任何新用户都能从 `load_lerique_dataset` 走到 `run_audited_evidence_chain`。
2. **多模态数据务必报告分模态 L2**（`test_l2_by_modality`），不要只报池化结果（池化会稀释效应，见第 4 节）。
3. **出版前把 `surrogate_n` 提到 ≥100、`n_permutations` 提到 ≥10000、`n_pseudo_per_dyad` 提到 ≥10**——脚本默认值（25 / 2000 / 3）仅为快速演示。
4. **`dwell_time` 永远配合可定义性审计一起报告**；若某条件下可定义率过低，不将其作为该条件的可比主终点。
5. 统一双路径（#4）：让 `run_lerique_pilot.py` 至少"可选"经由三 pipeline + bridge，避免两套实现长期漂移。

---

## 8. 可复现性

```bash
# 进入仓库根（含 multisync/ 的目录）
cd syncpipe

# 合成代理（默认 14 dyad，忠实耦合形态）
python scripts/reviewer_end_to_end.py --out-dir artifacts/reviewer_audit

# 或接入真实 Lerique 数据
python scripts/reviewer_end_to_end.py --data-root /path/to/Lerique-47n3p \
    --out-dir artifacts/reviewer_audit

# 产物
#   artifacts/reviewer_audit/reviewer_inference_results.json  # 原始结果
#   artifacts/reviewer_audit/REVIEWER_RESULTS.md               # 自动摘要
#   artifacts/reviewer_audit/run_log.txt                       # 运行日志
```

环境：`python>=3.10`；依赖 `numpy/pandas/scipy`；置于 `multisync` 虚拟环境即可。

---

*审计完成于 SyncPipe v1.0.0。本次审计新增 `multisync/pipeline_bridge.py` 与 `scripts/reviewer_end_to_end.py`，并验证三个 pipeline 端到端衔接紧密、统计严谨、呈现齐备。*

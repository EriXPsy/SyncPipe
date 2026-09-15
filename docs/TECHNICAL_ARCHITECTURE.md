# SyncPipe v1 —— 技术架构说明（面试/答辩用）

> 本文档把 SyncPipe v1 使用的每一项重要技术，按**重要性**与**可呈现性**排序，并逐项写出：
> **被选理由 / 与其他方案对比 / trade-off / 预判追问**。
> 所有依据来自代码实现与配套设计文档（`V1_PROTOCOL.md`、`V1_CLAIM_CEILING.md`、
> `DESIGN_PRINCIPLES.md`、`METHOD_LOG.md`、`LIMITATIONS.md`），不臆造。
>
> 配套：本文档是"为什么"，`V1_PROTOCOL.md` 是"锁定的科学对象"，
> `V1_CLAIM_CEILING.md` 是"能说什么/不能说什么"。

---

## 0. 一页纸科学对象

SyncPipe v1 提供一套**可审计的测量与推断程序**（auditable measurement & inference
procedure），对象是对偶（dyadic）连续低频同模态生理/行为信号的**同步性**。

- **它是什么**：标准化的 WCC 测量 + 三级 null 审计 + 设计控制 + 资格治理 + 可复现报告。
- **它不是什么**：不是"同步质量"这个心理构念的效度测量，不证明人际耦合/因果，不构建跨模态统一构念。
- **唯一验证性声明**：预注册的两条件、同 dyad 配对对比（principal endpoint `peak_amplitude`，dyad-paired permutation + BH-FDR）。
- **设计无关内核**（无需条件对比）：对任意对齐同模态 dyad，产出 WCC 轨迹、可解释描述符、L0/L1 存在性与结构审计——free-play / 连续同步设计同样是一等公民。

> 一句话锚点：**v1 把"synchrony 是否存在"这个问题，拆成三层各自有 null 的审计，而不是丢一个平均相关系数。**

---

## 1. 排序总览（重要性 × 可呈现性）

| # | 技术 | 重要性 | 可呈现性 |
|---|:--:|:--:|:--:|
| 1 | 三段证据链 L0→L1→L2（整体架构） | ★★★★★ | ★★★★ |
| 2 | IAAFT @ L0（signal-level surrogate） | ★★★★★ | ★★★★★ |
| 3 | Existence gate（预注册端点 + 模态集） | ★★★★★ | ★★★★ |
| 4 | Dyad-paired permutation @ L2 | ★★★★★ | ★★★★ |
| 5 | WCC（零滞后、leading window） | ★★★★ | ★★★★★ |
| 6 | BH-FDR @ L2（按 FDR 家族分组） | ★★★★ | ★★★ |
| 7 | level_preserving IAAFT（L0 备选 null） | ★★★★ | ★★★ |
| 8 | Design controls（pseudo-pair / time-shift / across-stim shuffle） | ★★★★ | ★★★ |
| 9 | Claim ceiling / governance（stage_status / claimable） | ★★★★ | ★★★ |
| 10 | Fail-loud 设计原则 | ★★★★ | ★★★ |

**面试策略**：
- **能口头展开 + 现场演示**：#2 IAAFT（demo 交互 proof）、#5 WCC（图）、#1 三段链（叙事主线）、#3 gate（清晰规则）。
- **能深度答辩（不靠图）**：#4 dyad-paired + #6 BH-FDR（为何只有此设计验证性）、#7 level_preserving 与 iaaft 取舍、#9 claim ceiling（诚实边界）。

---

## 2. 技术架构总览

```
对齐同模态 dyad 信号
   │
   ▼
[WCC] 零滞后滑动窗口互相关  →  连续 WCC 轨迹（测量内核）
   │
   ├─ L0  Signal-Level 存在性审计  ── IAAFT / level_preserving IAAFT（对原始信号重排相位）
   │      问："观测 WCC 是否超独立自相关信号？"
   │      → Existence gate（冻结端点 peak_amplitude × 预注册主模态 ECG/EDA，pass_rate>0.5）
   │
   ├─ L1  WCC-Level 结构审计  ── block permutation（对 WCC 轨迹块置换）
   │      问："WCC 轨迹是否呈现有序 episode（dwell_time / switching_rate）？"
   │
   └─ L2  Between-Condition 推断  ── dyad-paired permutation + BH-FDR
          问："特征在两预注册条件间是否有差异？"（唯一验证性声明）
                │
                └─ Design controls（pseudo-pair / time-shift / across-stim shuffle）
                   控 within-dyad 混淆（共享刺激、慢漂、周期）

全程：input 契约 fail-loud + definedness/eligibility/claimability 闸门 + 可复现报告
```

**关键 measured fact（必须记住，会被深挖）**：三层**不能共享同一个 null 模型，也不能共享同一个 BH 分母**。
IAAFT 在 L0 是正确的存在性 null，但在 L1 是**零功效**的——因为它保留功率谱，而功率谱正是生成 run-length 结构的来源，
`dwell_time` / `switching_rate` 因此无法移动（见 §4 与 `V1_PROTOCOL.md` §10）。

---

## 3. 逐项详述

### #1 三段证据链 L0→L1→L2（整体架构）

**被选理由**
单一相关系数掩盖了"synchrony 是否存在 / 结构是否有序 / 跨条件是否有差异"三个不同问题。
把三者拆成独立的审计层，每层有自己的 null 与claim类型，回答才可辩护。

**与其他方案对比**
- vs "一个平均 WCC 相关 + p 值"：后者把存在性与条件差异混为一谈，且 null 语义不清。
- vs 只做 L0：拿不到"跨条件差异"这个验证性声明。
- vs 只做 L2：没有存在性审计兜底，条件差异可能来自噪声/共享刺激而非耦合。

**trade-off**
- 三层串行，计算量随 surrogate 数线性增长；L0/L2 用 ProcessPool 并行（Windows 下 worker 必须模块级函数 + `__main__` guard，否则 BrokenProcessPool）。
- 层间不共享 BH 分母：L1 不发 per-dyad 判定（`L1_INFERENCE_UNIT = "group"`，见 `feature_definitions.py` FDR_FAMILIES 文档 2026-08-06 裁定），L0/L2 各自校正。

**预判追问**
- "三层为什么不能统一 null？" → 答 L1 零功效的 measured fact（IAAFT 保留功率谱 → run-length 不动）。
- "L1 不发 per-dyad 判定，那 L1 算什么？" → 答 group 级描述性结构审计，不发存在性裁决。

---

### #2 IAAFT @ L0（signal-level surrogate）

**被选理由**
存在性审计需要"独立于耦合的自相关信号"作为 null。IAAFT（Iterative Amplitude Adjusted
Fourier Transform）在保留原始信号振幅分布与功率谱（即自相关结构）的同时重排相位，
因此 surrogate 之间**没有**人际耦合，但有真实的自相关"底色"。观测 WCC 显著超过 surrogate 包络，
才说明信号不止是"各自自相关 + 噪声"。

**与其他方案对比**
- vs **block permutation（对原始信号）**：块置换会破坏自相关结构，null 太弱，容易假阳性。
- vs **phase randomization（一次性）**：不满足振幅分布约束，IAAFT 迭代到同时满足振幅+谱。
- vs **trace-level cyclic block-bootstrap**（早期方案，已退役，METHOD_LOG 7c）：trace 级 null 无法回答*存在性*——它保留正被测试的耦合。
- vs **PRTF**：PRTF 是同层备选 surrogate，IAAFT 为默认。

**trade-off**
- 假设**平稳性**（重排相位要求广义平稳）；共享慢漂时 iaaft 的 FPR≈30%（见 #7）。
- 计算较重（迭代收敛），用 surrogate 数（`--n-surrogates`，默认 1000）权衡。
- **最易演示**：surrogate 包络 vs 观测值的图，demo 有交互 proof。

**预判追问**
- "IAAFT 假设平稳，你的真实数据平稳吗？" → 答预处理到低频包络 + 不平稳用 level_preserving；并承认 peak-duration bias 是已知局限（见 LIMITATIONS §）。
- "FPR 30% 你还敢用？" → 答那是共享慢漂场景；默认 iaaft 在有慢漂时偏宽松，研究决策可切 level_preserving。

---

### #3 Existence gate（预注册端点 + 模态集）

**被选理由**
"同步是否存在"若从"哪个特征/模态先到 p<.05"来宣告，就是**未声明多重比较**。
gate 冻结**单一端点**（`peak_amplitude`）和**预注册主模态集**（`ECG`, `EDA`），
任一主模态的 cell pass_rate 严格 > 0.5 即判存在。自由度被提前锁死。

**与其他方案对比**
- vs "OR across all features / all channels"：未声明多重比较，p-hacking 温床。
- vs "要求所有主模态都过"：过度收紧——ECG 与 EDA 是同一自主神经同步构念的两个读数，一个达标即够。

**trade-off**
- **分母是 cell 不是 dyad**（2026-08-05 修正）：cell = dyad×modality×condition，是 L0 实际运行单位。
  多条件下二者不同；gate 定义在 cell-rate 上并预先固定，避免 any-pass 读法的多重性膨胀
  （每 dyad 多条件时 any-pass 期望 = `1 - 0.95**k`，随条件数膨胀，不能对固定 0.5 阈值比较）。
- 主模态集是**数据集相关**的：默认 Lerique 的 ECG/EDA/RESP 组成，其他数据集须通过
  `SyncPipeConfig.primary_modalities` **在看结果前**声明，否则继承 Lerique 默认。RESP 是负对照，进报告但**不进 gate**（常被任务结构整步/夹带）。

**预判追问**
- "cell 分母会不会高估存在性？" → 答 gate 定义在 cell-rate、预先冻结；any-pass 读法因多重性膨胀被明确排除，不因为答案"不方便"就采纳。
- "RESP 为什么排除？" → 答 RESP 多被任务结构整步，是敏感性/对照通道，不是自主神经同步主读数。

---

### #4 Dyad-paired permutation @ L2（唯一验证性声明）

**被选理由**
v1 唯一验证性声明是"同一 dyad 的两预注册条件对比"。dyad 是配对单位、condition 是 within-dyad 因子，
置换在 dyad 内打乱条件标签，天然控制 dyad 间个体差异（年龄、基线等）。

**与其他方案对比**
- vs **unpaired group comparison**：出 confirmatory 范围（仅探索）。
- vs **mixed-effects 模型**：出范围（探索）。
- vs **arbitrary covariate**：出范围。

**trade-off**
- 只覆盖"两预注册条件"这一设计；多条件/分阶段设计可把 phase 标成 condition 复用 L2 机器（探索→对比，须预注册）。
- 连续预测/结局走 `prediction.py`（回归族，探索性，**不在 FDR 声明内**，且该路径**无 null model**，见 #9 / `V1_CLAIM_CEILING.md` §6a）。

**预判追问**
- "只能比两条件，多条件研究怎么办？" → 答 phase 标 condition 复用 L2，但须预注册、否则探索性。
- "那 prediction.py 的 ΔAUC 能说显著吗？" → 答**不能**：无 surrogate/permutation/null，ΔAUC 无经验 null 分布、无 p 值、不在 FDR 族；只能当描述量报。

---

### #5 WCC（零滞后、leading window）

**被选理由**
WCC（Windowed Cross-Correlation）是测量内核：在滑动窗口上算两信号零滞后互相关系数，
得到一条连续 WCC 轨迹。选它是因为可解释、可可视化、可接描述符与三层审计。

**与其他方案对比**
- vs **lagged CCF**：v1 **只算零滞后**（有 `max_lag_sec=0` 的 fail-loud 守卫），无可用 lagged estimator；
  任何 lead-lag 对齐须在预处理/对齐步解决，不在 v1 内。
- vs **动态时间规整(DTW)等**：引入对齐自由度，破坏"对齐网格上 co-fluctuation"的干净测量对象。

**trade-off**
- `window_size` 是**自由参数，非最优声明**：须预注册 + 敏感性分析（Gate 4），报告必须写明取值。
- leading window：`output[i]` 覆盖 `[i, i+window_size-1]`，默认 `step=1`（最大重叠）、`window_type="rect"`。
- observation opportunity（`n_valid_wcc_points` 等）逐 dyad-condition 记录，约束可比性；不等观测机会跨条件直接 `raise`。

**预判追问**
- "window_size 怎么选的？" → 答预注册 + 敏感性分析，非数据驱动最优；报告载明。
- "零滞后够吗？" → 答 v1 锁零滞后；lead-lag 是预处理责任，morphology 描述符描述的是 WCC 轨迹形状而非人际时序方向。

---

### #6 BH-FDR @ L2（按 FDR 家族分组）

**被选理由**
多特征/多模态对比需多重校正。BH-FDR 在控制错误发现比例的同时比 Bonferroni 更有功效，
且按 `FDR_FAMILIES`（SSoT，来自 `feature_definitions.py`）**分组**校正，避免把异质特征塞进一个全局池。

**与其他方案对比**
- vs **Bonferroni**：过于保守，低功效。
- vs **不分组的 global FDR**：异质特征混合，校正粒度错。
- vs **不校正**：假阳性膨胀。

**trade-off**
- 分组依据是 SSoT 的 `FDR_FAMILIES`；**未归类到任何家族、又非 reference 的特征，原来会落入以自身命名的 singleton 组 → BH 对单元素恒等 → 静默逃逸多重校正**（已在 T6c 修：并入统一 catch-all 家族共同校正）。
- `fdr_scope` 可选 `global` / `within_modality`（PRDS 保持依赖 scope）。

**预判追问**
- "BH 分组依据是什么？未知特征怎么处理？" → 答 SSoT `FDR_FAMILIES`；未知特征不再各自逃逸，进 catch-all 共同校正（T6c）。
- "reference 特征（如 mean_synchrony）进 FDR 吗？" → 答不进，只报描述量。

---

### #7 level_preserving IAAFT（L0 备选 null）

**被选理由**
当两信号共享慢漂（common slow driver）时，iaaft 因重排相位仍保留慢漂结构，FPR≈30%（偏宽松）。
level_preserving 先**分离并保留慢趋势**，只对**残差**做 IAAFT，因此在共享慢漂下 FPR≈1.7%、功效>0.8。

**与其他方案对比**
- vs **iaaft（默认）**：iaaft 在共享慢漂下 FPR 高；level_preserving 更严，但改变了"null 回答了什么问题"（它假设慢趋势已知且共享）。
- 二者**语义不同**，切换是**研究决策**，不是工程优化。

**trade-off**
- `L0_DEFAULT_NULL_MODEL = "iaaft"`，**不许静默改**。
- 标签拼写 `"iaaft"` / `"level_preserving"` 经 `_L0_NULL_MODEL_LABELS`
  `{"iaaft": "signal_level_iaaft", "level_preserving": "signal_level_level_preserving_iaaft"}` 单一来源，已落地 artifact 与既有断言，不许"规范化"。
- L2 路径的 null 标签已贯通查此表（T4 修），artifact 字段必须反映真实所用 null，不许硬编码/f-string 拼装。

**预判追问**
- "两个 L0 null 谁对？" → 答不冲突，回答不同问题；默认 iaaft，有共享慢漂且研究问题容许时切 level_preserving，须显式声明。
- "artifact 里 null 标签怎么保证真实？" → 答查 `_L0_NULL_MODEL_LABELS` 单一来源，无硬编码（T4 对称审计）。

---

### #8 Design controls（pseudo-pair / time-shift / across-stim shuffle）

**被选理由**
L0/L1 分离信号与噪声，但不建立因果；L2 控 within-dyad 混淆。Design controls 进一步排除
"共享刺激 / 慢漂 / 周期"等**非 dyad 特有耦合**的 null 场景（Gate 3）。

**与其他方案对比**
- vs 只跑 L0/L1/L2：拿不到"排除共享刺激"的证据。
- 三控各有针对：pseudo-pair（打乱人际配对）、time-shift（时间错位）、across-stim shuffle（跨刺激洗牌，适用时）。

**trade-off**
- 是**必要非充分**证据：仍不建立因果，只排除特定混淆。
- 是否适用取决于实验设计（across-stim 仅特定设计可用）。

**预判追问**
- "过了 design controls 就能说耦合了吗？" → 答不能，仍是非因果；只是把非 dyad 特有混淆排除得更干净。

---

### #9 Claim ceiling / governance（stage_status / claimable）

**被选理由**
每条 L2 结果带 `definedness_status` / `eligibility_status` / `claimable` / `stage_status`，
只有所有闸门通过才是验证性。"探索性"**永不静默变验证性**（须预注册 estimand + 声明 null + 闸门通过 + 新 Gate-0 freeze 才能升级）。

**与其他方案对比**
- vs 宽松报告：多数工具把描述量当推断量；SyncPipe 显式标 tier。
- `prediction.py` 是反例教材：2014 行、默认关闭、无 null model，ΔAUC 无 p 值；代码已改掉"返回 0.5 假装 chance"的造假（改报 NaN，呼应 fail-loud）。

**trade-off**
- 牺牲"能说的范围"换"说的每句可辩护"。对面试是**加分项**：直接正面回应"0 用户/0 验证"类质疑——方法是审计过的，产品是另一回事（见 demo "我们站在哪"屏）。

**预判追问**
- "你到底证明了什么？" → 答：在锁定协议下，peak_amplitude 在两预注册条件间有 dyad-paired 差异（BH-FDR），且 L0/L1 存在性已审计；不证明耦合/因果/构念。
- "prediction 结果能用吗？" → 答只能当探索性描述量，无推断地位，绝不加"显著/优于基线"。

---

### #10 Fail-loud 设计原则

**被选理由**
v1 多数 bug 共性是"能跑就跑而非出错就响"（Blind Spot C）。原则要求每个边界（外部数据进入、阶段交接）**校验输入并 raise/warn**；`ValueError` 在摄入阶段很便宜，论文图里静默错数很贵。

**与其他方案对比**
- vs "运行期尽量不报错、给默认值"：默认值若改变科学含义（如把两独立时间线重锚到共享 0.0 假装已校正偏移）就是静默造假。
- 配套：每个出数路径有 contract test（`test_suite_health.py` 钉住收集基线，防 guard 测试变永久 skip 仍报成功）。

**trade-off**
- 牺牲一点"宽容度"换"可审计性"。对科研软件正确。

**预判追问**
- "你怎么保证测试真的在测？" → 答 suite_health 钉住收集基线 + 每个模块路径须解析；三处 `parents[1]` bug 曾把 guard 测试变永久 skip 仍绿，已钉死。

---

## 4. 面试答辩骨架

### 2 分钟讲稿主线
1. **对象**（20s）：我们做的是对偶连续低频同步性的**可审计测量程序**，不是构念效度测量。
2. **内核**（30s）：WCC 得到连续轨迹 → 三层 null 审计（L0 存在性 / L1 结构 / L2 跨条件）。
3. **为什么三层**（40s）：每层答不同问题；measured fact——IAAFT 在 L0 正确、在 L1 零功效，所以层间不能共享 null/BH 分母。
4. **验证性声明**（30s）：唯一验证性是 dyad-paired 两条件对比 + BH-FDR；existence gate 锁死端点与模态防 p-hacking。
5. **诚实边界**（20s）：不证耦合/因果；claim ceiling 把"探索性"与"验证性"分开；产品与方法分层（demo "我们站在哪"）。

### 预判深挖清单（按命中概率）
| 追问 | 落点 |
|---|---|
| IAAFT 为什么不在 L1 用？ | #1 / #2：L1 零功效 measured fact |
| 两个 L0 null 谁对？ | #7：语义不同，研究决策，默认 iaaft 不许静默改 |
| gate 分母 cell 还是 dyad？ | #3：cell-rate，预先冻结，any-pass 因多重性膨胀被排除 |
| 只能比两条件？多条件呢？ | #4：phase 标 condition 复用 L2，须预注册 |
| BH 分组依据 / 未知特征？ | #6：SSoT FDR_FAMILIES，T6c 修逃逸 |
| prediction 结果显著吗？ | #9 / #4：无 null model，只能描述量 |
| 你到底证明了什么？ | #9：条件差异 + 存在性审计，非耦合/因果 |
| window_size 怎么选？ | #5：预注册 + 敏感性，非数据驱动最优 |
| 测试真在测吗？ | #10：suite_health 钉基线，防 skip 变绿 |

---

## 5. 已知边界（诚实，呼应 claim ceiling）

- **Peak-duration bias**：靠观测元数据 + 严格拒不等长缓解；duration-aware 峰值校正**推迟**至独立仿真验证（不静默加最大统计量校正）。
- **NaN / dropout 策略**：有硬 NaN 比守卫；dropout vs segment-seam 区分、最短有限 WCC 点等**尚未最终确定**，须列为局限。
- **真实数据状态**：数据集分 `fully rerunnable` / `artifact-backed` / `diagnostic only`；Gordon/Andersen loader 不完整，不得声称完全可复现。Bizzego 目前唯一由 CI 强制 `fully rerunnable`。
- **prediction.py 无 null model**（见 #9 / `V1_CLAIM_CEILING.md` §6a）。
- **L1 不发 per-dyad 判定**（`L1_INFERENCE_UNIT = "group"`，2026-08-06 裁定）。

---

## 6. 与代码/文档的对应（便于现场查证）

| 文档 | 用途 |
|---|---|
| `V1_PROTOCOL.md` | 锁定的科学对象、三层 null、existence gate、FDR 治理 |
| `V1_CLAIM_CEILING.md` | 能说什么/不能说什么、claim tier、prediction 无推断地位 |
| `DESIGN_PRINCIPLES.md` | fail-loud、每出数路径有测试、docstring 契约 |
| `METHOD_LOG.md` | 决策史（如 7c 退役 trace-level bootstrap） |
| `LIMITATIONS.md` | 已知局限（peak-duration bias、NaN 策略等） |
| `feature_definitions.py` | `FDR_FAMILIES` / `L1_INFERENCE_UNIT` / `_L0_NULL_MODEL_LABELS` 等 SSoT |
| `inference_pipeline.py` | 三层审计编排、`_existence_gate_by_modality` |
| `dynamic_features.py` | IAAFT / level_preserving / L1 block_permutation surrogate |
| `validation/l2_between_condition.py` | dyad-paired permutation + BH-FDR（含 T6c catch-all 修正） |
| `artifacts/demo/syncpipe_investor_demo.html` | 现场演示（交互 existence proof + "我们站在哪"状态卡） |

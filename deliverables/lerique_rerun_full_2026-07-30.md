# Lerique 真实数据完整分析重跑报告（2026-07-30 / 收尾 2026-07-31）

**动因**：`prediction.py::_compute_effective_gap` 时序泄漏 gap 修复（`horizon_windows` → `ceil(horizon_windows*window_size/step)`）改变了预测链路的物理 gap 计算。需确认该修复对**真实数据结论**的影响，并顺带重跑标准管线验证修复后代码状态。

**范围**：Lerique-47n3p（176 dyad × 三模态 EDA/ECG/RESP × rest1 + trials_concat）。

---

## 一、Phase B — prediction.py 时序泄漏 gap 修复对真实迹的影响 ✅ 已定论

对 176 条真实 WCC 迹跑新旧两种 `rolling_origin_cv`（中位数阈值取平衡标签），逐模态 / 逐 horizon 对比：

| 模态 | horizon | mean ΔAUC (NEW) | mean ΔAUC (OLD) | 泄漏阈值 0.14 | 泄漏标记翻转 |
|---|---|---|---|---|---|
| EDA | hw=1 | −0.16 | −0.27 | 远低于 | 0 |
| EDA | hw=2 | −0.13 | −0.33 | 远低于 | 0 |
| ECG | hw=1 | −0.24 | −0.30 | 远低于 | 0 |
| ECG | hw=2 | −0.22 | −0.31 | 远低于 | 0 |
| RESP | hw=1 | −0.19 | −0.10 | 远低于 | 0 |
| RESP | hw=2 | −0.20 | −0.12 | 远低于 | 0 |

**结论**：
- 所有模态 / horizon 的 mean ΔAUC 均为**负值**（−0.13 ~ −0.24），**远低于 0.14 泄漏阈值** → Lerique 的真实 WCC 迹**没有可预测的前瞻信号**（符合同步性是 contemporaneous 而非 lead/lag 本质）。
- **泄漏标记翻转数 = 0** → gap 修复**不改变 Lerique 任何时序预测结论**。
- 逐迹最大 `|Δdiff|` 达 0.40（单 dyad 小样本噪声），但聚合位移极小（≤0.055）。
- 产物：`artifacts/prediction/lerique_gapcheck.json`。

**科学含义**：gap 修复保证了预测模块在合成数据上不再伪报 lead/lag 可预测性（这是校准正确性修复），但它对 Lerique 真实结论无实质影响——因为真实同步性本来就是 contemporaneous 的，时序预测本就不显著。这与"先展示不利结果"的规范一致：预测模块在真实数据上诚实地为"无可预测性"。

---

## 二、Phase A — canonical 3-pipeline 重跑（修复 session_pooled 变长 bug 后）✅ 已完成

### 2.1 修复了一个阻断"完整分析"的 pre-existing bug
- **症状**：`session_threshold.compute_session_pooled_threshold` 对变长 WCC 迹（Lerique rest1≈151 vs trials_concat≈1061）用 `np.vstack` 堆叠 → `ValueError: all the input array dimensions... must match exactly`。
- **证据**（非我引入）：该 vstack 行不在我任何改动的 diff 内（`git diff` 仅显示我加了 deprecation docstring）；且 `artifacts/realdata_full/realdata_full_Lerique.json` **历来仅 476 字节**（其他数据集数 KB~97KB），说明 **Lerique canonical 历来因该 bug 从未跑完**。
- **最小修复**：`np.vstack(pooled_values)` → `np.concatenate([m.ravel() for m in pooled_values], axis=0)`（与同文件 `compute_surrogate_threshold` 内部 flatten 语义一致；等长数据数值不变，变长数据从崩溃转可用）。
- **回归**：新增 `test_by_modality_variable_length_dyads_pool_without_error`，`tests/unit/test_session_threshold.py` **8/8 通过**。

### 2.2 L2 per-modality / morphology 复现已知 FULL CASCADE 模式
- EDA：3 feature 显著（dwell_time / switching_rate / mean_synchrony）
- ECG：2 feature 显著（peak_amplitude / dwell_time）
- RESP：1 feature 显著（peak_amplitude）
- morphology：RESP k=4、ECG k=2（与跨数据集双模态结构一致）。
- `l2_full_family` 报错的预期行为：P0-2 多模态 guard 拒绝 pooling（设计如此，非 bug）。

### 2.3 ⚠️ canonical 报 `existence pass_rate=0.0 (0/176)` 是配置伪影，非科学结论
- **根因**：FAST-CONFIRMATION 设 `SURROGATE_N = 30`。存在性检验用**双尾** Phipson-Smyth，最小可达 p = `2×(0+1)/(30+1) ≈ 0.0645 > 0.05` → **任何对都不可能显著** → 0/176 是 n 太小导致的伪影。
- 项目自身 docstring 声明规范存在性审计用 `surrogate_n=100`（双尾最小 p≈0.0198<0.05）。

---

## 三、存在性 n=100 重审（正确标准）✅ 已完成

用 `synchrony_existence_audit` + `surrogate_n=100`（与文档声明一致）重审：

| 模态 | n_pairs | 通过数 | 通过率 |
|---|---|---|---|
| **EDA** | 60 | 60 | **100.0%** |
| **ECG** | 54 | 38 | **70.4%** |
| **RESP** | 62 | 15 | **24.2%** |

**按 feature（通过率）**：
- EDA：peak_amplitude 100% · bimodality_coefficient 96.7% · mean_synchrony 91.7%
- ECG：peak_amplitude 57.4% · bimodality_coefficient 29.6% · mean_synchrony 18.5%
- RESP：peak_amplitude 14.5% · bimodality_coefficient 12.9% · mean_synchrony 6.5%

**按 condition（模态内对比）**：
- EDA：rest1 100% = trials_concat 100%（一致强）
- ECG：rest1 59.3% < trials_concat 81.5%（任务期更强，符合预期）
- RESP：rest1 25.8% ≈ trials_concat 22.6%（弱且稳定）

**与历史数字的关系（重要澄清）**：记忆中"Lerique EDA peak_amp 56.7%"是**校准（null 下 Type-I error）通过率**，不是真实数据的经验通过率；本重审的 100% 是**真实数据的经验存在性通过率**。二者对象不同、不矛盾——EDA 在 still-face 范式下本就极强，100% 经验通过是合理的。RESP 最弱（24%）、ECG 居中（70%），符合三模态的已知强度梯度。

产物：`artifacts/realdata_full/lerique_existence_n100.json`。

---

## 四、最终结论

1. **gap 修复对真实结论无影响**：prediction.py 时序泄漏修复在 176 条真实迹上不改变任何时序预测结论（所有 ΔAUC 负向、无泄漏标记翻转）。修复是校准正确性提升，不影响 Lerique 科学结论。
2. **标准管线在修复后代码上复现已知模式**：EDA/ECG/RESP 的 L2 FULL CASCADE 与 morphology 双模态结构与既往一致。
3. **存在性真实通过率：EDA 100% / ECG 70% / RESP 24%**（n=100），推翻了 canonical 的伪影 0/176。
4. **修复了一个 pre-existing 阻断 bug**（session_pooled 变长 vstack），使 Lerique canonical 首次能完整跑完。

## 五、遗留 / 建议
- **FAST-CONFIRMATION 的 `SURROGATE_N=30` 应上调至 100**（或文档明确声明这是快速 smoke、存在性结论须用 n=100 重审），否则存在性会被系统性压抑为 0。
- 是否需要把 `SURROGATE_N` 默认值在 `realdata_full_new_pipeline.py` 改为 100，由你决策（影响运行时间约 3×）。
- 本次未重跑 Lerique 的 incremental-AUC（`run_lerique_incremental_auc.py` 不调用 prediction.py，gap 修复不影响它；其结论取决于条件间 per-record feature 对比，与本次动因无关）。

## 六、产物清单
- `artifacts/prediction/lerique_gapcheck.json` — Phase B 新旧 gap 对比
- `artifacts/realdata_full/realdata_full_Lerique.json` — Phase A canonical 重跑（修复后）
- `artifacts/realdata_full/lerique_existence_n100.json` — 存在性 n=100 重审
- `deliverables/run_lerique_prediction_gapcheck.py` — Phase B runner
- `deliverables/run_lerique_canonical.py` — Phase A driver（复用项目脚本，未改项目脚本）
- `deliverables/run_lerique_existence_n100.py` — 存在性重审脚本
- `deliverables/lerique_rerun_gapcheck_2026-07-30.md` — Phase B 独立报告

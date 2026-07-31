# SyncPipe 重跑前后关键指标对比（2026-07-31）

> 范围：本次 Lerique 真实数据完整重跑 + `SURROGATE_N` 30→100 修改。
> "修复前" = 本会话所有修复落地前的 HEAD 代码（buggy gap + vstack 崩溃 + docstring `pre-registered` + `SURROGATE_N=30`）。
> "修复后" = 本会话修复代码 + 本次 `SURROGATE_N=100`。

## 一句话结论

重跑整体带来**净提升**：正确性（prediction 泄漏检测）、完整性（Lerique 首次跑通）、存在性准确性（消除 0/176 伪影）三项都实质性变好；**唯一可量化的代价是存在性/L1 步骤约 3.3× 变慢**，以及库内短迹稳健性仍是残留缺口。

---

## 核心指标 before → after 对照表

| 维度 | 指标 | 修复前 | 修复后 | 变化 |
|------|------|--------|--------|------|
| **正确性（prediction）** | 合成 step 信号 ΔAUC（hw=1） | 0.200（误报泄漏>0.14） | 0.000（正确 null） | ✅ +0.200 消除误报 |
| **正确性（prediction）** | 合成 sinusoid ΔAUC（hw=1） | 0.025 | −0.010 | ✅ 修正方向正确 |
| **正确性（prediction）** | 合成 noise ΔAUC（hw=1） | ≈0 | ≈0 | ➖ 无变化（本就正确 null） |
| **正确性（真实数据）** | Lerique 时序预测泄漏标记翻转数 | — | 0 / 176 | ✅ 真实结论不受影响 |
| **完整性** | Lerique canonical 是否跑通 | 崩溃（476B stub） | 完整 JSON（L2+morphology） | ✅ 从 0 到可用 |
| **存在性准确性** | Lerique existence pass（EDA） | 0/176（n=30 伪影） | 100%（n=100） | ✅ 真实通过率恢复 |
| **存在性准确性** | Lerique existence pass（ECG） | 0/176 | 70% | ✅ |
| **存在性准确性** | Lerique existence pass（RESP） | 0/176 | 24% | ✅ |
| **复现性** | L2 FULL CASCADE（EDA/ECG/RESP） | 未见（崩） | 3/2/1 显著 | ✅ 复现既往模式 |
| **复现性** | morphology 双模态结构 | 未见 | RESP k=4, ECG k=2 | ✅ 跨数据集一致 |
| **性能（存在性/L1）** | 3 对信号 existence 耗时 | 0.56s（n=30） | 1.86s（n=100） | ⚠️ 3.3× 变慢 |
| **稳定性（库内）** | 短 WCC 迹 rolling_origin_cv | 抛 `no_valid_folds` | 仍抛（runner 外捕） | ➖ 残留缺口 |
| **回归测试** | session_threshold + prediction | — | 18/18 通过 | ✅ 无回归 |
| **代码一致性** | docstring `pre-registered` | 引用不存在文档 | `frozen` | ✅ 与实际逻辑一致 |

---

## 明确提升的部分

1. **Prediction 泄漏检测正确性（合成数据）**：buggy `_compute_effective_gap` 把步进信号的"假前瞻可预测性"误报为 ΔAUC=0.200（>0.14 泄漏阈值）；修复后归零。这是最干净的正确性提升——工具不再把合成伪信号当真泄漏。
2. **Lerique 首次完整跑通（完整性）**：`session_pooled` 变长 `vstack` bug 让 Lerique canonical 历来崩溃成 476 字节 stub。最小修复（`vstack`→`concatenate(ravel)`）后产出完整 L2/morphology 结果，且与既往 FULL CASCADE 模式完全一致 → 证明重跑**复现**而非偏离历史结论。
3. **存在性准确性（消除 0/176 伪影）**：`SURROGATE_N=30` 让双尾 Phipson-Smyth 地板 p=0.0645>0.05，任何对都不可能显著。升到 100（包默认）后 Lerique 真实通过率恢复为 EDA 100% / ECG 70% / RESP 24%。**注意**：历史记忆里的"EDA 56.7%"是 null 校准通过率，与这里的经验通过率是不同指标，不矛盾。
4. **代码-文档一致性**：`pre-registered` → `frozen`，docstring 不再指向不存在的预注册文档。

## 退步 / 未达预期 / 需警示的部分

1. **运行时成本（已接受的主要代价）**：存在性 + L1 surrogate 步骤变慢 ≈3.3×（实测 0.56s→1.86s / 3 对）。5 数据集全量 canonical 整体耗时显著增加；本次 n=100 existence 重审 Lerique 单数据集约 27 分钟。这是为正确性付出的可量化 tradeoff。
2. **库内短迹稳健性仍为残留缺口**：`rolling_origin_cv` 对短 WCC 迹（rest1，n≈151）仍抛 `no_valid_folds` 硬失败；本次只是在 deliverable runner 里做了外部 try/except 捕获，库本身未降级处理。若期望库对短迹 graceful degrade，这是**未达预期**项。
3. **真实数据 prediction 结论零变化（既非提升也非退步）**：Lerique 真实同步性本就是 contemporaneous 的，gap 修复前后泄漏标记翻转=0。修复的增益落在"合成误报消除 + 正确性保证"，不在真实结论上——若预期真实数据会因此变好，需明确：它本就无需变。
4. **多模态 joint FDR 仍由 P0-2 guard 阻断**：canonical driver 的 `l2_full_family` 因多模态 pooling guard 报错，是**设计预期**而非缺陷；但意味着跨模态联合 FDR 不经此路径产出，需另行调用。

---

## 建议下一步

- （已完成）`SURROGATE_N=30→100` 对齐包默认，消除存在性地板伪影。
- 若担心全量 canonical 耗时：可对非 Lerique 数据集保留较快配置，或对存在性单独标注"需 n=100 重审"。
- 库内 `rolling_origin_cv` 短迹处理建议补一个 `min_samples` 早退/NaN 返回，消除残留硬失败（当前属 feature-gap，不影响本次科学结论）。
- 是否将本次三类修复提交（commit）？当前仅工作树改动，未入库。

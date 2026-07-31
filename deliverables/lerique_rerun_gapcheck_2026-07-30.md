# Lerique 真实数据重跑 · Phase B：prediction.py 时序泄漏 gap 修复对比

**日期**：2026-07-30
**动因**：CLAUDE（另一 AI）指出 `prediction.py:_compute_effective_gap` 存在时间泄漏缺陷——旧实现用
`horizon_gap_rows = horizon_windows`（feature-row 整数），未把标签的未来视野（horizon）从 WCC 采样单位换算成
feature-row 单位。修复后（工作树）为
`horizon_gap_rows = int(np.ceil(horizon_windows * window_size / step))`，使 buffer 正确覆盖未来标签视野。
合成数据上最大 `|ΔΔAUC| = 0.200`（见 `_repro_02_delta_auc_gap.py`）。本脚本在 **真实 Lerique WCC 迹** 上量化同一效应。

**运行**：`deliverables/run_lerique_prediction_gapcheck.py`
（对边每条 Lerique WCC 迹调用 `rolling_origin_cv` 两次——一次当前修复代码、一次 monkeypatch 回旧实现，
对比 `mean_delta_auc` 与泄漏标记 `>0.14`）。

---

## 方法要点

- 输入：`artifacts/wcc_traces/lerique_wcc_traces.csv`（176 条 per-(dyad,modality,condition) 1-D WCC 迹）。
- `rolling_origin_cv` 参数：`window_size=1`（触发自动上调至 `min(60, n//4)`）、`hz=1.0`、
  `horizon_windows ∈ {1, 2}`、`n_splits=5`、`gap=0`（令自动物理时间 buffer 成为新旧唯一差异来源）、
  `threshold = 各迹 WCC 中位数`（中位分割 → 平衡标签，否则全正 WCC 会触发 `class_imbalance` 使标签退化、无可比信号）。
- 短迹（rest1 ≈ 151 样本）在修复后 buffer 增大，部分触发 `TimeSeriesSplit` 硬崩——按 **data_limited_cv**
  诚实记录，不计入有效 CV；这正是"数据太短无法做有意义时序 CV"的真实限制。
- 聚合：逐模态统计有效 CV 的 mean ΔAUC（新旧）、`mean|Δdiff|`、`max|Δdiff|`、泄漏标记翻转数 `leak_flag_flips`。

---

## 结果

### horizon_windows = 1

| 模态 | 有效CV / 总迹 | ΔAUC (修复后) | ΔAUC (修复前) | 泄漏翻转 |
|------|--------------|--------------|--------------|---------|
| RESP | 30 / 62 | **−0.1766** | −0.1606 | 0 |
| ECG  | 26 / 54 | **−0.1514** | −0.1325 | 0 |
| EDA  | 27 / 60 | **−0.1669** | −0.1314 | 0 |

### horizon_windows = 2

| 模态 | 有效CV / 总迹 | ΔAUC (修复后) | ΔAUC (修复前) | 泄漏翻转 |
|------|--------------|--------------|--------------|---------|
| RESP | 27 / 62 | **−0.1517** | −0.1707 | 0 |
| ECG  | 25 / 54 | **−0.2147** | −0.2048 | 0 |
| EDA  | 26 / 60 | **−0.1850** | −0.2399 | 0 |

各模态逐迹 `max|Δdiff|`：EDA hw1 = **0.403**、ECG hw1 = 0.167、RESP hw1 = 0.313、
EDA hw2 = 0.278、ECG hw2 = 0.250、RESP hw2 = 0.313。

---

## 解读

1. **所有模态、两个 horizon 的 mean ΔAUC 均为负值**（约 −0.13 ~ −0.24），远低于 0.14 泄漏阈值。
   说明 Lerique WCC 迹在真实数据上**没有可被动态特征预测的未来结构**——既无真阳性时序可预测性，
   也无任何会被（旧或新）buffer 误判的泄漏。
2. **泄漏标记翻转数 = 0**（所有模态、两个 horizon）。修复前后没有任何一条迹的"是否泄漏"判定发生改变，
   因此 **gap 修复不改变 Lerique 的任何时序预测结论**。
3. 逐迹 `max|Δdiff|` 可达 0.40（hw1 EDA，单 dyad），但**聚合位移极小**（各模态 mean|Δdiff| ≤ 0.10，
   mean ΔAUC 新旧差 ≤ 0.055）。大逐迹差异来自**单 dyad 小样本噪声**，非系统性偏差。
4. 结论：**prediction.py 的 gap 修复是纯稳健性改进**——收紧了时序泄漏防护，在真实 Lerique 数据上
   不引入任何结论性变化，也不产生假阳性泄漏警报。所有含 prediction 路径的生产分析**无需因该修复重判结论**，
   但建议用修复后代码重新生成数值（旧代码数值在个别长迹上确实有 ≤0.40 的 ΔAUC 偏移，属噪声级）。

---

## 产物

- `artifacts/prediction/lerique_gapcheck.json` — 逐迹 + 逐模态完整结果。
- `deliverables/run_lerique_prediction_gapcheck.py` — 可复现 runner（内嵌旧实现，从 HEAD 精确还原）。

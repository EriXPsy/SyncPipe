# 脚本 → BRM 主干结论 映射表（用于剥离决策 + User Manual 登记）

> 用途：把 `scripts/` 下每个脚本定位到它支撑 BRM 哪条主干结论，从而决定它是
> **(K) 保留为主干结果生成器**（在 manual 登记、保留在 scripts/）还是
> **(E) 剥离到 experimental/**（一次性诊断 / 已被取代 / 阴性）。
>
> BRM 主干（你的 v1.0 要件）：
> - **[PIPE]** Feature/Computation/Inference Pipeline + feature family/table
> - **[GT]** GT1–5 ground-truth 检验
> - **[REAL]** 真实数据集验证（Gordon, Lerique, Andersen, Han）
> - **[AUC]** 正向 Lerique 结果 + 增量 AUC
> - **[AUDIT]** 三步审计证据链（existence → design control → group inference）
>
> 状态列：**K**=保留主干 / **E**=建议剥离 experimental / **?**=需你拍板

| 脚本 | 支撑主干 | 我的建议 | 理由 |
|---|---|---|---|
| `build_feature_table.py` | [PIPE] | **K** | 从 SSoT 生成权威 feature table（docs 引用 4 次）。主干基础设施。 |
| `fdr_family_impact.py` | [PIPE][AUDIT] | **K** | round-3 FDR 家族决策的影响分析依据。可复现的决策证据。 |
| `audit_timing_features.py` | [PIPE] | **K** | timing 描述符接入前的冗余审计依据。 |
| `validate_timing_descriptors.py` | [AUC] | **K** | round-4 block-bootstrap null + 增量 AUC。当前主验证脚本。 |
| `run_pgt2_grid.py` | [GT] | **K** | PGT-2 结构恢复网格（主干 GT 结果生成器）。 |
| `run_pgt2_surrogate.py` | [GT][AUDIT] | **K** | PGT-2 + per-dyad surrogate threshold（DECISION-01）。主干。 |
| `run_pgt3_grid.py` | [GT] | **K** | PGT-3 时间恢复网格（810 cells）。主干 GT。 |
| `run_egt4_matrix.py` | [GT] | **K** | EGT-4 emergent dynamics 2×2 验证矩阵。主干 GT。 |
| `run_gt5_simulation.py` | [GT][REAL] | **K** | GT-5 Gordon-conditions 模拟（46 dyads×5 cond）。主干。 |
| `run_kuramoto_l23_taxonomy.py` | [GT][AUC] | **K** | Kuramoto L2+L3 taxonomy（EGT 生成器源，被 validate_timing 复用）。主干。 |
| `run_gordon_case_study.py` | [REAL] | **K** | Gordon (2025) 案例研究 runner（BRM §6.x）。主干真实数据。 |
| `run_lerique_pilot.py` | [REAL] | **K** | Lerique (2024) pilot runner（有预注册+DECISION_LOG）。主干真实数据。 |
| `run_lerique_incremental_auc.py` | [AUC] | **K** | Lerique 增量 AUC（SSoT 驱动）。主干正向结果。 |
| `run_incremental_auc.py` | [AUC] | **K** | 通用增量 AUC（任意 realtest pipeline）。主干工具。 |
| `export_wcc_vif_morphology.py` | [REAL][PIPE] | **K** | 导出 WCC traces + VIF + 形态聚类（被 validate_timing 用作输入源）。主干。 |
| `rerun_real_datasets.py` | [REAL][AUDIT] | **K** | 用更新代码重跑真实数据 surrogate 检验（L0/L1 null + per-dyad threshold）。主干。 |
| `lerique_feature_analysis.py` | [AUC][REAL] | **? → 偏 K** | naive vs rigorous 两方法对照（leave-dyad-out CV + bootstrap CI）。是方法学严谨性展示，**建议保留**但请你确认是否进 BRM 主干。 |
| `run_lerique_shuffle.py` | [AUC] | **? → 偏 K** | AUC 加入顺序鲁棒性（100次随机顺序）。是对增量 AUC 的稳健性背书，**建议保留**。请确认。 |
| `surrogate_controls.py` | [REAL][AUDIT] | **? → 偏 K** | Lerique post-hoc specificity 检验（排除 nuisance 假设）。**建议保留**。请确认。 |
| `analyze_all_gt.py` | [GT] | **? → 偏 K** | 综合 GT 分析（PGT-2+PGT-3+EGT-4）。可能是主干汇总，也可能被单独 run_* 取代。**请你确认**是否冗余。 |
| `analyze_pgt2_fixed.py` | — | **E** | "fixed" 后缀，读 `pgt2_grid_results.csv` 测 H2.4。一次性临时分析。剥离。 |
| `diagnose_pgt2_drift.py` | — | **E** | 一次性诊断（peak_amplitude 随 n_epochs 漂移）。剥离。 |
| `diagnose_h2_switching_entropy.py` | — | **E** | 一次性诊断（switching/entropy 噪声来源）。剥离。 |

## 汇总

- **K（保留主干）**：16 个 —— 这些是你"所有代码依附在清晰研究主干上"的主干结果生成器，将在 User Manual 里按 [PIPE]/[GT]/[REAL]/[AUC]/[AUDIT] 分类登记。
- **E（建议剥离）**：3 个明确的一次性诊断/fixed 脚本 → `experimental/scripts/`。
- **?（需你拍板）**：4 个 —— `lerique_feature_analysis`、`run_lerique_shuffle`、`surrogate_controls`（我倾向保留，是严谨性背书）、`analyze_all_gt`（可能与单独 run_* 冗余，请你确认）。

## 待你确认的剥离动作
1. 明确的 3 个 **E** 是否批准剥离到 `experimental/scripts/`？
2. 4 个 **?** 里：`analyze_all_gt.py` 是不是被单独的 `run_pgt2/3_grid.py`+`run_egt4_matrix.py` 取代了（若是→E）？其余 3 个我倾向保留，你同意吗？
3. round-4 的 `_deprecated_nulls/circular_shift_timing_null_FALSIFIED.py` 是否一并移到 `experimental/scripts/` 统一管理？

"""
multisync.morphology — 同步性事件 Epoch 的形态学描述 (v0.1, 2026-06-08)
=====================================================================
设计原则 (根据用户 2026-06-08 的明确要求)
-----------------------------------------
1. **不门控计算**: 本模块只 *描述* WCC 轨迹的形态, 绝不阻止或修改
   onset/rise/recovery 等时序特征的计算。它是一个可选的旁路分析。
2. **形态无关优先**: 提供一组对任意形态都有良定义的描述符 (峰数、
   高于阈值占比、趋势、首峰时间、是否回落…), 用「时序特征描述形态」
   而非「假定单峰事件」。
3. **支持预注册假设**: 研究者可声明一个期望形态 (sustained / single_peak
   / plateau / antiphase / oscillatory), classify_morphology 会返回该
   假设的支持度, 供预注册式检验使用。

理论依据
--------
- daSilva & Wood (2024, Pers Soc Psychol Rev): 同步性沿多维度分类,
  不存在单一标准形态; "more synchrony is not always better"。
- Kelso 协调动力学 / metastability: 同步是间歇、亚稳的, 天然多峰。
- Chen et al. (2025, Psychophysiology): in-phase vs anti-phase linkage。
因此「单峰 baseline→rise→peak→recovery」(源自 SCR phasic 响应) 只是
众多形态之一, 不应作为同步性 Epoch 的默认假设。

本模块不对哪种形态"更好"做价值判断, 仅做结构描述。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

# 与 feature_definitions.ONSET_THRESHOLD 对齐的默认阈值 (仅用于形态描述,
# 不影响 SSoT 的特征计算)。
DEFAULT_THRESHOLD = 0.5

# 预注册假设可用的形态标签
MORPHOLOGY_LABELS = (
    "sustained",     # 持续型: 大部分时间高于阈值, 少有进出
    "single_peak",   # 单峰事件型: 一个主峰, 有升有落 (SCR 式)
    "plateau",       # 平台型: 升起后维持, 不回落
    "oscillatory",   # 振荡/间歇型: 多峰反复进出
    "subthreshold",  # 阈下型: 几乎从不跨越阈值
    "antiphase",     # 反相型: 以负相关为主 (需要带符号的 WCC)
)


@dataclass
class MorphologyProfile:
    """WCC 轨迹的形态描述 (全部形态无关, 对任意轨迹有定义)。"""
    n_samples: int
    finite_ratio: float
    above_ratio: float          # 高于阈值的样本占比
    n_episodes: int             # 高于阈值的连续段个数
    n_peaks: int                # 显著峰个数 (prominence-based)
    first_peak_time: float      # 首个显著峰的时间 (秒); NaN 若无
    peak_value: float           # 全局最大值
    returns_to_baseline: bool   # 主峰后是否回落到半高度
    has_baseline_phase: bool    # 是否存在低于阈值的起始段
    trend_slope: float          # 线性趋势斜率 (每秒)
    frac_negative: float        # WCC<0 的样本占比 (反相指示)
    peak_asymmetry: float       # 上升段/下降段时长比 (主峰); NaN 若无峰
    inter_peak_cv: float        # 峰间间隔变异系数; NaN 若峰数<2
    baseline_fraction: float    # 首峰前低于阈值的样本占比
    label: str                  # 主形态标签
    label_scores: Dict[str, float]  # 各形态的连续支持度 [0,1]

    def to_row(self) -> Dict:
        d = asdict(self)
        scores = d.pop("label_scores")
        for k, v in scores.items():
            d[f"score_{k}"] = v
        return d


def _episodes(above: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回高于阈值连续段的 (起点, 终点) 索引。"""
    padded = np.concatenate(([False], above, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return starts, ends


def _count_prominent_peaks(
    wcc: np.ndarray, threshold: float, min_prominence: float
) -> Tuple[int, float, list]:
    """简单的显著峰计数 (不依赖 scipy): 在高于阈值的段内找局部极大,
    要求相对两侧谷的 prominence >= min_prominence。
    返回 (峰数, 首峰索引, 全部峰索引列表) 。"""
    n = len(wcc)
    finite = np.isfinite(wcc)
    if not finite.any():
        return 0, float("nan"), []
    x = np.where(finite, wcc, -np.inf)
    all_peaks = []
    for i in range(1, n - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            left_min = np.min(x[max(0, i - 1)::-1][:50]) if i > 0 else x[i]
            right_min = np.min(x[i + 1: i + 51]) if i + 1 < n else x[i]
            base = max(left_min, right_min)
            if x[i] - base >= min_prominence:
                all_peaks.append(i)
    if not all_peaks:
        return 0, float("nan"), []
    return len(all_peaks), float(all_peaks[0]), all_peaks


def classify_morphology(
    wcc: np.ndarray,
    hz: float = 1.0,
    threshold: float = DEFAULT_THRESHOLD,
    min_prominence: float = 0.15,
    sustained_above_frac: float = 0.6,
    expected: Optional[str] = None,
) -> MorphologyProfile:
    """描述单条 WCC 轨迹的形态 (不修改/不门控任何特征计算)。

    Parameters
    ----------
    wcc : 1-D array
        窗口化互相关轨迹 (可含 NaN)。
    hz : float
        WCC 轨迹采样率。
    threshold : float
        形态描述用阈值 (默认 0.5, 与 SSoT 对齐)。
    min_prominence : float
        显著峰所需的最小 prominence。
    sustained_above_frac : float
        判为 "sustained" 所需的高于阈值占比下限。
    expected : str or None
        预注册的期望形态标签; 若提供, label_scores 仍返回全部分数,
        研究者可据此判断数据是否支持其假设。

    Returns
    -------
    MorphologyProfile
    """
    wcc = np.asarray(wcc, dtype=float)
    n = wcc.size
    finite = np.isfinite(wcc)
    finite_ratio = float(finite.mean()) if n else 0.0

    if n == 0 or not finite.any():
        scores = {k: 0.0 for k in MORPHOLOGY_LABELS}
        return MorphologyProfile(
            n_samples=n, finite_ratio=finite_ratio, above_ratio=0.0,
            n_episodes=0, n_peaks=0, first_peak_time=float("nan"),
            peak_value=float("nan"), returns_to_baseline=False,
            has_baseline_phase=False, trend_slope=float("nan"),
            frac_negative=float("nan"),
            peak_asymmetry=float("nan"), inter_peak_cv=float("nan"),
            baseline_fraction=float("nan"),
            label="subthreshold", label_scores=scores)

    vals = wcc[finite]
    above = (wcc >= threshold) & finite
    above_ratio = float(above.sum() / finite.sum())
    starts, ends = _episodes(above)
    n_episodes = int(len(starts))
    frac_negative = float((vals < 0).mean())

    n_peaks, first_peak_idx, all_peak_indices = _count_prominent_peaks(
        wcc, threshold, min_prominence)
    first_peak_time = (first_peak_idx / hz
                       if np.isfinite(first_peak_idx) else float("nan"))
    peak_value = float(np.nanmax(wcc))

    # 峰不对称比: 主峰上升段 / 下降段时长
    peak_asymmetry = float("nan")
    if all_peak_indices and np.isfinite(peak_value):
        main_peak = int(all_peak_indices[0])  # 首个显著峰
        amp = peak_value - threshold
        if amp > 0:
            # 上升段: 从首次跨越阈值到主峰
            pre = wcc[:main_peak + 1]
            cross_idx = np.where((pre >= threshold) & np.isfinite(pre))[0]
            rise_dur = main_peak - cross_idx[0] if len(cross_idx) > 0 else 0
            # 下降段: 从主峰到首次回落到半高度
            half = threshold + 0.5 * amp
            post = wcc[main_peak:]
            fall_idx = np.where((post <= half) & np.isfinite(post))[0]
            fall_dur = fall_idx[0] if len(fall_idx) > 0 else len(post)
            if rise_dur > 0 and fall_dur > 0:
                peak_asymmetry = float(rise_dur / fall_dur)

    # 峰间间隔变异系数
    inter_peak_cv = float("nan")
    if len(all_peak_indices) >= 2:
        gaps = np.diff(all_peak_indices).astype(float) / hz
        if gaps.mean() > 0:
            inter_peak_cv = float(gaps.std() / gaps.mean())

    # 首峰前基线占比
    baseline_fraction = float("nan")
    if all_peak_indices:
        fp = all_peak_indices[0]
        pre_peak = above[:fp]
        if len(pre_peak) > 0:
            baseline_fraction = float((~pre_peak).mean())

    # 趋势斜率 (每秒)
    t = np.arange(n)[finite]
    if t.size >= 2:
        slope = float(np.polyfit(t, vals, 1)[0] * hz)
    else:
        slope = float("nan")

    # 是否存在起始 baseline 段 (前 10% 主要低于阈值)
    head = above[: max(1, n // 10)]
    has_baseline_phase = bool(head.mean() < 0.5)

    # 主峰后是否回落到半高度
    returns_to_baseline = False
    if np.isfinite(first_peak_idx) and np.isfinite(peak_value):
        amp = peak_value - threshold
        if amp > 0:
            half = threshold + 0.5 * amp
            post = wcc[int(first_peak_idx):]
            returns_to_baseline = bool(
                np.any((post <= half) & np.isfinite(post)))

    scores = _score_labels(
        above_ratio=above_ratio, n_episodes=n_episodes, n_peaks=n_peaks,
        returns_to_baseline=returns_to_baseline,
        has_baseline_phase=has_baseline_phase, frac_negative=frac_negative,
        sustained_above_frac=sustained_above_frac)
    label = max(scores, key=scores.get)

    return MorphologyProfile(
        n_samples=n, finite_ratio=finite_ratio, above_ratio=above_ratio,
        n_episodes=n_episodes, n_peaks=n_peaks,
        first_peak_time=first_peak_time, peak_value=peak_value,
        returns_to_baseline=returns_to_baseline,
        has_baseline_phase=has_baseline_phase, trend_slope=slope,
        frac_negative=frac_negative,
        peak_asymmetry=peak_asymmetry, inter_peak_cv=inter_peak_cv,
        baseline_fraction=baseline_fraction,
        label=label, label_scores=scores)


def _score_labels(
    *, above_ratio, n_episodes, n_peaks, returns_to_baseline,
    has_baseline_phase, frac_negative, sustained_above_frac,
) -> Dict[str, float]:
    """把形态无关描述符映射为各形态的连续支持度 [0,1]。
    评分是启发式的、可解释的, 不做硬分类边界。"""
    s: Dict[str, float] = {}

    # subthreshold: 几乎不过阈值
    s["subthreshold"] = float(np.clip(1.0 - above_ratio / 0.1, 0.0, 1.0))

    # sustained: 高占比 + 少段数
    sus = np.clip((above_ratio - sustained_above_frac) /
                  (1.0 - sustained_above_frac), 0.0, 1.0)
    sus *= np.clip(1.0 - (n_episodes - 1) / 3.0, 0.0, 1.0)
    s["sustained"] = float(sus)

    # single_peak: 恰一个峰 + 有 baseline + 有回落
    sp = 1.0 if n_peaks == 1 else 0.0
    sp *= 0.5 + 0.25 * has_baseline_phase + 0.25 * returns_to_baseline
    s["single_peak"] = float(sp)

    # plateau: 一个段, 升起但不回落, 中高占比
    pl = 1.0 if (n_episodes == 1 and not returns_to_baseline) else 0.0
    pl *= np.clip(above_ratio / 0.5, 0.0, 1.0)
    s["plateau"] = float(pl)

    # oscillatory: 多峰或多段
    osc = np.clip((max(n_peaks, n_episodes) - 1) / 3.0, 0.0, 1.0)
    s["oscillatory"] = float(osc)

    # antiphase: 负相关占比高
    s["antiphase"] = float(np.clip((frac_negative - 0.3) / 0.4, 0.0, 1.0))

    return s

"""
形态依赖性验证 (morphology-dependence check) — 可在终端自行复跑。
================================================================
目的 (2026-06-08)
-----------------
用受控的、已知形态的合成 WCC 轨迹, 直接检验 8 个特征里:
  - 哪些是「形态无关」(对任意形态都给出有意义、稳定的值);
  - 哪些是「形态依赖」(隐含单峰 SCR 假设, 在振荡/持续/平台型上
    会 NaN 或给出误导值)。

这回答用户的问题: onset_latency / rise_time / recovery_time / dwell_time
的操作化定义到底"对不对", 以及它们如何依赖形态。

方法
----
合成 5 种典型形态各 N 条 (加噪声 + 多 seed):
  single_peak  : baseline → 上升 → 单峰 → 回落   (SCR 式, onset/rise/recovery 的隐含假设)
  oscillatory  : 多次进出阈值 (metastable, Lerique 真实数据里占 56%)
  sustained    : 起步即高、全程维持           (高基线 ISC)
  plateau      : 上升后维持不回落
  subthreshold : 几乎从不过阈值

对每条算 8 个特征, 然后按形态汇总:
  - defined-rate (非 NaN 比例): 形态依赖特征在非单峰上会很低;
  - 跨形态变异: 形态无关特征应在"该有信号的形态"上单调可解释。

用法
----
    python scripts/verify_morphology_dependence.py
    python scripts/verify_morphology_dependence.py --n 100 --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = r'<REPO>'
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from multisync.feature_definitions import (
    compute_onset_latency, compute_rise_time, compute_recovery_time,
    compute_peak_amplitude, compute_dwell_time, compute_switching_rate,
    compute_mean_synchrony, compute_synchrony_entropy, find_dominant_peak,
)

HZ = 1.0
DURATION = 180  # samples (= seconds at 1 Hz)
THRESHOLD = 0.5
SMOOTH_W = 3


def _smooth(x, w=SMOOTH_W):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def make_trace(kind: str, rng: np.random.Generator, noise=0.08) -> np.ndarray:
    """生成一条已知形态的 WCC 轨迹 (取值大致在 [-1, 1])。"""
    n = DURATION
    t = np.arange(n)
    base = 0.15

    if kind == "single_peak":
        center, width, height = 90, 22, 0.7
        x = base + height * np.exp(-0.5 * ((t - center) / width) ** 2)

    elif kind == "oscillatory":
        # 多次进出阈值: 正弦包络 + 随机相位, 约 6-10 个峰
        k = rng.integers(6, 11)
        phase = rng.uniform(0, 2 * np.pi)
        x = 0.45 + 0.45 * np.sin(2 * np.pi * k * t / n + phase)

    elif kind == "sustained":
        x = np.full(n, 0.75) + 0.05 * np.sin(2 * np.pi * 2 * t / n)

    elif kind == "plateau":
        x = base + (0.75 - base) / (1 + np.exp(-(t - 40) / 6.0))  # 升起不回落

    elif kind == "subthreshold":
        x = np.full(n, 0.2) + 0.05 * np.sin(2 * np.pi * 3 * t / n)

    else:
        raise ValueError(kind)

    x = x + rng.normal(0, noise, n)
    return np.clip(x, -1.0, 1.0)


def features_for(wcc: np.ndarray) -> dict:
    sm = _smooth(wcc)
    peak_val, peak_idx = compute_peak_amplitude(sm)
    if peak_idx is None:
        peak_idx = 0
    onset, onset_def = compute_onset_latency(wcc, HZ, DURATION, THRESHOLD)
    rise, rise_def = compute_rise_time(wcc, peak_idx, peak_val, HZ, THRESHOLD)
    recov, recov_def = compute_recovery_time(wcc, peak_idx, peak_val, HZ, THRESHOLD)
    return {
        "onset_latency": onset,
        "rise_time": rise,
        "recovery_time": recov,
        "peak_amplitude": peak_val,
        "dwell_time": compute_dwell_time(wcc, HZ, THRESHOLD),
        "switching_rate": compute_switching_rate(wcc, HZ, THRESHOLD),
        "mean_synchrony": compute_mean_synchrony(wcc),
        "synchrony_entropy": compute_synchrony_entropy(wcc),
    }


FEATURES = ["onset_latency", "rise_time", "recovery_time", "peak_amplitude",
            "dwell_time", "switching_rate", "mean_synchrony",
            "synchrony_entropy"]
KINDS = ["single_peak", "oscillatory", "sustained", "plateau", "subthreshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="每种形态的样本数")
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for kind in KINDS:
        for _ in range(args.n):
            wcc = make_trace(kind, rng, noise=args.noise)
            f = features_for(wcc)
            f["kind"] = kind
            rows.append(f)
    df = pd.DataFrame(rows)

    print(f"n per morphology = {args.n}, noise = {args.noise}\n")

    print("=== defined-rate (非 NaN 比例) by morphology ===")
    print("  形态依赖特征会在非单峰形态上明显偏低\n")
    defrate = df.groupby("kind")[FEATURES].apply(
        lambda g: g.notna().mean())
    print((defrate * 100).round(0).astype(int).to_string())

    print("\n=== median value by morphology (仅非 NaN) ===")
    med = df.groupby("kind")[FEATURES].median()
    print(med.round(2).to_string())

    print("\n=== 解读 ===")
    # 形态无关判据: 在所有 5 种形态上 defined-rate 都接近 100%
    agnostic, dependent = [], []
    for f in FEATURES:
        min_def = (df.groupby("kind")[f].apply(lambda s: s.notna().mean())).min()
        (agnostic if min_def >= 0.95 else dependent).append(f)
    print(f"形态无关 (所有形态 defined>=95%): {agnostic}")
    print(f"形态依赖 (某些形态 defined<95%): {dependent}")

    out = Path(REPO) / "artifacts" / "morphology_dependence_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved per-trace features -> {out}")


if __name__ == "__main__":
    main()

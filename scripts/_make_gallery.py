"""Build a single self-contained HTML gallery that inlines the 4 real-data
figures so they can be opened in the live preview panel and shared as one file.
"""
import pathlib

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "realdata_full" / "figs"
OUT = FIG_DIR.parent / "figs_gallery.html"

CAPTIONS = {
    "fig1_existence.svg": (
        "IAAFT 存在性审计通过率 (L0) — WCC 是否超越 IAAFT 零模型？通过 = 信号真实存在",
        "Lerique 0.618 (42/68 dyads) · Gordon 1.000 · Han 1.000 · Andersen 0.792。Bizzego 为组间设计，不纳入配对 L0。",
    ),
    "fig2_l2_forest.svg": (
        "L2 条件间检验 — −log10(p_fdr)",
        "红虚线 = p=0.05 (FDR 校正阈值)；红点 = 显著。Lerique·pooled 的 peak/dwell 达 1.59 (>0.05 线) 显著；其余多为非显著。",
    ),
    "fig3_dwell_definedness.svg": (
        "dwell_time 定义率 (p_definedness)",
        "低定义率 = 构造性零值（无持续 WCC 超阈），非 bug。RESP 0.22 / ECG 0.52 偏低，EDA/Gordon/Han 为 1.00。",
    ),
    "fig4_morphology.svg": (
        "WCC 形态学：簇稳定性 vs 外部效度",
        "稳定性 = 重采样 ARI（内部）；外部效度 = ARI vs 真实实验条件。Lerique/Andersen 形态稳定 (ARI 0.86–0.92)，且编码了条件 (RESP 0.56, EDA 0.65)。",
    ),
}

parts = ['<html lang="zh"><head><meta charset="utf-8">'
         '<title>SyncPipe 真实数据可视化</title>'
         '<style>body{font-family:Segoe UI,Arial;background:#f4f7fa;color:#1a3a5c;'
         'margin:0;padding:24px;}h1{font-size:20px;}section{background:#fff;border-radius:10px;'
         'padding:16px 20px;margin:18px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);}'
         'h2{font-size:15px;margin:0 0 4px;}p.cap{color:#5a6b7b;font-size:12px;margin:2px 0 12px;}'
         'svg{max-width:100%;height:auto;border:1px solid #eef2f6;border-radius:6px;}</style>'
         '</head><body><h1>SyncPipe · 真实数据集三级证据可视化</h1>']

for fname, (title, cap) in CAPTIONS.items():
    svg = (FIG_DIR / fname).read_text(encoding="utf-8")
    parts.append(f'<section><h2>{title}</h2><p class="cap">{cap}</p>{svg}</section>')

parts.append("</body></html>")
OUT.write_text("\n".join(parts), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size, "bytes")

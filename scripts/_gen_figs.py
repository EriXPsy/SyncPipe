import os
os.makedirs("artifacts/realdata_full/figs", exist_ok=True)

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")

# ---------- FIG 1: Existence pass rate ----------
data = [("Lerique", 0.618, "42/68"), ("Gordon", 1.00, "48/48"),
        ("Han", 1.00, "24/24"), ("Andersen", 0.792, "19/24")]
W, H = 680, 320
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI,Arial">']
svg.append(f'<rect width="{W}" height="{H}" fill="#fbfdff"/>')
svg.append(f'<text x="20" y="26" font-size="15" font-weight="700" fill="#1a3a5c">IAAFT 存在性审计通过率 (L0)</text>')
svg.append(f'<text x="20" y="44" font-size="11" fill="#5a6b7b">WCC 是否超越 IAAFT 零模型？通过=信号真实存在</text>')
svg.append(f'<line x1="70" y1="60" x2="70" y2="270" stroke="#ccc"/>')
svg.append(f'<line x1="70" y1="270" x2="650" y2="270" stroke="#ccc"/>')
for v in [0, 0.25, 0.5, 0.75, 1.0]:
    y = 270 - v * 200
    svg.append(f'<line x1="70" y1="{y}" x2="650" y2="{y}" stroke="#eee"/>')
    svg.append(f'<text x="64" y="{y+4}" font-size="10" fill="#888" text-anchor="end">{v:.2f}</text>')
bw, gap, x0 = 110, 20, 90
for i, (name, val, n) in enumerate(data):
    x = x0 + i * (bw + gap)
    bh = val * 200
    y = 270 - bh
    col = "#2e8b9e" if val >= 0.5 else "#d98a3a"
    svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{col}"/>')
    svg.append(f'<text x="{x+bw/2}" y="{y-6}" font-size="13" font-weight="700" fill="#1a3a5c" text-anchor="middle">{val:.3f}</text>')
    svg.append(f'<text x="{x+bw/2}" y="285" font-size="11" fill="#334" text-anchor="middle">{name}</text>')
    svg.append(f'<text x="{x+bw/2}" y="299" font-size="9" fill="#7a8" text-anchor="middle">n={n}</text>')
svg.append(f'<text x="90" y="318" font-size="9" fill="#99a">Bizzego=组间设计(不适用配对L2)，未纳入</text>')
svg.append('</svg>')
open("artifacts/realdata_full/figs/fig1_existence.svg", "w").write("\n".join(svg))

# ---------- FIG 2: L2 forest -log10(p) ----------
rows = [("Lerique·pooled", 1.59, True), ("Lerique·pooled", 1.59, True),
        ("Lerique·ECG", 0.87, False), ("Lerique·ECG", 0.29, False), ("Lerique·ECG", 0.29, False),
        ("Lerique·EDA", 0.53, False), ("Lerique·EDA", 1.19, False), ("Lerique·EDA", 0.74, False),
        ("Lerique·RESP", 0.42, False), ("Lerique·RESP", 0.34, False), ("Lerique·RESP", 0.00, False),
        ("Han·affect", 0.16, False), ("Han·affect", 0.29, False), ("Han·affect", 0.29, False),
        ("Andersen·hr", 0.93, False), ("Andersen·hr", 0.77, False), ("Andersen·hr", 0.77, False),
        ("Gordon·motion", 0.00, False), ("Gordon·motion", 0.00, False), ("Gordon·motion", 0.00, False)]
feats = ["peak", "dwell", "switch"] * 7
rows2 = [(f"{ds}·{f}", val, sig) for (ds, val, sig), f in zip(rows, feats)]
W, H = 680, 560
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI,Arial">']
svg.append(f'<rect width="{W}" height="{H}" fill="#fbfdff"/>')
svg.append(f'<text x="20" y="24" font-size="15" font-weight="700" fill="#1a3a5c">L2 条件间检验  minus log10(p_fdr)</text>')
svg.append(f'<text x="20" y="42" font-size="11" fill="#5a6b7b">红虚线 = p=0.05 (FDR校正阈值)；红点=显著</text>')
x0, x1, vmax = 120, 650, 1.6
def mx(v):
    return x0 + (v / vmax) * (x1 - x0)
rx = mx(1.30)
svg.append(f'<line x1="{rx}" y1="55" x2="{rx}" y2="{H-20}" stroke="#e0556b" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{rx+4}" y="68" font-size="10" fill="#e0556b">p=0.05</text>')
for v in [0, 0.5, 1.0, 1.5]:
    px = mx(v)
    svg.append(f'<line x1="{px}" y1="55" x2="{px}" y2="300" stroke="#f0f0f0"/>')
    svg.append(f'<text x="{px}" y="315" font-size="10" fill="#888" text-anchor="middle">{v}</text>')
rh, y0 = 22, 68
for i, (lab, val, sig) in enumerate(rows2):
    y = y0 + i * rh
    px = mx(val)
    col = "#e0556b" if sig else "#7a8ea0"
    svg.append(f'<text x="115" y="{y+4}" font-size="10" fill="#334" text-anchor="end">{esc(lab)}</text>')
    svg.append(f'<circle cx="{px}" cy="{y}" r="4.5" fill="{col}"/>')
    svg.append(f'<text x="{px+8}" y="{y+4}" font-size="9" fill="#667">{val:.2f}</text>')
svg.append('</svg>')
open("artifacts/realdata_full/figs/fig2_l2_forest.svg", "w").write("\n".join(svg))

# ---------- FIG 3: dwell definedness ----------
dwell = [("Lerique·EDA", 1.00), ("Lerique·ECG", 0.52), ("Lerique·RESP", 0.22),
         ("Gordon", 1.00), ("Han", 1.00), ("Andersen", 0.50)]
W, H = 680, 320
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI,Arial">']
svg.append(f'<rect width="{W}" height="{H}" fill="#fbfdff"/>')
svg.append(f'<text x="20" y="26" font-size="15" font-weight="700" fill="#1a3a5c">dwell_time 定义率 (p_definedness)</text>')
svg.append(f'<text x="20" y="44" font-size="11" fill="#5a6b7b">低定义率=构造性零值(无持续WCC超阈)，非bug</text>')
svg.append(f'<line x1="70" y1="60" x2="70" y2="265" stroke="#ccc"/>')
svg.append(f'<line x1="70" y1="265" x2="650" y2="265" stroke="#ccc"/>')
for v in [0, 0.25, 0.5, 0.75, 1.0]:
    y = 265 - v * 195
    svg.append(f'<line x1="70" y1="{y}" x2="650" y2="{y}" stroke="#eee"/>')
    svg.append(f'<text x="64" y="{y+4}" font-size="10" fill="#888" text-anchor="end">{v:.2f}</text>')
bw, gap, x0 = 82, 14, 85
for i, (name, val) in enumerate(dwell):
    x = x0 + i * (bw + gap)
    bh = val * 195
    y = 265 - bh
    col = "#2e8b9e" if val >= 0.8 else ("#d98a3a" if val >= 0.4 else "#c0504d")
    svg.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{col}"/>')
    svg.append(f'<text x="{x+bw/2}" y="{y-6}" font-size="13" font-weight="700" fill="#1a3a5c" text-anchor="middle">{val:.2f}</text>')
    svg.append(f'<text x="{x+bw/2}" y="280" font-size="10" fill="#334" text-anchor="middle">{esc(name)}</text>')
svg.append('</svg>')
open("artifacts/realdata_full/figs/fig3_dwell_definedness.svg", "w").write("\n".join(svg))

# ---------- FIG 4: morphology ----------
morph = [("Lerique·RESP", 0.861, 0.565, 4, 0.395), ("Lerique·ECG", 0.905, 0.0, 2, 0.643),
         ("Lerique·EDA", 0.869, 0.653, 2, 0.400), ("Andersen·hr", 0.92, 0.015, 2, 0.744)]
W, H = 680, 360
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI,Arial">']
svg.append(f'<rect width="{W}" height="{H}" fill="#fbfdff"/>')
svg.append(f'<text x="20" y="24" font-size="15" font-weight="700" fill="#1a3a5c">WCC 形态学：簇稳定性 vs 外部效度</text>')
svg.append(f'<text x="20" y="42" font-size="11" fill="#5a6b7b">稳定性=重采样ARI(内部)；外部效度=ARI vs 实验条件</text>')
svg.append(f'<line x1="70" y1="60" x2="70" y2="270" stroke="#ccc"/>')
svg.append(f'<line x1="70" y1="270" x2="650" y2="270" stroke="#ccc"/>')
for v in [0, 0.25, 0.5, 0.75, 1.0]:
    y = 270 - v * 195
    svg.append(f'<line x1="70" y1="{y}" x2="650" y2="{y}" stroke="#eee"/>')
    svg.append(f'<text x="64" y="{y+4}" font-size="10" fill="#888" text-anchor="end">{v:.2f}</text>')
bw, gap, x0 = 70, 18, 95
for i, (name, stab, cond, k, sil) in enumerate(morph):
    x = x0 + i * (2 * bw + gap + 10)
    bh1 = stab * 195
    y1 = 270 - bh1
    svg.append(f'<rect x="{x}" y="{y1}" width="{bw}" height="{bh1}" rx="4" fill="#3a7ca5"/>')
    svg.append(f'<text x="{x+bw/2}" y="{y1-5}" font-size="11" font-weight="700" fill="#1a3a5c" text-anchor="middle">{stab:.2f}</text>')
    svg.append(f'<text x="{x+bw/2}" y="285" font-size="9" fill="#334" text-anchor="middle">稳定</text>')
    x2 = x + bw + 8
    bh2 = cond * 195
    y2 = 270 - bh2
    col = "#2e8b9e" if cond >= 0.3 else "#bfae5a"
    svg.append(f'<rect x="{x2}" y="{y2}" width="{bw}" height="{bh2}" rx="4" fill="{col}"/>')
    svg.append(f'<text x="{x2+bw/2}" y="{y2-5}" font-size="11" font-weight="700" fill="#1a3a5c" text-anchor="middle">{cond:.2f}</text>')
    svg.append(f'<text x="{x2+bw/2}" y="285" font-size="9" fill="#334" text-anchor="middle">条件</text>')
    svg.append(f'<text x="{x+bw+4}" y="300" font-size="10" fill="#334" text-anchor="middle">{esc(name)}</text>')
    svg.append(f'<text x="{x+bw+4}" y="314" font-size="9" fill="#889" text-anchor="middle">k={k}, sil={sil:.2f}</text>')
svg.append(f'<text x="95" y="335" font-size="9" fill="#99a">稳定=重采样ARI(高=簇稳健)；条件=ARI vs 真实实验条件(高=形态编码了条件)</text>')
svg.append(f'<text x="95" y="348" font-size="9" fill="#99a">Gordon(KMeans NaN报错)与Han/Bizzego(每dyad仅2轨迹,k无法定)未绘</text>')
svg.append('</svg>')
open("artifacts/realdata_full/figs/fig4_morphology.svg", "w").write("\n".join(svg))
print("OK wrote 4 SVGs")
for f in ["fig1_existence.svg", "fig2_l2_forest.svg", "fig3_dwell_definedness.svg", "fig4_morphology.svg"]:
    print(f, os.path.getsize("artifacts/realdata_full/figs/" + f), "bytes")

"""Extract embedded base64 PNGs from the two Intro HTML files and compare.

Goal: determine whether the figures embedded in Intro-en.html are byte-identical
to those in Intro.html (i.e. still contain Chinese text), and dump each PNG to
disk so we can inspect / regenerate the ones that carry CJK labels.
"""
import base64
import hashlib
import re
from pathlib import Path

ROOT = Path(r"<REPO_ROOT>")
OUT = ROOT / "multisync-core" / "artifacts" / "extracted_figs"
OUT.mkdir(parents=True, exist_ok=True)

PAT = re.compile(r'data:image/png;base64,([A-Za-z0-9+/=]+)')


def extract(html_path):
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    return PAT.findall(text)


def main():
    en = extract(ROOT / "Intro-en.html")
    zh = extract(ROOT / "Intro.html")
    print(f"Intro-en.html : {len(en)} embedded PNGs")
    print(f"Intro.html    : {len(zh)} embedded PNGs")
    print("-" * 60)

    zh_hashes = {hashlib.md5(b.encode()).hexdigest(): i for i, b in enumerate(zh)}

    for i, b64 in enumerate(en):
        raw = base64.b64decode(b64)
        h = hashlib.md5(b64.encode()).hexdigest()
        same = zh_hashes.get(h)
        # crude PNG dimension read (IHDR at offset 16: width,height big-endian)
        w = int.from_bytes(raw[16:20], "big")
        ht = int.from_bytes(raw[20:24], "big")
        p = OUT / f"en_fig{i+1}_{w}x{ht}.png"
        p.write_bytes(raw)
        tag = f"IDENTICAL to zh fig#{same+1}" if same is not None else "unique to en"
        print(f"en fig#{i+1}: {w}x{ht}px  {len(raw)//1024}KB  [{tag}]  -> {p.name}")

    print("\nWrote decoded PNGs to:", OUT)


if __name__ == "__main__":
    main()

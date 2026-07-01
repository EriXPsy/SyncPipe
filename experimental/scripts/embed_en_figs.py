"""Replace the 6 Chinese embedded PNGs in Intro-en.html with the English
regenerations, in document order.

Order in Intro-en.html:
  1 GT three-axis architecture  -> en_fig1_gt_axes.png
  2 Feature validity (Lerique)  -> en_fig2_validity.png
  3 False-positive (Han)        -> en_fig3_false_positive.png
  4 Surrogate hourglass         -> en_fig4_hourglass.png
  5 Sensitivity envelopes       -> en_fig5_sensitivity.png
  6 Timing recovery (GT-3b)     -> en_fig6_timing_recovery.png
"""
import base64
import re
from pathlib import Path

HTML = Path(r"<REPO_ROOT>\Intro-en.html")
FIGS = Path(__file__).resolve().parents[1] / "artifacts" / "en_figs"

ORDER = [
    "en_fig1_gt_axes.png",
    "en_fig2_validity.png",
    "en_fig3_false_positive.png",
    "en_fig4_hourglass.png",
    "en_fig5_sensitivity.png",
    "en_fig6_timing_recovery.png",
]

PAT = re.compile(r'(data:image/png;base64,)([A-Za-z0-9+/=]+)')


def main():
    html = HTML.read_text(encoding="utf-8")
    matches = list(PAT.finditer(html))
    print(f"found {len(matches)} embedded PNGs in HTML")
    assert len(matches) == len(ORDER), \
        f"expected {len(ORDER)} images, found {len(matches)}"

    new_b64 = []
    for name in ORDER:
        raw = (FIGS / name).read_bytes()
        new_b64.append(base64.b64encode(raw).decode("ascii"))

    # Replace from the end so earlier offsets stay valid.
    out = html
    for m, b64 in zip(reversed(matches), reversed(new_b64)):
        out = out[:m.start()] + "data:image/png;base64," + b64 + out[m.end():]

    backup = HTML.with_suffix(".html.bak_zhfigs")
    if not backup.exists():
        backup.write_text(html, encoding="utf-8")
        print("backup written:", backup.name)
    HTML.write_text(out, encoding="utf-8")
    print("Intro-en.html updated with 6 English figures.")


if __name__ == "__main__":
    main()

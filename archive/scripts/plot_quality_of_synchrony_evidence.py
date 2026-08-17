"""Generate the "quality vs quantity" evidence figures (Blind Spot A).

Promotes L3_Temporal to a clean, falsifiable core result and visually
exposes why L2_Structure is *not* comparable evidence (its mean_sync
matching failed to suppress the level signal).

Outputs
-------
- artifacts/figures/l3_temporal_core.png   (flagship evidence)
- artifacts/figures/l2_vs_l3_auc.png       (contrast: clean vs contaminated)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUC = REPO / "artifacts" / "incremental_value" / "kuramoto_l23_v3_auc.json"
OUT = REPO / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

d = json.loads(AUC.read_text())


def _rows(name):
    return d["results"][name]


def _short(step: str) -> str:
    return step.split(":", 1)[1].strip().replace("+", "+\n")


def _aucs(name):
    rows = _rows(name)
    steps = [_short(r["step"]) for r in rows]
    aucs = [r["auc"] for r in rows]
    return steps, aucs


# ---------------------------------------------------------------------------
# Fig 1 — L3_Temporal: the clean "structure independent of quantity" result
# ---------------------------------------------------------------------------
steps, aucs = _aucs("L3_Temporal")
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(steps))
ax.plot(x, aucs, "-o", color="#1f77b4", lw=2)
ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(steps, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("Cross-validated AUC")
ax.set_title("L3_Temporal (early vs delayed peak): switching_rate isolates\n"
             "temporal structure after mean_synchrony is matched to chance")
# annotate the switching_rate jump
for i, s in enumerate(steps):
    if "switch" in s:
        ax.annotate(f"ΔAUC = +{aucs[i]-aucs[0]:.2f}",
                    xy=(i, aucs[i]), xytext=(i - 1.5, aucs[i] + 0.08),
                    fontsize=9, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "l3_temporal_core.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 — L2 vs L3 contrast: contaminated vs clean
# ---------------------------------------------------------------------------
s2, a2 = _aucs("L2_Structure")
s3, a3 = _aucs("L3_Temporal")
fig, ax = plt.subplots(figsize=(8, 4.2))
xx = np.arange(len(s3))
ax.plot(xx, a3, "-o", color="#1f77b4", lw=2, label="L3_Temporal (clean)")
ax.plot(xx, a2[:len(a3)], "-s", color="#d62728", lw=2, label="L2_Structure (contaminated)")
ax.axhline(0.5, ls="--", color="gray", lw=1)
ax.set_xticks(xx)
ax.set_xticklabels(s3, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("Cross-validated AUC")
ax.set_title("L2 matching failed to suppress mean_synchrony (0.91 at step 1)\n"
             "→ L2 structure claims are confounded; L3 is the valid core evidence")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "l2_vs_l3_auc.png", dpi=150)
plt.close(fig)

print("Wrote:", OUT / "l3_temporal_core.png")
print("Wrote:", OUT / "l2_vs_l3_auc.png")

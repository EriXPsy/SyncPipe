#!/usr/bin/env python3
"""Check which Rest segments have n_samples != 180000."""
from pathlib import Path
from scipy.io import loadmat
import re

DATA_ROOT = Path("<OSF_ROOT>/Lerique-47n3p")

results = []  # (dyad, modality, seg, n_samples)

for mod in ["ECG", "EDA", "RESP"]:
    mod_dir = DATA_ROOT / mod
    if not mod_dir.exists():
        continue
    for dyad_dir in sorted(mod_dir.iterdir()):
        if not dyad_dir.is_dir():
            continue
        dyad = dyad_dir.name[:5]
        for f in sorted(dyad_dir.iterdir()):
            if not f.name.endswith(".mat"):
                continue
            name = f.stem
            if "_Rest" not in name:
                continue
            m = re.search(r"_Rest(\d+)", name)
            if not m:
                continue
            seg = int(m.group(1))
            try:
                mat = loadmat(str(f))
                pk = [k for k in mat if not k.startswith("__")]
                if len(pk) != 1:
                    continue
                arr = mat[pk[0]]
                if arr.ndim == 2:
                    n = arr.shape[1] if arr.shape[0] == 1 else arr.shape[0]
                else:
                    continue
                if n != 180000:
                    results.append((dyad, mod, seg, n))
            except Exception:
                continue

print("Rest segments with n_samples != 180000:\n")
for (dyad, mod, seg, n) in sorted(results):
    fs = n / 180.0
    dev = n - 180000
    print(f"  {dyad}  {mod:4s}  Rest{seg}  n={n:>7d}  Fs={fs:.1f}  (dev={dev:+d})")

print("\n---")
print("Unique non-180000 values:")
unique_vals = sorted(set(n for _, _, _, n in results))
for v in unique_vals:
    fs = v / 180.0
    print(f"  n={v}  Fs={fs:.1f}  (dev from 180000: {v-180000:+d})")

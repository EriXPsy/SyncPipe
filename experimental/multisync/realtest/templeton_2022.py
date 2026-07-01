"""Templeton et al. (2022) — Subjective Connection Synchrony converter.

Templeton, E., Chang, L., Reynolds, E., Cone LeBeaumont, M., & Wheatley, T.
(2022). Fast response times signal social connection in conversation.
*Proceedings of the National Academy of Sciences*.

Paradigm
--------
- 10-minute unstructured free conversation between dyads
- Continuous dial rating (0–100) of "how connected do I feel right now?"
  from BOTH subjects simultaneously, sampled at 10 Hz
- Study 1: 66 strangers (3 convos each → ~198 recordings)
- Study 2: 65 friends (1 convo each → 65 recordings)

What this measures
------------------
WCC between two subjects' continuous connection ratings → **subjective
connection synchrony**: do dyads' feelings of connection rise and fall
together during conversation?

This is a novel synchrony dimension — all other SyncPipe datasets
measure physiological/behavioral synchrony.  This one measures
EXPERIENTIAL synchrony.

Usage
-----
    python realtest/templeton_2022.py --data-root "<OSF_ROOT>/Templeton-Twz3s"
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from multisync.dynamic_features import sliding_window_wcc, extract_dynamic_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("templeton")


# ── constants ────────────────────────────────────────────────────────
HZ = 10.0            # sampling rate of continuous ratings
WINDOW_SEC = 30.0    # WCC window in seconds
WINDOW = int(HZ * WINDOW_SEC)  # 300 samples
STEP = int(HZ * 10)           # 10s step = 100 samples


def _find_dyad_pairs(data_dir: Path) -> list[tuple[str, str, str, str]]:
    """Return list of (group, id_a, id_b, file_a, file_b) for all dyads."""
    pairs = []
    for group in ["friends", "strangers"]:
        gdir = data_dir / "Data" / "continuous_connection_ratings" / group
        if not gdir.exists():
            logger.warning(f"Directory not found: {gdir}")
            continue
        files = sorted(gdir.glob("*.csv"))
        # Group by dyad: files named "<id_a>_<id_b>.csv"
        seen = set()
        for f in files:
            stem = f.stem
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            id_a, id_b = parts
            dyad_key = tuple(sorted([id_a, id_b]))
            if dyad_key in seen:
                continue
            seen.add(dyad_key)
            # Look for reverse file
            f_rev = gdir / f"{id_b}_{id_a}.csv"
            if f_rev.exists():
                pairs.append((group, id_a, id_b, str(f), str(f_rev)))
    return pairs


def _load_dyad(file_a: str, file_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and align two subjects' continuous ratings."""
    s1 = pd.read_csv(file_a)
    s2 = pd.read_csv(file_b)
    # Trim to common length
    n = min(len(s1), len(s2))
    a = s1["Rating"].values[:n].astype(float)
    b = s2["Rating"].values[:n].astype(float)
    return a, b


def process_templeton(data_root: str, output_dir: str | None = None):
    """Full pipeline: load, compute WCC, extract features, save."""
    data_dir = Path(data_root)
    pairs = _find_dyad_pairs(data_dir)
    logger.info(f"Found {len(pairs)} dyad pairs")

    if output_dir is None:
        output_dir = data_dir / "multisync_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    rows_summary = []

    for group, id_a, id_b, fa, fb in pairs:
        try:
            a, b = _load_dyad(fa, fb)
            # Z-score per subject
            a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-10)
            b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-10)

            # WCC
            wcc = sliding_window_wcc(a, b, window_size=WINDOW, hz=HZ, step_samples=STEP)
            if len(wcc) < 3:
                continue

            # Extract features — WCC trace hz = HZ / STEP
            trace_hz = HZ / STEP
            feats = extract_dynamic_features(
                wcc, hz=trace_hz, wcc_window_sec=WINDOW_SEC,
            )

            d = feats.to_dict()
            d.update({
                "dyad_id": f"{id_a}_{id_b}",
                "group": group,
                "n_samples": len(a),
                "duration_sec": len(a) / HZ,
                "wcc_mean_raw": float(np.nanmean(wcc)),
                "wcc_std_raw": float(np.nanstd(wcc)),
            })
            rows.append(d)

            # Summary row
            rows_summary.append({
                "dyad_id": f"{id_a}_{id_b}",
                "group": group,
                "n_samples": len(a),
                "wcc_mean": d["wcc_mean_raw"],
                "peak_amplitude": d["peak_amplitude"],
                "dwell_time": d["dwell_time"],
                "switching_rate": d["switching_rate"],
                "onset_latency": d["onset_latency"],
                "recovery_time": d["recovery_time"],
                "rise_time": d["rise_time"],
                "mean_synchrony": d["mean_synchrony"],
                "synchrony_entropy": d["synchrony_entropy"],
                "onset_defined": d["onset_defined"],
                "recovery_defined": d["recovery_defined"],
            })

        except Exception as e:
            logger.warning(f"Skipping {id_a}_{id_b}: {e}")

    # Save
    df = pd.DataFrame(rows)
    df_sum = pd.DataFrame(rows_summary)

    fp_full = Path(output_dir) / "multisync_templeton_full.csv"
    fp_sum = Path(output_dir) / "multisync_templeton_summary.csv"
    df.to_csv(fp_full, index=False)
    df_sum.to_csv(fp_sum, index=False)

    logger.info(f"Saved {len(df)} dyads to {fp_full}")

    # ── Quick group comparison ──
    from scipy.stats import mannwhitneyu

    print("\n=== Friends vs Strangers (Wilcoxon) ===")
    for feat in ["peak_amplitude", "dwell_time", "switching_rate",
                 "mean_synchrony", "synchrony_entropy",
                 "onset_latency", "recovery_time", "rise_time"]:
        f_vals = df_sum[df_sum["group"] == "friends"][feat].dropna()
        s_vals = df_sum[df_sum["group"] == "strangers"][feat].dropna()
        if len(f_vals) < 3 or len(s_vals) < 3:
            print(f"  {feat:<22s} skip (n_f={len(f_vals)}, n_s={len(s_vals)})")
            continue
        try:
            stat, p = mannwhitneyu(f_vals, s_vals, alternative="two-sided")
            print(f"  {feat:<22s} friends={f_vals.median():.3f} strangers={s_vals.median():.3f} "
                  f"U={stat} p={p:.4f}")
        except Exception:
            print(f"  {feat:<22s} error")

    return df, df_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="<OSF_ROOT>/Templeton-Twz3s")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    process_templeton(args.data_root, args.output_dir)

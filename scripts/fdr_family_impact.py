"""Impact analysis: how does the primary-FDR family definition change which
features are 'significant' on the existing real-data (Lerique 2024) MAIN
contrasts?

This does NOT modify any code. It re-applies Benjamini-Hochberg FDR to the
existing per-feature p_raw values (artifacts/realtest/lerique_2024/
group_contrasts_paired.csv, 'main' contrast rows only) under two candidate
family definitions, and reports significance flips.

  Option A (external_wins): primary FDR family = {peak_amplitude}
  Option B (code_minus_bc): primary FDR family =
                            {peak_amplitude, dwell_time, switching_rate}
                            (mean_synchrony reported as reference, NOT corrected)

For reference we also show the status quo recorded in the artifact, whose
FDR family was {mean_synchrony, peak_amplitude, dwell_time, switching_rate}.

Run from multisync-core/:
    python scripts/fdr_family_impact.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "realtest" / "lerique_2024" / "group_contrasts_paired.csv"
)

FAMILY_A = {"peak_amplitude"}
FAMILY_B = {"peak_amplitude", "dwell_time", "switching_rate"}
FAMILY_STATUSQUO = {"mean_synchrony", "peak_amplitude", "dwell_time", "switching_rate"}

ALPHA = 0.05


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same convention as statsmodels)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity (step-up)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def correct_family(sub: pd.DataFrame, family: set) -> pd.DataFrame:
    """Within one modality's main contrast, BH-correct only the family
    members; non-members get NaN p_fdr (reported, not corrected)."""
    sub = sub.copy()
    in_fam = sub["feature"].isin(family) & sub["p_raw"].notna()
    sub["in_family"] = in_fam
    sub["p_fdr_new"] = np.nan
    if in_fam.any():
        sub.loc[in_fam, "p_fdr_new"] = bh_fdr(sub.loc[in_fam, "p_raw"].to_numpy())
    sub["sig_new"] = sub["p_fdr_new"] < ALPHA
    return sub


def main() -> None:
    df = pd.read_csv(CSV)
    main = df[df["contrast_role"] == "main"].copy()
    # drop the duplicated mean_synchrony row (it appears twice per modality)
    main = main.drop_duplicates(subset=["modality", "feature"])

    print("Lerique 2024 — MAIN contrast (rest1 vs trials_concat)")
    print("=" * 78)
    for label, fam in (
        ("STATUS QUO (artifact: mean+peak+dwell+switching)", FAMILY_STATUSQUO),
        ("OPTION A  (external_wins: peak only)", FAMILY_A),
        ("OPTION B  (code_minus_bc: peak+dwell+switching)", FAMILY_B),
    ):
        print(f"\n### {label}   [family size m varies per modality]")
        rows = []
        for modality, sub in main.groupby("modality"):
            res = correct_family(sub, fam)
            m = int(res["in_family"].sum())
            for _, r in res.iterrows():
                rows.append(
                    dict(
                        modality=modality,
                        feature=r["feature"],
                        in_family=r["in_family"],
                        m=m,
                        p_raw=r["p_raw"],
                        p_fdr_new=r["p_fdr_new"],
                        sig=bool(r["sig_new"]) if r["in_family"] else None,
                    )
                )
        out = pd.DataFrame(rows)
        with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
            print(
                out[["modality", "feature", "in_family", "m", "p_raw", "p_fdr_new", "sig"]]
                .to_string(index=False)
            )

    # Flip summary: compare A vs B significance for family-B members
    print("\n" + "=" * 78)
    print("SIGNIFICANCE FLIPS (Option A vs Option B), per modality x feature")
    print("=" * 78)
    flips = []
    for modality, sub in main.groupby("modality"):
        ra = correct_family(sub, FAMILY_A).set_index("feature")
        rb = correct_family(sub, FAMILY_B).set_index("feature")
        for feat in sorted(FAMILY_B):
            if feat not in rb.index:
                continue
            sig_a = bool(ra.loc[feat, "sig_new"]) if feat in FAMILY_A else None
            sig_b = bool(rb.loc[feat, "sig_new"])
            flips.append(
                dict(modality=modality, feature=feat,
                     sig_optionA=sig_a, sig_optionB=sig_b)
            )
    fdf = pd.DataFrame(flips)
    print(fdf.to_string(index=False))


if __name__ == "__main__":
    main()

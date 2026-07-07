"""
Real-data L2 audit — drive the *new* SyncPipe inference layer on the five
OSF datasets' existing feature CSVs.

Goal (per user request 2026-07-07):
  * Demonstrate the v1 SOP's L2 step (dyad-paired permutation + BH-FDR) on
    REAL data, not just the synthetic proxy.
  * Show per-modality L2 for multimodal datasets (mandatory reporting rule).
  * Surface feature-definedness rates so the dwell_time NaN discussion is
    grounded in real numbers.

Design notes:
  * Lerique & Gordon are PAIRED designs (same dyad in both conditions) ->
    `between_condition_fdr` applies directly.
  * Andersen / Han / Bizzego are between-dyad group or cross-pair designs ->
    the paired test is inappropriate; their L2 lives in the prior
    `diagnosis_bhfdr` CSVs, which this script *reads* but does not recompute.

Outputs: artifacts/realdata_audit/realdata_l2.json + .md
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from multisync.validation.l2_between_condition import between_condition_fdr
from multisync.feature_definitions import FDR_FEATURES

OSF = "E:/OSF"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "realdata_audit")
os.makedirs(OUT, exist_ok=True)

N_PERM = 10000
SEED = 42
ALPHA = 0.05

# Paired-design datasets the new L2 layer can consume directly.
PAIRED = {
    "Lerique": dict(
        path=f"{OSF}/Lerique-47n3p/multisync_results/lerique_dyads.csv",
        dyad="dyad_label", modality="modality", condition="condition",
        contrast=("rest1", "trials_concat"),
        fdr=["peak_amplitude", "dwell_time", "switching_rate"],
        extra=["mean_synchrony"],
        pooled=False,   # dyad_id encodes (dyad+modality+condition); real dyad key = dyad_label
        note="paired key = dyad_label (31 dyads); per-modality L2 is the mandated report",
    ),
    "Gordon": dict(
        path=f"{OSF}/Gordon-349su/multisync_results/multisync_gordon_46dyads.csv",
        dyad="dyad_id", modality="modality", condition="condition",
        contrast=(1, 4),   # bookend blocks (int); exploratory (paper uses pull_sync vs pull_seg)
        fdr=["peak_amplitude", "dwell_time", "switching_rate"],
        extra=["mean_synchrony"],
        pooled=False,
        note="condition bookend 1 vs 4 (exploratory); paper's primary contrast is pull_sync vs pull_seg (see gordon_diagnosis_bhfdr.csv)",
    ),
}

# Between-dyad / cross-pair datasets: read prior diagnosis outputs only.
PRIOR_DIAGNOSIS = {
    "Andersen": f"{OSF}/Andersen-hj4k6/multisync_results/andersen_diagnosis_bhfdr.csv",
    "Han":      f"{OSF}/Han-bzkdy/multisync_results/han_diagnosis_bhfdr.csv",
    "Gordon":   f"{OSF}/Gordon-349su/multisync_results/gordon_diagnosis_bhfdr.csv",
}


def _result_row(r) -> dict:
    return dict(
        feature=r.feature,
        observed_diff=round(float(r.observed_diff), 4) if np.isfinite(r.observed_diff) else None,
        p_raw=round(float(r.p_raw), 4) if np.isfinite(r.p_raw) else None,
        p_fdr=round(float(r.p_fdr), 4) if np.isfinite(r.p_fdr) else None,
        significant_05=bool(r.significant_05),
        perm_effect_size=round(float(r.perm_effect_size), 3) if np.isfinite(r.perm_effect_size) else None,
        defined_a=int(r.defined_a), defined_b=int(r.defined_b),
        p_definedness=round(float(r.p_definedness), 4) if np.isfinite(r.p_definedness) else None,
    )


def run_paired(name: str, cfg: dict):
    df = pd.read_csv(cfg["path"])
    present_fdr = [c for c in cfg["fdr"] if c in df.columns]
    present_extra = [c for c in cfg["extra"] if c in df.columns]
    feat_cols = present_fdr + present_extra

    out = {"dataset": name, "design": "paired", "contrast": list(cfg["contrast"]),
           "note": cfg.get("note"), "pooled": None, "per_modality": {}}

    # Pooled (all modalities mixed) — demonstrates the dilution effect.
    if cfg.get("pooled", True):
        try:
            res = between_condition_fdr(
                df, condition_col=cfg["condition"], dyad_col=cfg["dyad"],
                feature_cols=feat_cols, n_permutations=N_PERM, seed=SEED, alpha=ALPHA,
                condition_values=cfg["contrast"],
            )
            out["pooled"] = {
                "n_dyads": int(res["n_dyads"]),
                "n_significant": int(res["n_significant"]),
                "per_feature": [_result_row(r) for r in res["per_feature"]],
            }
        except Exception as e:
            out["pooled"] = {"error": str(e)}

    # Per-modality L2 (mandatory for multimodal data).
    if cfg.get("modality") and cfg["modality"] in df.columns:
        for mod in sorted(df[cfg["modality"]].dropna().unique()):
            sub = df[df[cfg["modality"]] == mod]
            try:
                res = between_condition_fdr(
                    sub, condition_col=cfg["condition"], dyad_col=cfg["dyad"],
                    feature_cols=feat_cols, n_permutations=N_PERM, seed=SEED, alpha=ALPHA,
                    condition_values=cfg["contrast"],
                )
                out["per_modality"][str(mod)] = {
                    "n_dyads": int(res["n_dyads"]),
                    "n_significant": int(res["n_significant"]),
                    "per_feature": [_result_row(r) for r in res["per_feature"]],
                }
            except Exception as e:
                out["per_modality"][str(mod)] = {"error": str(e)}
    return out


def read_prior_diagnosis(path: str):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({k: (None if (isinstance(row[k], float) and np.isnan(row[k])) else row[k])
                     for k in df.columns})
    return rows


def main():
    report = {"n_permutations": N_PERM, "alpha": ALPHA, "paired": {}, "prior_diagnosis": {}}
    for name, cfg in PAIRED.items():
        print(f"[run] {name} (paired L2) ...")
        report["paired"][name] = run_paired(name, cfg)
    for name, path in PRIOR_DIAGNOSIS.items():
        print(f"[read] {name} prior diagnosis_bhfdr")
        try:
            report["prior_diagnosis"][name] = read_prior_diagnosis(path)
        except Exception as e:
            report["prior_diagnosis"][name] = {"error": str(e)}

    with open(os.path.join(OUT, "realdata_l2.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Markdown ──
    md = ["# Real-data L2 audit (new SyncPipe inference layer)", "",
          f"n_permutations={N_PERM}, alpha={ALPHA}, seed={SEED}", ""]
    for name, r in report["paired"].items():
        md += [f"## {name} (paired design)", f"- contrast: {r['contrast']}", ""]
        if r.get("note"):
            md += [f"- note: {r['note']}", ""]
        p = r["pooled"]
        if p and "per_feature" in p:
            md += [f"### Pooled L2 (all modalities mixed) — n_dyads={p['n_dyads']}, "
                   f"significant={p['n_significant']}", "",
                   "| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |",
                   "|---|---|---|---|---|---|---|---|"]
            for fr in p["per_feature"]:
                md.append(f"| {fr['feature']} | {fr['observed_diff']} | {fr['p_raw']} | "
                          f"{fr['p_fdr']} | {fr['significant_05']} | {fr['perm_effect_size']} | "
                          f"{fr['defined_a']}/{fr['defined_b']} | {fr['p_definedness']} |")
            md.append("")
        for mod, mr in r["per_modality"].items():
            if "per_feature" not in mr:
                md += [f"### {mod}: {mr}", ""]; continue
            md += [f"### Per-modality L2 — {mod} (n_dyads={mr['n_dyads']}, "
                   f"significant={mr['n_significant']})", "",
                   "| feature | Δ(med) | p_raw | p_fdr | sig | d | def_a/def_b | p_def |",
                   "|---|---|---|---|---|---|---|---|"]
            for fr in mr["per_feature"]:
                md.append(f"| {fr['feature']} | {fr['observed_diff']} | {fr['p_raw']} | "
                          f"{fr['p_fdr']} | {fr['significant_05']} | {fr['perm_effect_size']} | "
                          f"{fr['defined_a']}/{fr['defined_b']} | {fr['p_definedness']} |")
            md.append("")

    md += ["## Prior diagnosis_bhfdr (between-dyad / cross-pair designs)", ""]
    for name, rows in report["prior_diagnosis"].items():
        md += [f"### {name}", ""]
        if isinstance(rows, dict) and "error" in rows:
            md += [f"  error: {rows['error']}", ""]; continue
        cols = [c for c in ("contrast", "effect", "feature", "n_hi", "n_lo",
                            "median_hi", "median_lo", "delta_median", "p_raw", "p_fdr",
                            "replication_status") if rows and c in rows[0]]
        if cols:
            md += ["| " + " | ".join(cols) + " |",
                   "|" + "|".join(["---"] * len(cols)) + "|"]
            for row in rows:
                md.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        md.append("")

    with open(os.path.join(OUT, "realdata_l2.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[done] wrote {OUT}/realdata_l2.json and realdata_l2.md")


if __name__ == "__main__":
    main()

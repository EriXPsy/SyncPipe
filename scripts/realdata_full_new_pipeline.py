"""
realdata_full_new_pipeline.py
================================

Re-run EVERY real dataset end-to-end through the NEW three-pipeline stack:

    records  ->  pipeline_bridge.records_to_inference_inputs
             ->  ComputationPipeline  (load -> WCC -> features)   [Pipeline 2]
             ->  InferencePipeline.run_audited_evidence_chain     [Pipeline 3]
             ->  test_l2_by_modality  (mandatory per-modality L2)
             ->  morphology.MorphologyAnalyzer  (WCC shape audit)

CRITICAL: features and inference are produced by the CURRENT code path, NOT by
any stale pre-computed CSV left over from the old DynamicAnalyzer path. Raw
signals are loaded directly from E:/OSF for every dataset.

NOTE on parameters: this is a FAST CONFIRMATION run (surrogate_n=30,
n_perm=2000) so the whole new pipeline can be demonstrated from RAW signals
for all five datasets within a session.  PUBLICATION-GRADE defaults are LOCKED
in the package itself:
    design_controls.synchrony_existence_audit  -> surrogate_n=100
    design_controls.design_control_audit       -> n_pseudo_per_dyad=10
    inference_pipeline.InferencePipeline        -> n_permutations=10000

Outputs (artifacts/realdata_full/):
    realdata_full_<dataset>.json   per-dataset full results
    realdata_full_summary.json     combined, visualization-ready
    STATUS_manifest.json           which datasets were raw-rerun + how
"""
from __future__ import annotations

import sys, os, json, traceback, warnings, glob
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ss
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multisync.pipeline_bridge import records_to_inference_inputs
from multisync.inference_pipeline import InferencePipeline
from multisync.morphology import MorphologyAnalyzer
from multisync.dynamic_features import _sliding_window_wcc_cumsum
from multisync.feature_definitions import FDR_FEATURES, REFERENCE_FEATURE
from multisync.realtest.lerique_2024 import load_lerique_dataset
from multisync.realtest.gordon_2025 import load_gordon_dataset

OSF = Path("E:/OSF")
OUT = ROOT / "artifacts" / "realdata_full"
OUT.mkdir(parents=True, exist_ok=True)

# FAST CONFIRMATION-run params (publication defaults are locked in the package).
SURROGATE_N = 30
N_PERM = 2000
N_PSEUDO = 8
SEED = 42

# Emit a supplementary L2 result that enters ALL 12 implemented features into
# a single BH-FDR step (reviewer-proof against the "cherry-picking 3/12"
# critique). Set False to skip the extra permutation pass.
EMIT_FULL_FAMILY_FDR = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RawRecord:
    """Universal bridge-compatible record produced from raw signals."""
    dyad_label: str
    modality: str
    condition: str
    person_a: np.ndarray
    person_b: np.ndarray
    target_hz: float
    incomplete: bool = False


def _as_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if "value" in x.columns:
            return x["value"].to_numpy(dtype=float)
        num = x.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return None
        return num.iloc[:, 0].to_numpy(dtype=float)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def to_hz(sig: np.ndarray, native_fs: float, target_hz: float) -> np.ndarray:
    """Resample a continuous signal to target_hz (anti-aliased)."""
    sig = np.asarray(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if sig.size < 4 or native_fs == target_hz:
        return sig
    n_target = int(round(len(sig) * target_hz / native_fs))
    if n_target < 4:
        return sig[:0]
    if target_hz < native_fs:
        q = int(round(native_fs / target_hz))
        if q >= 2:
            try:
                return ss.decimate(sig, q, ftype="fir", zero_phase=True)
            except Exception:
                pass
    return ss.resample(sig, n_target)


def _jsonify(o):
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
# Raw-data loaders (one per dataset)
# ---------------------------------------------------------------------------

def load_lerique():
    recs = load_lerique_dataset(
        str(OSF / "Lerique-47n3p"),
        modalities=["ECG", "EDA", "RESP"],
        condition_units=["rest1", "trials_concat"],
        preprocess=False,
    )
    out = []
    for r in recs:
        if getattr(r, "incomplete", False):
            continue
        a = _as_array(r.person_a)
        b = _as_array(r.person_b)
        if a is None or b is None or a.size < 200 or b.size < 200:
            continue
        a1 = to_hz(a, 1000.0, 1.0)
        b1 = to_hz(b, 1000.0, 1.0)
        out.append(RawRecord(dyad_label=r.dyad_label, modality=r.modality,
                             condition=r.condition, person_a=a1, person_b=b1,
                             target_hz=1.0))
    cfg = dict(hz=1.0, window=20, design_condition="trials_concat",
               l2_contrast=("rest1", "trials_concat"),
               status="clean_raw_loader",
               note="Raw .mat (1000 Hz) resampled to 1 Hz continuous; "
                    "paired rest1 vs trials_concat.")
    return out, cfg


def load_gordon():
    recs = load_gordon_dataset(str(OSF / "Gordon-349su"), target_hz=10)
    out = []
    for r in recs:
        a = _as_array(r.person_a)
        b = _as_array(r.person_b)
        if a is None or b is None or a.size < 10 or b.size < 10:
            continue
        out.append(RawRecord(dyad_label=r.pair_label, modality="motion",
                             condition=r.condition, person_a=a, person_b=b,
                             target_hz=10.0))
    cfg = dict(hz=10.0, window=10, design_condition="exp1",
               l2_contrast=("exp1", "exp4"),
               status="clean_raw_loader",
               note="Raw behavioral-data/<pair>/expN.csv (motion, 10 Hz); "
                    "paired exp1 (sync) vs exp4 (seg), exploratory.")
    return out, cfg


def load_bizzego():
    """Bizzego 2015: RawData_1/2/3 = Friends/Strangers/Lovers (between-group).

    Each dyad dir holds {eda,ecg}{F,M}.mat at 2048 Hz.  This is a BETWEEN-GROUP
    design (different dyads per group), so dyad-paired L2 is N/A; we report
    existence + morphology only and set l2_contrast=None.
    """
    out = []
    groups = {"RawData_1": "Friends", "RawData_2": "Strangers", "RawData_3": "Lovers"}
    for folder, grp in groups.items():
        base = OSF / "Bizzego" / folder
        if not base.exists():
            continue
        for dyad_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            dyad = dyad_dir.name
            for m in ("eda", "ecg"):
                fa = dyad_dir / f"{m}F.mat"
                fb = dyad_dir / f"{m}M.mat"
                if not (fa.exists() and fb.exists()):
                    continue
                try:
                    a = sio.loadmat(str(fa))[f"{m}F"].ravel().astype(float)
                    b = sio.loadmat(str(fb))[f"{m}M"].ravel().astype(float)
                except Exception:
                    continue
                a2 = to_hz(a, 2048.0, 2.0)
                b2 = to_hz(b, 2048.0, 2.0)
                if a2.size < 30 or b2.size < 30:
                    continue
                out.append(RawRecord(dyad_label=dyad, modality=m, condition=grp,
                                     person_a=a2, person_b=b2, target_hz=2.0))
    cfg = dict(hz=2.0, window=20, design_condition=None,
               l2_contrast=None,
               status="raw_adapter_heuristic",
               note="Raw .mat (2048 Hz) -> 2 Hz; condition = group "
                    "(Friends/Strangers/Lovers). BETWEEN-GROUP design: "
                    "paired L2 N/A; existence + morphology reported.")
    return out, cfg


def load_han():
    """Han: RawSCLData/<pid>_Stim_{A,B}.xls.

    Each file is (time, 8 affect-rating channels), all numeric.  We form a
    single composite affective-arousal series per person as the row-mean of the
    8 channels.  Dyads are formed by pairing consecutive participant IDs
    (cross-pair heuristic; documented as exploratory).  Stim_A vs Stim_B is the
    within-(cross)pair contrast.
    """
    folder = OSF / "Han-bzkdy" / "Data" / "RawSCLData"
    files = sorted(glob.glob(str(folder / "*.xls")))
    pdata = {}
    for f in files:
        name = Path(f).stem  # FO110_Stim_A
        parts = name.split("_Stim_")
        if len(parts) != 2:
            continue
        pid, cond = parts[0], parts[1]
        d = pd.read_excel(f, header=0)
        num = d.select_dtypes(include=[np.number]).drop(columns=["stim"], errors="ignore")
        if num.shape[1] == 0:
            continue
        arr = num.mean(axis=1).to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size > 30:
            pdata.setdefault(pid, {})[cond] = arr
    pids = sorted(pdata.keys())
    out = []
    for i in range(len(pids) - 1):
        pa, pb = pids[i], pids[i + 1]
        for cond in ("A", "B"):
            sa = pdata[pa].get(cond)
            sb = pdata[pb].get(cond)
            if sa is None or sb is None:
                continue
            out.append(RawRecord(dyad_label=f"{pa}__{pb}", modality="affect",
                                 condition=f"Stim_{cond}", person_a=sa,
                                 person_b=sb, target_hz=1.0))
    cfg = dict(hz=1.0, window=20, design_condition="Stim_A",
               l2_contrast=("Stim_A", "Stim_B"),
               status="raw_adapter_heuristic",
               note="Raw SCL/affect .xls -> composite (mean of 8 channels) at "
                    "1 Hz; cross-pair = consecutive participant IDs "
                    "(documented heuristic); paired Stim_A/B.")
    return out, cfg


def load_andersen():
    """Andersen: HeartRate_data/<ID>.csv (Time, HR @ 1 Hz).

    Dyad = (participant, Close nominee); not_close partner = first other
    participant.  HR series differ in length across partners -> truncate to the
    shorter of the two so WCC is well-defined.
    """
    meta = pd.read_csv(str(OSF / "Andersen-hj4k6" / "all_data.csv"))
    hr_dir = OSF / "Andersen-hj4k6" / "HeartRate_data"
    pdata = {}
    for pid in meta["ID"].dropna().unique():
        fp = hr_dir / f"{pid}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        col = "HR" if "HR" in d.columns else d.columns[1]
        arr = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        arr = arr[~np.isnan(arr)]
        if arr.size > 30:
            pdata[pid] = arr
    close_map = {}
    for _, row in meta.iterrows():
        pid = row.get("ID")
        cval = row.get("Close")
        if pd.isna(cval) or pid not in pdata:
            continue
        nom = [x.strip() for x in str(cval).replace(",", " ").split() if x.strip()]
        nom = [x for x in nom if x in pdata and x != pid]
        if nom:
            close_map[pid] = nom[0]
    ids = sorted(pdata.keys())
    out = []
    for pid, nom in close_map.items():
        partner = next((x for x in ids if x != pid and x != nom), None)
        if partner is None:
            continue
        for cond, (pA, pB) in (("close", (pid, nom)),
                                ("not_close", (pid, partner))):
            a, b = pdata[pA], pdata[pB]
            m = min(len(a), len(b))
            if m < 30:
                continue
            out.append(RawRecord(dyad_label=pid, modality="hr", condition=cond,
                                 person_a=a[:m], person_b=b[:m], target_hz=1.0))
    cfg = dict(hz=1.0, window=20, design_condition="close",
               l2_contrast=("close", "not_close"),
               status="raw_adapter_heuristic",
               note="Raw HR .csv (1 Hz); dyad = (participant, Close nominee); "
                    "not_close partner = first other participant (heuristic); "
                    "paired close vs not_close.")
    return out, cfg


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _run_morphology(raw_signals: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    hz: float, window: int) -> Dict[str, Any]:
    by_mod: Dict[str, List[np.ndarray]] = {}
    by_mod_cond: Dict[str, List[Any]] = {}
    for key, (a, b) in raw_signals.items():
        parts = key.split("__")
        mod = parts[1] if len(parts) >= 3 else parts[-1]
        cond = parts[2] if len(parts) >= 3 else None
        w = _sliding_window_wcc_cumsum(a, b, window)
        w = w[~np.isnan(w)]
        if w.size > window:
            by_mod.setdefault(mod, []).append(w)
            by_mod_cond.setdefault(mod, []).append(cond)
    out = {}
    for mod, traces in by_mod.items():
        try:
            ma = MorphologyAnalyzer(traces, hz=hz)
            r1 = ma.run_method1(max_k=4, seed=SEED)
            ft = ma.feature_table(prominence=0.1)
            ksel = r1.get("k_selection")
            subsample_ari = (float(ksel["subsample_ari"].max())
                             if (ksel is not None and not ksel.empty) else None)
            labels = r1.get("labels")
            ari_cond = None
            conds = by_mod_cond.get(mod)
            if (labels is not None and len(labels) == len(traces)
                    and conds is not None and len(set(conds)) >= 2):
                try:
                    ari_cond = float(adjusted_rand_score(
                        np.asarray(labels), np.asarray(conds)))
                except Exception:
                    ari_cond = None
            out[mod] = {
                "n_traces": len(traces),
                "k_best": r1.get("k_best"),
                "silhouette_best": r1.get("silhouette_best"),
                "subsample_ari": subsample_ari,
                "ari_vs_condition": ari_cond,
                "feature_table": (ft.to_dict(orient="list")
                                  if ft is not None else None),
            }
        except Exception as e:
            out[mod] = {"error": str(e)}
    return out


def _l2row(r) -> Dict[str, Any]:
    if isinstance(r, dict):
        g = r.get
    else:
        g = lambda k, d=None: getattr(r, k, d)
    return {
        "feature": g("feature"),
        "p_raw": (float(g("p_raw")) if g("p_raw") is not None else None),
        "p_fdr": (float(g("p_fdr")) if g("p_fdr") is not None else None),
        "significant_05": bool(g("significant_05")),
        "perm_effect_size": (float(g("perm_effect_size")) if g("perm_effect_size") is not None else None),
        "defined_a": (int(g("defined_a")) if g("defined_a") is not None else None),
        "defined_b": (int(g("defined_b")) if g("defined_b") is not None else None),
        "p_definedness": (float(g("p_definedness"))
                          if g("p_definedness") is not None else None),
    }


def _summarize_l2(group_inf: Dict[str, Any]) -> Dict[str, Any]:
    per = group_inf.get("per_feature", [])
    if not per:
        return {"n_significant": 0, "per_feature": [],
                "condition_a": group_inf.get("condition_a"),
                "condition_b": group_inf.get("condition_b")}
    feats = [_l2row(r) for r in per]
    return {"n_significant": int(group_inf.get("n_significant", 0)),
            "n_dyads": int(group_inf.get("n_dyads", 0)),
            "condition_a": group_inf.get("condition_a"),
            "condition_b": group_inf.get("condition_b"),
            "per_feature": feats}


def _summarize_pm(pm: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for mod, res in pm.items():
        if "per_feature" in res:
            out[mod] = _summarize_l2(res)
        else:
            out[mod] = {"error": str(res)}
    return out


def run_dataset(name: str, records: List[RawRecord], cfg: Dict[str, Any]) -> Dict[str, Any]:
    hz = cfg["hz"]
    window = cfg["window"]
    res = {
        "dataset": name,
        "status": cfg.get("status"),
        "note": cfg.get("note"),
        "hz": hz,
        "window": window,
        "n_records": len(records),
        "l2_contrast": cfg.get("l2_contrast"),
        "error": None,
    }
    try:
        inputs = records_to_inference_inputs(
            records, hz=hz, window_size=window, onset_threshold="session_pooled",
            design_condition=cfg.get("design_condition"),
        )
        res["n_feature_rows"] = int(len(inputs.features_df))
        res["modalities"] = sorted(inputs.features_df["modality"].unique().tolist())
        # Self-evidencing: record the onset-threshold policy actually used and
        # the resolved per-modality thresholds so the run is auditable.
        res["onset_threshold_policy"] = "session_pooled"
        res["onset_thresholds_by_modality"] = (
            inputs.thresholds_by_modality
            if inputs.thresholds_by_modality is not None
            else {}
        )

        pipe = InferencePipeline(inputs.features_df, hz=hz,
                                wcc_window_sec=float(window) / hz,
                                surrogate_n=SURROGATE_N, seed=SEED)

        chain = pipe.run_audited_evidence_chain(
            raw_signals=inputs.raw_signals, wcc_window_size=window,
            design_signal_pairs=(None if cfg.get("design_condition") is None
                                 else inputs.design_pairs),
            condition_col="condition", dyad_col="dyad_id",
            feature_cols=list(FDR_FEATURES), fdr_alpha=0.05,
            n_permutations=N_PERM,
            discontinuity_mask=inputs.discontinuity_mask,
        )

        # --- existence pass rate (correctly extract the per-pair results) ---
        exist_wrap = chain.get("synchrony_existence") or {}
        exist = exist_wrap.get("results", exist_wrap) if isinstance(exist_wrap, dict) else {}
        n_pairs = len(exist)
        n_sig = 0
        for pk, pv in exist.items():
            if isinstance(pv, dict):
                pfs = pv.get("per_feature_significant", {})
                if isinstance(pfs, dict) and any(pfs.values()):
                    n_sig += 1
        res["existence"] = {"n_pairs_audited": n_pairs,
                            "n_pairs_significant": n_sig,
                            "pass_rate": (n_sig / n_pairs) if n_pairs else None}

        res["design_controls"] = _jsonify(chain.get("design_controls", {}))

        # --- L2 (skip for between-group designs where paired L2 is N/A) ---
        if cfg.get("l2_contrast") is None:
            skip = "between-group design; dyad-paired L2 N/A in v1.0"
            res["l2_pooled"] = {"skipped": skip}
            res["l2_per_modality"] = {"skipped": skip}
        else:
            # group_condition_inference is a single pooled result for unimodal
            # data, but a per-modality dict (keyed by modality) when the
            # pipeline routed multimodal data through test_l2_by_modality
            # (P0-2 companion fix). Summarize accordingly.
            group_inf = chain.get("group_condition_inference", {})
            res["l2_pooled"] = (
                _summarize_pm(group_inf)
                if isinstance(group_inf, dict) and "per_feature" not in group_inf
                else _summarize_l2(group_inf)
            )
            try:
                pm = pipe.test_l2_by_modality(
                    modality_col="modality", condition_col="condition",
                    dyad_col="dyad_id",
                    feature_cols=list(FDR_FEATURES) + ["mean_synchrony"],
                    n_permutations=N_PERM)
                res["l2_per_modality"] = _summarize_pm(pm)
            except Exception as e:
                res["l2_per_modality"] = {"error": str(e)}

            # --- Supplementary: reviewer-proof full-family FDR (critique A) ---
            # Re-run the pooled L2 entering ALL 12 implemented features into a
            # single BH-FDR step. Strictly more conservative; if the
            # pre-registered core survives this, the "cherry-picking 3/12"
            # critique is answered. Opt-out via EMIT_FULL_FAMILY_FDR.
            if EMIT_FULL_FAMILY_FDR:
                try:
                    ff = pipe.test_l2_condition(
                        condition_col="condition", dyad_col="dyad_id",
                        feature_cols=None, fdr_alpha=0.05,
                        n_permutations=N_PERM, full_family_fdr=True)
                    res["l2_full_family"] = _summarize_l2(ff)
                except Exception as e:
                    res["l2_full_family"] = {"error": str(e)}

        res["morphology"] = _run_morphology(inputs.raw_signals, hz, window)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return res


def _stratified_cap(recs, cap, seed=42):
    """Cap dyads while keeping every condition represented (between-group safe)."""
    by_cond = defaultdict(list)
    for r in recs:
        by_cond[r.condition].append(r.dyad_label)
    uniq = {c: sorted(set(ds)) for c, ds in by_cond.items()}
    if len(uniq) <= 1:
        return set(sorted({r.dyad_label for r in recs})[:cap])
    per = max(1, cap // len(uniq))
    chosen = set()
    for c, ds in uniq.items():
        chosen.update(ds[:per])
    rest = [d for c, ds in uniq.items() for d in ds if d not in chosen]
    for d in rest:
        if len(chosen) >= cap:
            break
        chosen.add(d)
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # FAST CONFIRMATION MODE: cap dyads per dataset (stratified across
    # conditions) and skip the design-control IAAFT layer (the slowest part).
    # Existence + L2 + morphology below are all produced by the CURRENT
    # ComputationPipeline / InferencePipeline / MorphologyAnalyzer.
    MAX_DYADS = 12

    datasets = {
        "Lerique": load_lerique,
        "Gordon": load_gordon,
        "Bizzego": load_bizzego,
        "Han": load_han,
        "Andersen": load_andersen,
    }
    manifest = []
    combined = {}
    for name, loader in datasets.items():
        print(f"\n===== {name} =====", flush=True)
        try:
            recs, cfg = loader()
        except Exception as e:
            print(f"  LOADER ERROR: {e}", flush=True)
            manifest.append({"dataset": name, "loaded": False, "error": str(e)})
            combined[name] = {"error": f"loader: {e}"}
            continue
        if not recs:
            print(f"  LOADER returned 0 records.", flush=True)
            manifest.append({"dataset": name, "loaded": False,
                             "error": "0 records loaded"})
            combined[name] = {"error": "0 records"}
            continue
        chosen = _stratified_cap(recs, MAX_DYADS, seed=SEED)
        recs = [r for r in recs if r.dyad_label in chosen]
        print(f"  loaded {len(recs)} raw records (status={cfg['status']}) "
              f"from {len(chosen)} dyads", flush=True)
        res = run_dataset(name, recs, cfg)
        combined[name] = _jsonify(res)
        (OUT / f"realdata_full_{name}.json").write_text(
            json.dumps(_jsonify(res), indent=2, ensure_ascii=False))
        manifest.append({
            "dataset": name,
            "loaded": True,
            "n_records": len(recs),
            "status": cfg["status"],
            "hz": cfg["hz"],
            "window": cfg["window"],
            "l2_contrast": cfg["l2_contrast"],
            "error": res["error"],
        })
        print(f"  done. error={res['error']}", flush=True)
    (OUT / "realdata_full_summary.json").write_text(
        json.dumps(_jsonify(combined), indent=2, ensure_ascii=False))
    (OUT / "STATUS_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\nALL DONE. Manifest:", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()

#!/usr/bin/env python
"""
morphology_shape_analysis.py — two complementary, intensity-decoupled
morphology analyses, designed to answer "what does synchrony LOOK like"
(not just "how strong is it"), without presupposing oscillatory/single-peak/
sustained labels.

METHOD 1 — trace-level, SCALE-FREE shape clustering
  Cluster traces using only scale-free shape descriptors (skewness, kurtosis,
  peak density, inter-peak CV, lag-1 autocorr, frac-above-threshold). Removes
  the intensity axis that dominated the earlier k=2 (which was just high-vs-low
  synchrony). Asks: among traces, is there a SHAPE structure beyond intensity?

METHOD 2 — episode-level WAVEFORM archetypes
  Cut each high-synchrony episode out of the trace, normalise it, and cluster
  the actual waveforms. Asks: what does a single synchrony EVENT look like
  (rise/fall symmetry, single vs multi-peak)? Then AGGREGATE each trace's
  episode-archetype mixture back to a trace-level morphology profile — a
  bottom-up, data-driven alternative to preset shape labels.

Both episode definitions (fixed 0.5 threshold; per-trace percentile) and both
episode representations (resampled waveform; shape features) are run, so
sensitivity to those choices is visible.

USAGE
  python morphology_shape_analysis.py --traces lerique_wcc_traces.csv gordon_wcc_traces.csv andersen_wcc_traces.csv
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

RESAMPLE_LEN = 20          # episodes resampled to this many points
PCTL = 75                  # per-trace adaptive threshold percentile
MIN_EP_LEN = 4             # minimum episode length (samples) to keep
SEED = 42


# ───────────────────────── subsample-stability ARI (round-7 method) ────────
def subsample_ari(X, k, algo="kmeans", n_iter=150, frac=0.8, seed=SEED):
    n = X.shape[0]; m = max(k + 1, int(round(frac * n)))
    if n < 2 * k + 2 or m >= n:
        return float("nan"), 0
    rng = np.random.RandomState(seed)
    def cl(Xs):
        return (KMeans(k, random_state=rng.randint(0, 99999), n_init=10).fit_predict(Xs)
                if algo == "kmeans" else AgglomerativeClustering(n_clusters=k).fit_predict(Xs))
    a = []
    for _ in range(n_iter):
        i1 = rng.choice(n, m, replace=False); i2 = rng.choice(n, m, replace=False)
        l1 = dict(zip(i1, cl(X[i1]))); l2 = dict(zip(i2, cl(X[i2])))
        sh = np.intersect1d(i1, i2)
        if sh.size < k + 1:
            continue
        a.append(adjusted_rand_score([l1[i] for i in sh], [l2[i] for i in sh]))
    return (float(np.mean(a)), len(a)) if a else (float("nan"), 0)


def pick_k(X, krange=(2, 3, 4, 5)):
    out = []
    for k in krange:
        lab = KMeans(k, random_state=SEED, n_init=10).fit_predict(X)
        sil = silhouette_score(X, lab) if len(set(lab)) > 1 else float("nan")
        ari, nv = subsample_ari(X, k, "kmeans")
        out.append({"k": k, "silhouette": sil, "subsample_ari": ari})
    return pd.DataFrame(out)


# ───────────────────────── METHOD 1: trace-level scale-free shape ──────────
def scalefree_descriptors(w):
    w = w[np.isfinite(w)]
    if w.size < 10:
        return None
    peaks, _ = find_peaks(w, prominence=0.1, distance=3)
    ipc = float(np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-12)) if len(peaks) > 1 else 0.0
    ac1 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w.size > 2 else 0.0
    return {
        "skewness": float(stats.skew(w)),
        "kurtosis": float(stats.kurtosis(w)),
        "peak_density": len(peaks) / w.size,           # scale-free (per sample)
        "inter_peak_cv": ipc,
        "autocorr_lag1": ac1 if np.isfinite(ac1) else 0.0,
        "frac_above_median": float((w >= np.median(w)).mean()),  # scale-free
    }


def method1_traceshape(traces_by_ds):
    rows = []
    for ds, traces in traces_by_ds.items():
        for tid, w in traces:
            d = scalefree_descriptors(w)
            if d:
                d.update(dataset=ds, trace_id=tid)
                rows.append(d)
    df = pd.DataFrame(rows)
    cols = ["skewness", "kurtosis", "peak_density", "inter_peak_cv", "autocorr_lag1", "frac_above_median"]
    res = []
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        X = StandardScaler().fit_transform(sub[cols].fillna(sub[cols].median()).values)
        kt = pick_k(X); kt["dataset"] = ds
        res.append(kt)
    return df, pd.concat(res, ignore_index=True)


# ───────────────────────── METHOD 2: episode waveform archetypes ───────────
def extract_episodes(w, threshold_mode, min_len=MIN_EP_LEN):
    w = np.asarray(w, float)
    finite = np.isfinite(w)
    if finite.sum() < 10:
        return []
    wf = np.where(finite, w, np.nan)
    thr = 0.5 if threshold_mode == "fixed" else np.nanpercentile(wf, PCTL)
    above = (wf >= thr) & finite
    eps, cur = [], []
    for i, a in enumerate(above):
        if a:
            cur.append(i)
        elif cur:
            if len(cur) >= min_len:
                eps.append(wf[cur[0]:cur[-1] + 1])
            cur = []
    if len(cur) >= min_len:
        eps.append(wf[cur[0]:cur[-1] + 1])
    return [e for e in eps if np.isfinite(e).all()]


def resample_waveform(ep, L=RESAMPLE_LEN):
    x = np.linspace(0, 1, len(ep)); xi = np.linspace(0, 1, L)
    wf = np.interp(xi, x, ep)
    rng = wf.max() - wf.min()
    return (wf - wf.min()) / rng if rng > 1e-9 else wf - wf.min()  # amplitude-normalised


def episode_shape_features(ep):
    L = len(ep); pk = int(np.argmax(ep))
    peaks, _ = find_peaks(ep, prominence=0.05)
    return {
        "rise_frac": pk / max(L - 1, 1),               # 0=peak at start, 1=peak at end
        "fall_frac": (L - 1 - pk) / max(L - 1, 1),
        "asymmetry": (pk) / max(L - 1 - pk, 1),         # rise/fall ratio
        "n_subpeaks": len(peaks),                        # 1=single, >1=oscillatory
        "ep_skew": float(stats.skew(ep)) if L > 3 else 0.0,
        "duration": L,
    }


def method2_episodes(traces_by_ds, threshold_mode):
    wave_rows, feat_rows, ep_index = [], [], []
    for ds, traces in traces_by_ds.items():
        for tid, w in traces:
            for ep in extract_episodes(w, threshold_mode):
                wave_rows.append(resample_waveform(ep))
                feat_rows.append(episode_shape_features(ep))
                ep_index.append((ds, tid))
    if not wave_rows:
        return None
    waves = np.array(wave_rows)
    feats = pd.DataFrame(feat_rows)
    idx = pd.DataFrame(ep_index, columns=["dataset", "trace_id"])
    out = {"n_episodes": len(waves)}

    # (2a) cluster RAW resampled waveforms (purest data-driven)
    Xw = StandardScaler().fit_transform(waves)
    out["waveform_k"] = pick_k(Xw)
    kw = int(out["waveform_k"].sort_values("subsample_ari", ascending=False).iloc[0]["k"])
    out["waveform_labels"] = KMeans(kw, random_state=SEED, n_init=10).fit_predict(Xw)
    out["waveform_archetypes"] = np.vstack([waves[out["waveform_labels"] == c].mean(0)
                                            for c in range(kw)])
    out["waveform_k_chosen"] = kw

    # (2b) cluster episode SHAPE FEATURES (interpretable)
    fcols = ["rise_frac", "fall_frac", "asymmetry", "n_subpeaks", "ep_skew"]
    Xf = StandardScaler().fit_transform(feats[fcols].fillna(feats[fcols].median()).values)
    out["feature_k"] = pick_k(Xf)
    kf = int(out["feature_k"].sort_values("subsample_ari", ascending=False).iloc[0]["k"])
    fl = KMeans(kf, random_state=SEED, n_init=10).fit_predict(Xf)
    out["feature_labels"] = fl
    out["feature_k_chosen"] = kf
    out["feature_profiles"] = feats.assign(cl=fl).groupby("cl")[fcols].mean()

    # cross-check: do waveform clusters and feature clusters agree?
    out["wave_vs_feat_ari"] = float(adjusted_rand_score(out["waveform_labels"], fl))

    # (2c) AGGREGATE episode archetypes back to TRACE level (your idea):
    # each trace -> distribution over waveform archetypes -> trace-level profile
    idx = idx.assign(wlab=out["waveform_labels"])
    trace_profile = (idx.groupby(["dataset", "trace_id", "wlab"]).size()
                     .groupby(level=[0, 1]).apply(lambda s: (s / s.sum()).to_dict()))
    out["trace_archetype_mix"] = trace_profile
    out["idx"] = idx
    out["feats"] = feats.assign(**{"dataset": idx["dataset"].values, "trace_id": idx["trace_id"].values})
    return out


# ───────────────────────── main ────────────────────────────────────────────
def load_traces(csvs):
    by = {}
    for c in csvs:
        d = pd.read_csv(c); name = Path(c).stem.replace("_wcc_traces", "")
        tl = [(r["id"], np.asarray(json.loads(r["wcc_json"]), float)) for _, r in d.iterrows()]
        by[name] = tl
    return by


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--outdir", default="morphology_shape_out")
    args = ap.parse_args(argv)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    by = load_traces(args.traces)

    L = ["# Morphology: Method 1 (trace-level scale-free) + Method 2 (episode waveform)\n"]

    # METHOD 1
    m1_desc, m1_k = method1_traceshape(by)
    m1_desc.to_csv(out / "method1_scalefree_descriptors.csv", index=False)
    m1_k.to_csv(out / "method1_k_selection.csv", index=False)
    L.append("## METHOD 1 — trace-level, intensity-removed shape clustering\n")
    L.append("> k chosen by subsample-stability ARI; scale-free descriptors only.\n")
    L.append(m1_k.round(3).to_markdown(index=False))

    # METHOD 2 (both thresholds)
    for mode in ("fixed", "percentile"):
        res = method2_episodes(by, mode)
        L.append(f"\n## METHOD 2 — episode waveform archetypes (threshold = {mode})\n")
        if res is None:
            L.append("(no episodes extracted)\n"); continue
        L.append(f"- total episodes: **{res['n_episodes']}**")
        L.append(f"- waveform-cluster chosen k = **{res['waveform_k_chosen']}**, "
                 f"feature-cluster chosen k = **{res['feature_k_chosen']}**, "
                 f"agreement (ARI waveform vs feature) = **{res['wave_vs_feat_ari']:.3f}**\n")
        L.append("\n**(2a) waveform k-selection:**\n")
        L.append(res["waveform_k"].round(3).to_markdown(index=False))
        L.append("\n**(2b) episode shape-feature cluster profiles** "
                 "(rise_frac≈0.5 & n_subpeaks≈1 → symmetric single-peak; "
                 "n_subpeaks>1 → oscillatory; asymmetry≠1 → skewed rise/fall):\n")
        L.append(res["feature_profiles"].round(3).to_markdown())
        # save archetype waveforms
        np.savetxt(out / f"method2_{mode}_waveform_archetypes.csv",
                   res["waveform_archetypes"], delimiter=",")
        res["feats"].to_csv(out / f"method2_{mode}_episode_features.csv", index=False)

    (out / "MORPHOLOGY_SHAPE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("Done ->", out / "MORPHOLOGY_SHAPE_REPORT.md")


if __name__ == "__main__":
    sys.exit(main())

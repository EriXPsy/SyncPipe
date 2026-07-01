"""
multisync.morphology — data-driven morphology analysis for WCC traces.

Provides two complementary, intensity-decoupled approaches to characterise
"what synchrony looks like" beyond a single scalar:

1. Trace-level scale-free shape clustering (Method 1)
   - Cluster whole WCC traces using descriptors that do NOT depend on
     absolute amplitude (skewness, kurtosis, peak density, inter-peak CV,
     lag-1 autocorrelation, fraction above median).

2. Episode-level waveform archetypes (Method 2)
   - Extract high-synchrony episodes, normalise their waveforms, and cluster
     them. Aggregate each trace's episode-archetype mixture back to a
     trace-level morphology profile.

Also includes diagnostic helpers: collinearity/VIF, order-unbiased incremental
AUC (Shapley-style), and matched-mean-synchrony contrast.

All morphology analyses are EXPLORATORY in v1.0. They are not part of the
confirmatory FDR family.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .dynamic_features import extract_dynamic_features
from .feature_definitions import ONSET_THRESHOLD


__all__ = [
    "scalefree_descriptors",
    "trace_shape_cluster",
    "extract_episodes",
    "episode_shape_features",
    "resample_waveform",
    "episode_archetype_cluster",
    "morphology_feature_table",
    "collinearity_report",
    "incremental_value",
    "matched_mean_contrast",
    "MorphologyAnalyzer",
]


# ---------------------------------------------------------------------------
# Method 1: trace-level scale-free shape descriptors
# ---------------------------------------------------------------------------

def scalefree_descriptors(wcc: np.ndarray) -> Optional[Dict[str, float]]:
    """Return scale-free shape descriptors of a WCC trace.

    These descriptors intentionally avoid amplitude-dependent summaries so that
    clustering is driven by SHAPE, not by synchrony intensity.

    Parameters
    ----------
    wcc : np.ndarray
        WCC time series.

    Returns
    -------
    dict or None
        Keys: skewness, kurtosis, peak_density, inter_peak_cv, autocorr_lag1,
        frac_above_median. Returns None if trace is too short.
    """
    w = np.asarray(wcc, float)
    w = w[np.isfinite(w)]
    if w.size < 10:
        return None
    peaks, _ = find_peaks(w, prominence=0.1, distance=3)
    ipc = float(np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-12)) if len(peaks) > 1 else 0.0
    ac1 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w.size > 2 else 0.0
    return {
        "skewness": float(stats.skew(w)),
        "kurtosis": float(stats.kurtosis(w)),
        "peak_density": len(peaks) / w.size,
        "inter_peak_cv": ipc,
        "autocorr_lag1": ac1 if np.isfinite(ac1) else 0.0,
        "frac_above_median": float((w >= np.median(w)).mean()),
    }


def trace_shape_cluster(
    wcc_traces: List[np.ndarray],
    max_k: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Cluster WCC traces by scale-free shape descriptors (Method 1).

    Parameters
    ----------
    wcc_traces : list of np.ndarray
        One WCC trace per observation.
    max_k : int
        Maximum number of clusters to try.
    seed : int
        RNG seed for KMeans stability checks.

    Returns
    -------
    dict
        - ``descriptors``: DataFrame of shape descriptors
        - ``k_selection``: DataFrame of k, silhouette, subsample_ari
        - ``labels``: best cluster labels
        - ``k_best``: chosen k
        - ``silhouette_best``: silhouette of chosen k
    """
    rows = []
    for w in wcc_traces:
        d = scalefree_descriptors(w)
        if d:
            rows.append(d)
    if not rows:
        return {"descriptors": pd.DataFrame(), "k_selection": pd.DataFrame(),
                "labels": np.array([]), "k_best": None, "silhouette_best": np.nan}

    desc_df = pd.DataFrame(rows)
    cols = ["skewness", "kurtosis", "peak_density", "inter_peak_cv",
            "autocorr_lag1", "frac_above_median"]
    X = StandardScaler().fit_transform(desc_df[cols].fillna(desc_df[cols].median()).values)

    k_selection = []
    best = {"k": None, "sil": -1, "labels": None}
    rng = np.random.RandomState(seed)
    for k in range(2, min(max_k, X.shape[0] - 1) + 1):
        labels = KMeans(k, random_state=seed, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels)
        ari, n = _subsample_ari(X, k, rng)
        k_selection.append({"k": k, "silhouette": sil, "subsample_ari": ari})
        if sil > best["sil"]:
            best = {"k": k, "sil": sil, "labels": labels}

    return {
        "descriptors": desc_df,
        "k_selection": pd.DataFrame(k_selection),
        "labels": best["labels"] if best["labels"] is not None else np.array([]),
        "k_best": best["k"],
        "silhouette_best": float(best["sil"]) if best["sil"] is not None else np.nan,
    }


# ---------------------------------------------------------------------------
# Method 2: episode-level waveform archetypes
# ---------------------------------------------------------------------------

def extract_episodes(
    wcc: np.ndarray,
    threshold: float = ONSET_THRESHOLD,
    threshold_mode: str = "fixed",
    percentile: float = 75.0,
    min_len: int = 4,
) -> List[np.ndarray]:
    """Extract contiguous high-synchrony episodes from a WCC trace.

    Parameters
    ----------
    wcc : np.ndarray
        WCC time series.
    threshold : float
        Fixed threshold for ``threshold_mode="fixed"``.
    threshold_mode : {"fixed", "percentile"}
        "fixed" uses ``threshold``; "percentile" uses the given percentile of
        the trace.
    percentile : float
        Percentile for ``threshold_mode="percentile"``.
    min_len : int
        Minimum episode length in samples.

    Returns
    -------
    list of np.ndarray
        Episode waveforms.
    """
    w = np.asarray(wcc, float)
    finite = np.isfinite(w)
    if finite.sum() < 10:
        return []
    wf = np.where(finite, w, np.nan)
    thr = threshold if threshold_mode == "fixed" else float(np.nanpercentile(wf, percentile))
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


def resample_waveform(ep: np.ndarray, L: int = 20) -> np.ndarray:
    """Resample an episode to a fixed length L and amplitude-normalise."""
    x = np.linspace(0, 1, len(ep))
    xi = np.linspace(0, 1, L)
    wf = np.interp(xi, x, ep)
    rng = wf.max() - wf.min()
    return (wf - wf.min()) / rng if rng > 1e-9 else wf - wf.min()


def episode_shape_features(ep: np.ndarray) -> Dict[str, float]:
    """Return shape features of a single episode.

    Keys: rise_frac, fall_frac, asymmetry, n_subpeaks, ep_skew, duration.
    """
    L = len(ep)
    pk = int(np.argmax(ep))
    peaks, _ = find_peaks(ep, prominence=0.05)
    return {
        "rise_frac": pk / max(L - 1, 1),
        "fall_frac": (L - 1 - pk) / max(L - 1, 1),
        "asymmetry": pk / max(L - 1 - pk, 1),
        "n_subpeaks": len(peaks),
        "ep_skew": float(stats.skew(ep)) if L > 3 else 0.0,
        "duration": L,
    }


def episode_archetype_cluster(
    wcc_traces: List[np.ndarray],
    threshold: float = ONSET_THRESHOLD,
    threshold_mode: str = "fixed",
    percentile: float = 75.0,
    resample_len: int = 20,
    min_len: int = 4,
    k_range: Tuple[int, ...] = (2, 3, 4, 5),
    seed: int = 42,
) -> Dict[str, object]:
    """Cluster episodes into waveform archetypes (Method 2).

    Parameters
    ----------
    wcc_traces : list of np.ndarray
    threshold, threshold_mode, percentile, min_len
        Episode detection parameters.
    resample_len : int
        Length to normalise episodes to.
    k_range : tuple of int
        Candidate numbers of archetypes.
    seed : int

    Returns
    -------
    dict
        - ``n_episodes``: total episodes
        - ``waveform_archetypes``: mean waveform per archetype
        - ``waveform_labels``: episode archetype labels
        - ``waveform_k_best``: chosen k
        - ``feature_profiles``: mean shape features per archetype
        - ``trace_mixture``: mapping trace_id -> archetype distribution
    """
    wave_rows, feat_rows, trace_ids = [], [], []
    for tid, w in enumerate(wcc_traces):
        for ep in extract_episodes(w, threshold, threshold_mode, percentile, min_len):
            wave_rows.append(resample_waveform(ep, resample_len))
            feat_rows.append(episode_shape_features(ep))
            trace_ids.append(tid)

    if not wave_rows:
        return {"n_episodes": 0}

    waves = np.array(wave_rows)
    feats = pd.DataFrame(feat_rows)
    idx = pd.DataFrame({"trace_id": trace_ids})
    rng = np.random.RandomState(seed)

    # Cluster waveforms
    Xw = StandardScaler().fit_transform(waves)
    kw, wlabels, ari_w = _pick_k(Xw, k_range, rng)
    archetypes = np.vstack([waves[wlabels == c].mean(0) for c in range(kw)])

    # Cluster shape features
    fcols = ["rise_frac", "fall_frac", "asymmetry", "n_subpeaks", "ep_skew"]
    Xf = StandardScaler().fit_transform(feats[fcols].fillna(feats[fcols].median()).values)
    kf, flabels, ari_f = _pick_k(Xf, k_range, rng)
    profiles = feats.assign(cluster=flabels).groupby("cluster")[fcols].mean()

    # Trace-level mixture over waveform archetypes
    idx = idx.assign(wlab=wlabels)
    mixture = (idx.groupby(["trace_id", "wlab"]).size()
               .groupby(level=0).apply(lambda s: (s / s.sum()).to_dict()))

    return {
        "n_episodes": len(waves),
        "waveform_archetypes": archetypes,
        "waveform_labels": wlabels,
        "waveform_k_best": kw,
        "waveform_subsample_ari": ari_w,
        "feature_labels": flabels,
        "feature_k_best": kf,
        "feature_profiles": profiles,
        "feature_subsample_ari": ari_f,
        "trace_mixture": mixture,
    }


# ---------------------------------------------------------------------------
# Joint morphology feature table (shape descriptors + SyncPipe features)
# ---------------------------------------------------------------------------

def morphology_feature_table(
    wcc_traces: List[np.ndarray],
    hz: float = 1.0,
    wcc_window_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Return a DataFrame with shape descriptors + SyncPipe features per trace.

    Parameters
    ----------
    wcc_traces : list of np.ndarray
    hz : float
        Sampling rate of WCC trace.
    wcc_window_sec : float or None
        WCC window duration in seconds; used for sustained-crossing scaling.

    Returns
    -------
    pd.DataFrame
        One row per trace, with columns prefixed by ``shape_`` for shape
        descriptors and the 9 SyncPipe feature names.
    """
    rows = []
    for w in wcc_traces:
        sd = scalefree_descriptors(w)
        if sd is None:
            continue
        if wcc_window_sec is None:
            wcc_window_sec = len(w) / hz
        f = extract_dynamic_features(np.asarray(w, float), hz=hz, wcc_window_sec=wcc_window_sec)
        ms = {k: float(getattr(f, k, np.nan)) for k in ["mean_synchrony", "peak_amplitude",
            "dwell_time", "switching_rate", "bimodality_coefficient", "synchrony_entropy",
            "onset_latency", "rise_time", "recovery_time"]}
        rows.append({f"shape_{k}": v for k, v in sd.items()} | ms)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collinearity / incremental value / matched-mean contrast
# ---------------------------------------------------------------------------

def collinearity_report(
    feat_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return Spearman correlation matrix and VIF for a feature set."""
    sub = feat_df[features].dropna(axis=1, how="all")
    used = [c for c in features if c in sub.columns and sub[c].std() > 1e-9]
    corr = sub[used].corr(method="spearman")

    vif = {}
    X = sub[used].fillna(sub[used].median()).values
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    for i, c in enumerate(used):
        others = np.delete(Xc, i, axis=1)
        try:
            beta, *_ = np.linalg.lstsq(others, Xc[:, i], rcond=None)
            pred = others @ beta
            ss_res = np.sum((Xc[:, i] - pred) ** 2)
            ss_tot = np.sum((Xc[:, i] - Xc[:, i].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            vif[c] = float(1.0 / max(1e-6, 1 - r2))
        except Exception:
            vif[c] = float("nan")
    return corr, pd.Series(vif, name="VIF")


def incremental_value(
    feat_df: pd.DataFrame,
    y: np.ndarray,
    features: List[str],
    baseline_drop_meansync: bool = True,
    n_orders: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Random-order-averaged marginal AUC + LOFO.

    See morphology_analysis.py for full rationale. This is a small-n
    exploratory diagnostic, not a confirmatory test.
    """
    from itertools import permutations
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    rng = np.random.default_rng(seed)
    feats = [f for f in features if f in feat_df.columns and feat_df[f].std() > 1e-9]
    if baseline_drop_meansync and "mean_synchrony" in feats:
        feats = [f for f in feats if f != "mean_synchrony"]
        baseline_feats = []
    else:
        baseline_feats = ["mean_synchrony"] if "mean_synchrony" in feats else []
        feats = [f for f in feats if f not in baseline_feats]

    allcols = list(dict.fromkeys(baseline_feats + feats))
    Xall = feat_df[allcols].apply(pd.to_numeric, errors="coerce").fillna(feat_df[allcols].median())

    def auc_of(cols):
        if not cols:
            return 0.5
        X = Xall[cols].values
        classes, counts = np.unique(y, return_counts=True)
        n_splits = int(min(5, counts.min()))
        if n_splits < 2 or X.shape[1] == 0:
            return np.nan
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        scoring = "roc_auc_ovr" if len(classes) > 2 else "roc_auc"
        try:
            return float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring=scoring)))
        except Exception:
            return np.nan

    base_auc = auc_of(baseline_feats)
    full_auc = auc_of(baseline_feats + feats)

    marg = {f: [] for f in feats}
    k = len(feats)
    orders = list(permutations(feats)) if k <= 6 else [list(rng.permutation(feats)) for _ in range(n_orders)]
    for order in orders:
        cur = list(baseline_feats)
        prev = auc_of(cur)
        for f in order:
            cur.append(f)
            now = auc_of(cur)
            marg[f].append(now - prev)
            prev = now
    shap = {f: float(np.nanmean(v)) for f, v in marg.items()}

    lofo = {}
    for f in feats:
        without = baseline_feats + [x for x in feats if x != f]
        lofo[f] = float(full_auc - auc_of(without))

    rows = [{"feature": f, "shapley_marginal_auc": shap[f], "lofo_auc_drop": lofo[f]} for f in feats]
    out = pd.DataFrame(rows).sort_values("shapley_marginal_auc", ascending=False)
    meta = {"baseline": "none" if baseline_drop_meansync else "mean_synchrony",
            "baseline_auc": base_auc, "full_auc": full_auc}
    return out, meta


def matched_mean_contrast(
    feat_df: pd.DataFrame,
    y: np.ndarray,
    features: List[str],
    band: float = 0.1,
) -> pd.DataFrame:
    """Within a narrow mean_synchrony band, which feature separates morphology?"""
    if "mean_synchrony" not in feat_df:
        return pd.DataFrame()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    med = feat_df["mean_synchrony"].median()
    mask = (feat_df["mean_synchrony"] >= med - band) & (feat_df["mean_synchrony"] <= med + band)
    sub = feat_df[mask]
    ysub = np.asarray(y)[mask.values]
    rows = []
    for f in features:
        if f == "mean_synchrony" or f not in sub or sub[f].std() < 1e-9:
            continue
        col = pd.to_numeric(sub[f], errors="coerce").fillna(sub[f].median())
        classes, counts = np.unique(ysub, return_counts=True)
        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        scoring = "roc_auc_ovr" if len(classes) > 2 else "roc_auc"
        try:
            auc = float(np.mean(cross_val_score(pipe, col.values.reshape(-1, 1), ysub, cv=cv, scoring=scoring)))
        except Exception:
            auc = np.nan
        rows.append({"feature": f, "auc_within_mean_band": auc, "n_in_band": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("auc_within_mean_band", ascending=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subsample_ari(X, k, rng, n_iter=150, frac=0.8):
    n = X.shape[0]
    m = max(k + 1, int(round(frac * n)))
    if n < 2 * k + 2 or m >= n:
        return float("nan"), 0

    def cl(Xs):
        return KMeans(k, random_state=rng.randint(0, 99999), n_init=10).fit_predict(Xs)

    a = []
    for _ in range(n_iter):
        i1 = rng.choice(n, m, replace=False)
        i2 = rng.choice(n, m, replace=False)
        l1 = dict(zip(i1, cl(X[i1])))
        l2 = dict(zip(i2, cl(X[i2])))
        sh = np.intersect1d(i1, i2)
        if sh.size < k + 1:
            continue
        a.append(adjusted_rand_score([l1[i] for i in sh], [l2[i] for i in sh]))
    return (float(np.mean(a)), len(a)) if a else (float("nan"), 0)


def _pick_k(X, k_range, rng):
    best_ari = -1.0
    best_k = k_range[0]
    best_labels = None
    for k in k_range:
        labels = KMeans(k, random_state=rng.randint(0, 99999), n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        ari, _ = _subsample_ari(X, k, rng)
        if not np.isnan(ari) and ari > best_ari:
            best_ari = ari
            best_k = k
            best_labels = labels
    if best_labels is None:
        best_labels = KMeans(k_range[0], random_state=42, n_init=10).fit_predict(X)
    return best_k, best_labels, best_ari


# ---------------------------------------------------------------------------
# High-level MorphologyAnalyzer class
# ---------------------------------------------------------------------------

class MorphologyAnalyzer:
    """High-level wrapper for the two morphology methods.

    Parameters
    ----------
    wcc_traces : list of np.ndarray
        One WCC trace per observation.
    hz : float
        Sampling rate.
    """

    def __init__(self, wcc_traces: List[np.ndarray], hz: float = 1.0):
        self.wcc_traces = wcc_traces
        self.hz = hz
        self._method1: Optional[Dict] = None
        self._method2: Optional[Dict] = None

    def run_method1(self, max_k: int = 5, seed: int = 42) -> Dict[str, object]:
        """Run trace-level scale-free shape clustering."""
        self._method1 = trace_shape_cluster(self.wcc_traces, max_k=max_k, seed=seed)
        return self._method1

    def run_method2(
        self,
        threshold: float = ONSET_THRESHOLD,
        threshold_mode: str = "fixed",
        percentile: float = 75.0,
        resample_len: int = 20,
        k_range: Tuple[int, ...] = (2, 3, 4, 5),
        seed: int = 42,
    ) -> Dict[str, object]:
        """Run episode-level waveform archetype clustering."""
        self._method2 = episode_archetype_cluster(
            self.wcc_traces,
            threshold=threshold,
            threshold_mode=threshold_mode,
            percentile=percentile,
            resample_len=resample_len,
            k_range=k_range,
            seed=seed,
        )
        return self._method2

    def feature_table(self) -> pd.DataFrame:
        """Return shape descriptors + SyncPipe features per trace."""
        return morphology_feature_table(self.wcc_traces, hz=self.hz)

    def diagnostics(self, feature_cols: Optional[List[str]] = None) -> Dict[str, object]:
        """Collinearity, incremental value, and matched-mean contrast.

        Requires ``run_method1`` to have been called (uses Method 1 labels as y).
        """
        if self._method1 is None or self._method1["labels"] is None or len(self._method1["labels"]) == 0:
            raise ValueError("Call run_method1() before diagnostics().")
        feat_df = self.feature_table()
        if feature_cols is None:
            feature_cols = ["mean_synchrony", "peak_amplitude", "dwell_time", "switching_rate",
                             "bimodality_coefficient", "synchrony_entropy",
                             "onset_latency", "rise_time", "recovery_time"]
        y = self._method1["labels"]
        corr, vif = collinearity_report(feat_df, feature_cols)
        inc_drop, meta_drop = incremental_value(feat_df, y, feature_cols, baseline_drop_meansync=True)
        inc_keep, meta_keep = incremental_value(feat_df, y, feature_cols, baseline_drop_meansync=False)
        mm = matched_mean_contrast(feat_df, y, feature_cols)
        return {
            "correlation": corr,
            "vif": vif,
            "incremental_drop_meansync": (inc_drop, meta_drop),
            "incremental_keep_meansync": (inc_keep, meta_keep),
            "matched_mean_contrast": mm,
        }

    def to_report(self, feature_cols: Optional[List[str]] = None) -> Dict[str, object]:
        """Return a JSON-serializable morphology report dictionary.

        Combines Method 1, Method 2, feature table, and diagnostics into a
        single structure suitable for JSON export or downstream reporting.
        """
        if self._method1 is None:
            raise ValueError("Call run_method1() before to_report().")
        if self._method2 is None:
            self.run_method2()

        diag = self.diagnostics(feature_cols=feature_cols)
        feat_df = self.feature_table()
        labels = self._method1["labels"]

        # Per-cluster mean feature profile
        feat_df_clustered = feat_df.copy()
        feat_df_clustered["cluster"] = labels
        cluster_means = feat_df_clustered.groupby("cluster").mean()

        report = {
            "n_traces": len(self.wcc_traces),
            "hz": self.hz,
            "method1": {
                "k_best": _to_json_safe(self._method1["k_best"]),
                "silhouette_best": _to_json_safe(self._method1["silhouette_best"]),
                "k_selection": _df_to_dict(self._method1["k_selection"]),
                "labels": _to_json_safe(labels),
                "cluster_feature_means": _df_to_dict(cluster_means),
            },
            "method2": {
                "n_episodes": _to_json_safe(self._method2.get("n_episodes", 0)),
                "waveform_k_best": _to_json_safe(self._method2.get("waveform_k_best")),
                "waveform_subsample_ari": _to_json_safe(self._method2.get("waveform_subsample_ari")),
                "feature_k_best": _to_json_safe(self._method2.get("feature_k_best")),
                "feature_subsample_ari": _to_json_safe(self._method2.get("feature_subsample_ari")),
                "waveform_archetypes": _to_json_safe(self._method2.get("waveform_archetypes")),
                "feature_profiles": _df_to_dict(self._method2.get("feature_profiles")),
            },
            "diagnostics": {
                "correlation": _df_to_dict(diag["correlation"]),
                "vif": _series_to_dict(diag["vif"]),
                "incremental_drop_meansync": {
                    "table": _df_to_dict(diag["incremental_drop_meansync"][0]),
                    "meta": _to_json_safe(diag["incremental_drop_meansync"][1]),
                },
                "incremental_keep_meansync": {
                    "table": _df_to_dict(diag["incremental_keep_meansync"][0]),
                    "meta": _to_json_safe(diag["incremental_keep_meansync"][1]),
                },
                "matched_mean_contrast": _df_to_dict(diag["matched_mean_contrast"]),
            },
            "notes": [
                "All morphology analyses are EXPLORATORY in v1.0.",
                "Cluster stability should be confirmed with bootstrap ARI before any confirmatory claim.",
                "High-VIF features are redundant and should not be treated as independent tests.",
            ],
        }
        return report

    def write_report(self, path: Union[str, Path], feature_cols: Optional[List[str]] = None) -> None:
        """Write a JSON morphology report to ``path``."""
        import json

        path = Path(path)
        report = self.to_report(feature_cols=feature_cols)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON-serialisation helpers
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """Convert numpy/pandas objects to JSON-safe Python structures."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return _df_to_dict(obj)
    if isinstance(obj, pd.Series):
        return _series_to_dict(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _df_to_dict(df: Optional[pd.DataFrame]) -> Dict:
    if df is None or df.empty:
        return {}
    return {str(k): {str(kk): _to_json_safe(vv) for kk, vv in v.items()} for k, v in df.to_dict().items()}


def _series_to_dict(s: Optional[pd.Series]) -> Dict:
    if s is None or s.empty:
        return {}
    return {str(k): _to_json_safe(v) for k, v in s.items()}

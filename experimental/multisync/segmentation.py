"""
HMM / HSMM-lite state segmentation for synchrony traces.
========================================================

.. warning::

   **STATUS: EXPERIMENTAL (D4, 2026-06)**

   This module is fully implemented but NOT yet integrated into the main
   ``DynamicAnalyzer`` pipeline.  The locked-in feature definitions
   (``feature_definitions.py``) still use fixed-threshold binarization.
   Future work will add a ``segmenter="hmm"`` option to ``DynamicAnalyzer``.

   Do NOT rely on this API in production until promoted to stable.

Motivation (user, 2026-05-31):
    The locked-in feature definitions (feature_definitions.py) binarize the
    synchrony trace at a fixed threshold (WCC > 0.5) before computing
    dwell_time / switching_rate.  That forces a TWO-state view
    (synchronous / asynchronous).  Synchrony may instead visit MULTIPLE
    coupling regimes (strong / weak / anti-phase / decoupled), and dwell
    times may have a characteristic timescale rather than the memoryless
    (geometric) dwell that a plain Markov chain imposes.

Design (comparability + portability):
    - Self-contained Gaussian HMM (Baum-Welch EM + Viterbi), scipy/sklearn
      only.  No hmmlearn / pyhsmm dependency -> ships with SyncPipe.
    - K (number of states) chosen by BIC over a small grid -> "multi-state,
      not binary" is data-driven, not assumed (anti-HARKing: K is logged).
    - HSMM-lite: an explicit `min_dwell` duration floor post-processes the
      Viterbi path, removing sub-threshold runs.  This counters the
      geometric-dwell assumption of a pure HMM without a full
      explicit-duration EM (which is future work; see DECISION-14 follow-up).
    - Output is a [0,1] HIGH-SYNC POSTERIOR trace, so it feeds the SAME
      `extract_dynamic_features` as a WCC trace -> feature vectors are
      directly comparable across segmenters.

This module is a SEGMENTER (trace -> trace), conceptually one layer above
the METRICS in metrics.py (which are a,b -> trace).  It does NOT modify the
locked-in SSoT feature math.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────
# Compact Gaussian HMM (scaled forward-backward + Baum-Welch + Viterbi)
# ─────────────────────────────────────────────────────────────────────
def _gauss_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, _EPS)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def _init_params(x: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    qs = np.quantile(x, np.linspace(0.1, 0.9, k))
    mus = qs.astype(float)
    var = np.full(k, float(np.var(x)) + _EPS)
    trans = np.full((k, k), 1.0 / k)
    trans = 0.8 * np.eye(k) + 0.2 * trans
    trans /= trans.sum(axis=1, keepdims=True)
    pi = np.full(k, 1.0 / k)
    return mus, var, trans, pi


def _forward_backward(logB: np.ndarray, trans: np.ndarray, pi: np.ndarray):
    n, k = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    alpha = np.zeros((n, k))
    c = np.zeros(n)
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum() + _EPS
    alpha[0] /= c[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ trans) * B[t]
        c[t] = alpha[t].sum() + _EPS
        alpha[t] /= c[t]
    beta = np.zeros((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (trans @ (B[t + 1] * beta[t + 1]))
        beta[t] /= (beta[t].sum() + _EPS)
    gamma = alpha * beta
    gamma /= (gamma.sum(axis=1, keepdims=True) + _EPS)
    loglik = float(np.sum(np.log(c)))
    return gamma, alpha, beta, B, c, loglik


def fit_gaussian_hmm(x: np.ndarray, k: int, n_iter: int = 50,
                     tol: float = 1e-4, seed: int = 0):
    """Fit a K-state 1-D Gaussian HMM by Baum-Welch. Returns param dict."""
    x = np.asarray(x, float)
    n = len(x)
    mus, var, trans, pi = _init_params(x, k, seed)
    prev_ll = -np.inf
    for _ in range(n_iter):
        logB = np.column_stack([_gauss_logpdf(x, mus[j], var[j]) for j in range(k)])
        gamma, alpha, beta, B, c, ll = _forward_backward(logB, trans, pi)
        # xi (transition posteriors), summed over time
        xi_sum = np.zeros((k, k))
        for t in range(n - 1):
            num = (alpha[t][:, None] * trans) * (B[t + 1] * beta[t + 1])[None, :]
            xi_sum += num / (num.sum() + _EPS)
        # M-step
        pi = gamma[0] + _EPS
        pi /= pi.sum()
        trans = xi_sum + _EPS
        trans /= trans.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0) + _EPS
        mus = (gamma * x[:, None]).sum(axis=0) / w
        var = (gamma * (x[:, None] - mus[None, :]) ** 2).sum(axis=0) / w
        var = np.maximum(var, 1e-6)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return {"mus": mus, "var": var, "trans": trans, "pi": pi,
            "loglik": ll, "k": k, "gamma": gamma}


def viterbi(x: np.ndarray, params: dict) -> np.ndarray:
    mus, var, trans, pi = params["mus"], params["var"], params["trans"], params["pi"]
    k = len(mus)
    n = len(x)
    logB = np.column_stack([_gauss_logpdf(x, mus[j], var[j]) for j in range(k)])
    logT = np.log(trans + _EPS)
    delta = np.zeros((n, k))
    psi = np.zeros((n, k), dtype=int)
    delta[0] = np.log(pi + _EPS) + logB[0]
    for t in range(1, n):
        for j in range(k):
            seq = delta[t - 1] + logT[:, j]
            psi[t, j] = int(np.argmax(seq))
            delta[t, j] = seq[psi[t, j]] + logB[t, j]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def bic(params: dict, n: int) -> float:
    """BIC = -2*loglik + n_params*log(N). Lower is better."""
    k = params["k"]
    n_params = 2 * k + k * (k - 1) + (k - 1)  # mus+var + trans + init
    return -2 * params["loglik"] + n_params * np.log(max(n, 2))


# ─────────────────────────────────────────────────────────────────────
# HSMM-lite: min-dwell duration floor on the Viterbi path
# ─────────────────────────────────────────────────────────────────────
def enforce_min_dwell(path: np.ndarray, min_dwell: int) -> np.ndarray:
    """Merge runs shorter than `min_dwell` into the preceding state.

    A transparent explicit-duration constraint (HSMM-lite): a pure HMM
    imposes geometric (memoryless) dwell times; this floor removes
    physically implausible flicker without a full explicit-duration EM.
    """
    if min_dwell <= 1 or len(path) == 0:
        return path
    out = path.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_dwell and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def select_k_by_bic(x: np.ndarray, k_grid=(2, 3, 4),
                    n_iter: int = 50, seed: int = 0):
    """Fit HMMs for each K in k_grid, return (best_params, bic_table)."""
    x = np.asarray(x, float)
    n = len(x)
    best, best_bic = None, np.inf
    table = []
    for k in k_grid:
        if n < k * 3:
            continue
        params = fit_gaussian_hmm(x, k, n_iter=n_iter, seed=seed)
        b = bic(params, n)
        table.append({"k": k, "bic": b, "loglik": params["loglik"]})
        if b < best_bic:
            best, best_bic = params, b
    return best, table


def hsmm_high_sync_trace(
    trace: np.ndarray,
    k_grid=(2, 3, 4),
    min_dwell: int = 3,
    high_frac: float = 0.5,
    n_iter: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """Segment a synchrony trace into states, return a [0,1] high-sync trace.

    The returned trace is the soft posterior probability of being in the
    "high-synchrony" state set (states whose mean exceeds the `high_frac`
    quantile of state means), so it can feed `extract_dynamic_features`
    EXACTLY like a WCC trace -> feature vectors stay comparable across
    segmenters.

    Returns
    -------
    high_trace : np.ndarray   # posterior P(high-sync state) in [0,1]
    info : dict               # {k, state_means, min_dwell, bic_table, path}
    """
    x = np.nan_to_num(np.asarray(trace, float), nan=0.0)
    if len(x) < 6 or np.std(x) < _EPS:
        return np.clip(x, 0, 1), {"k": 1, "state_means": [float(np.mean(x))],
                                  "min_dwell": min_dwell, "bic_table": [],
                                  "path": np.zeros(len(x), dtype=int)}
    params, table = select_k_by_bic(x, k_grid, n_iter=n_iter, seed=seed)
    if params is None:
        return np.clip(x, 0, 1), {"k": 1, "state_means": [float(np.mean(x))],
                                  "min_dwell": min_dwell, "bic_table": table,
                                  "path": np.zeros(len(x), dtype=int)}
    path = viterbi(x, params)
    path = enforce_min_dwell(path, min_dwell)
    mus = params["mus"]
    gamma = params["gamma"]
    # high-sync state set: state means above the high_frac quantile of means
    thr = np.quantile(mus, high_frac) if len(mus) > 1 else mus[0]
    high_states = np.where(mus >= thr)[0]
    high_trace = gamma[:, high_states].sum(axis=1)
    high_trace = np.clip(high_trace, 0, 1)
    info = {"k": params["k"], "state_means": [float(m) for m in mus],
            "min_dwell": min_dwell, "bic_table": table, "path": path}
    return high_trace, info


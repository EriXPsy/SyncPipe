"""
Surrogate-based significance testing (Level 3).

Implements the statistical testing framework for the dynamic features
extracted from WCC time series.  Provides two single-signal surrogate
null models: an FT surrogate (Fourier-phase randomization; Theiler et al.
1992) and IAAFT (Schreiber & Schmitz 1996).  IAAFT is the primary,
more conservative null: SyncPipe's implementation preserves the empirical
amplitude distribution exactly and approximates the power spectrum / linear
autocorrelation. The FT surrogate is provided as a robustness comparator
(preserves the power spectrum only).

Feature partition (SSoT Option B, revised 2026-06-29)
-----------------------------------------------------
- **FDR family** (``FEATURE_TAILS``, 3 features, grouped by mathematical
  invariance tier):

  *Family L0* (signal-level IAAFT null, existence test):
    peak_amplitude

  *Family L1* (WCC-level IAAFT null, structural increment test):
    dwell_time, switching_rate

  These define the family that :func:`apply_bh_fdr_within_noise`
  controls at level *q*.

  mean_synchrony is reported as a reference (REFERENCE_TAILS) and
  bimodality_coefficient is exploratory; both remain L0 features for the
  separate synchrony-existence audit but are NOT in the confirmatory FDR
  family.  L2 features (onset_latency, rise_time, recovery_time) and
  synchrony_entropy are likewise EXCLUDED (exploratory).
  See ``docs/METHOD_LOG.md`` for the v1 audited null-model stance
  architecture.

- **Reference** (``REFERENCE_TAILS``):
  As of 2026-06-23, ``REFERENCE_TAILS`` is empty.  mean_synchrony
  is classified as "reference" on the functional tier (Axis A) but is
  included in FDR Family L0 for the existence test (Axis C).  Functional
  tier and FDR membership are independent axes.

Theoretical background
----------------------
- Surrogate data testing: Schreiber & Schmitz (2000), ``Physica D``
- Unbiased p-values: Phipson & Smyth (2010), ``SAGMB``
- FDR control: Benjamini & Hochberg (1995), ``JRSS B``

FT surrogate vs IAAFT for Level 3
------------------------------------------
Both null models preserve each signal's own power spectrum (linear
autocorrelation) and destroy only the cross-signal phase relationship.
IAAFT additionally preserves the amplitude distribution, making it the
more conservative and field-standard choice; it is used as the primary
null.  The FT surrogate (phase randomization only) is reported as a
robustness comparator.  When both yield the same conclusion, the result
is shown to be insensitive to the surrogate choice.

Note on burst-dominated signals: IAAFT preserves high-amplitude burst
peaks, which under random phase realignment can occasionally coincide and
inflate the null; the FT surrogate disperses burst energy across the time
axis.  Agreement between the two is therefore the relevant robustness check.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .recovery import _extract_six_features, ONSET_THRESHOLD_DEFAULT
from ..dynamic_features import sliding_window_wcc
from ..synthetic import generate_ground_truth_dyad


# ---------------------------------------------------------------------------
# Feature directionality (one- vs two-tailed)
# ---------------------------------------------------------------------------
# FDR family (3 features: L0 peak_amplitude + L1 dwell/switching).
# ---------------------------------------------------------------------------
# mean_synchrony is Reference (reported, NOT in FDR) and bimodality_
# coefficient is exploratory (NOT in FDR) per the 2026-06-29 SSoT
# decision (Option B).  Dwell/switching use two-tailed tests;
# peak_amplitude uses upper-tailed.
# SSoT guard: the feature list for FEATURE_TAILS is locked to
# ``FDR_FEATURES`` in ``feature_definitions.py``.
# The tail directions below are pgt1_intensity-specific; the list of
# features is validated at import time against the SSoT.
from ..feature_definitions import FDR_FEATURES  # noqa: E402 (late import — SSoT guard)

FEATURE_TAILS: Dict[str, str] = {
    # Family L0 (signal-level null): existence test
    "peak_amplitude":        "upper",
    # Family L1 (WCC-level null): structural increment test
    "dwell_time":            "two",
    "switching_rate":        "two",
}
"""FDR family for Level 3 — 3 features (SSoT Option B, 2026-06-29).

L0: peak_amplitude.  L1: dwell_time, switching_rate.
mean_synchrony (reference) and bimodality_coefficient (exploratory) were
removed from the confirmatory FDR family on 2026-06-29; both remain L0
features for the synchrony-existence audit.

Tail directions
---------------
- ``"upper"``:  H₁: ``obs > null``  (feature higher = more sync).
- ``"two"``:    H₁: ``|obs - null_mean| > 0``  (conservative).

L2 features (onset_latency, rise_time, recovery_time) and
synchrony_entropy are EXCLUDED (exploratory, pending proper null model).

The feature list **must** match ``FDR_FEATURES`` (SSoT).
A runtime assertion on import enforces this sync.
"""

# -- runtime SSoT sync guard -------------------------------------------------
_FDR_FEATURE_SET = set(FEATURE_TAILS.keys())
_SSOT_FDR_SET = set(FDR_FEATURES)
assert _FDR_FEATURE_SET == _SSOT_FDR_SET, (
    f"FEATURE_TAILS keys {sorted(_FDR_FEATURE_SET)} "
    f"≠ FDR_FEATURES {sorted(_SSOT_FDR_SET)}.  "
    "Update FEATURE_TAILS in pgt1_intensity.py to match the SSoT."
)


REFERENCE_TAILS: Dict[str, str] = {
    "mean_synchrony": "upper",
}
"""Reference features — reported with a surrogate p-value but NOT included
in the confirmatory BH-FDR family.

2026-06-29 (SSoT Option B): mean_synchrony was demoted from FDR Family L0
back to a reported reference comparator.  Its existence/power p-value is
still computed and reported (so its response curve is inspectable), but it
does not consume the confirmatory family's multiplicity budget.
"""


FEATURE_P_COLUMNS: Tuple[str, ...] = tuple(f"p_{f}" for f in FEATURE_TAILS.keys())
"""Column names for the raw p-values that DEFINE the FDR family
(3 features: peak_amplitude, dwell_time, switching_rate).
``apply_bh_fdr_within_noise`` uses this tuple as the default
``feature_p_columns``.
"""


REFERENCE_P_COLUMNS: Tuple[str, ...] = tuple(f"p_{f}" for f in REFERENCE_TAILS.keys())
"""Column names for Reference p-values that are *reported* but **not**
included in BH-FDR."""

# Legacy aliases
DIAGNOSTIC_TAILS: Dict[str, str] = REFERENCE_TAILS
DIAGNOSTIC_P_COLUMNS: Tuple[str, ...] = REFERENCE_P_COLUMNS


# ---------------------------------------------------------------------------
# FT / IAAFT surrogates — imported from shared public module
# ---------------------------------------------------------------------------

from ..surrogate import ft_surrogate, iaaft_surrogate, prtf_surrogate


# ---------------------------------------------------------------------------
# Phiper-Smyth unbiased p-value
# ---------------------------------------------------------------------------

def phipson_smyth_p(
    observed: float,
    null_values: np.ndarray,
    tail: str = "upper",
) -> float:
    """
    Compute the Phipson & Smyth (2010) unbiased p‑value.

    The unbiased estimator eliminates the downward bias of the naive formula
    ``p = (# null >= obs) / N``, which can produce ``p = 0`` and
    causes ``-log10(p) = inf``.

    Formula
    -------
    ``p_unbiased = (1 + #{null >= obs}) / (1 + N)``   (upper‑tailed)

    The minimum achievable p‑value is ``1 / (1 + N)``, so with
    ``N = 999`` the minimum p is ``0.001``.

    Parameters
    ----------
    observed : float
        Observed feature value.
    null_values : np.ndarray
        Array of surrogate feature values.  NaN entries are automatically
        excluded.
    tail : {"upper", "lower", "two"}
        - ``"upper"``:  H₁: obs > null  (peak_amplitude, mean_synchrony)
        - ``"lower"``:  H₁: obs < null
        - ``"two"``:    H₁: |obs − null_mean| > 0  (conservative)

    Returns
    -------
    float
        p‑value in ``(0, 1]``.  Returns ``NaN`` if ``observed`` is
        non‑finite or ``null_values`` has no finite entries.
    """
    null_clean = null_values[np.isfinite(null_values)]
    n = null_clean.size
    if n == 0 or not np.isfinite(observed):
        return float("nan")

    if tail == "upper":
        k = int(np.sum(null_clean >= observed))
        return (1.0 + k) / (1.0 + n)
    if tail == "lower":
        k = int(np.sum(null_clean <= observed))
        return (1.0 + k) / (1.0 + n)
    if tail == "two":
        k_up = int(np.sum(null_clean >= observed))
        k_lo = int(np.sum(null_clean <= observed))
        p_up = (1.0 + k_up) / (1.0 + n)
        p_lo = (1.0 + k_lo) / (1.0 + n)
        return min(1.0, 2.0 * min(p_up, p_lo))

    raise ValueError(f"Unknown tail: {tail!r}")


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR control
# ---------------------------------------------------------------------------

def bh_fdr(
    p_values: np.ndarray,
    q: float = 0.05,
) -> np.ndarray:
    """
    Benjamini‑Hochberg step‑up FDR control.

    Returns a boolean array indicating which p‑values are rejected at
    FDR level ``q``.  NaN p‑values are treated as **not rejected**.

    Parameters
    ----------
    p_values : np.ndarray
        1‑D array of p‑values (may contain NaN).
    q : float
        Target FDR level (default 0.05).

    Returns
    -------
    np.ndarray (bool)
        ``True`` for rejected tests.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    out = np.zeros(n, dtype=bool)

    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return out

    p_finite = p[finite_mask]
    order = np.argsort(p_finite)
    m = p_finite.size
    thresholds = q * np.arange(1, m + 1) / m

    sorted_p = p_finite[order]
    below = sorted_p <= thresholds

    if not below.any():
        return out

    k_max = int(np.max(np.where(below)[0]))
    cutoff = sorted_p[k_max]

    rejected_finite = p_finite <= cutoff
    finite_indices = np.where(finite_mask)[0]
    out[finite_indices[rejected_finite]] = True

    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level3Config:
    """
    Frozen configuration for a Level 3 surrogate testing experiment.

    The default sweep covers:
    - ``coupling=(0.0,)``  → Type I error (Level 3a)
    - ``coupling=(0.3,)`` → Statistical power  (Level 3b)
    - ``coupling=(0.7,)`` → Sanity check      (Level 3c, small seeds)

    Surrogate method
    ----------------
    ``surrogate_method`` selects the null model:
    - ``"ft"`` (default; ``"prtf"`` accepted as a deprecated alias):
      Fourier-phase randomization, preserves power spectrum only
      (Theiler et al. 1992).  Default for this **synthetic** burst-dominated
      Level 3 sweep, where IAAFT can inflate the null by preserving
      coincident burst peaks.
    - ``"iaaft"``: preserves the empirical amplitude distribution exactly and approximates the power spectrum
      (Schreiber & Schmitz 1996).  The more conservative, field-standard
      null; used as the **primary** method for the real-dataset main results
      and as the conservative comparator here.

    References
    ----------
    - Schreiber & Schmitz (2000), Physica D, 142(3‑4), 346‑382.
    - Phipson & Smyth (2010), SAGMB, 9(1), Article 39.
    - Benjamini & Hochberg (1995), JRSS B, 57(1), 289‑300.
    """

    duration_sec: float = 300.0
    hz: float = 1.0
    n_bursts: int = 5
    burst_sigma: float = 3.0
    true_lag_sec: float = 0.0
    morphology: str = "identical"
    gap_prob: float = 0.0
    wcc_window_sec: float = 30.0

    onset_threshold: float = ONSET_THRESHOLD_DEFAULT

    noise_ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 1.0)
    couplings: Sequence[float] = (0.0,)          # default: Type I only
    seeds: Sequence[int] = tuple(range(1000, 1030))  # 30 seeds

    n_surrogates: int = 999
    fdr_q: float = 0.05
    surrogate_method: str = "ft"   # "ft" (alias "prtf") | "iaaft"
    iaaft_max_iter: int = 200
    iaaft_tol: float = 1e-8

    # DECISION-01 (revised 2026-06-21): surrogate-derived threshold
    use_surrogate_threshold: bool = False
    n_threshold_surrogates: int = 200
    threshold_percentile: float = 95.0

    @property
    def wcc_window_samples(self) -> int:
        return max(2, int(round(self.wcc_window_sec * self.hz)))

    @property
    def n_cells(self) -> int:
        return len(self.noise_ratios) * len(self.couplings) * len(self.seeds)


# ---------------------------------------------------------------------------
# Single-cell surrogate test
# ---------------------------------------------------------------------------

def _surrogate_cell(
    noise_ratio: float,
    coupling: float,
    seed: int,
    cfg: Level3Config,
) -> Dict[str, float]:
    """
    Run surrogate test for one (noise_ratio, coupling, seed) cell.

    Generates the synthetic dyad, computes the 6 observed features,
    then builds an empirical null distribution by surrogate‑sampling
    ``person_a`` and ``person_b`` independently and recomputing features
    on each surrogate WCC.

    Returns
    -------
    dict
        Row with experimental knobs + observed features + per‑feature
        p‑values + count‑based definedness p‑values + per‑feature
        null‑distribution means/SDs.
    """
    ds = generate_ground_truth_dyad(
        lead_modality="behavior",
        lag_modality="neural",
        true_lag_sec=cfg.true_lag_sec,
        noise_ratio=noise_ratio,
        duration_sec=cfg.duration_sec,
        hz=cfg.hz,
        seed=seed,
        n_bursts=cfg.n_bursts,
        burst_sigma=cfg.burst_sigma,
        gap_prob=cfg.gap_prob,
        morphology=cfg.morphology,
        coupling=coupling,
    )
    df_lead = ds.modalities["behavior"]
    a = df_lead["person_a"].to_numpy()
    b = df_lead["person_b"].to_numpy()

    # ------------------------------------------------------------------
    # Compute onset threshold (per-dyad surrogate-derived or fixed)
    # ------------------------------------------------------------------
    if cfg.use_surrogate_threshold:
        onset_threshold = compute_dyad_surrogate_threshold(
            a, b,
            hz=cfg.hz,
            wcc_window_samples=cfg.wcc_window_samples,
            n_surrogates=cfg.n_threshold_surrogates,
            percentile=cfg.threshold_percentile,
            seed=seed,
        )
    else:
        onset_threshold = cfg.onset_threshold

    # ------------------------------------------------------------------
    # Observed features
    # ------------------------------------------------------------------
    wcc_obs = sliding_window_wcc(
        a, b,
        window_size=cfg.wcc_window_samples,
        hz=cfg.hz,
    )
    obs = _extract_six_features(
        wcc_obs,
        hz=cfg.hz,
        onset_threshold=onset_threshold,
        wcc_window_sec=cfg.wcc_window_sec,
    )

    # ------------------------------------------------------------------
    # Choose surrogate generator
    # ------------------------------------------------------------------
    if cfg.surrogate_method in ("ft", "prtf"):
        def _gen(z: np.ndarray, r: np.random.Generator) -> np.ndarray:
            return ft_surrogate(z, r)
    elif cfg.surrogate_method == "iaaft":
        def _gen(z: np.ndarray, r: np.random.Generator) -> np.ndarray:
            return iaaft_surrogate(
                z, r,
                max_iter=cfg.iaaft_max_iter,
                tol=cfg.iaaft_tol,
            )
    else:
        raise ValueError(f"Unknown surrogate_method: {cfg.surrogate_method!r}")

    # ------------------------------------------------------------------
    # Build empirical null — TIERED by mathematical invariance
    # ------------------------------------------------------------------

    # Feature lists by tier
    _L0_FEATURES = {"mean_synchrony", "peak_amplitude",
                   "synchrony_entropy", "bimodality_coefficient"}
    _L1_FEATURES = {"dwell_time", "switching_rate"}
    
    fdr_feature_names = list(FEATURE_TAILS.keys())
    ref_feature_names = list(REFERENCE_TAILS.keys())
    all_feature_names = fdr_feature_names + ref_feature_names
    
    # Null value collectors (separate for L0 and L1)
    null_vals_l0: Dict[str, List[float]] = {f: [] for f in all_feature_names if f in _L0_FEATURES or f in ref_feature_names}
    null_vals_l1: Dict[str, List[float]] = {f: [] for f in all_feature_names if f in _L1_FEATURES}
    null_defined_count: Dict[str, int] = {
        "onset_defined": 0,
        "recovery_defined": 0,
    }

    rng = np.random.default_rng(seed + 10_000)

    for _ in range(cfg.n_surrogates):
        # --- L0 null: signal-level surrogate ---
        a_surr = _gen(a, rng)
        b_surr = _gen(b, rng)
        wcc_l0 = sliding_window_wcc(
            a_surr, b_surr,
            window_size=cfg.wcc_window_samples,
            hz=cfg.hz,
        )
        feats_l0 = _extract_six_features(
            wcc_l0,
            hz=cfg.hz,
            onset_threshold=onset_threshold,
            wcc_window_sec=cfg.wcc_window_sec,
        )
        for f in null_vals_l0:
            null_vals_l0[f].append(feats_l0[f])
        
        # --- L1 null: WCC-level surrogate (preserve L0 moments) ---
        # Use IAAFT on the OBSERVED WCC to generate null WCC that preserves
        # mean, peak, distribution shape (L0 moments), only shuffling time structure.
        wcc_l1 = iaaft_surrogate(wcc_obs, rng)
        feats_l1 = _extract_six_features(
            wcc_l1,
            hz=cfg.hz,
            onset_threshold=onset_threshold,
            wcc_window_sec=cfg.wcc_window_sec,
        )
        for f in null_vals_l1:
            null_vals_l1[f].append(feats_l1[f])
        
        # Definedness count (use L0 null, since onset/recovery are L2 features)
        null_defined_count["onset_defined"] += int(np.isfinite(feats_l0.get("onset_latency", np.nan)))
        null_defined_count["recovery_defined"] += int(np.isfinite(feats_l0.get("recovery_time", np.nan)))

    # Merge null values
    null_vals: Dict[str, List[float]] = {}
    null_vals.update(null_vals_l0)
    null_vals.update(null_vals_l1)

    # ------------------------------------------------------------------
    # Value‑based p‑values (per feature, Phipson-Smyth)
    # ------------------------------------------------------------------
    # FDR family uses FEATURE_TAILS, Reference uses REFERENCE_TAILS.
    tails: Dict[str, str] = {**FEATURE_TAILS, **REFERENCE_TAILS}
    p_values: Dict[str, float] = {}
    null_means: Dict[str, float] = {}
    null_sds: Dict[str, float] = {}
    for f in all_feature_names:
        arr = np.array(null_vals[f], dtype=float)
        p_values[f] = phipson_smyth_p(
            obs[f], arr, tail=tails[f],
        )
        clean = arr[np.isfinite(arr)]
        null_means[f] = float(np.mean(clean)) if clean.size > 0 else float("nan")
        null_sds[f] = float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan")

    # ------------------------------------------------------------------
    # Count‑based definedness p‑values
    # ------------------------------------------------------------------
    # Under H0, the per‑seed P(defined) is approximately
    #   null_defined_count / N.
    # If the observed seed has the feature defined, we ask:
    #   "What is P(>= observed_defined_seeds | H0)?"
    p_def: Dict[str, float] = {}
    _DEFINED_TO_FEATURE = {"onset_defined": "onset_latency", "recovery_defined": "recovery_time"}
    for key in ("onset_defined", "recovery_defined"):
        n_def = null_defined_count[key]
        obs_key = _DEFINED_TO_FEATURE[key]
        if np.isfinite(obs.get(obs_key, np.nan)):
            # observed = defined → upper‑tailed: how extreme is the observed count?
            p_def[key] = (1.0 + n_def) / (1.0 + cfg.n_surrogates)
        else:
            p_def[key] = 1.0  # observed undefined → no claim of significance

    # ------------------------------------------------------------------
    # Assemble output row
    # ------------------------------------------------------------------
    row: Dict[str, float] = {
        "noise_ratio": noise_ratio,
        "coupling": coupling,
        "seed": seed,
        "onset_threshold": cfg.onset_threshold,
        "n_surrogates": cfg.n_surrogates,
        "surrogate_method": cfg.surrogate_method,
    }
    for f in all_feature_names:
        row[f"obs_{f}"] = obs[f]
        row[f"p_{f}"] = p_values[f]
        row[f"null_mean_{f}"] = null_means[f]
        row[f"null_sd_{f}"] = null_sds[f]
    row["p_onset_defined"] = p_def["onset_defined"]
    row["p_recovery_defined"] = p_def["recovery_defined"]

    # Also store raw definedness for downstream FDR
    row["obs_onset_defined"] = 1.0 if np.isfinite(obs.get("onset_latency", np.nan)) else 0.0
    row["obs_recovery_defined"] = 1.0 if np.isfinite(obs.get("recovery_time", np.nan)) else 0.0

    return row


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_level3_grid(
    cfg: Optional[Level3Config] = None,
) -> pd.DataFrame:
    """
    Run the full Level 3 (noise_ratio, coupling, seed) grid.

    Returns one row per cell containing observed features, surrogate
    null statistics, and Phipson‑Smyth raw p‑values.  FDR correction
    is applied downstream by ``apply_bh_fdr_within_noise``.

    Parameters
    ----------
    cfg : Level3Config, optional
        Configuration object.  Uses defaults if ``None``.

    Returns
    -------
    pd.DataFrame
        Raw results (one row per experimental cell).
    """
    cfg = cfg or Level3Config()
    rows: List[dict] = []
    for noise_ratio in cfg.noise_ratios:
            for coupling in cfg.couplings:
                for seed in cfg.seeds:
                    rows.append(
                        _surrogate_cell(
                            float(noise_ratio), float(coupling),
                            int(seed), cfg,
                        )
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FDR correction within noise_ratio families
# ---------------------------------------------------------------------------

def apply_bh_fdr_within_noise(
    df: pd.DataFrame,
    feature_p_columns: Optional[Sequence[str]] = None,
    q: float = 0.05,
) -> pd.DataFrame:
    """
    Apply BH‑FDR per ``(noise_ratio, seed)`` group, treating the 5 FEATURE_TAILS features as one family.

    Adds boolean columns ``reject_<feature>`` and a summary ``n_reject``.
    The family definition ("within noise_ratio") follows the rationale that
    reviewers care about "does the method work at a given noise level?",
    not "does it work across all noise levels simultaneously?".

    Parameters
    ----------
    df : pd.DataFrame
        Raw results from ``run_level3_grid``.
    feature_p_columns : sequence of str, optional
        p‑value column names.  Defaults to ``FEATURE_P_COLUMNS``.
    q : float
        Target FDR level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with rejection columns appended.
    """
    if feature_p_columns is None:
        feature_p_columns = FEATURE_P_COLUMNS
    out = df.copy()
    reject_cols = [c.replace("p_", "reject_") for c in feature_p_columns]
    for c in reject_cols:
        out[c] = False

    for idx, row in out.iterrows():
        p_arr = np.array([row[c] for c in feature_p_columns], dtype=float)
        rej = bh_fdr(p_arr, q=q)
        for c, r in zip(reject_cols, rej):
            out.at[idx, c] = bool(r)

    out["n_reject"] = out[reject_cols].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_level3(
    df_with_fdr: pd.DataFrame,
    feature_p_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Per ``(noise_ratio, coupling)`` summary of rejection rates.

    For Level 3a (``coupling=0``), each ``reject_*`` rate is the
    empirical false positive rate.  The acceptance criterion is
    **FPR ≤ q** (e.g. 0.05).

    For Level 3b (``coupling=0.3``) it becomes **statistical power**.

    Parameters
    ----------
    df_with_fdr : pd.DataFrame
        Results with FDR columns added by ``apply_bh_fdr_within_noise``.
    feature_p_columns : sequence of str, optional
        p‑value column names (used to derive reject column names).

    Returns
    -------
    pd.DataFrame
        One row per ``(noise_ratio, coupling)`` with rejection rates.
    """
    if feature_p_columns is None:
        feature_p_columns = FEATURE_P_COLUMNS
    reject_cols = [c.replace("p_", "reject_") for c in feature_p_columns]

    agg_dict = {f"{c}_rate": (c, "mean") for c in reject_cols}
    agg_dict["n_seeds"] = ("seed", "count")

    grouped = df_with_fdr.groupby(["noise_ratio", "coupling"], as_index=False)
    summary = grouped.agg(**agg_dict)
    return summary


# ---------------------------------------------------------------------------
# Per-dyad surrogate-derived threshold (DECISION-01 revised 2026-06-21)
# ---------------------------------------------------------------------------

def compute_dyad_surrogate_threshold(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    *,
    hz: float,
    wcc_window_samples: int,
    n_surrogates: int = 200,
    percentile: float = 95.0,
    seed: int = 0,
    surrogate_method: str = "iaaft",
) -> float:
    """Compute a per-dyad IAAFT-based surrogate threshold for WCC binarisation.

    Runs ``n_surrogates`` IAAFT replicates of the two raw signals, computes
    WCC for each surrogate pair, pools all finite WCC values, and returns the
    ``percentile``-th quantile.  The result is the WCC level this dyad would
    reach by chance at the chosen false-positive rate.

    Delegates to :func:`feature_definitions.compute_surrogate_threshold` for
    the quantile step.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Raw physiological signals (finite, same length).
    hz : float
        Sampling rate of ``sig_a`` / ``sig_b`` (Hz).
    wcc_window_samples : int
        WCC window length in samples.
    n_surrogates : int, optional
        Number of IAAFT replicates (default 200).
    percentile : float, optional
        Quantile for the threshold (default 95).
    seed : int, optional
        RNG seed for reproducibility.
    surrogate_method : str, optional
        "iaaft" (default) or "ft".

    Returns
    -------
    float
        Surrogate-derived threshold.  Falls back to 0.5 if fewer than 10
        finite surrogate WCC values are available.

    Notes
    -----
    Session-level use: pass the full-session ``sig_a`` / ``sig_b`` to obtain
    a single threshold shared across conditions (Task A comparability in
    docs/METHOD_LOG.md).
    Condition-level use: pass the condition-specific signal slices.
    """
    from ..dynamic_features import sliding_window_wcc
    from ..feature_definitions import compute_surrogate_threshold

    rng = np.random.default_rng(seed)
    surrogate_wccs: list[np.ndarray] = []

    _gen = iaaft_surrogate if surrogate_method == "iaaft" else ft_surrogate

    for _ in range(n_surrogates):
        a_surr = _gen(sig_a, rng)
        b_surr = _gen(sig_b, rng)
        wcc_s = sliding_window_wcc(
            a_surr, b_surr,
            window_size=wcc_window_samples,
            hz=hz,
        )
        surrogate_wccs.append(wcc_s)

    surrogate_matrix = np.vstack(surrogate_wccs)  # (n_surrogates, n_timepoints)
    return compute_surrogate_threshold(surrogate_matrix, percentile=percentile)

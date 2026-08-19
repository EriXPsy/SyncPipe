"""
Pipeline 3: Inference pipeline.

Purpose: audited statistical evidence chain for WCC-derived synchrony
measurement.

The recommended v1 public workflow is:
1. synchrony-existence audit (signal-level IAAFT);
2. design-control audit (pseudo-pair, time-shift, and when applicable
   across-stimulus shuffle);
3. group condition inference (dyad-paired permutation + BH-FDR).

The older L0/L1/L2 method names remain for backward compatibility, but their
results must be interpreted as audits of specific null hypotheses, not as proof
of dyad-specific interpersonal coupling or psychological mechanism.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .design_controls import (
    DEFAULT_AUDIT_FEATURES,
    SignalPair,
    design_control_audit,
    synchrony_existence_audit,
)
from .dynamic_features import sliding_window_wcc, wcc_surrogate_test
from .feature_definitions import (
    EXISTENCE_GATE_ALPHA,
    FDR_FAMILIES,
    FDR_FEATURES,
    ONSET_THRESHOLD,
    PRIMARY_EXISTENCE_ENDPOINT,
    PRIMARY_EXISTENCE_MODALITIES,
    PRIMARY_FDR_FAMILY,
    REFERENCE_FEATURE,
    extract_features,
    get_fdr_features,
)
from .validation.across_stim_shuffle import across_stim_shuffle_test
from .validation.l2_between_condition import (
    between_condition_fdr,
    between_condition_by_modality,
    _bh_fdr,
)

UNSPECIFIED_MODALITY = "__unspecified__"
"""Key used for L2 results whose modality cannot be honestly named.

L2 results are ALWAYS keyed by modality (P0-2: pooling across modalities can
cancel real effects, so the modality is the only admissible index unit for a
between-condition claim). When the feature table carries no usable modality
column — or carries one with missing values, so we cannot assert that every row
belongs to the single observed modality — we label the result with this explicit
sentinel rather than inventing a modality name.
"""


def _apply_global_modality_fdr(results: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    """Apply BH within each SSoT FDR family, pooled across modalities.

    L0 (signal-level IAAFT null) and L1 (WCC-level IAAFT null) are different
    null models and must NOT share one BH denominator. We therefore group the
    modality × feature hypotheses by their SSoT family (Axis D of
    feature_definitions) and run BH *within* each family, pooling across
    modalities so a family tested on M modalities controls the joint
    family-wise error at M hypotheses (not M separate chances). Reference
    features are reported (p_raw) but never enter any BH denominator.
    """
    family_of: Dict[str, str] = {
        feat: fam for fam, feats in FDR_FAMILIES.items() for feat in feats
    }
    reference_set = set(REFERENCE_FEATURE)

    # Collect (modality, result) per family; reference features excluded.
    by_family: Dict[str, List[tuple]] = {}
    all_items = []
    for modality, payload in results.items():
        if isinstance(payload, dict) and "per_feature" in payload:
            for result in payload["per_feature"]:
                all_items.append((modality, result))
                if result.feature in reference_set:
                    continue
                fam = family_of.get(result.feature, result.feature)
                by_family.setdefault(fam, []).append((modality, result))
    if not all_items:
        return results

    # BH within each family, pooled across that family's modalities.
    for fam, items in by_family.items():
        adjusted = _bh_fdr(np.asarray([r.p_raw for _, r in items], dtype=float))
        for (_, result), p_adj in zip(items, adjusted):
            result.p_fdr = float(p_adj) if np.isfinite(p_adj) else 1.0
            result.significant_05 = bool(result.p_fdr < alpha and result.claimable)

    # Reference features: report p_raw, never corrected, never significant.
    for _, result in all_items:
        if result.feature in reference_set:
            result.p_fdr = float("nan")
            result.significant_05 = False

    for payload in results.values():
        if isinstance(payload, dict) and "per_feature" in payload:
            payload["fdr_scope"] = "family_pooled_across_modality"
            payload["fdr_family_size"] = {
                fam: len(items) for fam, items in by_family.items()
            }
            payload["summary_df"] = pd.DataFrame([
                {
                    "feature": r.feature, "observed_diff": r.observed_diff,
                    "null_mean": r.null_mean, "null_sd": r.null_sd,
                    "p_raw": r.p_raw, "p_fdr": r.p_fdr,
                    "significant_05": r.significant_05,
                    "perm_effect_size": r.perm_effect_size,
                    "difference_q25": r.difference_q25,
                    "difference_q75": r.difference_q75,
                    "median_ci_low": r.median_ci_low,
                    "median_ci_high": r.median_ci_high,
                    "median_ci_confidence": r.median_ci_confidence,
                    "median_ci_bounded": r.median_ci_bounded,
                    "median_ci_method": r.median_ci_method,
                    "permutation_method": r.permutation_method,
                    "n_null_draws": r.n_null_draws,
                    "min_attainable_p": r.min_attainable_p,
                    "approx_monte_carlo_se": r.approx_monte_carlo_se,
                    "n_dyads": r.n_dyads, "defined_a": r.defined_a,
                    "defined_b": r.defined_b, "p_definedness": r.p_definedness,
                    "definedness_status": r.definedness_status,
                    "claimable": r.claimable,
                }
                for r in payload["per_feature"]
            ])
    return results


def _existence_audit_one(task: Tuple[Any, ...]) -> Tuple[str, Dict[str, Any]]:
    """Run one pair's existence audit. Module-level so it is picklable.

    Defined at module scope (not as a closure or lambda) because
    ProcessPoolExecutor pickles the callable by qualified name; a nested
    function would raise PicklingError on the Windows spawn start method.
    """
    (label, sig_a, sig_b, hz, window_size, surrogate_n, seed,
     window_type, dm) = task
    return label, synchrony_existence_audit(
        sig_a,
        sig_b,
        hz=hz,
        window_size=window_size,
        surrogate_n=surrogate_n,
        seed=seed,
        window_type=window_type,
        discontinuity_mask=dm,
    )


def _modality_from_label(label: str) -> str:
    """Extract the modality token from a raw_signals label.

    Labels are ``"<dyad>__<modality>"`` or ``"<dyad>__<modality>__<condition>"``
    (double-underscore separated). The modality is the second segment. Labels
    without a separator carry no modality and return "" so they group under a
    single unnamed bucket rather than crashing the gate.
    """
    parts = str(label).split("__")
    return parts[1] if len(parts) >= 2 else ""


def _existence_gate_by_modality(
    existence_results: Dict[str, Any],
    primary_modalities: Sequence[str],
    alpha: float = EXISTENCE_GATE_ALPHA,
) -> Dict[str, Any]:
    """Second-order group existence test per modality.

    The observed group statistic for a modality is the mean over dyads of the
    per-dyad observed ``PRIMARY_EXISTENCE_ENDPOINT`` (peak_amplitude). The
    null distribution is built by draw-wise aggregation of the per-dyad
    signal-level IAAFT surrogate peaks: null draw i = the mean, across dyads,
    of each dyad's i-th surrogate peak (NaN-masked). This treats dyads as a
    random effect — each dyad's autocorrelation is preserved while cross-signal
    coupling is destroyed, so between-dyad heterogeneity enters the null
    rather than being averaged away. A two-tailed Phipson-Smyth p-value is
    computed per modality. Primary modalities are BH-corrected (m = number of
    primary modalities) so the any-of-k gate controls its family-wise error;
    the gate passes if at least one primary modality is significant after
    correction. Per-dyad descriptive statistics (fraction significant) are
    reported alongside but do not decide the gate.
    """
    obs_peaks: Dict[str, List[float]] = {}
    null_peaks: Dict[str, List[np.ndarray]] = {}
    frac_sig: Dict[str, List[bool]] = {}
    for label, r in existence_results.items():
        if not isinstance(r, dict):
            continue
        mod = _modality_from_label(label)
        ov = r.get("observed", {}).get(PRIMARY_EXISTENCE_ENDPOINT, np.nan)
        if not np.isfinite(ov):
            ov = r.get("obs_peak_amplitude", np.nan)
        if not np.isfinite(ov):
            continue
        obs_peaks.setdefault(mod, []).append(float(ov))
        null_arr = np.asarray(r.get("null_peak_amplitude", []), dtype=float)
        if null_arr.size:
            null_peaks.setdefault(mod, []).append(null_arr)
        frac_sig.setdefault(mod, []).append(
            bool(r.get("per_feature_significant", {}).get(PRIMARY_EXISTENCE_ENDPOINT, False))
        )

    per_modality: Dict[str, Dict[str, Any]] = {}
    for mod in sorted(obs_peaks):
        obs_vec = np.asarray(obs_peaks[mod], dtype=float)
        obs_mean = float(np.nanmean(obs_vec))
        p_group = float("nan")
        p_min_attainable = float("nan")
        p_monte_carlo_se = float("nan")
        n_null_draws = 0
        stack = null_peaks.get(mod, [])
        if stack:
            width = min(a.size for a in stack)
            mat = np.vstack([a[:width] for a in stack])  # (n_dyads, width)
            group_null = np.nanmean(mat, axis=0)
            finite = group_null[np.isfinite(group_null)]
            n_null_draws = int(finite.size)
            if finite.size:
                p_ge = (np.sum(finite >= obs_mean) + 1) / (finite.size + 1)
                p_le = (np.sum(finite <= obs_mean) + 1) / (finite.size + 1)
                q_tail = float(min(p_ge, p_le))
                p_group = float(min(1.0, 2.0 * q_tail))
                p_min_attainable = float(min(1.0, 2.0 / (finite.size + 1)))
                p_monte_carlo_se = float(
                    2.0 * np.sqrt(q_tail * (1.0 - q_tail) / finite.size)
                )
        n_dyads = int(obs_vec.size)
        sig = frac_sig.get(mod, [])
        per_modality[mod] = {
            "n_dyads": n_dyads,
            "group_observed_mean": obs_mean,
            "n_null_draws": n_null_draws,
            "p_group": p_group,
            "min_attainable_two_sided_p": p_min_attainable,
            "approx_monte_carlo_se": p_monte_carlo_se,
            # Descriptive only: fraction of dyads whose per-dyad test passed.
            "frac_dyads_significant": float(np.mean(sig)) if sig else float("nan"),
            "is_primary": mod in set(primary_modalities),
        }

    # BH-FDR across the primary modalities controls the any-of-k gate.
    prim_mods = [
        m for m in per_modality
        if per_modality[m]["is_primary"] and np.isfinite(per_modality[m]["p_group"])
    ]
    primary_pass = False
    if prim_mods:
        adjusted = _bh_fdr(
            np.asarray([per_modality[m]["p_group"] for m in prim_mods], dtype=float)
        )
        for m, p_adj in zip(prim_mods, adjusted):
            per_modality[m]["p_fdr"] = float(p_adj)
            per_modality[m]["supports"] = bool(p_adj < alpha)
        primary_pass = any(per_modality[m]["supports"] for m in prim_mods)

    return {
        "primary_pass": primary_pass,
        "per_modality": per_modality,
        "primary_modalities": list(primary_modalities),
        "alpha": float(alpha),
        "endpoint": PRIMARY_EXISTENCE_ENDPOINT,
        "test": "second_order_group_surrogate",
    }


class InferencePipeline:
    """Audited inference pipeline for WCC-derived synchrony descriptors.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with one row per observation, containing feature columns
        and metadata (dyad_id, condition, modality, etc.).
    hz : float
        Signal sampling rate (Hz).
    wcc_window_sec : float
        WCC window duration in seconds.
    surrogate_n : int
        Number of surrogate iterations for L0/L1 tests. Default 100.
    seed : int
        Random seed for reproducibility.
    n_workers : int
        Number of worker processes for the per-pair existence audit. Default 1
        (serial). Values > 1 distribute *pairs* across processes; results are
        bit-identical to serial because each pair's audit seeds its own
        Generator from ``seed`` and shares no RNG state with any other pair
        (see :meth:`run_synchrony_existence_audit`).

    Examples
    --------
    >>> pipe = InferencePipeline(df, hz=4.0, wcc_window_sec=10.0)
    >>> # Per-observation tests (legacy L0/L1/L2 API):
    >>> l0 = pipe.test_l0_signal(wcc, (sig_a, sig_b), wcc_window_size=40)
    >>> l1 = pipe.test_l1_structure(wcc, label="dyad_01")
    >>> l2 = pipe.test_l2_condition(condition_col="condition", dyad_col="dyad_id")
    >>> # Or run the full v1 evidence chain in three steps:
    >>> pipe.run_synchrony_existence_audit(raw_signals, wcc_window_size=40)
    >>> pipe.run_design_control_audit(signal_pairs, wcc_window_size=40)
    >>> pipe.run_group_condition_inference(condition_col="condition", dyad_col="dyad_id")
    >>> report = pipe.summarize()
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        hz: float = 4.0,
        wcc_window_sec: Optional[float] = None,
        surrogate_n: int = 100,
        seed: int = 42,
        n_workers: int = 1,
    ):
        if int(n_workers) < 1:
            raise ValueError("n_workers must be >= 1")
        self.df = features_df.copy()
        self.hz = hz
        self.wcc_window_sec = wcc_window_sec
        self.surrogate_n = surrogate_n
        self.seed = seed
        self.n_workers = int(n_workers)

        self._l0_results: Dict[str, Any] = {}
        self._l1_results: Dict[str, Any] = {}
        self._l2_results: Optional[Dict[str, Any]] = None

        # v1 audited evidence-chain results.  These are the recommended
        # public API going forward; the older L0/L1/L2 methods remain for
        # backward compatibility and regression tests.
        self._synchrony_existence_results: Dict[str, Any] = {}
        self._design_control_results: Optional[Dict[str, Any]] = None
        self._across_stim_results: Optional[Dict[str, Any]] = None
        self._group_inference_results: Optional[Dict[str, Any]] = None

    # ---- v1 evidence chain: synchrony-existence → design controls → group inference ----

    def run_synchrony_existence_audit(
        self,
        raw_signals: Dict[str, SignalPair],
        *,
        wcc_window_size: int,
        labels: Optional[Sequence[str]] = None,
        window_type: str = "rect",
        discontinuity_mask: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Step 1: test whether each pair shows synchrony above signal-level null.

        This is a synchrony-existence audit, not proof of dyad-specific
        interpersonal coupling.  Shared-stimulus and co-presence alternatives
        are evaluated in :meth:`run_design_control_audit` and
        :meth:`run_across_stimulus_shuffle_audit`.

        Parameters
        ----------
        raw_signals : dict
            Mapping from observation label -> (sig_a, sig_b).
        discontinuity_mask : dict or None
            Optional mapping from the same labels -> per-sample boundary mask
            (signal-resolution). When provided, each pair's existence audit
            gates out coupling windows straddling a segment seam.

        Notes
        -----
        With ``n_workers > 1`` the *pairs* are distributed across processes.
        This is bit-exact rather than merely reproducible: every pair calls
        ``synchrony_existence_audit(..., seed=self.seed)``, which builds its own
        ``default_rng(seed)`` inside ``_signal_level_surrogate_test``, so no RNG
        state is carried from one pair to the next and completion order cannot
        enter any number. Results are reassembled in ``selected`` order, so the
        returned dict has the same key order as the serial path too.

        The pair is the right parallel granularity: surrogate generation is
        argsort-bound and does not release the GIL (threads capped at 1.39x and
        degraded with more of them), while a per-surrogate process pool loses to
        Windows spawn overhead (~1s vs 1.5-2.6s of work per pair). One pair is a
        large enough unit of work to amortise spawn.
        """
        selected = list(labels) if labels is not None else list(raw_signals.keys())
        tasks = []
        for label in selected:
            if label not in raw_signals:
                continue
            sig_a, sig_b = raw_signals[label]
            dm = (
                discontinuity_mask.get(label)
                if discontinuity_mask is not None
                else None
            )
            tasks.append((
                label, sig_a, sig_b, self.hz, wcc_window_size,
                self.surrogate_n, self.seed, window_type, dm,
            ))

        if self.n_workers > 1 and len(tasks) > 1:
            n = min(self.n_workers, len(tasks))
            with ProcessPoolExecutor(max_workers=n) as pool:
                # `map` preserves input order, so `results` is keyed in
                # `selected` order regardless of which worker finishes first.
                completed = list(pool.map(_existence_audit_one, tasks))
        else:
            completed = [_existence_audit_one(t) for t in tasks]

        results: Dict[str, Any] = {label: res for label, res in completed}
        self._synchrony_existence_results = results
        return {
            "step": "synchrony_existence_audit",
            "null_model": "signal_level_iaaft",
            "n_pairs": len(results),
            "results": results,
            "interpretation": (
                "Tests whether aligned WCC features exceed independent "
                "autocorrelated signals. Necessary but not sufficient for "
                "dyad-specific interpersonal coupling."
            ),
        }

    def run_design_control_audit(
        self,
        signal_pairs: Dict[str, SignalPair],
        *,
        wcc_window_size: int,
        feature_names: Sequence[str] = DEFAULT_AUDIT_FEATURES,
        n_pseudo_per_dyad: int = 10,
        shift_lags_sec: Sequence[float] = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0),
        window_type: str = "rect",
        threshold: Any = 0.5,
        discontinuity_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Step 2a: run pseudo-pair and time-shift design controls.

        Pseudo-pair controls ask whether real partners exceed mismatched
        partners.  Time-shift controls ask whether the effect depends on the
        original temporal alignment.  These are formal API methods, not just
        dataset-specific scripts.
        For publication, keep ``n_pseudo_per_dyad`` at >= 10 for stable
        null distributions; the default is 10. Reduce to 3 for quick demos only.
        """
        result = design_control_audit(
            signal_pairs,
            hz=self.hz,
            window_size=wcc_window_size,
            feature_names=feature_names,
            n_pseudo_per_dyad=n_pseudo_per_dyad,
            shift_lags_sec=shift_lags_sec,
            seed=self.seed,
            window_type=window_type,
            threshold=threshold,
            discontinuity_masks=discontinuity_masks,
        )
        self._design_control_results = result
        return result

    def run_across_stimulus_shuffle_audit(
        self,
        segments: List[Tuple[str, np.ndarray, np.ndarray]],
        *,
        wcc_window_size: int,
        feature_names: Sequence[str] = DEFAULT_AUDIT_FEATURES,
        n_shuffles: Optional[int] = None,
        window_type: str = "rect",
    ) -> Dict[str, Any]:
        """Step 2b: run across-stimulus shuffle for segmented stimulus designs.

        Use this when both partners experienced the same ordered stimulus
        sequence (e.g., video clips, repeated trials).  It permutes stimulus
        segments independently across partners, making shared-stimulus timing
        auditable.  It is not appropriate for unsegmented free interaction.
        """
        window_sec = self.wcc_window_sec or (wcc_window_size / self.hz)

        def _wcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return sliding_window_wcc(a, b, window_size=wcc_window_size, hz=self.hz, window_type=window_type)

        def _features(wcc: np.ndarray) -> Dict[str, float]:
            feats = extract_features(wcc, hz=self.hz, wcc_window_sec=window_sec)
            return {name: float(getattr(feats, name, np.nan)) for name in feature_names}

        result = across_stim_shuffle_test(
            segments,
            wcc_func=_wcc,
            feature_func=_features,
            n_surr=self.surrogate_n if n_shuffles is None else n_shuffles,
            seed=self.seed,
            feature_names=list(feature_names),
        )
        payload = {
            "step": "across_stimulus_shuffle_audit",
            "n_segments": len(segments),
            "results": result,
            "interpretation": (
                "For segmented shared-stimulus designs, tests whether observed "
                "features exceed a null that breaks shared stimulus order."
            ),
        }
        self._across_stim_results = payload
        return payload

    def run_group_condition_inference(
        self,
        condition_col: str = "condition",
        dyad_col: str = "dyad_id",
        feature_cols: Optional[List[str]] = None,
        fdr_alpha: float = 0.05,
        n_permutations: int = 10000,
        contrast: Optional[Tuple[str, str]] = None,
        full_family_fdr: bool = False,
        threshold_scope: str = "unknown",
        modality_col: str = "modality",
        fdr_scope: str = "global",
        undefined_policy: str = "gate",
        observation_policy: str = "warn",
        eligibility_policy: str = "warn",
        n_min_dyads: int = 10,
    ) -> Dict[str, Any]:
        """Step 3: test whether features differentiate experimental conditions.

        L2 is ALWAYS computed per modality and the return value is ALWAYS
        ``{modality: l2_result_dict}``, including when the dataset carries a
        single modality (then the dict has exactly one entry). Rationale: pooled
        L2 across modalities averages cross-modality rows and can cancel real
        effects (P0-2), so the modality is the only admissible index unit for a
        between-condition claim. A single-modality dataset is not a different
        kind of result — it is the same result with M=1.

        The key is the real modality label when one can be read from
        ``modality_col``; if that column is absent, or has any missing value (so
        we cannot assert every row belongs to the one observed modality), the key
        is ``UNSPECIFIED_MODALITY`` rather than an invented modality name.

        Returns
        -------
        dict
            ``{modality: l2_result_dict}``. Each value matches the return format
            of ``between_condition_fdr``, or is ``{"error": str}`` if that
            modality could not be tested.
        """
        has_mod_col = modality_col in self.df.columns
        mod_values = (
            self.df[modality_col].dropna().unique() if has_mod_col else []
        )
        fully_labelled = (
            has_mod_col
            and len(mod_values) > 0
            and bool(self.df[modality_col].notna().all())
        )
        if len(mod_values) > 1:
            # P1-R1: forward contrast / threshold_scope / seed — previously
            # dropped, so multimodal scientific path ignored caller contrast
            # and fell back to sorted condition labels.
            result = self.test_l2_by_modality(
                modality_col=modality_col,
                condition_col=condition_col,
                dyad_col=dyad_col,
                feature_cols=feature_cols,
                fdr_alpha=fdr_alpha,
                n_permutations=n_permutations,
                contrast=contrast,
                full_family_fdr=full_family_fdr,
                threshold_scope=threshold_scope,
                seed=self.seed,
                fdr_scope=fdr_scope,
                undefined_policy=undefined_policy,
                observation_policy=observation_policy,
                eligibility_policy=eligibility_policy,
                n_min_dyads=n_min_dyads,
            )
        else:
            # Single modality (or none nameable): still key the result so the
            # shape is invariant. Note the deliberate asymmetry with the M>1
            # branch: `between_condition_by_modality` traps a per-modality
            # ValueError so one untestable modality does not destroy the others,
            # but with M=1 there is nothing to protect, so an untestable design
            # must propagate (fail-loud) rather than become a silent
            # ``{"error": ...}`` payload that reads as "0 significant".
            key = str(mod_values[0]) if fully_labelled else UNSPECIFIED_MODALITY
            result = {
                key: self.test_l2_condition(
                    condition_col=condition_col,
                    dyad_col=dyad_col,
                    feature_cols=feature_cols,
                    fdr_alpha=fdr_alpha,
                    n_permutations=n_permutations,
                    contrast=contrast,
                    full_family_fdr=full_family_fdr,
                    threshold_scope=threshold_scope,
                    undefined_policy=undefined_policy,
                    observation_policy=observation_policy,
                    eligibility_policy=eligibility_policy,
                    n_min_dyads=n_min_dyads,
                )
            }
        self._group_inference_results = result
        return result

    def run_audited_evidence_chain(
        self,
        raw_signals: Dict[str, SignalPair],
        *,
        wcc_window_size: int,
        design_signal_pairs: Optional[Dict[str, SignalPair]] = None,
        across_stim_segments: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
        condition_col: str = "condition",
        dyad_col: str = "dyad_id",
        feature_cols: Optional[List[str]] = None,
        fdr_alpha: float = 0.05,
        n_permutations: int = 10000,
        window_type: str = "rect",
        full_family_fdr: bool = False,
        discontinuity_mask: Optional[Dict[str, np.ndarray]] = None,
        design_discontinuity_mask: Optional[Dict[str, np.ndarray]] = None,
        threshold_scope: str = "unknown",
        contrast: Optional[Tuple[str, str]] = None,
        modality_col: str = "modality",
        n_pseudo_per_dyad: int = 10,
        design_threshold: Any = 0.5,
        fdr_scope: str = "global",
        undefined_policy: str = "gate",
        observation_policy: str = "warn",
        eligibility_policy: str = "warn",
        n_min_dyads: int = 10,
        primary_modalities: Optional[Sequence[str]] = None,
        existence_alpha: float = EXISTENCE_GATE_ALPHA,
    ) -> Dict[str, Any]:
        """Run the recommended v1 evidence chain end-to-end.

        Chain:
        1. synchrony-existence audit (signal-level IAAFT)
        2. design-control audit (pseudo-pair/time-shift; optional across-stim)
        3. group condition inference (paired permutation + FDR)

        Parameters
        ----------
        discontinuity_mask : dict or None
            Optional label -> per-sample boundary mask (signal-resolution).
            Forwarded to the synchrony-existence audit so L0 gating respects
            segment seams (see ``discontinuity_mask`` on the audit for detail).
        """
        existence = self.run_synchrony_existence_audit(
            raw_signals, wcc_window_size=wcc_window_size, window_type=window_type,
            discontinuity_mask=discontinuity_mask,
        )
        design = None
        if design_signal_pairs is not None:
            design = self.run_design_control_audit(
                design_signal_pairs, wcc_window_size=wcc_window_size, window_type=window_type,
                n_pseudo_per_dyad=n_pseudo_per_dyad, threshold=design_threshold,
                discontinuity_masks=design_discontinuity_mask,
            )
        across = None
        if across_stim_segments is not None:
            across = self.run_across_stimulus_shuffle_audit(
                across_stim_segments, wcc_window_size=wcc_window_size, window_type=window_type,
            )
        group = self.run_group_condition_inference(
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            fdr_alpha=fdr_alpha,
            n_permutations=n_permutations,
            contrast=contrast,
            full_family_fdr=full_family_fdr,
            threshold_scope=threshold_scope,
            modality_col=modality_col,
            fdr_scope=fdr_scope,
            undefined_policy=undefined_policy,
            observation_policy=observation_policy,
            eligibility_policy=eligibility_policy,
            n_min_dyads=n_min_dyads,
        )
        existence_results = existence.get("results", {})
        prim_mods = (
            list(primary_modalities)
            if primary_modalities is not None
            else list(PRIMARY_EXISTENCE_MODALITIES)
        )
        gate = _existence_gate_by_modality(
            existence_results,
            primary_modalities=prim_mods,
            alpha=existence_alpha,
        )
        primary_pass = gate["primary_pass"]
        return {
            "evidence_chain_version": "v1",
            "synchrony_existence": existence,
            "existence_gate": gate,
            "design_controls": design,
            "across_stimulus_shuffle": across,
            "group_condition_inference": group,
            "stage_status": {
                "existence": "passed" if primary_pass else "not_supported",
                "design_controls": "completed" if design is not None else "not_run",
                "across_stimulus": "completed" if across is not None else "not_run",
                "group_inference": "completed" if group is not None else "not_run",
            },
            "claim_ceiling": (
                "Existence support is present but does not establish dyad-specific "
                "interpersonal coupling." if primary_pass else
                "Group differences are descriptive only because the primary "
                "existence audit was not supported."
            ),
            "summary": self._build_audited_chain_summary(existence, design, across, group),
        }

    @staticmethod
    def _build_audited_chain_summary(
        existence: Dict[str, Any],
        design: Optional[Dict[str, Any]],
        across: Optional[Dict[str, Any]],
        group: Optional[Dict[str, Any]],
    ) -> str:
        parts = [
            f"Synchrony-existence audit completed for {existence.get('n_pairs', 0)} pair(s)."
        ]
        if design is None:
            parts.append("Pseudo-pair/time-shift design controls were not run.")
        else:
            parts.append(
                f"Design controls completed for {design.get('n_dyads', 0)} dyad(s)."
            )
        if across is None:
            parts.append("Across-stimulus shuffle was not run.")
        else:
            parts.append(
                f"Across-stimulus shuffle completed for {across.get('n_segments', 0)} segment(s)."
            )
        if group is None:
            parts.append("Group condition inference was not run.")
        else:
            # 1c: `group` is ALWAYS {modality: l2_dict} (M=1 included), so there
            # is a single code path here — no shape sniffing, hence no way to
            # read a modality-keyed dict as if it were one pooled result and
            # silently report zero significant features.
            n_sig = 0
            mod_bits = []
            for mod, sub in sorted(group.items(), key=lambda kv: str(kv[0])):
                if not isinstance(sub, dict) or "error" in sub:
                    mod_bits.append(f"{mod}=error")
                    continue
                k = int(sub.get("n_significant", 0))
                n_sig += k
                mod_bits.append(f"{mod}:{k}")
            parts.append(
                "Group condition inference (per-modality) found "
                f"{n_sig} significant feature-test(s) "
                f"[{', '.join(mod_bits)}]."
            )
        parts.append(
            "Admissible claims per step — "
            "L0 (signal-level IAAFT): establishes only EXISTENCE of synchrony "
            "beyond independent autocorrelated surrogates, NOT dyad-specific "
            "interpersonal coupling. "
            "Design controls (pseudo-pair / time-shift / across-stimulus): "
            "pass/fail on dyad-specificity and alignment-dependence, with residual "
            "alternatives (shared stimulus, co-presence) still requiring "
            "domain-specific controls. "
            "L2 (group inference): only condition DIFFERENCES in audited "
            "descriptors, NOT direction, mechanism, or causality. "
            "Interpret all positive findings as audited evidence, not causal proof; "
            "shared-stimulus and co-presence alternatives require design-specific controls."
        )
        return " | ".join(parts)

    # ---- L0: signal-level existence test ---------------------------------

    def test_l0_signal(
        self,
        wcc: np.ndarray,
        raw_signals: Tuple[np.ndarray, np.ndarray],
        wcc_window_size: int,
        label: str = "",
        window_type: str = "rect",
        discontinuity_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run L0 signal-level IAAFT surrogate test.

        H0: The two signals are independent. Any observed WCC pattern
        is explainable by the auto-correlation structure of each signal alone.

        This is the most fundamental test — if a dyad fails L0, their
        synchrony cannot be distinguished from independent noise.

        Parameters
        ----------
        wcc : np.ndarray
            Observed WCC series.
        raw_signals : tuple of (sig_a, sig_b)
            Raw signal arrays for IAAFT shuffling.
        wcc_window_size : int
            WCC window size in samples (needed for correct recomputation).
        label : str
            Optional label for results tracking.
        discontinuity_mask : np.ndarray of bool or None
            Per-sample boundary mask (signal-resolution). Forwarded to the
            L0 signal-level null so recomputed surrogate WCC NaN out windows
            straddling a seam, matching the observed WCC gating.

        Returns
        -------
        dict with keys: surrogate_method, per_feature_significant, p_*.
        """
        result = wcc_surrogate_test(
            wcc,
            hz=self.hz,
            surrogate_n=self.surrogate_n,
            seed=self.seed,
            raw_signals=raw_signals,
            wcc_window_size=wcc_window_size,
            wcc_window_sec=self.wcc_window_sec,
            window_type=window_type,
            discontinuity_mask=discontinuity_mask,
        )
        result["label"] = label
        self._l0_results[label] = result
        return result

    # ---- L1: WCC-level temporal structure test ---------------------------

    def test_l1_structure(
        self,
        wcc: np.ndarray,
        label: str = "",
        null_model: str = "state_shuffle",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run L1 WCC-level surrogate test.

        H0: The WCC series has no temporal structure beyond its local
        autocorrelation and amplitude distribution.

        Defaults to 'state_shuffle' in v1.0 (revised from 'iaaft')
        to better preserve the exact dwell-time distribution while
        testing temporal organization.

        Parameters
        ----------
        wcc : np.ndarray
            Observed WCC series.
        label : str
            Optional label for results tracking.
        null_model : {"state_shuffle", "block_permutation", "iaaft"}
            L1 null model.
        threshold : float or None
            Threshold for state binarization. Defaults to ONSET_THRESHOLD.

        Returns
        -------
        dict with keys: surrogate_method, per_feature_significant, p_dwell_time, p_switching_rate.
        """
        result = wcc_surrogate_test(
            wcc,
            hz=self.hz,
            surrogate_n=self.surrogate_n,
            seed=self.seed,
            raw_signals=None,
            wcc_window_sec=self.wcc_window_sec,
            null_model=null_model,
            threshold=threshold if threshold is not None else ONSET_THRESHOLD,
        )
        result["label"] = label
        self._l1_results[label] = result
        return result

    # ---- L2: between-condition differentiation test ---------------------

    def test_l2_condition(
        self,
        condition_col: str = "condition",
        dyad_col: str = "dyad_id",
        feature_cols: Optional[List[str]] = None,
        fdr_alpha: float = 0.05,
        n_permutations: int = 10000,
        contrast: Optional[Tuple[str, str]] = None,
        full_family_fdr: bool = False,
        threshold_scope: str = "unknown",
        undefined_policy: str = "gate",
        observation_policy: str = "warn",
        eligibility_policy: str = "warn",
        n_min_dyads: int = 10,
    ) -> Dict[str, Any]:
        """Run L2 between-condition permutation test with BH-FDR correction.

        H0: Feature values come from the same distribution in both conditions.
        The observed condition difference is due to random assignment.

        This is the final tier — even if synchrony is real (L0) and structured
        (L1), it only matters scientifically if it differentiates conditions.

        Uses dyad-paired permutation: shuffles condition labels within each
        dyad to preserve the dyad-level correlation structure, then computes
        null distributions for the between-condition difference.

        Parameters
        ----------
        condition_col : str
            Column name for condition labels.
        dyad_col : str
            Column name for dyad/pair identifiers.
        feature_cols : list of str or None
            Features to test. Default: the FDR-family features (FDR_FEATURES).
            If ``feature_cols`` is None and ``full_family_fdr=True``, all 12
            implemented features enter a single BH-FDR step (strictly more
            conservative; reviewer-proof against "cherry-picking 3/12").
        fdr_alpha : float
            BH-FDR significance threshold (default 0.05).
        n_permutations : int
            Number of permutation iterations.
        contrast : tuple of (cond_a, cond_b) or None
            Specific contrast to test. If None, tests all pairwise.
        full_family_fdr : bool, default False
            When True and ``feature_cols`` is None, test the full 12-feature
            family under one BH-FDR correction.  Has no effect if
            ``feature_cols`` is supplied explicitly.

        Returns
        -------
        dict with per-feature p_raw, p_fdr, significant_05, effect_size.
        """
        if feature_cols is None:
            feature_cols = get_fdr_features(full_family_fdr)

        self._l2_results = between_condition_fdr(
            self.df,
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            alpha=fdr_alpha,            # was fdr_alpha= (wrong kwarg name)
            n_permutations=n_permutations,
            seed=self.seed,
            condition_values=contrast,  # was contrast= (wrong kwarg name)
            threshold_scope=threshold_scope,
            undefined_policy=undefined_policy,
            observation_policy=observation_policy,
            eligibility_policy=eligibility_policy,
            n_min_dyads=n_min_dyads,
        )
        return self._l2_results

    def test_l2_by_modality(
        self,
        modality_col: str = "modality",
        condition_col: str = "condition",
        dyad_col: str = "dyad_id",
        feature_cols: Optional[List[str]] = None,
        fdr_alpha: float = 0.05,
        n_permutations: int = 10000,
        window_type: str = "rect",
        full_family_fdr: bool = False,
        contrast: Optional[Tuple[str, str]] = None,
        threshold_scope: str = "unknown",
        seed: Optional[int] = None,
        fdr_scope: str = "global",
        undefined_policy: str = "gate",
        observation_policy: str = "warn",
        eligibility_policy: str = "warn",
        n_min_dyads: int = 10,
    ) -> Dict[str, Any]:
        """Run L2 tests separately for each modality.

        Useful for multimodal datasets (EDA/ECG/RESP) where synchrony
        patterns may differ by physiological channel.

        Parameters
        ----------
        contrast : tuple of (cond_a, cond_b) or None
            Forwarded as ``condition_values`` to each per-modality L2 test.
            If None, each modality falls back to sorted unique labels (with
            warning). Must be forwarded from the scientific path (P1-R1).
        threshold_scope : str
            Forwarded for structure-feature threshold governance warnings.
        seed : int or None
            Base RNG seed. Defaults to ``self.seed`` so multimodal runs are
            reproducible under the pipeline seed (not a hard-coded 42).

        Returns dict mapping modality → L2 results.
        """
        if feature_cols is None:
            feature_cols = get_fdr_features(full_family_fdr)
        if seed is None:
            seed = self.seed

        results = between_condition_by_modality(
            self.df,
            modality_col=modality_col,
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            alpha=fdr_alpha,
            n_permutations=n_permutations,
            seed=seed,
            condition_values=contrast,
            threshold_scope=threshold_scope,
            undefined_policy=undefined_policy,
            observation_policy=observation_policy,
            eligibility_policy=eligibility_policy,
            n_min_dyads=n_min_dyads,
        )
        if fdr_scope not in {"global", "within_modality"}:
            raise ValueError("fdr_scope must be 'global' or 'within_modality'")
        if fdr_scope == "global":
            results = _apply_global_modality_fdr(results, fdr_alpha)
        else:
            for payload in results.values():
                if isinstance(payload, dict):
                    payload["fdr_scope"] = "within_modality"
                    payload["fdr_family_size"] = len(payload.get("per_feature", []))
        return results

    # ---- full cascade ---------------------------------------------------

    def run_full_cascade(
        self,
        raw_signals_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        wcc_dict: Dict[str, np.ndarray],
        wcc_window_size: int,
        condition_col: str = "condition",
        dyad_col: str = "dyad_id",
        feature_cols: Optional[List[str]] = None,
        fdr_alpha: float = 0.05,
        n_permutations: int = 10000,
        window_type: str = "rect",
        full_family_fdr: bool = False,
        discontinuity_mask: Optional[Dict[str, np.ndarray]] = None,
        contrast: Optional[Tuple[str, str]] = None,
        modality_col: str = "modality",
    ) -> Dict[str, Any]:
        """Run the complete L0 → L1 → L2 cascade.

        Parameters
        ----------
        raw_signals_dict : dict
            Mapping from observation label → (sig_a, sig_b) tuple.
        wcc_dict : dict
            Mapping from observation label → WCC array.
        wcc_window_size : int
            WCC window size in samples.
        condition_col, dyad_col, feature_cols, fdr_alpha, n_permutations :
            Passed to L2 test.

        Returns
        -------
        dict with keys: l0_summary, l1_summary, l2_results, cascade_summary.
        """
        l0_pass = 0
        l0_total = 0
        l1_pass = 0
        l1_total = 0
        l0_feature_pass: Dict[str, int] = {}
        l1_feature_pass: Dict[str, int] = {}
        # Pre-registered PRIMARY endpoint per level (NOT an OR across the family).
        # L1 primary = switching_rate, NOT dwell_time: dwell_time is undefined in
        # a large fraction of real dyads (~40% NaN in Lerique), so it cannot be a
        # primary endpoint that must cover every dyad. See manuscript Methods.
        L0_PRIMARY = PRIMARY_EXISTENCE_ENDPOINT
        L1_PRIMARY = "switching_rate"

        for label in wcc_dict:
            if label in raw_signals_dict:
                l0_result = self.test_l0_signal(
                    wcc_dict[label],
                    raw_signals_dict[label],
                    wcc_window_size,
                    label=label,
                    window_type=window_type,
                    discontinuity_mask=(
                        discontinuity_mask.get(label)
                        if discontinuity_mask is not None
                        else None
                    ),
                )
                l0_total += 1
                pfs0 = l0_result.get("per_feature_significant", {})
                for f, sig in pfs0.items():
                    l0_feature_pass[f] = l0_feature_pass.get(f, 0) + int(bool(sig))
                if pfs0.get(L0_PRIMARY, False):
                    l0_pass += 1

            if label in wcc_dict:
                l1_result = self.test_l1_structure(wcc_dict[label], label=label)
                # gstack Finding 6 (corrected): a WCC too short for a valid L1
                # test is NOT "L1 not significant" — it is "test not applicable"
                # and must be excluded from the denominator (l1_total) so it does
                # not silently deflate the reported L1 pass rate.
                if not l1_result.get("applicable", True):
                    continue
                l1_total += 1
                pfs1 = l1_result.get("per_feature_significant", {})
                for f, sig in pfs1.items():
                    l1_feature_pass[f] = l1_feature_pass.get(f, 0) + int(bool(sig))
                if pfs1.get(L1_PRIMARY, False):
                    l1_pass += 1

        # Use scientific group router (multimodal per-modality + seed/contrast).
        l2_results = self.run_group_condition_inference(
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            fdr_alpha=fdr_alpha,
            n_permutations=n_permutations,
            contrast=contrast,
            full_family_fdr=full_family_fdr,
            threshold_scope="unknown",
            modality_col=modality_col,
        )

        return {
            "l0_summary": {
                "pass": l0_pass,
                "total": l0_total,
                "pass_rate": l0_pass / max(l0_total, 1),
                "primary_feature": "peak_amplitude",
                "per_feature_pass": l0_feature_pass,
                "description": (
                    "L0 signal-level IAAFT: tests whether WCC exceeds what "
                    "independent signals with identical spectra could produce."
                ),
            },
            "l1_summary": {
                "pass": l1_pass,
                "total": l1_total,
                "pass_rate": l1_pass / max(l1_total, 1),
                "primary_feature": "switching_rate",
                "per_feature_pass": l1_feature_pass,
                "description": (
                    "L1 WCC-level IAAFT: tests whether temporal structure "
                    "(dwell/switching) exceeds chance given the WCC distribution."
                ),
            },
            "l2_results": l2_results,
            "cascade_summary": _build_cascade_summary(
                l0_pass, l0_total, l1_pass, l1_total, l2_results
            ),
        }

    # ---- reporting ------------------------------------------------------


    @staticmethod
    def _format_one_l2_block(l2: Dict[str, Any], *, heading: str) -> List[str]:
        """Format a single unimodal L2 result dict into summarize() lines."""
        lines: List[str] = []
        if not isinstance(l2, dict) or "error" in l2:
            err = l2.get("error", "unknown") if isinstance(l2, dict) else "invalid"
            lines.append(f"{heading}: ERROR ({err})")
            return lines
        n_sig = int(l2.get("n_significant", 0))
        n_total = int(l2.get("n_tested", len(PRIMARY_FDR_FAMILY)))
        fam = "all-features" if n_total > len(PRIMARY_FDR_FAMILY) else "primary-FDR"
        ca = l2.get("condition_a", "?")
        cb = l2.get("condition_b", "?")
        lines.append(
            f"{heading} [{ca} vs {cb}] (BH-FDR, {fam}): "
            f"{n_sig}/{n_total} significant"
        )
        lines.append("  Method: dyad-paired permutation + BH-FDR correction")
        lines.append("  Significant features:")
        any_sig = False
        for feat in l2.get("per_feature", []) or []:
            if getattr(feat, "significant_05", False):
                any_sig = True
                if getattr(feat, "median_ci_bounded", False):
                    ci_text = (
                        f"95% exact median CI=[{feat.median_ci_low:.3g}, "
                        f"{feat.median_ci_high:.3g}]"
                    )
                else:
                    ci_text = "95% exact median CI=unbounded at this n"
                lines.append(
                    f"    {feat.feature}: median Δ={feat.observed_diff:.3g}, "
                    f"{ci_text}, p_raw={feat.p_raw:.4f}, "
                    f"p_fdr={feat.p_fdr:.4f}, method={feat.permutation_method}, "
                    f"MCSE={feat.approx_monte_carlo_se:.3g}"
                )
            p_def = getattr(feat, "p_definedness", 1.0)
            if p_def is not None and float(p_def) < 0.05:
                lines.append(
                    f"    [WARN] {feat.feature} definedness diff: "
                    f"{getattr(feat, 'defined_a', '?')} vs "
                    f"{getattr(feat, 'defined_b', '?')} (p={float(p_def):.4f})"
                )
        if not any_sig:
            lines.append("    (none)")
        lines.append(
            "  Claim ceiling: group inference = condition differences in "
            "audited descriptors; it does not establish direction, mechanism, "
            "or causality."
        )
        return lines

    @classmethod
    def _format_l2_summary_lines(cls, l2_display: Dict[str, Any]) -> List[str]:
        """Format modality-keyed L2 results for summarize().

        Expects the invariant shape ``{modality: l2_dict}`` produced by
        ``run_group_condition_inference`` (1c). The caller is responsible for
        wrapping a bare legacy ``test_l2_condition`` result, so this formatter
        never has to guess which shape it was handed.
        """
        if not isinstance(l2_display, dict):
            return ["L2: (unavailable)"]
        lines: List[str] = ["L2 (between-condition + BH-FDR, per-modality):"]
        for mod in sorted(l2_display.keys(), key=lambda x: str(x)):
            sub = l2_display[mod]
            lines.extend(
                cls._format_one_l2_block(
                    sub if isinstance(sub, dict) else {"error": "invalid"},
                    heading=f"  [{mod}]",
                )
            )
        return lines

    def summarize(self) -> str:
        """Return a human-readable summary of all test results."""
        lines = ["=" * 60, "SyncPipe Inference Pipeline Summary", "=" * 60, ""]

        if self._l0_results:
            n_l0 = len(self._l0_results)
            n_l0_sig = sum(
                1 for r in self._l0_results.values()
                if r.get("per_feature_significant", {}).get(PRIMARY_EXISTENCE_ENDPOINT, False)
            )
            lines.append(f"L0 (signal-level IAAFT): {n_l0_sig}/{n_l0} significant")
            lines.append("  Tests: mean_synchrony, peak_amplitude, bimodality_coefficient")
            lines.append("  H0: signals are independent")
            lines.append(
                "  Claim ceiling: IAAFT shows synchrony above independent "
                "autocorrelated surrogates — existence, not dyad-specific coupling."
            )

        if self._l1_results:
            applicable = [r for r in self._l1_results.values()
                          if r.get("applicable", True)]
            n_l1 = len(applicable)
            n_l1_sig = sum(
                1 for r in applicable
                if r.get("per_feature_significant", {}).get("switching_rate", False)
            )
            lines.append(f"\nL1 (WCC-level IAAFT): {n_l1_sig}/{n_l1} significant")
            lines.append("  Tests: dwell_time, switching_rate")
            lines.append("  H0: WCC temporal structure is random")
            lines.append(
                "  Claim ceiling: WCC-level structure test rejects random "
                "temporal organization, not co-presence / shared-stimulus alternatives."
            )

        # The scientific path stores the modality-keyed result (1c invariant).
        # The legacy `test_l2_condition` path fills `_l2_results` with a single
        # bare L2 dict, so wrap it here into the same modality-keyed shape rather
        # than making the formatter sniff which one it received.
        _l2_display = self._group_inference_results
        if _l2_display is None and self._l2_results:
            _l2_display = {UNSPECIFIED_MODALITY: self._l2_results}
        if _l2_display:
            lines.append("")
            lines.extend(self._format_l2_summary_lines(_l2_display))

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self, path: Optional[str] = None) -> str:
        """Export all results as JSON.

        Parameters
        ----------
        path : str or None
            If provided, write to this file path.

        Returns
        -------
        JSON string.
        """
        payload = {
            "l0_results": self._l0_results,
            "l1_results": self._l1_results,
            "l2_results": self._l2_results,
            "synchrony_existence_results": self._synchrony_existence_results,
            "design_control_results": self._design_control_results,
            "across_stimulus_results": self._across_stim_results,
            "group_inference_results": self._group_inference_results,
        }

        def _sanitize(o):
            """Recursively convert results to JSON-safe structures.

            Critical: dataclasses (L2Result) must become dicts — the previous
            default=str path serialized them as unusable repr strings (release bug).
            Non-finite floats become null (strict JSON).
            """
            if o is None or isinstance(o, (str, bool)):
                return o
            if isinstance(o, dict):
                return {str(k): _sanitize(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_sanitize(v) for v in o]
            if isinstance(o, (int, np.integer)) and not isinstance(o, bool):
                return int(o)
            if isinstance(o, (float, np.floating)):
                x = float(o)
                if x != x or x in (float("inf"), float("-inf")):
                    return None
                return x
            if isinstance(o, np.ndarray):
                return _sanitize(o.tolist())
            if isinstance(o, pd.DataFrame):
                return _sanitize(o.to_dict(orient="records"))
            if isinstance(o, pd.Series):
                return _sanitize(o.to_dict())
            try:
                from dataclasses import is_dataclass, asdict
                if is_dataclass(o) and not isinstance(o, type):
                    return _sanitize(asdict(o))
            except Exception:
                pass
            if hasattr(o, "to_dict") and callable(getattr(o, "to_dict")):
                try:
                    return _sanitize(o.to_dict())
                except Exception:
                    pass
            raise TypeError(
                f"Object of type {type(o).__name__} is not JSON serializable"
            )

        json_str = json.dumps(
            _sanitize(payload), indent=2, ensure_ascii=False, allow_nan=False,
        )

        if path:
            Path(path).write_text(json_str, encoding="utf-8")

        return json_str


def _build_cascade_summary(
    l0_pass: int,
    l0_total: int,
    l1_pass: int,
    l1_total: int,
    l2_results: Dict[str, Any],
) -> str:
    """Build a narrative summary of the L0→L1→L2 cascade.

    ``l2_results`` is the invariant modality-keyed shape ``{modality: l2_dict}``
    produced by ``run_group_condition_inference`` (1c), with one entry when the
    dataset carries a single modality.
    """
    l0_rate = l0_pass / max(l0_total, 1)
    l1_rate = l1_pass / max(l1_total, 1)
    n_l2_sig = 0
    n_l2_total = 0
    widest_family = 0
    for _sub in (l2_results or {}).values():
        if isinstance(_sub, dict) and "error" not in _sub:
            n_l2_sig += int(_sub.get("n_significant", 0))
            _tested = int(_sub.get("n_tested", 0))
            n_l2_total += _tested
            widest_family = max(widest_family, _tested)
    # Family label: the frozen PRIMARY FDR family (single descriptor) is the
    # primary endpoint; full_family_fdr=True enters all features into one
    # BH-FDR step (reviewer-proof against "cherry-picking"). Judge this from the
    # widest single-modality family, not the cross-modality sum, so M modalities
    # x the primary family is not mislabelled "all-features".
    fam = "all-features" if widest_family > len(PRIMARY_FDR_FAMILY) else "primary-FDR"
    if len(l2_results or {}) > 1:
        fam = f"{fam}, per-modality"

    parts = []

    if l0_rate >= 0.5:
        parts.append(
            f"L0: {l0_pass}/{l0_total} ({l0_rate:.0%}) dyads show above-chance synchrony. "
            "This supports synchrony-like evidence above the signal-level null, "
            "but does not by itself prove dyad-specific coupling."
        )
    elif l0_rate > 0:
        parts.append(
            f"L0: {l0_pass}/{l0_total} ({l0_rate:.0%}) dyads show above-chance synchrony. "
            "Coupling evidence is present but limited."
        )
    else:
        parts.append(
            "L0: No dyads exceeded the signal-level null. "
            "The dataset may lack sufficient coupling signal."
        )

    if l1_rate >= 0.3:
        parts.append(
            f"L1: {l1_pass}/{l1_total} ({l1_rate:.0%}) dyads show structured temporal patterns. "
            "Synchrony episodes have non-random dwell/ switching organization."
        )
    elif l1_rate > 0:
        parts.append(
            f"L1: {l1_pass}/{l1_total} ({l1_rate:.0%}) dyads show structured patterns. "
            "Temporal structure evidence is preliminary."
        )
    else:
        parts.append(
            "L1: No dyads showed significant temporal structure. "
            "This may reflect short WCC series or weak episode patterning."
        )

    if n_l2_sig >= 4:
        parts.append(
            f"L2 (between-condition BH-FDR, {fam}): {n_l2_sig}/{n_l2_total} features "
            "are condition-differentiated. "
            "Strong evidence that synchrony is modulated by task context."
        )
    elif n_l2_sig > 0:
        parts.append(
            f"L2 (between-condition BH-FDR, {fam}): {n_l2_sig}/{n_l2_total} features "
            "are condition-differentiated. "
            "Selective modulation evidence."
        )
    else:
        parts.append(
            f"L2 (between-condition BH-FDR, {fam}): No features survived BH-FDR. "
            "Synchrony may exist (L0) but not vary by condition."
        )

    return " | ".join(parts)

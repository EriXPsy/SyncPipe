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
from .feature_definitions import FDR_FEATURES, extract_features
from .validation.across_stim_shuffle import across_stim_shuffle_test
from .validation.l2_between_condition import (
    between_condition_fdr,
    between_condition_by_modality,
)


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

    Examples
    --------
    >>> pipe = InferencePipeline(df, hz=4.0, wcc_window_sec=10.0)
    >>> pipe.run_l0(sig_a, sig_b)          # per-observation
    >>> pipe.run_l1(wcc_series)            # per-observation
    >>> pipe.run_l2("condition", "dyad_id")  # group-level
    >>> report = pipe.summarize()
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        hz: float = 4.0,
        wcc_window_sec: Optional[float] = None,
        surrogate_n: int = 100,
        seed: int = 42,
    ):
        self.df = features_df.copy()
        self.hz = hz
        self.wcc_window_sec = wcc_window_sec
        self.surrogate_n = surrogate_n
        self.seed = seed

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
    ) -> Dict[str, Any]:
        """Step 1: test whether each pair shows synchrony above signal-level null.

        This is a synchrony-existence audit, not proof of dyad-specific
        interpersonal coupling.  Shared-stimulus and co-presence alternatives
        are evaluated in :meth:`run_design_control_audit` and
        :meth:`run_across_stimulus_shuffle_audit`.
        """
        selected = list(labels) if labels is not None else list(raw_signals.keys())
        results: Dict[str, Any] = {}
        for label in selected:
            if label not in raw_signals:
                continue
            sig_a, sig_b = raw_signals[label]
            results[label] = synchrony_existence_audit(
                sig_a,
                sig_b,
                hz=self.hz,
                window_size=wcc_window_size,
                surrogate_n=self.surrogate_n,
                seed=self.seed,
            )
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
        n_pseudo_per_dyad: int = 3,
        shift_lags_sec: Sequence[float] = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0),
    ) -> Dict[str, Any]:
        """Step 2a: run pseudo-pair and time-shift design controls.

        Pseudo-pair controls ask whether real partners exceed mismatched
        partners.  Time-shift controls ask whether the effect depends on the
        original temporal alignment.  These are formal API methods, not just
        dataset-specific scripts.
        """
        result = design_control_audit(
            signal_pairs,
            hz=self.hz,
            window_size=wcc_window_size,
            feature_names=feature_names,
            n_pseudo_per_dyad=n_pseudo_per_dyad,
            shift_lags_sec=shift_lags_sec,
            seed=self.seed,
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
    ) -> Dict[str, Any]:
        """Step 2b: run across-stimulus shuffle for segmented stimulus designs.

        Use this when both partners experienced the same ordered stimulus
        sequence (e.g., video clips, repeated trials).  It permutes stimulus
        segments independently across partners, making shared-stimulus timing
        auditable.  It is not appropriate for unsegmented free interaction.
        """
        window_sec = self.wcc_window_sec or (wcc_window_size / self.hz)

        def _wcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return sliding_window_wcc(a, b, window_size=wcc_window_size, hz=self.hz)

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
    ) -> Dict[str, Any]:
        """Step 3: test whether features differentiate experimental conditions."""
        result = self.test_l2_condition(
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            fdr_alpha=fdr_alpha,
            n_permutations=n_permutations,
            contrast=contrast,
        )
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
    ) -> Dict[str, Any]:
        """Run the recommended v1 evidence chain end-to-end.

        Chain:
        1. synchrony-existence audit (signal-level IAAFT)
        2. design-control audit (pseudo-pair/time-shift; optional across-stim)
        3. group condition inference (paired permutation + FDR)
        """
        existence = self.run_synchrony_existence_audit(
            raw_signals, wcc_window_size=wcc_window_size,
        )
        design = None
        if design_signal_pairs is not None:
            design = self.run_design_control_audit(
                design_signal_pairs, wcc_window_size=wcc_window_size,
            )
        across = None
        if across_stim_segments is not None:
            across = self.run_across_stimulus_shuffle_audit(
                across_stim_segments, wcc_window_size=wcc_window_size,
            )
        group = self.run_group_condition_inference(
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            fdr_alpha=fdr_alpha,
            n_permutations=n_permutations,
        )
        return {
            "evidence_chain_version": "v1",
            "synchrony_existence": existence,
            "design_controls": design,
            "across_stimulus_shuffle": across,
            "group_condition_inference": group,
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
            parts.append(
                f"Group condition inference found {group.get('n_significant', 0)} significant feature(s)."
            )
        parts.append(
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
        )
        result["label"] = label
        self._l0_results[label] = result
        return result

    # ---- L1: WCC-level temporal structure test ---------------------------

    def test_l1_structure(
        self,
        wcc: np.ndarray,
        label: str = "",
    ) -> Dict[str, Any]:
        """Run L1 WCC-level IAAFT surrogate test.

        H0: The WCC series has no temporal structure beyond its amplitude
        distribution. Any dwell/switching patterns are random.

        This test preserves the mean and variance of the WCC (so L0 must
        pass first) but destroys temporal ordering. It asks: given that
        this level of synchrony exists, is its temporal organization real?

        Parameters
        ----------
        wcc : np.ndarray
            Observed WCC series.
        label : str
            Optional label for results tracking.

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
        fdr_alpha : float
            BH-FDR significance threshold (default 0.05).
        n_permutations : int
            Number of permutation iterations.
        contrast : tuple of (cond_a, cond_b) or None
            Specific contrast to test. If None, tests all pairwise.

        Returns
        -------
        dict with per-feature p_raw, p_fdr, significant_05, effect_size.
        """
        if feature_cols is None:
            feature_cols = list(FDR_FEATURES)

        self._l2_results = between_condition_fdr(
            self.df,
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            alpha=fdr_alpha,            # was fdr_alpha= (wrong kwarg name)
            n_permutations=n_permutations,
            condition_values=contrast,  # was contrast= (wrong kwarg name)
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
    ) -> Dict[str, Any]:
        """Run L2 tests separately for each modality.

        Useful for multimodal datasets (EDA/ECG/RESP) where synchrony
        patterns may differ by physiological channel.

        Returns dict mapping modality → L2 results.
        """
        if feature_cols is None:
            feature_cols = list(FDR_FEATURES)

        return between_condition_by_modality(
            self.df,
            modality_col=modality_col,
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            alpha=fdr_alpha,            # was fdr_alpha= (wrong kwarg name)
            n_permutations=n_permutations,
        )

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
        L0_PRIMARY = "peak_amplitude"
        L1_PRIMARY = "switching_rate"

        for label in wcc_dict:
            if label in raw_signals_dict:
                l0_result = self.test_l0_signal(
                    wcc_dict[label],
                    raw_signals_dict[label],
                    wcc_window_size,
                    label=label,
                )
                l0_total += 1
                pfs0 = l0_result.get("per_feature_significant", {})
                for f, sig in pfs0.items():
                    l0_feature_pass[f] = l0_feature_pass.get(f, 0) + int(bool(sig))
                if pfs0.get(L0_PRIMARY, False):
                    l0_pass += 1

            if label in wcc_dict:
                l1_result = self.test_l1_structure(wcc_dict[label], label=label)
                l1_total += 1
                pfs1 = l1_result.get("per_feature_significant", {})
                for f, sig in pfs1.items():
                    l1_feature_pass[f] = l1_feature_pass.get(f, 0) + int(bool(sig))
                if pfs1.get(L1_PRIMARY, False):
                    l1_pass += 1

        l2_results = self.test_l2_condition(
            condition_col=condition_col,
            dyad_col=dyad_col,
            feature_cols=feature_cols,
            fdr_alpha=fdr_alpha,
            n_permutations=n_permutations,
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

    def summarize(self) -> str:
        """Return a human-readable summary of all test results."""
        lines = ["=" * 60, "SyncPipe Inference Pipeline Summary", "=" * 60, ""]

        if self._l0_results:
            n_l0 = len(self._l0_results)
            n_l0_sig = sum(
                1 for r in self._l0_results.values()
                if r.get("per_feature_significant", {}).get("peak_amplitude", False)
            )
            lines.append(f"L0 (signal-level IAAFT): {n_l0_sig}/{n_l0} significant")
            lines.append("  Tests: mean_synchrony, peak_amplitude, bimodality_coefficient")
            lines.append("  H0: signals are independent")

        if self._l1_results:
            n_l1 = len(self._l1_results)
            n_l1_sig = sum(
                1 for r in self._l1_results.values()
                if r.get("per_feature_significant", {}).get("switching_rate", False)
            )
            lines.append(f"\nL1 (WCC-level IAAFT): {n_l1_sig}/{n_l1} significant")
            lines.append("  Tests: dwell_time, switching_rate")
            lines.append("  H0: WCC temporal structure is random")

        if self._l2_results:
            n_sig = self._l2_results.get("n_significant", 0)
            n_total = self._l2_results.get("n_tested", len(FDR_FEATURES))
            lines.append(f"\nL2 (between-condition + BH-FDR): {n_sig}/{n_total} significant")
            lines.append("  Method: dyad-paired permutation + BH-FDR correction")
            lines.append("  Significant features:")
            for feat in self._l2_results.get("features", []):
                if feat.get("significant_05"):
                    lines.append(
                        f"    {feat['feature']}: p_raw={feat['p_raw']:.4f}, "
                        f"p_fdr={feat['p_fdr']:.4f}"
                    )

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

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            return str(obj)

        json_str = json.dumps(payload, default=_convert, indent=2, ensure_ascii=False)

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
    """Build a narrative summary of the L0→L1→L2 cascade."""
    l0_rate = l0_pass / max(l0_total, 1)
    l1_rate = l1_pass / max(l1_total, 1)
    n_l2_sig = l2_results.get("n_significant", 0)

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
            f"L2: {n_l2_sig}/{len(FDR_FEATURES)} FDR features are condition-differentiated. "
            "Strong evidence that synchrony is modulated by task context."
        )
    elif n_l2_sig > 0:
        parts.append(
            f"L2: {n_l2_sig}/{len(FDR_FEATURES)} FDR features are condition-differentiated. "
            "Selective modulation evidence."
        )
    else:
        parts.append(
            "L2: No features survived BH-FDR. "
            "Synchrony may exist (L0) but not vary by condition."
        )

    return " | ".join(parts)

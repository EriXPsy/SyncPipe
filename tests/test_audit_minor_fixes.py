"""Minor-round regression tests (adversarial-review m2/m3/m4/m6/m8/m11,
2026-09-14). Document-only minors (m1/m5/m7/m9/m12/m13) carry no behaviour
to test; they are verified by inspection in the audit checklist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# m2 — second-order gate consumes ALL valid draws (no shortest-width chop).
# ---------------------------------------------------------------------------

class TestGateFullWidthNull:
    def test_unequal_null_lengths_are_padded_not_chopped(self):
        from syncpipe.inference_pipeline import _existence_gate_by_modality
        results = {
            "d1__ECG": {
                "observed": {"peak_amplitude": 0.8},
                "null_peak_amplitude": np.array([0.1, 0.2, 0.3, 0.4]),
                "per_feature_significant": {"peak_amplitude": True},
            },
            "d2__ECG": {
                "observed": {"peak_amplitude": 0.75},
                "null_peak_amplitude": np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.9]),
                "per_feature_significant": {"peak_amplitude": True},
            },
        }
        gate = _existence_gate_by_modality(results, primary_modalities=("ECG",))
        # n_null_draws must reflect the FULL width (6), not the min (4).
        assert gate["per_modality"]["ECG"]["n_null_draws"] == 6


# ---------------------------------------------------------------------------
# m8 — vectorised Schmitt trigger is bit-identical to the reference loop.
# ---------------------------------------------------------------------------

class TestVectorizedHysteresisParity:
    @staticmethod
    def _reference(wcc, threshold, delta):
        finite = np.isfinite(wcc)
        n = wcc.shape[0]
        states = np.zeros(n, dtype=bool)
        if not finite.any() or n == 0:
            return states
        if delta <= 0:
            return (wcc >= threshold) & finite
        enter, exit_ = threshold + delta, threshold - delta
        state = False
        for i in range(n):
            if not finite[i]:
                states[i] = False
                state = False
                continue
            if not state and wcc[i] >= enter:
                state = True
            elif state and wcc[i] < exit_:
                state = False
            states[i] = state
        return states

    def test_parity_random_500(self):
        from syncpipe.feature_definitions import _binarize_with_hysteresis
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = int(rng.integers(3, 60))
            w = rng.normal(0.5, 0.35, n)
            if rng.random() < 0.5:
                w[rng.integers(0, n, size=int(rng.integers(0, 5)))] = np.nan
            thr = float(rng.uniform(0.3, 0.7))
            delta = float(rng.choice([0.0, 0.05, 0.15]))
            np.testing.assert_array_equal(
                _binarize_with_hysteresis(w, thr, delta),
                self._reference(w, thr, delta),
            )

    def test_nan_resets_state_across_gap(self):
        from syncpipe.feature_definitions import _binarize_with_hysteresis
        # Elevated run -> NaN gap -> values inside the hysteresis band:
        # memory must reset, so post-gap band values are baseline (False).
        w = np.array([0.9, np.nan, 0.52, 0.52, 0.9])
        states = _binarize_with_hysteresis(w, 0.5, 0.05)
        np.testing.assert_array_equal(
            states, [True, False, False, False, True]
        )


# ---------------------------------------------------------------------------
# m4 — fold-internal imputation (no whole-data median leak).
# ---------------------------------------------------------------------------

class TestFoldMedianImputerNoLeak:
    def test_imputer_fits_train_median_only(self):
        from syncpipe.morphology import _FoldMedianImputer
        rng = np.random.default_rng(3)
        X_train = rng.normal(size=(50, 2))
        X_train[:10, 0] = np.nan
        imp = _FoldMedianImputer().fit(X_train)
        expected0 = float(np.nanmedian(X_train[:, 0]))
        assert imp.medians_[0] == pytest.approx(expected0)
        X_test = np.array([[np.nan, 0.0]])
        out = imp.transform(X_test)
        assert out[0, 0] == pytest.approx(expected0)

    def test_incremental_value_runs_with_nan_features(self):
        from syncpipe.morphology import incremental_value
        rng = np.random.default_rng(5)
        n = 40
        y = (np.arange(n) % 2).astype(str)
        df = pd.DataFrame({
            "mean_synchrony": rng.normal(0.3, 0.1, n),
            "peak_amplitude": rng.normal(0.5, 0.2, n),
            "dwell_time": rng.normal(3.0, 1.0, n),
        })
        df.loc[df.index[:8], "dwell_time"] = np.nan
        out, _meta = incremental_value(
            df, y, ["mean_synchrony", "peak_amplitude", "dwell_time"],
            n_orders=3,
        )
        assert np.isfinite(out["shapley_marginal_auc"]).all()


# ---------------------------------------------------------------------------
# m6 — Dyad coerces non-string dyad_id with a warning (never silent swap).
# ---------------------------------------------------------------------------

class TestDyadIdCoercion:
    def test_int_id_warns_and_coerces(self):
        from syncpipe.core import Dyad
        df = pd.DataFrame({"time": [0.0, 1.0], "v": [1.0, 2.0]})
        with pytest.warns(UserWarning, match="coerced"):
            d = Dyad(dyad_id=123, hz=1.0, eeg=df)
        assert d.dyad_id == "123"

    def test_string_id_unchanged_silent(self, recwarn):
        from syncpipe.core import Dyad
        df = pd.DataFrame({"time": [0.0, 1.0], "v": [1.0, 2.0]})
        d = Dyad(dyad_id="abcd", hz=1.0, eeg=df)
        assert d.dyad_id == "abcd"
        assert len(recwarn) == 0


# ---------------------------------------------------------------------------
# m3 — SIGNFLIP_MAX_DRAWS is the single source for the gate resolution.
# ---------------------------------------------------------------------------

class TestSignflipConstantCoupling:
    def test_builder_resolution_derives_from_constant(self):
        from syncpipe.design_controls import SIGNFLIP_MAX_DRAWS
        from syncpipe.evidence.builder import _design_resolution
        design = {
            "feature_summary": {
                "peak_amplitude": {"n_real": 15},
            }
        }
        res = _design_resolution(design, "peak_amplitude")
        assert res == pytest.approx(1.0 / (SIGNFLIP_MAX_DRAWS + 1.0))


# ---------------------------------------------------------------------------
# m11 — L1 result surfaces the actual window duration + heuristic flag.
# ---------------------------------------------------------------------------

class TestL1WindowSecVisibility:
    def test_heuristic_flag_and_value_in_result(self):
        from syncpipe.dynamic_features import _wcc_level_surrogate_test
        rng = np.random.default_rng(2)
        w = rng.uniform(0.2, 0.9, 120)
        res = _wcc_level_surrogate_test(
            w, hz=1.0, surrogate_n=5, seed=1, wcc_window_sec=None,
        )
        assert res["wcc_window_sec_heuristic"] is True
        assert res["wcc_window_sec"] == pytest.approx(len(w) / 10.0)

        res2 = _wcc_level_surrogate_test(
            w, hz=1.0, surrogate_n=5, seed=1, wcc_window_sec=30.0,
        )
        assert res2["wcc_window_sec_heuristic"] is False
        assert res2["wcc_window_sec"] == pytest.approx(30.0)

"""A2: per-dyad L0 audit counters follow the gate's tail policy.

Before the fix, the per-dyad `l0_pass` / `l0_feature_pass` audit counters
counted the LEGACY two-sided per_feature_significant flags while the gate
itself is one-sided upper (R4a) — an anti-phase dyad could count as a
"pass" in the audit while the gate rejected its modality. The counters are
descriptive (V1_PROTOCOL: a p-value is a population statement), but they
must never contradict the decision layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from syncpipe.inference_pipeline import InferencePipeline




def _l0(obs, null, *, p_upper, p_two_sided):
    return {
        "observed": {"peak_amplitude": obs},
        "null_peak_amplitude": np.asarray(null, dtype=float),
        "per_feature_significant": {"peak_amplitude": p_two_sided},
        "direction_resolved": {
            "peak_amplitude": {
                "p_upper": p_upper, "p_lower": 1.0,
                "significant_upper": p_upper < 0.05,
                "significant_lower": False,
            }
        },
    }


class TestAuditCountersFollowGateTail:
    def _run(self, l0_results, monkeypatch):
        # Zero-row feature frame: L2 has nothing to test and returns its
        # empty-cohort shape; the L0 counters are what we assert on.
        pipe = InferencePipeline(
            features_df=pd.DataFrame(
                {"condition": [], "dyad_id": [], "mean_synchrony": []}
            )
        )
        monkeypatch.setattr(
            pipe, "test_l0_signal",
            lambda *a, **k: l0_results.get(k.get("label", a[0] if a else None)),
        )
        monkeypatch.setattr(
            pipe, "run_group_condition_inference", lambda **k: {"skipped": True},
        )
        out = pipe.run_full_cascade(
            raw_signals_dict={k: (np.zeros(10), np.zeros(10)) for k in l0_results},
            wcc_dict={k: np.zeros(10) for k in l0_results},
            wcc_window_size=5,
        )
        return out["l0_summary"]

    def test_antiphase_dyad_not_counted_as_pass(self, monkeypatch):
        """Two-sided legacy flag True (below-null), upper False: the audit
        must NOT count this dyad — the gate would reject it."""
        results = {
            "d0__ECG__rest": _l0(0.6, np.linspace(0.8, 0.9, 40), p_upper=1.0, p_two_sided=True),
            "d1__ECG__rest": _l0(0.9, np.linspace(0.3, 0.5, 40), p_upper=0.01, p_two_sided=True),
        }
        summary = self._run(results, monkeypatch)
        assert summary["pass"] == 1  # only the true-upper dyad

    def test_fallback_without_direction_resolved(self, monkeypatch):
        """Legacy results without direction_resolved keep the two-sided
        flag so historical artifacts remain interpretable."""
        results = {
            "d0__ECG__rest": {
                "observed": {"peak_amplitude": 0.9},
                "null_peak_amplitude": np.linspace(0.3, 0.5, 40),
                "per_feature_significant": {"peak_amplitude": True},
            },
        }
        summary = self._run(results, monkeypatch)
        assert summary["pass"] == 1

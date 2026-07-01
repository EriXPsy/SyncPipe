"""
Band-pass PLV control: does PLV's apparent edge survive honest band selection?

In run_metrics_benchmark_v2.py, PLV ran on BROADBAND Hilbert phase
(freq_band=None). Broadband phase is high for almost any smooth signal, so
PLV's detection score is inflated. This control re-runs the same three coupling
structures and compares three PLV variants against WCC:

  - plv_broadband : freq_band=None            (what the v2 benchmark used)
  - plv_correct   : band-pass at the signal's TRUE frequency band
  - plv_wrong     : band-pass at a plausible-but-wrong band

The point: PLV is only trustworthy when you have THE right frequency band.
Pick wrong (or have no single band across HR/EDA/RESP/behaviour) and PLV
collapses or manufactures spurious locking. WCC needs no such choice.

Reuses the generators / scoring of run_metrics_benchmark_v2.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"<REPO>")

import numpy as np
import pandas as pd
from multisync.metrics import plv_synchrony
from multisync.dynamic_features import extract_dynamic_features, sliding_window_wcc
from multisync.feature_definitions import CONFIRMATORY_FEATURES

N_SEEDS = 15
COUPLINGS = [0.0, 0.6]
DURATION = 300
HZ = 1.0
WINDOW = 60
STEP = 10
LAG = 30

FEATS = list(CONFIRMATORY_FEATURES) + ["mean_synchrony", "synchrony_entropy"]
KEY_FEATS = ["peak_amplitude", "mean_synchrony", "switching_rate", "synchrony_entropy"]

# True dominant periods of each generator (see v2): linear 60/20s, lagged 30s,
# nonlinear 40s. Convert to a band-pass window around the true frequency.
TRUE_BAND = {            # Hz
    "linear": (1 / 80.0, 1 / 15.0),
    "lagged": (1 / 45.0, 1 / 20.0),
    "nonlinear": (1 / 55.0, 1 / 30.0),
}
WRONG_BAND = (0.15, 0.45)  # a "physiology-plausible" but mismatched fast band


def generate_linear(c, seed):
    rng = np.random.default_rng(seed)
    n = DURATION * int(HZ)
    t = np.linspace(0, DURATION, n)
    shared = np.sin(2 * np.pi * t / 60) + 0.5 * np.sin(2 * np.pi * t / 20)
    shared /= np.std(shared) + 1e-10
    a = shared + 0.3 * rng.normal(0, 1, n)
    b = c * shared + (1 - c) * rng.normal(0, 1, n) + 0.3 * rng.normal(0, 0.2, n)
    return a, b


def generate_lagged(c, seed, lag=LAG):
    rng = np.random.default_rng(seed)
    n = DURATION * int(HZ)
    t = np.linspace(0, DURATION, n)
    p1 = np.sin(2 * np.pi * t / 30) + 0.3 * rng.normal(0, 1, n)
    p2_base = np.zeros(n)
    p2_base[lag:] = p1[:-lag]
    p2 = c * p2_base + (1 - c) * rng.normal(0, 1, n) + 0.3 * rng.normal(0, 0.2, n)
    return p1, p2


def generate_nonlinear(c, seed):
    rng = np.random.default_rng(seed)
    n = DURATION * int(HZ)
    t = np.linspace(0, DURATION, n)
    p1 = np.sin(2 * np.pi * t / 40) + 0.3 * rng.normal(0, 1, n)
    p1_norm = p1 / (np.std(p1) + 1e-10)
    sigmoid = 1.0 / (1.0 + np.exp(-3 * p1_norm))
    p2 = c * sigmoid + (1 - c) * rng.normal(0, 1, n) + 0.3 * rng.normal(0, 0.2, n)
    return p1, p2


GENERATORS = {
    "linear": generate_linear,
    "lagged": generate_lagged,
    "nonlinear": generate_nonlinear,
}


def metrics_for(structure):
    tb = TRUE_BAND[structure]
    return {
        "wcc": lambda a, b: sliding_window_wcc(a, b, window_size=WINDOW, hz=HZ, step_samples=STEP),
        "plv_broadband": lambda a, b: plv_synchrony(a, b, window_size=WINDOW, step=STEP, fs=HZ),
        "plv_correct": lambda a, b: plv_synchrony(a, b, window_size=WINDOW, step=STEP, fs=HZ, freq_band=tb),
        "plv_wrong": lambda a, b: plv_synchrony(a, b, window_size=WINDOW, step=STEP, fs=HZ, freq_band=WRONG_BAND),
    }


def main():
    rows = []
    for gen_name, gen in GENERATORS.items():
        mets = metrics_for(gen_name)
        for c in COUPLINGS:
            for seed in range(N_SEEDS):
                a, b = gen(c, 1000 + seed)
                for m_name, m_func in mets.items():
                    trace = m_func(a, b)
                    if len(trace) < 10:
                        continue
                    trace = np.asarray(trace, float)
                    trace[np.isnan(trace)] = 0
                    feats = extract_dynamic_features(trace, hz=HZ / STEP, wcc_window_sec=WINDOW)
                    for feat in FEATS:
                        v = getattr(feats, feat, None)
                        if v is not None and np.isfinite(v):
                            rows.append({
                                "structure": gen_name, "coupling": c, "seed": seed,
                                "metric": m_name, "feature": feat, "value": float(v),
                            })

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "plv_bandpass_control.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    metric_order = ["wcc", "plv_broadband", "plv_correct", "plv_wrong"]
    print("=" * 78)
    print("BAND-PASS PLV CONTROL  (summed detection power over 4 features, 0-400%)")
    print("=" * 78)
    header = f"{'structure':>10s}" + "".join(f"{m:>15s}" for m in metric_order)
    print(header)
    print("-" * len(header))
    summary = {}
    for structure in GENERATORS:
        sub = df[df.structure == structure]
        line = f"{structure:>10s}"
        summary[structure] = {}
        for m in metric_order:
            total = 0.0
            for feat in KEY_FEATS:
                c0 = sub[(sub.metric == m) & (sub.coupling == 0.0) & (sub.feature == feat)]["value"]
                c6 = sub[(sub.metric == m) & (sub.coupling == 0.6) & (sub.feature == feat)]["value"]
                if len(c0) >= 2 and len(c6) >= 2:
                    total += np.mean(c6.values > np.percentile(c0, 95))
            summary[structure][m] = total
            line += f"{total*100:14.0f}%"
        print(line)
    print("-" * len(header))
    print("\nINTERPRETATION:")
    print("  plv_broadband is the inflated number used in the v2 benchmark.")
    print("  plv_correct = PLV WITH the right band; plv_wrong = PLV with a wrong band.")
    print("  If plv_correct/plv_wrong drop below or near WCC, PLV's 'edge' was a")
    print("  broadband artifact and depends entirely on knowing the right band.")
    print(f"\nfull table -> {out}")


if __name__ == "__main__":
    main()

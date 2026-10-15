"""Regenerate pair.csv (the deterministic input of the bsync conformance fixture).

    python tests/fixtures/bsync_golden/make_pair.py

300 samples at 1 Hz; y follows x by 3 samples. Values are written with ``repr(float(v))`` (17 significant
digits, plain numbers) so both Python and R read back exactly the same doubles.
"""
from __future__ import annotations

import pathlib

import numpy as np

SEED, N = 20261008, 300


def make_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    driver = np.convolve(rng.normal(size=N + 40), np.ones(7) / 7, mode="same")
    x = driver[20:20 + N] + 0.4 * rng.normal(size=N)
    y = driver[17:17 + N] + 0.4 * rng.normal(size=N)
    return x, y


if __name__ == "__main__":
    x, y = make_pair()
    out = pathlib.Path(__file__).with_name("pair.csv")
    out.write_text("x,y\n" + "".join(f"{float(a)!r},{float(b)!r}\n" for a, b in zip(x, y)), encoding="utf-8")
    print(f"wrote {out} ({N} rows)")

"""Cross-implementation consistency lock for the (formerly 3, now 1 canonical
+ thin wrappers) BH-FDR routines.

Claude's review found 3 independent BH-FDR implementations:
  - multisync.batch._bh_fdr_correction  (canonical; returns (adjusted, rejected))
  - multisync.validation.l2_between_condition._bh_fdr  (now delegates to batch)
  - multisync.validation.pgt1_intensity.bh_fdr  (returns boolean rejected array)

All three must agree on (a) adjusted p-values and (b) rejected flags.  This
test guards against any future drift between the copies.
"""
from __future__ import annotations

import numpy as np

from multisync.batch import _bh_fdr_correction
from multisync.validation.l2_between_condition import _bh_fdr
from multisync.validation.pgt1_intensity import bh_fdr


def test_bh_fdr_implementations_agree():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        p = rng.uniform(0.0, 1.0, size=n)
        # Occasionally inject NaN / boundary values.
        if rng.random() < 0.3:
            p[int(rng.integers(0, n))] = np.nan
        if rng.random() < 0.2:
            p[0] = 0.0
        if rng.random() < 0.2:
            p[0] = 1.0

        adj_batch, rej_batch = _bh_fdr_correction(list(p), alpha=0.05)
        adj_l2 = _bh_fdr(p)
        rej_pgt1 = bh_fdr(p, q=0.05)

        # Adjusted p-values: batch (canonical) vs l2 (thin wrapper).
        np.testing.assert_allclose(
            np.asarray(adj_batch, dtype=float),
            np.asarray(adj_l2, dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
        # Rejected flags: batch vs pgt1 (both at FDR 0.05).
        np.testing.assert_array_equal(
            np.asarray(rej_batch, dtype=bool),
            np.asarray(rej_pgt1, dtype=bool),
        )

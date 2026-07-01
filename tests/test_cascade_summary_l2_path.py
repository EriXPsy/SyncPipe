"""Regression test for the _build_cascade_summary() NameError on the
n_l2_sig > 0 path.

Background: _build_cascade_summary references FDR_FEATURES, but FDR_FEATURES was
only imported locally inside two methods, never at module top level. The
reference lives ONLY in the `n_l2_sig > 0` branch, so the crash was invisible
unless at least one feature passed BH-FDR — i.e. exactly the "we have a finding"
case. These tests exercise BOTH branches so the bug cannot silently return.
"""
import pytest

from multisync.inference_pipeline import _build_cascade_summary


def test_cascade_summary_no_l2_significant():
    """n_l2_sig == 0 branch (was always safe)."""
    s = _build_cascade_summary(l0_pass=3, l0_total=5, l1_pass=2, l1_total=5,
                               l2_results={"n_significant": 0})
    assert isinstance(s, str) and "L2" in s


def test_cascade_summary_some_l2_significant():
    """n_l2_sig > 0 branch — used to raise NameError: name 'FDR_FEATURES'."""
    s = _build_cascade_summary(l0_pass=3, l0_total=5, l1_pass=2, l1_total=5,
                               l2_results={"n_significant": 2})
    assert isinstance(s, str) and "L2" in s


def test_cascade_summary_strong_l2_significant():
    """n_l2_sig >= 4 branch (also references FDR_FEATURES)."""
    s = _build_cascade_summary(l0_pass=4, l0_total=5, l1_pass=3, l1_total=5,
                               l2_results={"n_significant": 4})
    assert isinstance(s, str) and "L2" in s


def test_fdr_features_importable_at_module_top():
    """FDR_FEATURES must be bound at module global scope (the actual fix)."""
    import multisync.inference_pipeline as ip
    assert hasattr(ip, "FDR_FEATURES")
    assert len(ip.FDR_FEATURES) >= 1

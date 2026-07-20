"""Regression tests for Finding 13: clock-offset / co-start caveat.

The temporal_alignment QC stage verifies the *loaded* time axis is internally
consistent (matching sample counts, identical time vectors, monotonicity) but
CANNOT detect a true between-file clock offset already absorbed at import — two
relative-time 0.0 starts that were never co-triggered are merged at t=0 by
construction. `run_quality_check` must surface this as a non-blocking NOTE
(`clock_offset_caveat` / `co_start_verified`) rather than silently promising
co-start, and must stay silent once the dataset explicitly marks co-start as
verified.
"""
from multisync.synthetic import generate_ground_truth_dyad
from multisync.qc import run_quality_check


def _clean_dyad():
    return generate_ground_truth_dyad(duration_sec=120, noise_ratio=0.1)


def test_clock_offset_caveat_emitted_by_default():
    report = run_quality_check(_clean_dyad())
    assert report.co_start_verified is False
    assert report.clock_offset_caveat  # non-empty caveat string
    assert any(
        "co-start" in n or "clock offset" in n.lower() for n in report.notes
    )


def test_co_start_verified_suppresses_caveat():
    ds = _clean_dyad()
    ds.co_start_verified = True
    report = run_quality_check(ds)
    assert report.co_start_verified is True
    assert report.clock_offset_caveat == ""


def test_summary_surfaces_caveat():
    report = run_quality_check(_clean_dyad())
    assert "CLOCK OFFSET CAVEAT" in report.summary()

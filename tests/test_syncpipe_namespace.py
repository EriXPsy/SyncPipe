def test_syncpipe_namespace_exposes_v1_public_api():
    import syncpipe as sp

    for name in [
        "Dyad",
        "DynamicAnalyzer",
        "InferencePipeline",
        "feature_status_table",
        "feature_status_latex",
        "explain_feature",
        "run_quality_check",
        "format_qc_report",
        "DataQualityError",
        "compute_session_pooled_threshold",
    ]:
        assert hasattr(sp, name), name

def test_syncpipe_version_available():
    import syncpipe as sp

    assert isinstance(sp.__version__, str)


def test_legacy_multisync_namespace_still_available():
    import multisync as ms

    assert hasattr(ms, "Dyad")
    assert hasattr(ms, "DynamicAnalyzer")
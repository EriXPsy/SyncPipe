def test_syncpipe_namespace_exposes_v1_public_api():
    import syncpipe as sp

    assert hasattr(sp, "Dyad")
    assert hasattr(sp, "DynamicAnalyzer")
    assert hasattr(sp, "InferencePipeline")
    assert hasattr(sp, "feature_status_table")


def test_legacy_multisync_namespace_still_available():
    import multisync as ms

    assert hasattr(ms, "Dyad")
    assert hasattr(ms, "DynamicAnalyzer")

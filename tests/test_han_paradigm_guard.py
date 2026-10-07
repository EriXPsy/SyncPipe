"""Prevent the unsupported Han composite/consecutive-ID WCC path."""
import numpy as np
import pandas as pd
import pytest

from scripts import realdata_full_new_pipeline as pipeline


@pytest.mark.parametrize("time_column", ["stim", "time", None])
def test_han_native_loader_refuses_unsupported_paradigm(monkeypatch, time_column):
    # A/B identify different people; eight columns are different messages.
    data = pd.DataFrame({f"message_{i}": np.arange(40) + i for i in range(8)})
    if time_column is not None:
        data[time_column] = np.arange(40) / 10
    files = [f"FO{pid}_Stim_{person}.xls" for pid in (110, 111) for person in "AB"]
    monkeypatch.setattr(pipeline.glob, "glob", lambda _: files)
    monkeypatch.setattr(pipeline.pd, "read_excel", lambda *a, **k: data.copy())
    with pytest.raises(NotImplementedError, match="Han.*CRQA"):
        pipeline.load_han()

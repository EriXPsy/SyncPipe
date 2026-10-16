"""Run the docstring examples of the surface package (documentation that cannot rot)."""
from __future__ import annotations

import doctest
import importlib

import pytest


@pytest.mark.parametrize("module", ["syncpipe.surface.model", "syncpipe.surface.engine"])
def test_docstring_examples_run(module):
    result = doctest.testmod(importlib.import_module(module), optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0, f"{module} has no runnable examples"
    assert result.failed == 0

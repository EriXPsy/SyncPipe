"""Thin re-export so ``python -m syncpipe`` and the ``syncpipe`` console script resolve.

The real CLI implementation lives in :mod:`multisync.cli`. ``syncpipe`` is a
namespace package that mirrors ``multisync`` for packaging/CLI symmetry, so the
entry point ``syncpipe.cli:main`` must resolve to the same ``main`` object.
"""

from multisync.cli import main

__all__ = ["main"]

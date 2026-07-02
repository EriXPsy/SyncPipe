"""SyncPipe — preferred public namespace.

Use ``import syncpipe as sp`` for the clean public API. ``import multisync`` is
the legacy compatibility alias and remains available during the transition.
"""

from multisync import *  # noqa: F401,F403
from multisync import __all__ as _MULTISYNC_ALL
from multisync.__about__ import __version__  # noqa: F401

__all__ = list(_MULTISYNC_ALL)  # __version__ already in multisync.__all__
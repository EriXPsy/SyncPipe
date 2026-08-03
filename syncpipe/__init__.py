"""SyncPipe — preferred public namespace.

Use ``import syncpipe as sp`` for the clean public API. ``import multisync`` is
the legacy compatibility alias and remains available during the transition.

Submodule aliasing
------------------
``from multisync import *`` only re-exports the names in ``multisync.__all__``,
so it makes the *top-level* API available but leaves ``syncpipe.<submodule>``
unimportable. That asymmetry is user-visible and awkward to document: the README
would have to tell readers to use ``syncpipe`` for top-level objects but switch
to ``multisync`` the moment they need ``feature_definitions`` or
``pipeline_bridge``.

The finder installed below maps any ``syncpipe.X[.Y...]`` to ``multisync.X[.Y...]``
on demand, so both spellings resolve to the *same module objects*.

It must be installed at the *front* of ``sys.meta_path``. Appending it is not
enough: once ``syncpipe.validation`` is aliased, its ``__path__`` points at the
real ``multisync/validation`` directory, so the standard ``PathFinder`` — which
runs earlier — happily loads ``syncpipe.validation.l2_between_condition``
straight from that file as a *second, independent* module object. Two live copies
of a module that owns FDR family definitions is exactly the kind of split state
this alias exists to prevent, so the alias has to win first.

Winning first means the finder must then step aside for modules that genuinely
belong to this package (``syncpipe.cli``, ``syncpipe.__main__``); those names are
discovered by scanning this package's own directory rather than hard-coded, and
likewise the alias is a name mapping rather than an enumeration of ``multisync``
submodules — so neither side can drift out of sync as modules are added.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import pathlib
import sys

from multisync import *  # noqa: F401,F403
from multisync import __all__ as _MULTISYNC_ALL
from multisync.__about__ import __version__  # noqa: F401

__all__ = list(_MULTISYNC_ALL)  # __version__ already in multisync.__all__

_ALIAS_PREFIX = "syncpipe."
_TARGET_PREFIX = "multisync."


class _AliasLoader(importlib.abc.Loader):
    """Loader that returns the already-imported target module unchanged."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec):
        # Returning the target module makes `syncpipe.X is multisync.X` true, so
        # isinstance checks and module-level state (caches, registries) cannot
        # diverge between the two spellings.
        return importlib.import_module(self._target_name)

    def exec_module(self, module):
        # Already executed when the target was imported; re-executing it would
        # duplicate module-level side effects.
        pass


def _own_top_level_modules() -> frozenset:
    """Top-level module names that really live in this package.

    Scanned from the directory instead of hard-coded, so adding a real module
    beside ``cli.py`` does not silently get shadowed by the alias.
    """
    here = pathlib.Path(__file__).resolve().parent
    names = {p.stem for p in here.glob("*.py")} | {
        p.name for p in here.iterdir() if (p / "__init__.py").is_file()
    }
    return frozenset(names - {"__init__"})


_OWN_MODULES = _own_top_level_modules()


class _SyncpipeAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``syncpipe.X`` to the module ``multisync.X``."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_ALIAS_PREFIX):
            return None
        suffix = fullname[len(_ALIAS_PREFIX):]
        # Step aside for modules that genuinely belong to this package, so that
        # sitting at the front of sys.meta_path does not shadow them.
        if suffix.split(".", 1)[0] in _OWN_MODULES:
            return None
        target_name = _TARGET_PREFIX + suffix
        try:
            if importlib.util.find_spec(target_name) is None:
                return None
        except (ImportError, AttributeError, ValueError):
            # No such module under multisync either: defer to normal machinery
            # so the user gets the usual ModuleNotFoundError for their name.
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(target_name))


# Prepended: see the module docstring. PathFinder would otherwise load a second,
# independent copy of any nested submodule via the aliased parent's __path__.
if not any(isinstance(f, _SyncpipeAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SyncpipeAliasFinder())

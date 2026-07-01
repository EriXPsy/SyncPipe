"""SyncPipe real-test dataset loaders.

Each sub-module loads one public OSF dataset into the SyncPipe
``Dyad`` format used by :class:`multisync.core.DynamicAnalyzer`.

Currently shipped
-----------------
* ``gordon_2025``: Mayo & Gordon (2025) motion-tracking dyads.
* ``lerique_2024``: Lerique (2024) ECG/EDA/RESP dyad dataset (P2 pilot;
  loader complete, preprocessing stubs pending — see pre-registration
  ``docs/PRE_REGISTRATION_PILOTS.md``).
* (planned) ``han_2021``: Han, Lang & Amon (2021) media-induced SCR.
* (planned) ``andersen_2026``: Andersen et al. (2026) horror HR.
"""

from multisync.realtest.gordon_2025 import (
    GordonDyadCondition,
    load_gordon_dataset,
    gordon_record_to_multisync_dyad,
)
from multisync.realtest.lerique_2024 import (
    LeriqueDyadCondition,
    load_lerique_dataset,
    lerique_record_to_multisync_dyad,
)

__all__ = [
    "GordonDyadCondition",
    "load_gordon_dataset",
    "gordon_record_to_multisync_dyad",
    "LeriqueDyadCondition",
    "load_lerique_dataset",
    "lerique_record_to_multisync_dyad",
]

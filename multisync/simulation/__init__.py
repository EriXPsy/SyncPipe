"""
simulation/ — Computational dyad simulators for ground-truth validation.

EXPERIMENTAL — NOT FULLY VALIDATED
====================================
The Treur dyad simulator (v1) has known limitations:

1. **Symmetric weight update**: The step() function updates W_AB and W_BA
   with identical formulas. When initial weights are equal, W_AB(t) ≡ W_BA(t)
   for all t, making leader-follower asymmetry *impossible* without external
   perturbation.  GT-3 validation of directional effects is NOT supported.

2. **Non-simulated v1 scenarios**: The five scenario_* functions in
   treur_dyad.py set eta=0 and adaptation_rate=0, constructing synthetic
   data by hand rather than running the simulator. They are suitable for
   GT-1/2 (deterministic feature recovery) only, NOT for GT-4/5 (emergent
   dynamics from a running model).

3. **v2 scenarios** (treur_dyad_v2) provide a genuine simulator-driven
   scenario (scenario_emergent_sync) for GT-4 validation.  Now exported
   from this package via ``from multisync.simulation import scenario_emergent_sync``.

Modules
-------
treur_dyad
    Three-level controlled adaptive network (Hendrikse, Treur et al. 2023).
    Produces emergent dyadic dynamics with known synchrony parameters.
"""

from .treur_dyad import (
    TreurDyadResult,
    TreurDyadSimulator,
    scenario_constant_high_sync,
    scenario_frequent_switching,
    scenario_leader_follower,
    scenario_gradual_emergence,
    scenario_isc_confound,
)
from .treur_dyad_v2 import scenario_emergent_sync
from .shared_signal_model import (
    PGTResult,
    generate_signals,
    constant_coupling,
    alternating_coupling,
    trapezoidal_coupling,
    smooth_trapezoidal_coupling,
)

__all__ = [
    "TreurDyadResult",
    "TreurDyadSimulator",
    "scenario_constant_high_sync",
    "scenario_frequent_switching",
    "scenario_leader_follower",
    "scenario_gradual_emergence",
    "scenario_isc_confound",
    "scenario_emergent_sync",
    # Shared signal model (PGT backbone)
    "PGTResult",
    "generate_signals",
    "constant_coupling",
    "alternating_coupling",
    "trapezoidal_coupling",
    "smooth_trapezoidal_coupling",
]

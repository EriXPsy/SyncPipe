"""Kuramoto phase-difference simulator — extracted ground-truth generator.

This module is the single source of truth for the two-oscillator
phase-difference ODE used in the L2/L3 taxonomy validation
(``scripts/run_kuramoto_l23_taxonomy.py``).  Previously the solver was
inlined in that script, which meant the generator producing the
validation numbers had **no unit test** guarding it against silent
regressions (Blind Spot B of the v1.0 review: a mis-wired coupling
parameter or broken noise injection would still yield plausible-looking
output while the ground truth underneath was wrong).

The instantaneous synchrony of two phase oscillators at phase difference
:math:`\\Delta\\theta` is :math:`|\\cos(\\Delta\\theta/2)|` (Kuramoto order
parameter for the *difference* oscillator).  For a constant coupling ``K``
larger ``K`` pulls the phase difference toward 0, so the time-averaged
synchrony is a **monotonically increasing** function of ``K`` — that is
exactly the invariant the regression test pins down.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["solve_phase_difference", "mean_sync_from_K"]


def solve_phase_difference(
    K_func: Union[float, Callable[[float], float]],
    delta_omega: float,
    theta_0: float,
    T: float,
    n_fine: int = 2000,
) -> np.ndarray:
    """Integrate the two-oscillator phase-difference ODE.

    d/dt[Δθ] = Δω − K(t)·sin(Δθ)

    Parameters
    ----------
    K_func : float or callable
        Coupling strength.  Either a constant ``K`` or a function ``K(t)``
        of time returning the instantaneous coupling.
    delta_omega : float
        Natural frequency difference Δω.
    theta_0 : float
        Initial phase difference.
    T : float
        Integration horizon (seconds).
    n_fine : int
        Number of evaluation points (ODE solver resolution).

    Returns
    -------
    np.ndarray
        Instantaneous synchrony ``|cos(Δθ(t)/2)|`` sampled over ``[0, T]``,
        values in ``[0, 1]``.
    """
    def ode(t, y):
        K = K_func(t) if callable(K_func) else K_func
        return [delta_omega - K * np.sin(y[0])]

    t_eval = np.linspace(0, T, n_fine)
    sol = solve_ivp(
        ode, [0, T], [theta_0], t_eval=t_eval,
        method="RK45", rtol=1e-9, atol=1e-12,
    )
    delta_theta = sol.y[0]
    return np.abs(np.cos(delta_theta / 2.0))


def mean_sync_from_K(
    K_const: float,
    delta_omega: float = 0.7,
    theta_0: float = 0.0,
    T: float = 60.0,
    n_fine: int = 2000,
) -> float:
    """Time-averaged synchrony for a *constant* coupling ``K_const``.

    Convenience wrapper used by the monotonicity test: the mean of
    ``|cos(Δθ/2)|`` over a constant-K trajectory increases with ``K``.
    """
    r = solve_phase_difference(K_const, delta_omega, theta_0, T, n_fine=n_fine)
    return float(np.mean(r))

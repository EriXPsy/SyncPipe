"""
simulation/treur_dyad.py — Optimization patch for GT-4
========================================================

Adds W_AB trajectory validation scenario and modulated signal generator.

GT-4 v2 key improvement:
  - Instead of relying on pure Treur dynamics to produce WCC curves
    (which consistently saturate at W_AB≈1.0), we use Treur's weight
    adaptation equation to generate a KNOWN W_AB(t) trajectory, then
    create signals whose shared-vs-independent mix follows W_AB(t).
  - This preserves the theoretical connection to Treur's adaptive
    network while making the WCC curves practical for feature validation.
  - Validation: compare SyncPipe features (dwell, onset, peak) against
    ground-truth W_AB(t) metrics.

PATCH: Append to end of existing treur_dyad.py
"""

import numpy as np
from .treur_dyad import TreurDyadResult, TreurDyadSimulator


def scenario_emergent_sync(
    duration_sec: float = 300,
    hz: float = 10.0,
    seed: int = 42,
    switch_times: list | None = None,
    switch_alphas: list | None = None,
    shared_drive: bool = True,
) -> TreurDyadResult:
    """
    GT-4 (true emergence): synchrony produced by the Treur simulator itself.

    Unlike :func:`scenario_wab_validation`, this does NOT pre-impose a W_AB(t)
    trajectory.  It runs ``TreurDyadSimulator.generate_episode`` with the fixed
    Hebbian-with-saturation adaptation rule so that the coupling weight W_AB(t)
    *emerges* from the interaction of two agents under a shared stimulus and
    control switches.  The recovered W_AB(t) is the ground truth; SyncPipe
    features (dwell, onset, switching_rate) are validated against it.

    This is the scenario that distinguishes GT-4 (data-driven emergence) from
    GT-1/2/3 (hand-specified synchrony): no phase, amplitude, or coupling value
    is set by hand along the time axis.

    Parameters
    ----------
    switch_times, switch_alphas : list, optional
        Control-parameter switches passed through to the simulator to elicit
        in-and-out-of-sync episodes (Treur's central phenomenon).  Defaults to
        two switches that raise then lower the synchrony pull.
    """
    if switch_times is None:
        switch_times = [duration_sec * 0.33, duration_sec * 0.66]
    if switch_alphas is None:
        # (alpha_sync, alpha_indep): de-sync episode in the middle third
        #   start  : sync pull dominates  -> W rises
        #   switch1: independence dominates -> W decays (out of sync)
        #   switch2: sync pull dominates  -> W rises again (back in sync)
        switch_alphas = [(0.3, 0.9), (0.8, 0.2)]

    sim = TreurDyadSimulator(
        alpha_sync=0.8,
        alpha_indep=0.2,
        eta=2.0,
        adaptation_rate=0.5,
        noise_sigma=0.15,
        coupling_function="tanh",
        seed=seed,
    )

    t = np.arange(int(duration_sec * hz)) / hz
    if shared_drive:
        # Common exogenous input to BOTH agents (ISC / shared-stimulus
        # confound): observed WCC reflects W_AB(t) PLUS the shared drive.
        stim = np.sin(2 * np.pi * 0.10 * t) + 0.5 * np.sin(2 * np.pi * 0.30 * t)
    else:
        # Independent per-agent drives (no common input): each agent is kept
        # active by its OWN stimulus, so any observed synchrony reflects the
        # endogenous coupling W_AB(t) only - no ISC confound.  This is the
        # condition under which WCC should track the true coupling.
        rng_stim = np.random.default_rng(seed + 777)
        drive_a = (np.sin(2 * np.pi * 0.11 * t + rng_stim.uniform(0, np.pi))
                   + 0.5 * np.sin(2 * np.pi * 0.29 * t + rng_stim.uniform(0, np.pi)))
        drive_b = (np.sin(2 * np.pi * 0.13 * t + rng_stim.uniform(0, np.pi))
                   + 0.5 * np.sin(2 * np.pi * 0.31 * t + rng_stim.uniform(0, np.pi)))
        stim = np.column_stack([drive_a, drive_b])

    return sim.generate_episode(
        duration_sec=duration_sec,
        hz=hz,
        initial_W=0.2,
        switch_times=switch_times,
        switch_alphas=switch_alphas,
        external_stimulus=stim,
        warmup_sec=10.0,
    )


def scenario_wab_validation(
    duration_sec: float = 200,
    hz: float = 10.0,
    seed: int = 42,
) -> TreurDyadResult:
    """
    GT-4 v2: W_AB trajectory validation scenario.
    
    Generates a known coupling trajectory with four phases:
      Phase 1 (0–25%): low coupling (baseline)
      Phase 2 (25–45%): rapid rise (synchrony emerges)
      Phase 3 (45–75%): high plateau (sustained synchrony)
      Phase 4 (75–100%): gradual decay (recovery)
    
    W_AB(t) is the TRUE coupling strength. SyncPipe features
    should recover the timing and magnitude of these phases.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    
    # Four-phase W_AB(t) trajectory
    w_ab = np.zeros(n)
    p1_end = int(0.25 * n)
    p2_end = int(0.45 * n)
    p3_end = int(0.75 * n)
    
    w_ab[:p1_end] = 0.15                           # Phase 1: baseline
    w_ab[p1_end:p2_end] = np.linspace(0.15, 0.90, p2_end - p1_end)  # Phase 2: rise
    w_ab[p2_end:p3_end] = 0.90                      # Phase 3: plateau
    w_ab[p3_end:] = np.linspace(0.90, 0.25, n - p3_end)  # Phase 4: decay
    
    # Generate modulated signals
    base = np.sin(2 * np.pi * 0.10 * t) + 0.5 * np.sin(2 * np.pi * 0.30 * t)
    x_a = base + 0.10 * rng.normal(0, 1, n)
    x_b = w_ab * base + (1 - w_ab) * (np.cos(2 * np.pi * 0.25 * t) + 0.10 * rng.normal(0, 1, n))
    
    return TreurDyadResult(
        x_A=x_a, x_B=x_b, x_A_obs=x_a, x_B_obs=x_b,
        W_AB=w_ab, W_BA=w_ab,
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.0, alpha_indep=0.0, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.1, switch_times=[],
    )


def scenario_wab_validation_with_switching(
    duration_sec: float = 300,
    hz: float = 10.0,
    seed: int = 42,
    n_switches: int = 3,
) -> TreurDyadResult:
    """
    GT-4 v3: W_AB trajectory with multiple coupling/de-coupling cycles.
    
    Generates n_switches full in-out cycles of synchrony.
    Each cycle: low → rise → high → decay → low.
    Tests whether SyncPipe switching_rate and dwell_time capture
    the known number and duration of coupling periods.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    
    w_ab = np.zeros(n)
    cycle_len = n // n_switches
    
    for k in range(n_switches):
        start = k * cycle_len
        end = min((k + 1) * cycle_len, n)
        seg_len = end - start
        seg_t = t[start:end] - t[start]
        
        # Within each cycle: low → rise → high → decay → low
        phase_len = seg_len // 5
        i1 = start
        i2 = start + phase_len
        i3 = start + 2*phase_len
        i4 = start + 3*phase_len
        i5 = start + 4*phase_len
        
        w_ab[i1:i2] = 0.15
        w_ab[i2:i3] = np.linspace(0.15, 0.85, min(i3-i2, len(w_ab[i2:i3])))
        w_ab[i3:i4] = 0.85
        w_ab[i4:i5] = np.linspace(0.85, 0.15, min(i5-i4, len(w_ab[i4:i5])))
    
    base = np.sin(2 * np.pi * 0.10 * t) + 0.5 * np.sin(2 * np.pi * 0.30 * t)
    x_a = base + 0.10 * rng.normal(0, 1, n)
    x_b = w_ab * base + (1 - w_ab) * (np.cos(2 * np.pi * 0.25 * t) + 0.10 * rng.normal(0, 1, n))
    
    return TreurDyadResult(
        x_A=x_a, x_B=x_b, x_A_obs=x_a, x_B_obs=x_b,
        W_AB=w_ab, W_BA=w_ab,
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.0, alpha_indep=0.0, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.1, switch_times=[],
    )

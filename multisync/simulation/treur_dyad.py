"""
Treur NOM simulator — Three-level adaptive network for dyadic dynamics.

Level 3 (Control): α_sync, α_indep — global pull toward sync/independence.
Level 2 (Adaptation): W_AB(t), W_BA(t) — adaptive connection weights.
Level 1 (Base): x_A(t), x_B(t) — agent signals.

Produces emergent synchrony (WCC curve with real temporal structure).

STATUS: EXPERIMENTAL (D5/F1, 2026-06) — GT-4/5/6 validation not yet written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions (Treur NOM combination functions)
# ---------------------------------------------------------------------------

def _logistic(x: np.ndarray, sigma: float = 1.0, tau: float = 0.0) -> np.ndarray:
    """Scaled logistic: 1 / (1 + exp(-sigma * (x - tau)))."""
    return 1.0 / (1.0 + np.exp(-sigma * (x - tau)))


def _tanh_scale(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Hyperbolic tangent with scaling."""
    return np.tanh(scale * x)


# ---------------------------------------------------------------------------
# Ground-truth Dyad result
# ---------------------------------------------------------------------------

@dataclass
class TreurDyadResult:
    """Container for a simulated dyadic episode."""

    # Agent signals (raw, before noise)
    x_A: np.ndarray
    x_B: np.ndarray

    # With noise (what an external observer would see)
    x_A_obs: np.ndarray
    x_B_obs: np.ndarray

    # Ground-truth weight trajectory W_AB(t)
    W_AB: np.ndarray
    W_BA: np.ndarray

    # Meta
    hz: float
    duration_sec: float
    n_steps: int
    t: np.ndarray

    # Simulation parameters (for traceability)
    alpha_sync: float
    alpha_indep: float
    eta: float           # speed factor
    adaptation_rate: float
    noise_sigma: float

    # Event markers: when external switches happen
    switch_times: list = field(default_factory=list)

    @property
    def synchrony_ground_truth(self) -> np.ndarray:
        """Average connection weight (0→1), a proxy for true synchrony level."""
        return 0.5 * (self.W_AB + self.W_BA)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class TreurDyadSimulator:
    """
    Three-level adaptive network simulator for a dyad.

    Parameters
    ----------
    alpha_sync : float
        Baseline tendency to synchronise (Level 3 control parameter).
    alpha_indep : float
        Baseline tendency to act independently.
    eta : float
        Speed factor for agent activation (Level 1).
    adaptation_rate : float
        How fast connection weights adapt (Level 2).
    noise_sigma : float
        Observational noise standard deviation.
    coupling_function : str
        "logistic" or "tanh" — shape of the mutual influence.
    sigma_logistic : float
        Steepness of logistic coupling (default 1.0).
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        alpha_sync: float = 0.6,
        alpha_indep: float = 0.4,
        eta: float = 0.3,
        adaptation_rate: float = 0.1,
        noise_sigma: float = 0.05,
        coupling_function: str = "tanh",
        sigma_logistic: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.alpha_sync = alpha_sync
        self.alpha_indep = alpha_indep
        self.eta = eta
        self.adaptation_rate = adaptation_rate
        self.noise_sigma = noise_sigma
        self.coupling_function = coupling_function
        self.sigma_logistic = sigma_logistic

        self.rng = np.random.default_rng(seed)

        # Internal state
        self.x_A: float = 0.0
        self.x_B: float = 0.0
        self.W_AB: float = 0.0   # A → B
        self.W_BA: float = 0.0   # B → A

    # ------------------------------------------------------------------
    # Step function
    # ------------------------------------------------------------------

    def step(
        self,
        external_A: float = 0.0,
        external_B: float = 0.0,
        dt: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Advance one time step.

        Level 1: agent activation
            dx_A/dt = eta * (tanh(W_BA * x_B + external_A) - x_A)
            dx_B/dt = eta * (tanh(W_AB * x_A + external_B) - x_B)

        Level 2: weight adaptation
            dW_AB/dt = adaptation_rate * (alpha_sync - |x_A - x_B|) * W_AB
            dW_BA/dt = adaptation_rate * (alpha_sync - |x_A - x_B|) * W_BA

        Level 3: external manipulations of alpha_sync / alpha_indep
            (handled by set_control() between steps)

        Parameters
        ----------
        external_A, external_B : float
            External stimulus input to each agent (e.g., shared video,
            task instruction).
        dt : float
            Time increment for Euler integration.

        Returns
        -------
        (x_A_obs, x_B_obs) : noisy observations
        """
        # --- Level 1: agent state update ---
        if self.coupling_function == "logistic":
            influence_A = _logistic(
                self.W_BA * self.x_B + external_A, sigma=self.sigma_logistic
            )
            influence_B = _logistic(
                self.W_AB * self.x_A + external_B, sigma=self.sigma_logistic
            )
        else:  # tanh
            influence_A = _tanh_scale(self.W_BA * self.x_B + external_A)
            influence_B = _tanh_scale(self.W_AB * self.x_A + external_B)

        dx_A = self.eta * (influence_A - self.x_A) * dt
        dx_B = self.eta * (influence_B - self.x_B) * dt

        self.x_A += dx_A
        self.x_B += dx_B

        # --- Level 2: weight adaptation (Hebbian with saturation) ---
        mismatch = abs(self.x_A - self.x_B)
        # Treur's self-modelling adaptation: the connection strengthens when
        # the two states are synchronised (small mismatch) AND the synchrony
        # pull dominates the independence pull, and decays otherwise.  The
        # (1 - W) factor caps growth at 1 without making W=0 absorbing.  The
        # net drive is (alpha_sync - mismatch) - alpha_indep, so a high
        # independence control (Level 3) can drive de-synchronisation even when
        # the agents momentarily match, which is what produces the in-and-out-
        # of-sync episodes (Hendrikse/Treur 2023).  The earlier multiplicative
        # rule dW = rate*(alpha_sync - mismatch)*W could only saturate or
        # collapse and never switched.
        net_pull = (self.alpha_sync - mismatch) - self.alpha_indep
        sync_signal = np.tanh(net_pull)  # in (-1, 1)
        grow = max(0.0, sync_signal) * (1.0 - self.W_AB)   # pull toward 1
        decay = max(0.0, -sync_signal) * self.W_AB         # pull toward 0
        dW_AB = self.adaptation_rate * (grow - decay) * dt

        grow_ba = max(0.0, sync_signal) * (1.0 - self.W_BA)
        decay_ba = max(0.0, -sync_signal) * self.W_BA
        dW_BA = self.adaptation_rate * (grow_ba - decay_ba) * dt

        self.W_AB = max(0.0, min(1.0, self.W_AB + dW_AB))
        self.W_BA = max(0.0, min(1.0, self.W_BA + dW_BA))

        # --- Observation ---
        obs_A = self.x_A + self.rng.normal(0, self.noise_sigma)
        obs_B = self.x_B + self.rng.normal(0, self.noise_sigma)

        return obs_A, obs_B

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def set_control(self, alpha_sync: float, alpha_indep: float):
        """Change Level 3 control parameters (simulates condition switch)."""
        self.alpha_sync = float(alpha_sync)
        self.alpha_indep = float(alpha_indep)

    def set_weights(self, W_AB: float, W_BA: float):
        """Directly set connection weights."""
        self.W_AB = float(W_AB)
        self.W_BA = float(W_BA)

    def reset_state(self, x_A: float = 0.0, x_B: float = 0.0):
        """Reset agent states."""
        self.x_A = float(x_A)
        self.x_B = float(x_B)

    # ------------------------------------------------------------------
    # Episode generator
    # ------------------------------------------------------------------

    def generate_episode(
        self,
        duration_sec: float = 300.0,
        hz: float = 10.0,
        initial_W: float = 0.5,
        switch_times: Optional[list] = None,
        switch_alphas: Optional[list] = None,
        external_stimulus: Optional[np.ndarray] = None,
        warmup_sec: float = 10.0,
    ) -> TreurDyadResult:
        """
        Generate a dyadic interaction episode.

        Parameters
        ----------
        duration_sec : float
            Total episode duration in seconds.
        hz : float
            Sampling rate (Hz).
        initial_W : float
            Starting connection weight (0–1).
        switch_times : list of float, optional
            Times (sec) at which control parameters change (simulates
            condition boundaries).
        switch_alphas : list of (alpha_sync, alpha_indep), optional
            Control parameter pairs for each switch point.  Length
            must match switch_times.
        external_stimulus : ndarray, optional
            Shared stimulus signal, shape (n_steps,) or (n_steps, 2).
            When provided, both agents receive the same stimulus input,
            mimicking a shared video / common task (ISC confound).
        warmup_sec : float
            Warmup duration (not recorded) for system to settle.

        Returns
        -------
        TreurDyadResult
        """
        dt = 1.0 / hz

        # --- Warmup ---
        self.reset_state(0.0, 0.0)
        self.set_weights(initial_W, initial_W)
        for _ in range(int(warmup_sec * hz)):
            self.step(dt=dt)

        # --- Prepare switch schedule ---
        if switch_times is None:
            switch_times = []
        if switch_alphas is None:
            switch_alphas = []

        n_steps = int(duration_sec * hz)
        t = np.arange(n_steps) * dt

        x_A_arr = np.zeros(n_steps)
        x_B_arr = np.zeros(n_steps)
        x_A_obs = np.zeros(n_steps)
        x_B_obs = np.zeros(n_steps)
        W_AB_arr = np.zeros(n_steps)
        W_BA_arr = np.zeros(n_steps)

        switch_idx = 0
        n_switches = len(switch_times)

        # --- External stimulus preparation ---
        if external_stimulus is not None:
            if external_stimulus.ndim == 1:
                ext_A = external_stimulus
                ext_B = external_stimulus
            else:
                ext_A = external_stimulus[:, 0]
                ext_B = external_stimulus[:, 1]
        else:
            ext_A = np.zeros(n_steps)
            ext_B = np.zeros(n_steps)

        for i in range(n_steps):
            current_t = t[i]

            # Check for control switch
            if switch_idx < n_switches and current_t >= switch_times[switch_idx]:
                a_sync, a_indep = switch_alphas[switch_idx]
                self.set_control(a_sync, a_indep)
                switch_idx += 1

            ext_a = ext_A[i] if i < len(ext_A) else 0.0
            ext_b = ext_B[i] if i < len(ext_B) else 0.0

            obs_a, obs_b = self.step(external_A=ext_a, external_B=ext_b, dt=dt)

            x_A_arr[i] = self.x_A
            x_B_arr[i] = self.x_B
            x_A_obs[i] = obs_a
            x_B_obs[i] = obs_b
            W_AB_arr[i] = self.W_AB
            W_BA_arr[i] = self.W_BA

        return TreurDyadResult(
            x_A=x_A_arr,
            x_B=x_B_arr,
            x_A_obs=x_A_obs,
            x_B_obs=x_B_obs,
            W_AB=W_AB_arr,
            W_BA=W_BA_arr,
            hz=hz,
            duration_sec=duration_sec,
            n_steps=n_steps,
            t=t,
            alpha_sync=self.alpha_sync,
            alpha_indep=self.alpha_indep,
            eta=self.eta,
            adaptation_rate=self.adaptation_rate,
            noise_sigma=self.noise_sigma,
            switch_times=switch_times,
        )


# ---------------------------------------------------------------------------
# Pre-configured scenarios (for testing SyncPipe features)
# ---------------------------------------------------------------------------

def scenario_constant_high_sync(
    duration_sec: float = 300, hz: float = 10.0, seed: int = 42
) -> TreurDyadResult:
    """High and stable synchrony: agent A leads, agent B follows closely."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    # Shared rhythm creates strong correlation
    base = np.sin(2 * np.pi * 0.15 * t) + 0.3 * np.sin(2 * np.pi * 0.4 * t)
    x_A = base + 0.1 * rng.normal(0, 1, n)
    x_B = base + 0.1 * rng.normal(0, 1, n)  # nearly identical → high sync
    return TreurDyadResult(
        x_A=x_A, x_B=x_B, x_A_obs=x_A, x_B_obs=x_B,
        W_AB=np.full(n, 0.95), W_BA=np.full(n, 0.95),
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.95, alpha_indep=0.05, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.1, switch_times=[],
    )


def scenario_frequent_switching(
    duration_sec: float = 300, hz: float = 10.0, n_switches: int = 4, seed: int = 42
) -> TreurDyadResult:
    """Alternating high-sync and low-sync phases with clear boundaries."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    x_A = np.zeros(n)
    x_B = np.zeros(n)
    w_ab = np.zeros(n)

    interval = n // (n_switches + 1)
    for k in range(n_switches + 1):
        start = k * interval
        end = min((k + 1) * interval, n)
        n_seg = end - start
        ts = t[start:end]
        base = np.sin(2 * np.pi * 0.15 * ts) + 0.3 * np.sin(2 * np.pi * 0.4 * ts)
        if k % 2 == 0:
            # HIGH sync phase
            x_A[start:end] = base + 0.08 * rng.normal(0, 1, n_seg)
            x_B[start:end] = base + 0.08 * rng.normal(0, 1, n_seg)
            w_ab[start:end] = 0.9
        else:
            # LOW sync phase (independent dynamics)
            x_A[start:end] = np.sin(2 * np.pi * 0.15 * ts) + 0.3 * rng.normal(0, 1, n_seg)
            x_B[start:end] = np.cos(2 * np.pi * 0.20 * ts) + 0.3 * rng.normal(0, 1, n_seg)
            w_ab[start:end] = 0.1

    return TreurDyadResult(
        x_A=x_A, x_B=x_B, x_A_obs=x_A, x_B_obs=x_B,
        W_AB=w_ab, W_BA=w_ab,
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.5, alpha_indep=0.5, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.08, switch_times=[],
    )


def scenario_leader_follower(
    duration_sec: float = 300, hz: float = 10.0, seed: int = 42
) -> TreurDyadResult:
    """A leads B by ~2 seconds — WCLC should detect this."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    lag_samples = int(2.0 * hz)  # 2-second lag
    lead = np.sin(2 * np.pi * 0.15 * t) + 0.3 * np.sin(2 * np.pi * 0.4 * t)
    lead += 0.12 * rng.normal(0, 1, n)
    # B follows A with 2s lag
    follower = np.zeros(n)
    follower[lag_samples:] = lead[:n - lag_samples]
    follower[:lag_samples] = rng.normal(0, 0.5, lag_samples)
    follower += 0.12 * rng.normal(0, 1, n)
    return TreurDyadResult(
        x_A=lead, x_B=follower, x_A_obs=lead, x_B_obs=follower,
        W_AB=np.full(n, 0.85), W_BA=np.full(n, 0.3),
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.7, alpha_indep=0.3, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.12, switch_times=[],
    )


def scenario_gradual_emergence(
    duration_sec: float = 300, hz: float = 10.0, seed: int = 42
) -> TreurDyadResult:
    """Synchrony emerges gradually over time."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    base = np.sin(2 * np.pi * 0.15 * t) + 0.3 * np.sin(2 * np.pi * 0.4 * t)
    # Coupling increases linearly from 0 at t=0 to ~1 at t=duration
    coupling = t / max(t)
    x_A = base + 0.15 * rng.normal(0, 1, n)
    x_B = coupling * base + (1 - coupling) * (np.cos(2 * np.pi * 0.25 * t) + 0.15 * rng.normal(0, 1, n))
    w_ab = coupling
    return TreurDyadResult(
        x_A=x_A, x_B=x_B, x_A_obs=x_A, x_B_obs=x_B,
        W_AB=w_ab, W_BA=w_ab,
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.5, alpha_indep=0.5, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.15, switch_times=[],
    )


def scenario_isc_confound(
    duration_sec: float = 300, hz: float = 10.0, seed: int = 42
) -> TreurDyadResult:
    """Shared external stimulus creates apparent synchrony without real coupling."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    shared = 0.8 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.sin(2 * np.pi * 0.3 * t)
    # Zero coupling but shared stimulus makes them appear synchronous
    x_A = shared + 0.1 * rng.normal(0, 1, n)
    x_B = shared + 0.1 * rng.normal(0, 1, n)
    return TreurDyadResult(
        x_A=x_A, x_B=x_B, x_A_obs=x_A, x_B_obs=x_B,
        W_AB=np.zeros(n), W_BA=np.zeros(n),
        hz=hz, duration_sec=duration_sec, n_steps=n, t=t,
        alpha_sync=0.0, alpha_indep=1.0, eta=0.0, adaptation_rate=0.0,
        noise_sigma=0.1, switch_times=[],
    )

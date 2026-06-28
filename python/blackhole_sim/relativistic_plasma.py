"""Relativistic charged-particle pushers and guiding-center validation.

The module provides two explicit relativistic particle pushers commonly used as
validation baselines and a gyro-averaged relativistic guiding-center pusher for
uniform/slowly varying fields. It is intentionally separated from the GR metric
code so local tetrad-frame plasma algorithms can be tested before coupling to a
full GRPIC/GRMHD field solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

VectorField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RelativisticParticleState:
    x: np.ndarray  # position
    u: np.ndarray  # proper velocity gamma * v, with c configurable


@dataclass(frozen=True)
class RelativisticGuidingCenterState:
    R: np.ndarray
    u_parallel: np.ndarray
    mu: np.ndarray


def _asvec(a: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("Expected vectors with trailing dimension 3")
    return arr


def gamma_from_u(u: np.ndarray, c: float = 1.0) -> np.ndarray:
    u = _asvec(u)
    return np.sqrt(1.0 + np.sum(u * u, axis=-1, keepdims=True) / (c * c))


def velocity_from_u(u: np.ndarray, c: float = 1.0) -> np.ndarray:
    u = _asvec(u)
    return u / gamma_from_u(u, c=c)


def relativistic_boris_step(
    x: np.ndarray,
    u: np.ndarray,
    q_over_m: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
    c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Relativistic Boris step using proper velocity u = gamma v."""
    x = _asvec(x)
    u = _asvec(u)
    E = _asvec(e_field)
    B = _asvec(b_field)

    u_minus = u + 0.5 * q_over_m * E * dt
    gamma_minus = gamma_from_u(u_minus, c=c)
    t = 0.5 * q_over_m * B * dt / gamma_minus
    t2 = np.sum(t * t, axis=-1, keepdims=True)
    s = 2.0 * t / (1.0 + t2)
    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)
    u_new = u_plus + 0.5 * q_over_m * E * dt
    x_new = x + velocity_from_u(u_new, c=c) * dt
    return x_new, u_new


def vay_step(
    x: np.ndarray,
    u: np.ndarray,
    q_over_m: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
    c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vay relativistic particle pusher.

    The implementation follows the common normalized-c=1 formulation with u as
    proper velocity. It is useful in relativistic E x B drift tests where the
    ordinary Boris pusher can show frame-dependent drift errors.
    """
    if c != 1.0:
        # Keep this implementation unambiguous. Users needing SI can normalize
        # fields and time first or add the c-rescaled form with tests.
        raise NotImplementedError("vay_step currently expects normalized c=1 units")
    x = _asvec(x)
    u = _asvec(u)
    E = _asvec(e_field)
    B = _asvec(b_field)

    u_minus = u + 0.5 * q_over_m * E * dt
    gamma_minus = gamma_from_u(u_minus, c=1.0)
    tau = 0.5 * q_over_m * B * dt
    u_prime = u_minus + np.cross(u_minus, tau) / gamma_minus
    tau2 = np.sum(tau * tau, axis=-1, keepdims=True)
    sigma = gamma_minus * gamma_minus - tau2
    u_dot_tau = np.sum(u_prime * tau, axis=-1, keepdims=True)
    gamma_new = np.sqrt(0.5 * (sigma + np.sqrt(sigma * sigma + 4.0 * (tau2 + u_dot_tau * u_dot_tau))))
    t = tau / gamma_new
    t2 = np.sum(t * t, axis=-1, keepdims=True)
    u_plus = (u_prime + np.sum(u_prime * t, axis=-1, keepdims=True) * t + np.cross(u_prime, t)) / (1.0 + t2)
    u_new = u_plus + 0.5 * q_over_m * E * dt
    x_new = x + velocity_from_u(u_new, c=1.0) * dt
    return x_new, u_new


def push_particles(
    state: RelativisticParticleState,
    q_over_m: float,
    e_func: VectorField,
    b_func: VectorField,
    dt: float,
    steps: int,
    method: str = "vay",
    c: float = 1.0,
) -> RelativisticParticleState:
    x = np.asarray(state.x, dtype=float).copy()
    u = np.asarray(state.u, dtype=float).copy()
    stepper = vay_step if method.lower() == "vay" else relativistic_boris_step
    for _ in range(steps):
        E = e_func(x)
        B = b_func(x)
        x, u = stepper(x, u, q_over_m, E, B, dt, c=c)
    return RelativisticParticleState(x=x, u=u)


def relativistic_guiding_center_from_particle(
    x: np.ndarray,
    u: np.ndarray,
    q_over_m: float,
    b_field: np.ndarray,
    c: float = 1.0,
) -> RelativisticGuidingCenterState:
    x = _asvec(x)
    u = _asvec(u)
    B = _asvec(b_field)
    Bmag = np.linalg.norm(B, axis=-1, keepdims=True)
    if np.any(Bmag <= 0):
        raise ValueError("Guiding-center initialization requires nonzero B")
    b = B / Bmag
    u_par = np.sum(u * b, axis=-1, keepdims=True) * b
    u_perp = u - u_par
    gamma = gamma_from_u(u, c=c)
    # Relativistic magnetic moment proxy in normalized units.
    mu = np.sum(u_perp * u_perp, axis=-1, keepdims=True) / (2.0 * Bmag * gamma)
    # Larmor radius vector for uniform B in normalized variables.
    rho = np.cross(b, u_perp) / (q_over_m * Bmag)
    return RelativisticGuidingCenterState(R=x - rho, u_parallel=u_par, mu=mu)


def relativistic_guiding_center_step(
    state: RelativisticGuidingCenterState,
    q_over_m: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
    c: float = 1.0,
) -> RelativisticGuidingCenterState:
    R = _asvec(state.R)
    u_par = _asvec(state.u_parallel)
    E = _asvec(e_field)
    B = _asvec(b_field)
    Bmag = np.linalg.norm(B, axis=-1, keepdims=True)
    if np.any(Bmag <= 0):
        raise ValueError("Guiding-center step requires nonzero B")
    b = B / Bmag
    E_par = np.sum(E * b, axis=-1, keepdims=True) * b
    u_par_new = u_par + q_over_m * E_par * dt
    gamma_par = gamma_from_u(u_par_new, c=c)
    v_par = u_par_new / gamma_par
    # Relativistic E x B drift in normalized units; valid for magnetized regime
    # and subluminal drift |E_perp| < |B| in c=1 units.
    v_exb = np.cross(E, B) / np.sum(B * B, axis=-1, keepdims=True)
    speed = np.linalg.norm(v_exb, axis=-1, keepdims=True)
    if np.any(speed >= c):
        # Clamp to remain causal instead of silently producing nonsense.
        v_exb = v_exb * ((0.999999 * c) / np.maximum(speed, 1.0e-30))
    R_new = R + (v_par + v_exb) * dt
    return RelativisticGuidingCenterState(R=R_new, u_parallel=u_par_new, mu=state.mu)


def push_guiding_center(
    state: RelativisticGuidingCenterState,
    q_over_m: float,
    e_func: VectorField,
    b_func: VectorField,
    dt: float,
    steps: int,
    c: float = 1.0,
) -> RelativisticGuidingCenterState:
    R = np.asarray(state.R, dtype=float).copy()
    u_par = np.asarray(state.u_parallel, dtype=float).copy()
    mu = np.asarray(state.mu, dtype=float).copy()
    gc = RelativisticGuidingCenterState(R=R, u_parallel=u_par, mu=mu)
    for _ in range(steps):
        gc = relativistic_guiding_center_step(gc, q_over_m, e_func(gc.R), b_func(gc.R), dt, c=c)
    return gc


def relativistic_gyro_dt(q_over_m: float, Bmag: float, gamma: float, points_per_gyration: int = 96) -> float:
    omega = abs(q_over_m) * Bmag / max(gamma, 1.0e-30)
    if omega <= 0:
        raise ValueError("Gyro timestep requires nonzero q/m and B")
    return 2.0 * np.pi / (omega * points_per_gyration)

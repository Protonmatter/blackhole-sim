"""Charged-particle algorithms for black-hole plasma scaffolding.

The OpenAI article focuses on a hard plasma-simulation bottleneck: charged
particles spiral rapidly around magnetic field lines, which forces tiny
timesteps if every gyro-orbit is explicitly resolved. This module includes two
contrasting algorithms:

1. Boris pusher: robust explicit particle pusher that resolves gyromotion.
2. Guiding-center pusher: advances the averaged gyro-center instead of every
   tiny spiral, useful when the magnetic field varies slowly over one gyroradius.

This is intentionally non-relativistic and flat-spacetime. It is a clean place
to test algorithmic ideas before replacing the fields/metric with GR versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

VectorField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParticleState:
    x: np.ndarray  # shape (..., 3)
    v: np.ndarray  # shape (..., 3)


def _asvec(a: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("Expected vectors with trailing dimension 3")
    return arr


def boris_step(
    x: np.ndarray,
    v: np.ndarray,
    q_over_m: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one non-relativistic charged particle step with Boris rotation.

    The Boris method splits acceleration into E/2, B rotation, E/2. It is widely
    used in particle-in-cell codes because it handles magnetic rotation without
    secular energy growth in the E=0 case.
    """
    x = _asvec(x)
    v = _asvec(v)
    e = _asvec(e_field)
    b = _asvec(b_field)

    v_minus = v + 0.5 * q_over_m * e * dt
    t = 0.5 * q_over_m * b * dt
    t2 = np.sum(t * t, axis=-1, keepdims=True)
    s = 2.0 * t / (1.0 + t2)
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_new = v_plus + 0.5 * q_over_m * e * dt
    x_new = x + v_new * dt
    return x_new, v_new


def boris_push(
    state: ParticleState,
    q_over_m: float,
    e_func: VectorField,
    b_func: VectorField,
    dt: float,
    steps: int,
) -> ParticleState:
    """Push particles by resolving gyromotion."""
    x = np.asarray(state.x, dtype=float).copy()
    v = np.asarray(state.v, dtype=float).copy()
    for _ in range(steps):
        e = e_func(x)
        b = b_func(x)
        x, v = boris_step(x, v, q_over_m, e, b, dt)
    return ParticleState(x=x, v=v)


def gyrofrequency(q_over_m: float, b_magnitude: float) -> float:
    return abs(q_over_m) * abs(b_magnitude)


def gyroradius(v_perp: float, q_over_m: float, b_magnitude: float) -> float:
    omega = gyrofrequency(q_over_m, b_magnitude)
    if omega == 0.0:
        return np.inf
    return abs(v_perp) / omega


def recommended_boris_dt(q_over_m: float, b_magnitude: float, points_per_gyration: int = 64) -> float:
    """Timestep small enough to resolve a gyro-orbit with N points."""
    omega = gyrofrequency(q_over_m, b_magnitude)
    if omega == 0.0:
        return np.inf
    return 2.0 * np.pi / (omega * points_per_gyration)


@dataclass(frozen=True)
class GuidingCenterState:
    R: np.ndarray          # gyro-center position, shape (..., 3)
    v_parallel: np.ndarray # speed along b-hat, shape (..., 1) or (...,)
    mu: np.ndarray         # magnetic moment proxy, m v_perp^2/(2B); m=1 here


def guiding_center_step_uniform(
    R: np.ndarray,
    v_parallel: np.ndarray,
    mu: np.ndarray,
    q_over_m: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-field guiding-center step.

    For uniform B and E, the gyro-averaged perpendicular drift is E x B / B².
    This skips the orbital timestep constraint and advances the center directly.
    The magnetic moment is held constant.
    """
    R = _asvec(R)
    e = _asvec(e_field)
    b = _asvec(b_field)
    B2 = np.sum(b * b, axis=-1, keepdims=True)
    Bmag = np.sqrt(np.maximum(B2, 1e-30))
    bhat = b / Bmag
    vp = np.asarray(v_parallel, dtype=float)
    if vp.ndim == R.ndim - 1:
        vp = np.expand_dims(vp, axis=-1)

    e_parallel = np.sum(e * bhat, axis=-1, keepdims=True)
    drift_exb = np.cross(e, b) / np.maximum(B2, 1e-30)
    R_new = R + (vp * bhat + drift_exb) * dt
    vp_new = vp + q_over_m * e_parallel * dt
    return R_new, np.squeeze(vp_new, axis=-1), mu


def guiding_center_push(
    state: GuidingCenterState,
    q_over_m: float,
    e_func: VectorField,
    b_func: VectorField,
    dt: float,
    steps: int,
) -> GuidingCenterState:
    """Push gyro-centers without resolving every small spiral.

    This implementation uses the uniform-field local approximation at each
    position. For production use, add grad-B, curvature, polarization drift, and
    relativistic/metric corrections.
    """
    R = np.asarray(state.R, dtype=float).copy()
    vp = np.asarray(state.v_parallel, dtype=float).copy()
    mu = np.asarray(state.mu, dtype=float).copy()
    for _ in range(steps):
        e = e_func(R)
        b = b_func(R)
        R, vp, mu = guiding_center_step_uniform(R, vp, mu, q_over_m, e, b, dt)
    return GuidingCenterState(R=R, v_parallel=vp, mu=mu)


def decompose_velocity(v: np.ndarray, b_field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return v_parallel scalar, v_parallel vector, v_perp vector."""
    v = _asvec(v)
    b = _asvec(b_field)
    bhat = b / np.sqrt(np.maximum(np.sum(b * b, axis=-1, keepdims=True), 1e-30))
    vp_scalar = np.sum(v * bhat, axis=-1)
    vp_vec = np.expand_dims(vp_scalar, axis=-1) * bhat
    return vp_scalar, vp_vec, v - vp_vec


def initial_guiding_center_from_particle(
    x: np.ndarray,
    v: np.ndarray,
    q_over_m: float,
    b_field: np.ndarray,
) -> GuidingCenterState:
    """Approximate initial guiding center for a uniform magnetic field."""
    x = _asvec(x)
    v = _asvec(v)
    b = _asvec(b_field)
    Bmag = np.sqrt(np.maximum(np.sum(b * b, axis=-1, keepdims=True), 1e-30))
    bhat = b / Bmag
    vp_scalar, _vp_vec, vperp = decompose_velocity(v, b)
    # Larmor radius vector for positive q/m convention.
    rho_vec = np.cross(vperp, bhat) / np.maximum(q_over_m * Bmag, 1e-30)
    R = x + rho_vec
    mu = np.sum(vperp * vperp, axis=-1) / (2.0 * np.squeeze(Bmag, axis=-1))
    return GuidingCenterState(R=R, v_parallel=vp_scalar, mu=mu)

"""Null geodesics in Schwarzschild spacetime.

Units
-----
The code uses geometrized units: G = c = M = 1 by default. In these units:
- Event horizon radius is r_h = 2M.
- Photon sphere radius is r_ph = 3M.
- Critical photon impact parameter is b_crit = sqrt(27) M.

Algorithm
---------
For a Schwarzschild black hole, spherical symmetry lets each photon path live in
one plane. In that plane the orbit equation for a null geodesic is

    d²u/dphi² + u = 3 M u²,        where u = 1/r.

We integrate this second-order ODE with RK4. This is good enough for a compact,
readable educational renderer and avoids symbolic Christoffel code in the inner
render loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Status = Literal["captured", "escaped", "max_steps", "radial"]


@dataclass(frozen=True)
class RayTraceResult:
    """Output of one geodesic trace."""

    status: Status
    positions: np.ndarray  # shape: (N, 3)
    phi: np.ndarray        # shape: (N,)
    r: np.ndarray          # shape: (N,)
    impact_parameter: float


def normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n


def schwarzschild_f(r: float | np.ndarray, mass: float = 1.0) -> float | np.ndarray:
    return 1.0 - 2.0 * mass / r


def horizon_radius(mass: float = 1.0) -> float:
    return 2.0 * mass


def photon_sphere_radius(mass: float = 1.0) -> float:
    return 3.0 * mass


def critical_impact_parameter(mass: float = 1.0) -> float:
    """Critical null-geodesic impact parameter for Schwarzschild capture."""
    return float(np.sqrt(27.0) * mass)


def _rk4_step(u: float, v: float, h: float, mass: float) -> tuple[float, float]:
    """Advance u'=v, v'=3 M u^2 - u by one RK4 step in phi."""

    def deriv(y: tuple[float, float]) -> tuple[float, float]:
        uu, vv = y
        return vv, 3.0 * mass * uu * uu - uu

    k1u, k1v = deriv((u, v))
    k2u, k2v = deriv((u + 0.5 * h * k1u, v + 0.5 * h * k1v))
    k3u, k3v = deriv((u + 0.5 * h * k2u, v + 0.5 * h * k2v))
    k4u, k4v = deriv((u + h * k3u, v + h * k3v))

    u_new = u + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    v_new = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return u_new, v_new


def trace_null_geodesic(
    camera_position: np.ndarray,
    ray_direction: np.ndarray,
    *,
    mass: float = 1.0,
    dphi: float = 0.0025,
    max_steps: int = 12000,
    escape_radius: float = 90.0,
    horizon_epsilon: float = 1e-4,
) -> RayTraceResult:
    """Trace a photon from a camera through Schwarzschild spacetime.

    Parameters
    ----------
    camera_position:
        3D Cartesian camera position in black-hole-centered coordinates.
    ray_direction:
        Initial local ray direction. This is interpreted in the camera's local
        Euclidean/orthonormal frame approximation. For distant cameras this is
        accurate enough for visualization.
    mass:
        Black-hole mass in geometrized units.
    dphi:
        Angular integration step. Smaller is more accurate and slower.
    max_steps:
        Safety cap.
    escape_radius:
        Stop after the ray bends back outward and exceeds this radius.
    horizon_epsilon:
        Stop as captured just outside r=2M.

    Returns
    -------
    RayTraceResult
        Contains the path, final status, and impact parameter.
    """
    cam = np.asarray(camera_position, dtype=float)
    direction = normalize(np.asarray(ray_direction, dtype=float))

    r0 = float(np.linalg.norm(cam))
    if r0 <= horizon_radius(mass):
        raise ValueError("Camera must be outside the event horizon")

    e_radial = cam / r0
    radial_component = float(np.dot(direction, e_radial))
    tangent = direction - radial_component * e_radial
    tangent_norm = float(np.linalg.norm(tangent))

    # Exactly radial rays do not define an orbital plane. Integrate the obvious
    # straight radial limit: inward rays are captured, outward rays escape.
    if tangent_norm < 1e-12:
        status: Status = "captured" if radial_component < 0 else "escaped"
        end_r = horizon_radius(mass) * (1.0 + horizon_epsilon) if status == "captured" else escape_radius
        positions = np.vstack([cam, e_radial * end_r])
        return RayTraceResult(
            status="radial" if status == "escaped" else "captured",
            positions=positions,
            phi=np.array([0.0, 0.0]),
            r=np.array([r0, end_r]),
            impact_parameter=0.0,
        )

    e_tangent = tangent / tangent_norm

    # Impact parameter b = L/E for a photon emitted by a static observer.
    # The sqrt(f) factor converts the local orthonormal angle to the conserved b.
    f0 = float(schwarzschild_f(r0, mass))
    if f0 <= 0:
        raise ValueError("Camera must be outside the horizon")
    impact = r0 * tangent_norm / np.sqrt(f0)

    u = 1.0 / r0
    # From the first integral: (du/dphi)^2 = 1/b^2 - u^2 + 2 M u^3.
    v_mag_sq = max((1.0 / (impact * impact)) - u * u + 2.0 * mass * u * u * u, 0.0)
    # If the ray initially points inward, r decreases and u increases.
    v = -np.sign(radial_component) * np.sqrt(v_mag_sq)
    if abs(radial_component) < 1e-14:
        v = 0.0

    positions: list[np.ndarray] = []
    phis: list[float] = []
    radii: list[float] = []

    status: Status = "max_steps"
    phi = 0.0
    r_prev = r0

    for _ in range(max_steps):
        if u <= 0.0:
            status = "escaped"
            break

        r = 1.0 / u
        pos = r * (np.cos(phi) * e_radial + np.sin(phi) * e_tangent)
        positions.append(pos)
        phis.append(phi)
        radii.append(r)

        if r <= horizon_radius(mass) * (1.0 + horizon_epsilon):
            status = "captured"
            break

        # Escaped after reaching a turning point and moving back outward.
        if r > escape_radius and len(radii) > 8 and r > r_prev:
            status = "escaped"
            break

        r_prev = r
        u, v = _rk4_step(u, v, dphi, mass)
        phi += dphi

    if not positions:
        positions = [cam]
        phis = [0.0]
        radii = [r0]

    return RayTraceResult(
        status=status,
        positions=np.asarray(positions),
        phi=np.asarray(phis),
        r=np.asarray(radii),
        impact_parameter=float(impact),
    )

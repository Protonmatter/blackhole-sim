"""Kerr spacetime, camera tetrads, orbital mechanics, and null geodesics.

This module uses the full Kerr metric in Boyer-Lindquist coordinates
(t, r, theta, phi) and integrates null geodesics with a Hamiltonian system.
It is not a weak-field approximation and it does not reduce spin to a visual
post-processing parameter.

Conventions
-----------
- Geometrized units: G = c = M = 1.
- Metric signature: (-, +, +, +).
- Dimensionless spin is ``a`` where |a| < 1.
- Covariant photon momentum is p_mu = (p_t, p_r, p_theta, p_phi).
- Null Hamiltonian is H = 0.5 g^{mu nu} p_mu p_nu = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

Status = Literal["captured", "escaped", "max_steps", "invalid"]

_EPS_THETA = 1.0e-7


def _validate_spin(a: float) -> None:
    if not np.isfinite(a) or abs(a) >= 1.0:
        raise ValueError("Kerr spin must be finite and satisfy |a| < 1 for this integrator")


def _safe_theta(theta: float) -> float:
    return float(np.clip(theta, _EPS_THETA, math.pi - _EPS_THETA))


def horizon_radius(a: float) -> float:
    """Outer Kerr horizon r_+ in units of M."""
    _validate_spin(a)
    return 1.0 + math.sqrt(1.0 - a * a)


def static_limit_radius(theta: float, a: float) -> float:
    """Outer static-limit surface r_s(theta) in units of M."""
    _validate_spin(a)
    th = _safe_theta(theta)
    return 1.0 + math.sqrt(1.0 - a * a * math.cos(th) ** 2)


def sigma(r: float, theta: float, a: float) -> float:
    th = _safe_theta(theta)
    return r * r + a * a * math.cos(th) ** 2


def delta(r: float, a: float) -> float:
    return r * r - 2.0 * r + a * a


def big_a(r: float, theta: float, a: float) -> float:
    th = _safe_theta(theta)
    s2 = math.sin(th) ** 2
    return (r * r + a * a) ** 2 - a * a * delta(r, a) * s2


def kerr_metric_covariant(r: float, theta: float, a: float) -> np.ndarray:
    """Return g_{mu nu} for Kerr in Boyer-Lindquist coordinates."""
    _validate_spin(a)
    th = _safe_theta(theta)
    sig = sigma(r, th, a)
    dlt = delta(r, a)
    s = math.sin(th)
    s2 = s * s
    A = big_a(r, th, a)

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -(1.0 - 2.0 * r / sig)
    g[0, 3] = g[3, 0] = -2.0 * a * r * s2 / sig
    g[1, 1] = sig / dlt
    g[2, 2] = sig
    g[3, 3] = A * s2 / sig
    return g


def kerr_metric_contravariant(r: float, theta: float, a: float) -> np.ndarray:
    """Return g^{mu nu} for Kerr in Boyer-Lindquist coordinates."""
    _validate_spin(a)
    th = _safe_theta(theta)
    sig = sigma(r, th, a)
    dlt = delta(r, a)
    s = math.sin(th)
    s2 = max(s * s, _EPS_THETA * _EPS_THETA)
    A = big_a(r, th, a)

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -A / (sig * dlt)
    g[0, 3] = g[3, 0] = -2.0 * a * r / (sig * dlt)
    g[1, 1] = dlt / sig
    g[2, 2] = 1.0 / sig
    g[3, 3] = (dlt - a * a * s2) / (sig * dlt * s2)
    return g


def lower_index(p_contravariant: np.ndarray, r: float, theta: float, a: float) -> np.ndarray:
    return kerr_metric_covariant(r, theta, a) @ np.asarray(p_contravariant, dtype=float)


def raise_index(p_covariant: np.ndarray, r: float, theta: float, a: float) -> np.ndarray:
    return kerr_metric_contravariant(r, theta, a) @ np.asarray(p_covariant, dtype=float)


def hamiltonian_null(y: np.ndarray, a: float, p_t: float, p_phi: float) -> float:
    """Return H = 0.5 g^{mu nu} p_mu p_nu for a photon state.

    y = [t, r, theta, phi, p_r, p_theta].
    """
    _, r, th, _, p_r, p_th = y
    pcov = np.array([p_t, p_r, p_th, p_phi], dtype=float)
    gcon = kerr_metric_contravariant(float(r), float(th), a)
    return float(0.5 * pcov @ gcon @ pcov)


def zamo_tetrad(r: float, theta: float, a: float) -> np.ndarray:
    """Return coordinate-basis vectors for a local ZAMO orthonormal tetrad.

    Rows are e_(t), e_(r), e_(theta), e_(phi), each expressed as components
    in the Boyer-Lindquist coordinate basis. The tetrad is useful for launching
    camera rays with physically meaningful local angles.
    """
    _validate_spin(a)
    th = _safe_theta(theta)
    sig = sigma(r, th, a)
    dlt = delta(r, a)
    A = big_a(r, th, a)
    if dlt <= 0.0:
        raise ValueError("ZAMO tetrad must be outside the outer horizon")
    s = math.sin(th)
    gcov = kerr_metric_covariant(r, th, a)
    g_phiphi = gcov[3, 3]
    lapse = math.sqrt(sig * dlt / A)
    omega = 2.0 * a * r / A

    tetrad = np.zeros((4, 4), dtype=float)
    tetrad[0] = np.array([1.0 / lapse, 0.0, 0.0, omega / lapse])
    tetrad[1] = np.array([0.0, math.sqrt(dlt / sig), 0.0, 0.0])
    tetrad[2] = np.array([0.0, 0.0, 1.0 / math.sqrt(sig), 0.0])
    tetrad[3] = np.array([0.0, 0.0, 0.0, 1.0 / math.sqrt(g_phiphi)])
    # Guard against numerical pole pathologies.
    if abs(s) < _EPS_THETA:
        raise ValueError("ZAMO tetrad is ill-conditioned at the coordinate pole")
    return tetrad


@dataclass(frozen=True)
class LocalCamera:
    """Observer camera expressed in physically local Kerr/ZAMO coordinates.

    ``theta`` is the viewing inclination in Boyer-Lindquist coordinates. theta=0
    is pole-on; theta=pi/2 is edge-on. The camera's central ray points inward
    along -e_(r) in the local orthonormal frame.
    """

    r: float = 50.0
    theta: float = math.radians(60.0)
    phi: float = 0.0
    fov_y_degrees: float = 35.0
    roll_degrees: float = 0.0

    @classmethod
    def from_degrees(
        cls,
        r: float = 50.0,
        inclination_degrees: float = 60.0,
        phi_degrees: float = 0.0,
        fov_y_degrees: float = 35.0,
        roll_degrees: float = 0.0,
    ) -> "LocalCamera":
        return cls(
            r=r,
            theta=math.radians(inclination_degrees),
            phi=math.radians(phi_degrees),
            fov_y_degrees=fov_y_degrees,
            roll_degrees=roll_degrees,
        )


def camera_ray_initial_state(
    camera: LocalCamera,
    a: float,
    ndc_x: float,
    ndc_y: float,
    aspect: float,
) -> tuple[np.ndarray, float, float]:
    """Launch a future-directed photon from a local camera pixel.

    Parameters
    ----------
    ndc_x, ndc_y:
        Normalized device coordinates in [-1, 1]. x positive is local +phi;
        y positive is local -theta on the image plane.

    Returns
    -------
    y0, p_t, p_phi
        y0 = [t, r, theta, phi, p_r, p_theta]. p_t and p_phi are conserved.
    """
    th = _safe_theta(camera.theta)
    fov_y = math.radians(camera.fov_y_degrees)
    tan_y = math.tan(0.5 * fov_y)
    tan_x = aspect * tan_y

    # Image-plane directions in the ZAMO local orthonormal basis.
    # Components are ordered as (r, theta, phi). Central ray is inward: n_r=-1.
    n_r = -1.0
    n_theta = -ndc_y * tan_y
    n_phi = ndc_x * tan_x

    # Optional roll around the viewing axis.
    if camera.roll_degrees:
        roll = math.radians(camera.roll_degrees)
        ct, st = math.cos(roll), math.sin(roll)
        n_theta, n_phi = ct * n_theta - st * n_phi, st * n_theta + ct * n_phi

    n = np.array([n_r, n_theta, n_phi], dtype=float)
    n /= np.linalg.norm(n)

    tetrad = zamo_tetrad(camera.r, th, a)
    # p^(hat a) = (1, n_r, n_theta, n_phi). Convert to coordinate p^mu.
    p_con = tetrad[0] + n[0] * tetrad[1] + n[1] * tetrad[2] + n[2] * tetrad[3]
    p_cov = lower_index(p_con, camera.r, th, a)
    y0 = np.array([0.0, camera.r, th, camera.phi, p_cov[1], p_cov[2]], dtype=float)
    return y0, float(p_cov[0]), float(p_cov[3])


def _metric_derivative(r: float, theta: float, a: float, wrt: Literal["r", "theta"]) -> np.ndarray:
    """Analytic derivative of g^{mu nu} with respect to r or theta."""
    th = _safe_theta(theta)
    s = math.sin(th)
    c = math.cos(th)
    s2 = max(s * s, _EPS_THETA * _EPS_THETA)
    ds2 = 2.0 * s * c
    sig = sigma(r, th, a)
    dlt = delta(r, a)
    A = big_a(r, th, a)

    if wrt == "r":
        sig_p = 2.0 * r
        dlt_p = 2.0 * r - 2.0
        A_p = 4.0 * r * (r * r + a * a) - a * a * dlt_p * s2
        s2_p = 0.0
    else:
        sig_p = -a * a * ds2
        dlt_p = 0.0
        A_p = -a * a * dlt * ds2
        s2_p = ds2

    out = np.zeros((4, 4), dtype=float)

    # g^tt = -A/(Sigma Delta)
    den = sig * dlt
    den_p = sig_p * dlt + sig * dlt_p
    out[0, 0] = -((A_p * den - A * den_p) / (den * den))

    # g^tphi = -2 a r /(Sigma Delta)
    num = -2.0 * a * r
    num_p = -2.0 * a if wrt == "r" else 0.0
    out[0, 3] = out[3, 0] = (num_p * den - num * den_p) / (den * den)

    # g^rr = Delta/Sigma
    out[1, 1] = (dlt_p * sig - dlt * sig_p) / (sig * sig)

    # g^thetatheta = 1/Sigma
    out[2, 2] = -sig_p / (sig * sig)

    # g^phiphi = (Delta - a^2 sin^2 theta)/(Sigma Delta sin^2 theta)
    num = dlt - a * a * s2
    num_p = dlt_p - a * a * s2_p
    den = sig * dlt * s2
    den_p = sig_p * dlt * s2 + sig * dlt_p * s2 + sig * dlt * s2_p
    out[3, 3] = (num_p * den - num * den_p) / (den * den)
    return out


def kerr_geodesic_rhs(y: np.ndarray, a: float, p_t: float, p_phi: float) -> np.ndarray:
    """Hamiltonian RHS for null geodesics in Kerr spacetime.

    State y is [t, r, theta, phi, p_r, p_theta]. Conserved p_t and p_phi are
    passed separately. The equations are:

        dx^alpha/dlambda = g^{alpha beta} p_beta
        dp_alpha/dlambda = -1/2 partial_alpha(g^{mu nu}) p_mu p_nu
    """
    _, r, th, _, p_r, p_th = y
    th = _safe_theta(float(th))
    pcov = np.array([p_t, p_r, p_th, p_phi], dtype=float)
    gcon = kerr_metric_contravariant(float(r), th, a)
    xdot = gcon @ pcov
    dg_dr = _metric_derivative(float(r), th, a, "r")
    dg_dth = _metric_derivative(float(r), th, a, "theta")
    dp_r = -0.5 * float(pcov @ dg_dr @ pcov)
    dp_th = -0.5 * float(pcov @ dg_dth @ pcov)
    return np.array([xdot[0], xdot[1], xdot[2], xdot[3], dp_r, dp_th], dtype=float)


def rk4_step(y: np.ndarray, h: float, a: float, p_t: float, p_phi: float) -> np.ndarray:
    k1 = kerr_geodesic_rhs(y, a, p_t, p_phi)
    k2 = kerr_geodesic_rhs(y + 0.5 * h * k1, a, p_t, p_phi)
    k3 = kerr_geodesic_rhs(y + 0.5 * h * k2, a, p_t, p_phi)
    k4 = kerr_geodesic_rhs(y + h * k3, a, p_t, p_phi)
    out = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[2] = _safe_theta(float(out[2]))
    return out


@dataclass(frozen=True)
class KerrTraceResult:
    status: Status
    states: np.ndarray
    p_t: float
    p_phi: float
    hamiltonian_error_max: float

    @property
    def r(self) -> np.ndarray:
        return self.states[:, 1]

    @property
    def theta(self) -> np.ndarray:
        return self.states[:, 2]

    @property
    def phi(self) -> np.ndarray:
        return self.states[:, 3]


def trace_kerr_null_geodesic(
    y0: np.ndarray,
    p_t: float,
    p_phi: float,
    a: float,
    step: float = 0.05,
    max_steps: int = 8000,
    escape_radius: float = 200.0,
    store_stride: int = 1,
) -> KerrTraceResult:
    """Trace one null ray in Kerr spacetime."""
    _validate_spin(a)
    y = np.asarray(y0, dtype=float).copy()
    r_plus = horizon_radius(a)
    states: list[np.ndarray] = [y.copy()]
    H0 = abs(hamiltonian_null(y, a, p_t, p_phi))
    max_h_err = 0.0
    status: Status = "max_steps"

    prev_r = float(y[1])
    for i in range(1, max_steps + 1):
        try:
            y = rk4_step(y, step, a, p_t, p_phi)
            H = abs(hamiltonian_null(y, a, p_t, p_phi))
        except (FloatingPointError, ValueError, OverflowError):
            status = "invalid"
            break
        if not np.all(np.isfinite(y)):
            status = "invalid"
            break
        max_h_err = max(max_h_err, abs(H - H0))
        if i % store_stride == 0:
            states.append(y.copy())
        r = float(y[1])
        if r <= r_plus * (1.0 + 2.0e-4):
            status = "captured"
            break
        # Escape after it has turned around and crossed the escape radius.
        if r > escape_radius and r > prev_r:
            status = "escaped"
            break
        prev_r = r
    else:
        status = "max_steps"

    if len(states) == 0 or not np.allclose(states[-1], y):
        states.append(y.copy())
    return KerrTraceResult(status=status, states=np.vstack(states), p_t=p_t, p_phi=p_phi, hamiltonian_error_max=max_h_err)


def isco_radius(a: float, prograde: bool = True) -> float:
    """Equatorial circular-orbit ISCO radius for Kerr, in units of M."""
    _validate_spin(a)
    aa = abs(a)
    z1 = 1.0 + (1.0 - aa * aa) ** (1.0 / 3.0) * ((1.0 + aa) ** (1.0 / 3.0) + (1.0 - aa) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * aa * aa + z1 * z1)
    sign = -1.0 if prograde else 1.0
    return 3.0 + z2 + sign * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def keplerian_omega(r: float, a: float, prograde: bool = True) -> float:
    """Angular velocity dphi/dt of an equatorial circular geodesic."""
    _validate_spin(a)
    if r <= horizon_radius(a):
        raise ValueError("Circular orbital frequency requested inside horizon")
    s = 1.0 if prograde else -1.0
    return s / (r ** 1.5 + s * a)


def circular_orbit_four_velocity(r: float, a: float, prograde: bool = True) -> np.ndarray:
    """Four-velocity u^mu for a circular equatorial Kerr orbit."""
    theta = math.pi / 2.0
    omega = keplerian_omega(r, a, prograde=prograde)
    g = kerr_metric_covariant(r, theta, a)
    norm = -(g[0, 0] + 2.0 * omega * g[0, 3] + omega * omega * g[3, 3])
    if norm <= 0.0:
        raise ValueError("No timelike circular orbit with requested parameters")
    ut = 1.0 / math.sqrt(norm)
    return np.array([ut, 0.0, 0.0, omega * ut], dtype=float)


def orbital_period_code(r: float, a: float, prograde: bool = True) -> float:
    """Coordinate orbital period at infinity in units of M."""
    return 2.0 * math.pi / abs(keplerian_omega(r, a, prograde=prograde))


def redshift_factor_to_observer(p_cov_emit: np.ndarray, u_emit_con: np.ndarray, observed_energy: float = 1.0) -> float:
    """Frequency shift g = nu_obs / nu_emit for a photon hitting the observer.

    The geodesics are launched with local observed photon energy of 1 by default,
    so nu_obs is ``observed_energy``. At emission, nu_emit = -p_mu u_emit^mu.
    """
    denom = -float(np.asarray(p_cov_emit) @ np.asarray(u_emit_con))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return observed_energy / denom

"""Thermal and non-thermal synchrotron coefficient models.

The coefficient interface returns all polarized transfer coefficients used by a
Stokes-vector transport equation: emissivity, absorptivity, Faraday rotation,
and Faraday conversion. Calculations use cgs units internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

try:  # scipy is used for Bessel functions and stable transfer matrices.
    from scipy import integrate, special  # type: ignore
except Exception as exc:  # pragma: no cover
    integrate = None
    special = None

from .calibration import C_CGS, E_CHARGE_ESU, K_BOLTZMANN_CGS, M_ELECTRON_CGS
from .calibration import PhysicalScaling
from .grmhd import FluidSample


@dataclass(frozen=True)
class LocalPlasmaFrame:
    """Physical plasma quantities needed by synchrotron coefficient models."""

    n_e_cm3: float
    theta_e: float
    b_gauss: float
    cos_pitch_to_los: float = 0.0
    evpa_rad: float = 0.0

    @property
    def b_perp_gauss(self) -> float:
        return abs(self.b_gauss) * math.sqrt(max(0.0, 1.0 - self.cos_pitch_to_los**2))

    @property
    def b_parallel_gauss(self) -> float:
        return self.b_gauss * self.cos_pitch_to_los


@dataclass(frozen=True)
class PolarizedCoefficients:
    """Polarized radiative-transfer coefficients in a local plasma frame.

    Units are cgs-like transfer units per cm. The Stokes convention is local:
    positive Q is aligned with ``evpa_rad`` before the final rotation into the
    image basis.
    """

    j_i: float = 0.0
    j_q: float = 0.0
    j_u: float = 0.0
    j_v: float = 0.0
    alpha_i: float = 0.0
    alpha_q: float = 0.0
    alpha_u: float = 0.0
    alpha_v: float = 0.0
    rho_v: float = 0.0  # Faraday rotation Q<->U
    rho_q: float = 0.0  # Faraday conversion U<->V in chosen basis
    rho_u: float = 0.0

    def rotated(self, chi: float) -> "PolarizedCoefficients":
        """Rotate linear polarization coefficients by EVPA angle chi."""
        c = math.cos(2.0 * chi)
        s = math.sin(2.0 * chi)
        jq = self.j_q * c - self.j_u * s
        ju = self.j_q * s + self.j_u * c
        aq = self.alpha_q * c - self.alpha_u * s
        au = self.alpha_q * s + self.alpha_u * c
        rq = self.rho_q * c - self.rho_u * s
        ru = self.rho_q * s + self.rho_u * c
        return PolarizedCoefficients(self.j_i, jq, ju, self.j_v, self.alpha_i, aq, au, self.alpha_v, self.rho_v, rq, ru)

    def as_emission_vector(self) -> np.ndarray:
        return np.array([self.j_i, self.j_q, self.j_u, self.j_v], dtype=float)

    def propagation_matrix(self) -> np.ndarray:
        """Return the 4x4 polarized transfer matrix K for dS/ds = j - K S."""
        ai, aq, au, av = self.alpha_i, self.alpha_q, self.alpha_u, self.alpha_v
        rv, rq, ru = self.rho_v, self.rho_q, self.rho_u
        return np.array(
            [
                [ai, aq, au, av],
                [aq, ai, rv, -ru],
                [au, -rv, ai, rq],
                [av, ru, -rq, ai],
            ],
            dtype=float,
        )

    def scaled(self, factor: float) -> "PolarizedCoefficients":
        return PolarizedCoefficients(
            self.j_i * factor, self.j_q * factor, self.j_u * factor, self.j_v * factor,
            self.alpha_i * factor, self.alpha_q * factor, self.alpha_u * factor, self.alpha_v * factor,
            self.rho_v * factor, self.rho_q * factor, self.rho_u * factor,
        )


def local_plasma_from_sample(sample: FluidSample, scaling: PhysicalScaling, p_cov: np.ndarray | None = None) -> LocalPlasmaFrame:
    b_vec = np.asarray(sample.b_con[1:], dtype=float)
    b_code = float(np.linalg.norm(b_vec))
    b_gauss = float(scaling.magnetic_field_gauss(b_code))
    n_e = float(scaling.electron_number_density_cm3(sample.rho))
    cos_los = 0.0
    if p_cov is not None and b_code > 0.0:
        # Coordinate-basis proxy for angle between B and photon spatial covector.
        k = np.asarray(p_cov[1:], dtype=float)
        nk = float(np.linalg.norm(k))
        if nk > 0.0:
            cos_los = float(np.clip(np.dot(b_vec, k) / (b_code * nk), -1.0, 1.0))
    evpa = math.atan2(float(b_vec[2]), float(b_vec[1])) if b_code > 0.0 else 0.0
    return LocalPlasmaFrame(n_e, max(float(sample.theta_e), 1.0e-8), max(b_gauss, 0.0), cos_los, evpa)


def _synchrotron_f_approx(x: np.ndarray | float) -> np.ndarray:
    """Fast positive approximation to F(x)=x∫_x^∞K_{5/3}(t)dt.

    The previous implementation evaluated the improper Bessel integral by
    adaptive quadrature while rendering. That is accurate but pathological for
    tiny ``x`` values common in high-temperature/low-frequency cells and can
    make the full suite appear to hang. This approximation preserves the
    correct x^(1/3) low-frequency trend and exp(-x) high-frequency cutoff,
    which is sufficient for the project-level coefficient and renderer tests.
    Production coefficient modules can still replace this function with a
    tabulated or library-backed kernel.
    """
    xx = np.asarray(x, dtype=float)
    xx = np.clip(xx, 1.0e-12, 1.0e3)
    # Crusius-Schlickeiser-style smooth fit with proper asymptotic behavior.
    out = 1.25 * np.power(xx, 1.0 / 3.0) * np.exp(-xx) * np.power(648.0 + xx * xx, 1.0 / 12.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _synchrotron_g_kernel(x: np.ndarray | float) -> np.ndarray:
    """Return G(x)=x K_{2/3}(x) with finite asymptotic fallbacks."""
    xx = np.asarray(x, dtype=float)
    xx = np.clip(xx, 1.0e-12, 1.0e3)
    if special is not None:
        out = xx * special.kv(2.0 / 3.0, xx)
        bad = ~np.isfinite(out)
        if np.any(bad):
            small = 1.0747641207672393 * np.power(xx[bad], 1.0 / 3.0)
            large = np.sqrt(np.pi * xx[bad] / 2.0) * np.exp(-xx[bad])
            out = np.asarray(out)
            out[bad] = np.where(xx[bad] < 1.0e-3, small, large)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    small = 1.0747641207672393 * np.power(xx, 1.0 / 3.0)
    large = np.sqrt(np.pi * xx / 2.0) * np.exp(-xx)
    return np.where(xx < 1.0, small / np.power(1.0 + xx, 1.0 / 6.0), large)


@lru_cache(maxsize=8)
def _synchrotron_kernel_table(n: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.geomspace(1.0e-8, 1.0e3, n)
    F = _synchrotron_f_approx(xs)
    G = _synchrotron_g_kernel(xs)
    return xs, F, G


def _kernel_interp(x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    xs, F, G = _synchrotron_kernel_table()
    xx = np.asarray(x, dtype=float)
    lx = np.log(np.clip(xx, xs[0], xs[-1]))
    lxs = np.log(xs)
    outF = np.exp(np.interp(lx, lxs, np.log(np.maximum(F, 1.0e-300))))
    outG = np.exp(np.interp(lx, lxs, np.log(np.maximum(G, 1.0e-300))))
    return outF, outG


def _planck_nu_cgs(nu_hz: float, theta_e: float) -> float:
    # T_e = theta_e m_e c^2/k_B.
    h = 6.62607015e-27
    T = theta_e * M_ELECTRON_CGS * C_CGS * C_CGS / K_BOLTZMANN_CGS
    x = h * nu_hz / max(K_BOLTZMANN_CGS * T, 1.0e-300)
    if x < 1.0e-3:
        return 2.0 * nu_hz * nu_hz * K_BOLTZMANN_CGS * T / (C_CGS * C_CGS)
    return 2.0 * h * nu_hz**3 / (C_CGS * C_CGS) / max(math.expm1(min(x, 700.0)), 1.0e-300)


def _single_particle_prefactor(B_perp_gauss: float) -> float:
    return math.sqrt(3.0) * E_CHARGE_ESU**3 * B_perp_gauss / (M_ELECTRON_CGS * C_CGS**2)


def _thermal_distribution(gamma: np.ndarray, theta_e: float) -> np.ndarray:
    if special is None:
        raise RuntimeError("scipy is required for thermal synchrotron coefficients")
    beta = np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(gamma, 1.0) ** 2, 0.0))
    norm = theta_e * special.kv(2.0, 1.0 / max(theta_e, 1.0e-12))
    f = gamma * gamma * beta * np.exp(-gamma / max(theta_e, 1.0e-12)) / max(norm, 1.0e-300)
    integ = np.trapezoid(f, gamma)
    return f / max(integ, 1.0e-300)


def _powerlaw_distribution(gamma: np.ndarray, p: float, gamma_min: float, gamma_max: float) -> np.ndarray:
    mask = (gamma >= gamma_min) & (gamma <= gamma_max)
    f = np.zeros_like(gamma)
    f[mask] = gamma[mask] ** (-p)
    integ = np.trapezoid(f, gamma)
    return f / max(integ, 1.0e-300)


def _integrated_synchrotron_j(
    frame: LocalPlasmaFrame,
    nu_hz: float,
    gamma: np.ndarray,
    f_gamma: np.ndarray,
    pitch_nodes: int = 16,
) -> tuple[float, float]:
    """Return local j_I and linearly polarized j_Q in cgs emissivity units."""
    if frame.n_e_cm3 <= 0.0 or frame.b_gauss <= 0.0:
        return 0.0, 0.0
    # Isotropic pitch-angle average over mu=cos(alpha), weight 1/2.
    mu, w = np.polynomial.legendre.leggauss(pitch_nodes)
    total_I = 0.0
    total_Q = 0.0
    nu_B = E_CHARGE_ESU * frame.b_gauss / (2.0 * math.pi * M_ELECTRON_CGS * C_CGS)
    for m, ww in zip(mu, w):
        sin_a = math.sqrt(max(0.0, 1.0 - float(m) ** 2))
        Bp = frame.b_gauss * sin_a
        if Bp <= 0.0:
            continue
        nu_c = 1.5 * gamma * gamma * nu_B * sin_a
        x = nu_hz / np.maximum(nu_c, 1.0e-300)
        F, G = _kernel_interp(x)
        pref = _single_particle_prefactor(Bp)
        pI = pref * F
        pQ = pref * G
        total_I += 0.5 * ww * float(np.trapezoid(f_gamma * pI, gamma))
        total_Q += 0.5 * ww * float(np.trapezoid(f_gamma * pQ, gamma))
    # 4pi assumes isotropic emission direction distribution.
    return frame.n_e_cm3 * total_I / (4.0 * math.pi), frame.n_e_cm3 * total_Q / (4.0 * math.pi)


@dataclass(frozen=True)
class ThermalSynchrotronCoefficients:
    """Thermal Maxwell-Juttner synchrotron Stokes coefficients."""

    gamma_grid_size: int = 192
    pitch_nodes: int = 16
    faraday_rotation_scale: float = 1.0
    faraday_conversion_scale: float = 1.0

    def coefficients(self, frame: LocalPlasmaFrame, nu_hz: float) -> PolarizedCoefficients:
        if frame.n_e_cm3 <= 0.0 or frame.b_gauss <= 0.0 or nu_hz <= 0.0:
            return PolarizedCoefficients()
        gmax = max(10.0, 1.0 + 45.0 * frame.theta_e)
        gamma = 1.0 + np.geomspace(1.0e-5, gmax - 1.0, self.gamma_grid_size)
        f = _thermal_distribution(gamma, frame.theta_e)
        jI, jQ0 = _integrated_synchrotron_j(frame, nu_hz, gamma, f, self.pitch_nodes)
        # Thermal absorption through Kirchhoff's law in the local plasma frame.
        Bnu = _planck_nu_cgs(nu_hz, frame.theta_e)
        alphaI = jI / max(Bnu, 1.0e-300)
        alphaQ0 = jQ0 / max(Bnu, 1.0e-300)
        # Thermal Faraday terms; constants are the cold-plasma scale with
        # relativistic suppression/enhancement factors retained explicitly.
        rel_supp = max(1.0 / (frame.theta_e * frame.theta_e + 1.0), 1.0e-8)
        rhoV = self.faraday_rotation_scale * (E_CHARGE_ESU**3 / (math.pi * M_ELECTRON_CGS**2 * C_CGS**2))
        rhoV *= frame.n_e_cm3 * frame.b_parallel_gauss / max(nu_hz * nu_hz, 1.0e-300) * rel_supp
        rhoQ = self.faraday_conversion_scale * (E_CHARGE_ESU**4 / (4.0 * math.pi**2 * M_ELECTRON_CGS**3 * C_CGS**3))
        rhoQ *= frame.n_e_cm3 * frame.b_perp_gauss**2 / max(nu_hz**3, 1.0e-300) * max(frame.theta_e, 1.0e-8)
        coeff = PolarizedCoefficients(
            j_i=max(jI, 0.0),
            j_q=max(jQ0, 0.0),
            alpha_i=max(alphaI, 0.0),
            alpha_q=max(alphaQ0, 0.0),
            rho_v=float(rhoV),
            rho_q=float(rhoQ),
        )
        return coeff.rotated(frame.evpa_rad)


@dataclass(frozen=True)
class NonthermalPowerLawSynchrotronCoefficients:
    """Non-thermal power-law synchrotron coefficients from numerical integration."""

    p: float = 3.5
    gamma_min: float = 10.0
    gamma_max: float = 1.0e5
    nonthermal_fraction: float = 0.02
    gamma_grid_size: int = 224
    pitch_nodes: int = 16
    absorption_scale: float = 1.0
    faraday_scale: float = 0.15

    def coefficients(self, frame: LocalPlasmaFrame, nu_hz: float) -> PolarizedCoefficients:
        if frame.n_e_cm3 <= 0.0 or frame.b_gauss <= 0.0 or nu_hz <= 0.0 or self.nonthermal_fraction <= 0.0:
            return PolarizedCoefficients()
        gamma = np.geomspace(self.gamma_min, self.gamma_max, self.gamma_grid_size)
        f = _powerlaw_distribution(gamma, self.p, self.gamma_min, self.gamma_max)
        scaled = LocalPlasmaFrame(frame.n_e_cm3 * self.nonthermal_fraction, frame.theta_e, frame.b_gauss, frame.cos_pitch_to_los, frame.evpa_rad)
        jI, jQ0 = _integrated_synchrotron_j(scaled, nu_hz, gamma, f, self.pitch_nodes)
        # Numerical absorption via the standard synchrotron self-absorption
        # scaling for power-law distributions, normalized to emissivity.
        characteristic_gamma = max(self.gamma_min, math.sqrt(max(nu_hz / max(4.2e6 * frame.b_perp_gauss, 1.0e-300), 1.0)))
        electron_energy = characteristic_gamma * M_ELECTRON_CGS * C_CGS * C_CGS
        alphaI = self.absorption_scale * jI / max(electron_energy * nu_hz * nu_hz / (C_CGS * C_CGS), 1.0e-300)
        pol_frac = min((self.p + 1.0) / (self.p + 7.0 / 3.0), 0.85)
        jQ0 = min(jQ0, pol_frac * jI)
        alphaQ0 = min(alphaI * pol_frac, alphaI)
        rhoV = self.faraday_scale * (E_CHARGE_ESU**3 / (math.pi * M_ELECTRON_CGS**2 * C_CGS**2))
        rhoV *= scaled.n_e_cm3 * frame.b_parallel_gauss / max(nu_hz * nu_hz, 1.0e-300) / max(self.gamma_min**2, 1.0)
        rhoQ = self.faraday_scale * (E_CHARGE_ESU**4 / (4.0 * math.pi**2 * M_ELECTRON_CGS**3 * C_CGS**3))
        rhoQ *= scaled.n_e_cm3 * frame.b_perp_gauss**2 / max(nu_hz**3, 1.0e-300) * math.log(max(self.gamma_max / self.gamma_min, 1.0))
        coeff = PolarizedCoefficients(j_i=max(jI,0.0), j_q=max(jQ0,0.0), alpha_i=max(alphaI,0.0), alpha_q=max(alphaQ0,0.0), rho_v=float(rhoV), rho_q=float(rhoQ))
        return coeff.rotated(frame.evpa_rad)


@dataclass(frozen=True)
class HybridSynchrotronCoefficients:
    thermal: ThermalSynchrotronCoefficients = ThermalSynchrotronCoefficients()
    nonthermal: NonthermalPowerLawSynchrotronCoefficients = NonthermalPowerLawSynchrotronCoefficients()

    def coefficients(self, frame: LocalPlasmaFrame, nu_hz: float) -> PolarizedCoefficients:
        a = self.thermal.coefficients(frame, nu_hz)
        b = self.nonthermal.coefficients(frame, nu_hz)
        return PolarizedCoefficients(
            a.j_i + b.j_i, a.j_q + b.j_q, a.j_u + b.j_u, a.j_v + b.j_v,
            a.alpha_i + b.alpha_i, a.alpha_q + b.alpha_q, a.alpha_u + b.alpha_u, a.alpha_v + b.alpha_v,
            a.rho_v + b.rho_v, a.rho_q + b.rho_q, a.rho_u + b.rho_u,
        )

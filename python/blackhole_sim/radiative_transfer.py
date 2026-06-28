"""General-relativistic radiative-transfer integration along Kerr rays.

The transfer integrator uses invariant intensity along null geodesics and samples
fluid variables from a GRMHD snapshot. The default emissivity model is a compact
thermal synchrotron fitting model suitable for testing the pipeline. Production
use should replace it with a coefficient table or a vetted synchrotron package
while keeping the same invariant-transfer interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .grmhd import FluidSample, GRMHDSnapshot
from .kerr import KerrTraceResult
from .physics import C_SI


class CoefficientModel(Protocol):
    def coefficients(self, sample: FluidSample, nu_emit_hz: float, p_cov: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return emission j_nu, absorption alpha_nu, and RGB source color."""


@dataclass(frozen=True)
class ThermalSynchrotronFit:
    """Thermal synchrotron coefficient fit for optically thin GRRT tests.

    The field variables are dimensionless by default because most public GRMHD
    dumps need model-specific density, electron-temperature, and magnetic-field
    scaling before direct Jy images are meaningful. ``emissivity_scale`` and
    ``absorptivity_scale`` are explicit calibration knobs rather than hidden
    constants.
    """

    observing_frequency_hz: float = 230.0e9
    emissivity_scale: float = 1.2
    absorptivity_scale: float = 0.08
    spectral_index: float = 1.2
    color_temperature_bias: float = 0.18

    def coefficients(self, sample: FluidSample, nu_emit_hz: float, p_cov: np.ndarray) -> tuple[float, float, np.ndarray]:
        if not sample.valid or sample.rho <= 0.0 or sample.theta_e <= 0.0:
            return 0.0, 0.0, np.zeros(3)
        b_mag = _magnetic_magnitude_proxy(sample.b_con)
        theta_e = max(sample.theta_e, 1.0e-8)
        # Critical-frequency proxy in dimensionless field units. Real datasets
        # must provide the physical B and density scale for absolute flux.
        nu_c = max(1.0, 2.8e10 * b_mag * theta_e * theta_e)
        x = max(nu_emit_hz / nu_c, 1.0e-12)
        shape = x ** (1.0 / 3.0) * math.exp(-x ** (1.0 / 3.0))
        j = self.emissivity_scale * sample.rho * b_mag * theta_e * theta_e * shape
        # Kirchhoff-style absorption proxy. Keeps optically thick behavior sane.
        alpha = self.absorptivity_scale * j / max(theta_e * nu_emit_hz ** 2, 1.0e-60)
        alpha *= self.observing_frequency_hz ** 2
        hot = np.array([0.78, 0.88, 1.0])
        warm = np.array([1.0, 0.48, 0.16])
        m = np.clip(math.log1p(theta_e) * self.color_temperature_bias, 0.0, 1.0)
        return float(max(j, 0.0)), float(max(alpha, 0.0)), (1.0 - m) * warm + m * hot


@dataclass(frozen=True)
class TransferConfig:
    observing_frequency_hz: float = 230.0e9
    path_length_scale: float = 0.045
    max_optical_depth: float = 18.0
    min_redshift: float = 1.0e-6
    intensity_floor: float = 0.0


@dataclass(frozen=True)
class TransferResult:
    intensity_rgb: np.ndarray
    optical_depth: float
    emitting_steps: int
    valid_steps: int
    redshift_min: float
    redshift_max: float


def _magnetic_magnitude_proxy(b_con: np.ndarray) -> float:
    # Positive definite proxy for local field strength. A production adapter can
    # replace this with b^2 = b_mu b^mu in the local tetrad frame.
    b = np.asarray(b_con, dtype=float)
    return float(np.linalg.norm(b[1:]) + 1.0e-30)


def photon_energy_in_fluid_frame(p_cov: np.ndarray, u_con: np.ndarray) -> float:
    return -float(np.asarray(p_cov, dtype=float) @ np.asarray(u_con, dtype=float))


def invariant_redshift(p_cov: np.ndarray, u_emit: np.ndarray, observed_energy: float = 1.0) -> float:
    e_emit = photon_energy_in_fluid_frame(p_cov, u_emit)
    if e_emit <= 0.0 or not np.isfinite(e_emit):
        return 0.0
    return observed_energy / e_emit


def integrate_kerr_grrt(
    trace: KerrTraceResult,
    snapshot: GRMHDSnapshot,
    coeffs: CoefficientModel | None = None,
    cfg: TransferConfig | None = None,
) -> TransferResult:
    """Integrate unpolarized GRRT along a traced Kerr null geodesic.

    The observed contribution from a comoving emission coefficient is accumulated
    as dI_obs = g^3 j_emit exp(-tau) d\\ell_emit, while optical depth is updated
    by d_tau = alpha_emit d\\ell_emit. d\\ell_emit is approximated from the photon
    energy in the fluid frame times the affine step in code units.
    """
    model = coeffs or ThermalSynchrotronFit()
    tc = cfg or TransferConfig()
    states = trace.states
    if len(states) < 2:
        return TransferResult(np.zeros(3), 0.0, 0, 0, math.inf, 0.0)

    I = np.zeros(3, dtype=float)
    tau = 0.0
    emit_steps = 0
    valid_steps = 0
    gmin = math.inf
    gmax = 0.0

    p_t = trace.p_t
    p_phi = trace.p_phi
    for i in range(1, len(states)):
        prev = states[i - 1]
        cur = states[i]
        mid = 0.5 * (prev + cur)
        r = float(mid[1])
        th = float(mid[2])
        ph = float(mid[3])
        sample = snapshot.sample(r, th, ph)
        if not sample.valid:
            continue
        valid_steps += 1
        p_cov = np.array([p_t, float(mid[4]), float(mid[5]), p_phi], dtype=float)
        g_shift = invariant_redshift(p_cov, sample.u_con, observed_energy=1.0)
        if g_shift < tc.min_redshift:
            continue
        nu_emit = tc.observing_frequency_hz / g_shift
        j, alpha, color = model.coefficients(sample, nu_emit, p_cov)
        if j <= 0.0 and alpha <= 0.0:
            continue
        dlambda = float(np.linalg.norm(cur[1:4] - prev[1:4]))
        e_emit = max(photon_energy_in_fluid_frame(p_cov, sample.u_con), 0.0)
        dl_emit = tc.path_length_scale * e_emit * max(dlambda, 1.0e-12)
        dtau = alpha * dl_emit
        trans = math.exp(-min(tau, tc.max_optical_depth))
        # Invariant transfer: I_nu/nu^3 accumulates with g^3 weighting.
        dI_scalar = (g_shift ** 3) * j * trans * dl_emit
        I += dI_scalar * color
        tau += dtau
        emit_steps += 1
        gmin = min(gmin, g_shift)
        gmax = max(gmax, g_shift)
        if tau >= tc.max_optical_depth:
            break

    return TransferResult(
        intensity_rgb=np.maximum(I, tc.intensity_floor),
        optical_depth=float(tau),
        emitting_steps=emit_steps,
        valid_steps=valid_steps,
        redshift_min=float(gmin if gmin < math.inf else 0.0),
        redshift_max=float(gmax),
    )

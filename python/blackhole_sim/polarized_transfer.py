"""Stokes I,Q,U,V polarized radiative transfer along Kerr geodesics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from scipy.linalg import expm  # type: ignore
except Exception as exc:  # pragma: no cover
    expm = None

from .calibration import PhysicalScaling
from .grmhd import GRMHDSnapshot
from .kerr import KerrTraceResult
from .radiative_transfer import invariant_redshift, photon_energy_in_fluid_frame
from .synchrotron import HybridSynchrotronCoefficients, PolarizedCoefficients, local_plasma_from_sample


@dataclass(frozen=True)
class PolarizedTransferConfig:
    observing_frequency_hz: float = 230.0e9
    affine_to_length_cm: float | None = None
    max_optical_depth: float = 30.0
    min_redshift: float = 1.0e-8
    use_exact_matrix_step: bool = True


@dataclass(frozen=True)
class PolarizedTransferResult:
    stokes: np.ndarray
    optical_depth: float
    emitting_steps: int
    valid_steps: int
    faraday_depth_abs: float
    redshift_min: float
    redshift_max: float

    @property
    def linear_polarization_fraction(self) -> float:
        I = max(float(self.stokes[0]), 1.0e-300)
        return float(math.sqrt(float(self.stokes[1]) ** 2 + float(self.stokes[2]) ** 2) / I)

    @property
    def circular_polarization_fraction(self) -> float:
        return float(self.stokes[3] / max(self.stokes[0], 1.0e-300))


def stokes_step_exact(stokes: np.ndarray, coeff: PolarizedCoefficients, ds_cm: float) -> np.ndarray:
    """Solve constant-coefficient polarized transfer exactly over one step.

    Uses an augmented matrix exponential for dS/ds = j - K S, avoiding explicit
    inversion of K when Faraday or absorption terms are small/singular.
    """
    if ds_cm <= 0.0:
        return np.asarray(stokes, dtype=float)
    if expm is None:
        return stokes_step_rk2(stokes, coeff, ds_cm)
    K = coeff.propagation_matrix()
    j = coeff.as_emission_vector()
    A = np.zeros((5, 5), dtype=float)
    A[:4, :4] = -K
    A[:4, 4] = j
    M = expm(A * ds_cm)
    y = np.ones(5, dtype=float)
    y[:4] = np.asarray(stokes, dtype=float)
    out = M @ y
    return np.asarray(out[:4], dtype=float)


def stokes_step_rk2(stokes: np.ndarray, coeff: PolarizedCoefficients, ds_cm: float) -> np.ndarray:
    K = coeff.propagation_matrix()
    j = coeff.as_emission_vector()
    S = np.asarray(stokes, dtype=float)
    k1 = j - K @ S
    mid = S + 0.5 * ds_cm * k1
    k2 = j - K @ mid
    return S + ds_cm * k2


def integrate_polarized_kerr_grrt(
    trace: KerrTraceResult,
    snapshot: GRMHDSnapshot,
    scaling: PhysicalScaling,
    coeff_model: HybridSynchrotronCoefficients | None = None,
    cfg: PolarizedTransferConfig | None = None,
) -> PolarizedTransferResult:
    """Integrate Stokes I,Q,U,V along a Kerr null geodesic through a GRMHD dump."""
    model = coeff_model or HybridSynchrotronCoefficients()
    pcfg = cfg or PolarizedTransferConfig()
    states = trace.states
    if len(states) < 2:
        return PolarizedTransferResult(np.zeros(4), 0.0, 0, 0, 0.0, math.inf, 0.0)
    length_scale = scaling.rg_cm if pcfg.affine_to_length_cm is None else float(pcfg.affine_to_length_cm)
    S = np.zeros(4, dtype=float)
    tau = 0.0
    fd = 0.0
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
        sample = snapshot.sample(float(mid[1]), float(mid[2]), float(mid[3]))
        if not sample.valid:
            continue
        valid_steps += 1
        p_cov = np.array([p_t, float(mid[4]), float(mid[5]), p_phi], dtype=float)
        g_shift = invariant_redshift(p_cov, sample.u_con, observed_energy=1.0)
        if g_shift < pcfg.min_redshift:
            continue
        nu_emit = pcfg.observing_frequency_hz / g_shift
        frame = local_plasma_from_sample(sample, scaling, p_cov, spin_a=float(snapshot.spin_a))
        coeff = model.coefficients(frame, nu_emit)
        # Invariant transfer: j/nu^2 and alpha*nu are invariant; for a compact
        # screen-frame Stokes integral we use g^2 on emission and g^-1 on K.
        j_factor = g_shift * g_shift * scaling.flux_scale_jy
        k_factor = 1.0 / max(g_shift, 1.0e-300)
        coeff = PolarizedCoefficients(
            coeff.j_i * j_factor, coeff.j_q * j_factor, coeff.j_u * j_factor, coeff.j_v * j_factor,
            coeff.alpha_i * k_factor, coeff.alpha_q * k_factor, coeff.alpha_u * k_factor, coeff.alpha_v * k_factor,
            coeff.rho_v * k_factor, coeff.rho_q * k_factor, coeff.rho_u * k_factor,
        )
        dlambda = float(np.linalg.norm(cur[1:4] - prev[1:4]))
        e_emit = max(photon_energy_in_fluid_frame(p_cov, sample.u_con), 0.0)
        ds_cm = length_scale * e_emit * max(dlambda, 1.0e-14)
        if pcfg.use_exact_matrix_step:
            S = stokes_step_exact(S, coeff, ds_cm)
        else:
            S = stokes_step_rk2(S, coeff, ds_cm)
        tau += max(coeff.alpha_i, 0.0) * ds_cm
        fd += (abs(coeff.rho_v) + abs(coeff.rho_q) + abs(coeff.rho_u)) * ds_cm
        emit_steps += 1
        gmin = min(gmin, g_shift)
        gmax = max(gmax, g_shift)
        if tau >= pcfg.max_optical_depth:
            break
    return PolarizedTransferResult(np.asarray(S, dtype=float), float(tau), emit_steps, valid_steps, float(fd), float(gmin if gmin < math.inf else 0.0), float(gmax))

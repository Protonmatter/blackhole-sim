"""Physical scaling and flux calibration for GRMHD post-processing.

GRMHD dumps are usually scale-free in code units. This module keeps the mapping
from code density/magnetic-field units to physical cgs units explicit, because
absolute images in Jy cannot be interpreted without this calibration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .physics import BlackHoleSystem, C_SI

# cgs constants used by synchrotron/radiative-transfer modules.
C_CGS = 2.99792458e10
G_CGS = 6.67430e-8
M_SUN_CGS = 1.98847e33
M_PROTON_CGS = 1.67262192369e-24
M_ELECTRON_CGS = 9.1093837015e-28
K_BOLTZMANN_CGS = 1.380649e-16
E_CHARGE_ESU = 4.80320471257e-10
JY_CGS = 1.0e-23  # erg s^-1 cm^-2 Hz^-1


@dataclass(frozen=True)
class PhysicalScaling:
    """Map dimensionless GRMHD fields into cgs quantities.

    Attributes
    ----------
    mass_bh_g:
        Black-hole mass in grams.
    distance_cm:
        Source distance in centimeters.
    rho_cgs_per_code:
        Multiplicative factor converting code rho to mass density g cm^-3.
    b_gauss_per_code:
        Multiplicative factor converting code magnetic field to Gauss.
    ion_to_electron_temperature_ratio:
        Optional Ti/Te metadata knob used by post-processing models. The snapshot
        already stores Thetae, so this is mainly retained for adapter provenance.
    """

    mass_bh_g: float
    distance_cm: float
    rho_cgs_per_code: float
    b_gauss_per_code: float
    ion_to_electron_temperature_ratio: float = 3.0
    mean_molecular_weight_per_electron: float = 1.0
    flux_scale_jy: float = 1.0

    @classmethod
    def from_black_hole_system(
        cls,
        system: BlackHoleSystem,
        rho_cgs_per_code: float,
        b_gauss_per_code: float,
        **kwargs: float,
    ) -> "PhysicalScaling":
        return cls(
            mass_bh_g=system.mass_kg * 1.0e3,
            distance_cm=system.distance_m * 100.0,
            rho_cgs_per_code=float(rho_cgs_per_code),
            b_gauss_per_code=float(b_gauss_per_code),
            **kwargs,
        )

    @classmethod
    def from_mdot(
        cls,
        system: BlackHoleSystem,
        mdot_msun_per_year: float,
        density_geometry_factor: float = 4.0 * math.pi,
        b_plasma_beta: float = 10.0,
        pressure_over_rho_c2: float = 1.0e-3,
        **kwargs: float,
    ) -> "PhysicalScaling":
        """Estimate density and B-unit from accretion-rate scale.

        The density unit is set by mdot/(geometry_factor*r_g^2*c). The magnetic
        unit is set from beta = p_gas / p_mag and p_gas = pressure_over_rho_c2
        * rho c^2. Users should replace these priors with calibration against
        observational flux or a simulation paper's published unit conversion.
        """
        mass_g = system.mass_kg * 1.0e3
        rg_cm = G_CGS * mass_g / (C_CGS * C_CGS)
        seconds_per_year = 365.25 * 24.0 * 3600.0
        mdot_g_s = mdot_msun_per_year * M_SUN_CGS / seconds_per_year
        rho_unit = mdot_g_s / max(density_geometry_factor * rg_cm * rg_cm * C_CGS, 1.0e-300)
        p_gas = pressure_over_rho_c2 * rho_unit * C_CGS * C_CGS
        # beta = p_gas / (B^2 / 8pi)
        b_unit = math.sqrt(max(8.0 * math.pi * p_gas / max(b_plasma_beta, 1.0e-300), 0.0))
        return cls(
            mass_bh_g=mass_g,
            distance_cm=system.distance_m * 100.0,
            rho_cgs_per_code=rho_unit,
            b_gauss_per_code=b_unit,
            **kwargs,
        )

    @property
    def rg_cm(self) -> float:
        return G_CGS * self.mass_bh_g / (C_CGS * C_CGS)

    @property
    def tg_s(self) -> float:
        return G_CGS * self.mass_bh_g / (C_CGS ** 3)

    def electron_number_density_cm3(self, rho_code: float | np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_code, dtype=float) * self.rho_cgs_per_code
        return rho / max(self.mean_molecular_weight_per_electron * M_PROTON_CGS, 1.0e-300)

    def magnetic_field_gauss(self, b_code: float | np.ndarray) -> np.ndarray:
        return np.asarray(b_code, dtype=float) * self.b_gauss_per_code

    def solid_angle_per_pixel_sr(self, fov_y_degrees: float, width: int, height: int) -> float:
        fov_y = math.radians(fov_y_degrees)
        aspect = width / height
        fov_x = 2.0 * math.atan(aspect * math.tan(0.5 * fov_y))
        return (fov_x / width) * (fov_y / height)


@dataclass(frozen=True)
class FluxCalibrationResult:
    target_flux_jy: float
    measured_flux_jy: float
    multiplicative_scale: float
    calibrated_scaling: PhysicalScaling


def image_flux_jy(image_intensity_cgs: np.ndarray, pixel_solid_angle_sr: float) -> float:
    """Integrate a specific-intensity image into Jansky.

    The input image is expected in cgs specific intensity units
    erg s^-1 cm^-2 Hz^-1 sr^-1. For RGB images this sums the channels first;
    for scalar images it integrates directly.
    """
    arr = np.asarray(image_intensity_cgs, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return float(np.sum(arr) * pixel_solid_angle_sr / JY_CGS)


def calibrate_flux_scale(
    scaling: PhysicalScaling,
    rendered_intensity_cgs: np.ndarray,
    target_flux_jy: float,
    pixel_solid_angle_sr: float,
) -> FluxCalibrationResult:
    measured = image_flux_jy(rendered_intensity_cgs, pixel_solid_angle_sr)
    factor = float(target_flux_jy / measured) if measured > 0.0 else math.inf
    calibrated = PhysicalScaling(
        mass_bh_g=scaling.mass_bh_g,
        distance_cm=scaling.distance_cm,
        rho_cgs_per_code=scaling.rho_cgs_per_code,
        b_gauss_per_code=scaling.b_gauss_per_code,
        ion_to_electron_temperature_ratio=scaling.ion_to_electron_temperature_ratio,
        mean_molecular_weight_per_electron=scaling.mean_molecular_weight_per_electron,
        flux_scale_jy=scaling.flux_scale_jy * factor,
    )
    return FluxCalibrationResult(float(target_flux_jy), measured, factor, calibrated)


def fit_density_scale_from_flux_powerlaw(
    current_scaling: PhysicalScaling,
    measured_flux_jy: float,
    target_flux_jy: float,
    emissivity_density_exponent: float = 1.0,
    b_density_exponent: float = 0.5,
    synchrotron_b_exponent: float = 1.0,
) -> PhysicalScaling:
    """Estimate a new density/B scale from a flux target.

    This is a practical calibration helper for iterative rendering. If B is tied
    to density through B ∝ rho^b_density_exponent and synchrotron emissivity has
    j ∝ rho^emissivity_density_exponent B^synchrotron_b_exponent, then flux
    scales as rho^total_exponent.
    """
    if measured_flux_jy <= 0.0 or target_flux_jy <= 0.0:
        raise ValueError("measured_flux_jy and target_flux_jy must be positive")
    total_exp = emissivity_density_exponent + b_density_exponent * synchrotron_b_exponent
    density_factor = (target_flux_jy / measured_flux_jy) ** (1.0 / max(total_exp, 1.0e-12))
    return PhysicalScaling(
        mass_bh_g=current_scaling.mass_bh_g,
        distance_cm=current_scaling.distance_cm,
        rho_cgs_per_code=current_scaling.rho_cgs_per_code * density_factor,
        b_gauss_per_code=current_scaling.b_gauss_per_code * (density_factor ** b_density_exponent),
        ion_to_electron_temperature_ratio=current_scaling.ion_to_electron_temperature_ratio,
        mean_molecular_weight_per_electron=current_scaling.mean_molecular_weight_per_electron,
        flux_scale_jy=current_scaling.flux_scale_jy,
    )

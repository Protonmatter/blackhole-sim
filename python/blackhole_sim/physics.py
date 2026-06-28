"""Physical constants and unit conversions for black-hole simulations.

The numerical geodesic code uses geometrized units with G = c = M = 1. This
module maps physical inputs such as black-hole mass, distance, observer radius,
and orbital period into those dimensionless units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# CODATA exact/standard values where applicable.
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_SI = 299_792_458.0        # m s^-1
M_SUN_SI = 1.98847e30       # kg
PARSEC_SI = 3.085677581491367e16  # m
KPC_SI = 1.0e3 * PARSEC_SI
MPC_SI = 1.0e6 * PARSEC_SI


@dataclass(frozen=True)
class BlackHoleSystem:
    """Physical black-hole system parameters.

    Parameters
    ----------
    mass_kg:
        Black-hole mass in kg.
    spin_a:
        Dimensionless Kerr spin a/M. Must satisfy |a| < 1 for a sub-extremal
        Kerr black hole.
    distance_m:
        Observer distance in meters. Used for angular conversion and physically
        meaningful camera placement, not for the local Kerr metric itself.
    """

    mass_kg: float
    spin_a: float = 0.0
    distance_m: float = 1.0 * KPC_SI

    @classmethod
    def sgr_a_star(cls, spin_a: float = 0.0) -> "BlackHoleSystem":
        # Representative values; expose as defaults, not fitted parameters.
        return cls(mass_kg=4.297e6 * M_SUN_SI, spin_a=spin_a, distance_m=8.277 * KPC_SI)

    @classmethod
    def m87_star(cls, spin_a: float = 0.5) -> "BlackHoleSystem":
        return cls(mass_kg=6.5e9 * M_SUN_SI, spin_a=spin_a, distance_m=16.8 * MPC_SI)

    @property
    def gravitational_radius_m(self) -> float:
        """Return r_g = GM/c^2 in meters."""
        return G_SI * self.mass_kg / (C_SI * C_SI)

    @property
    def gravitational_time_s(self) -> float:
        """Return t_g = GM/c^3 in seconds."""
        return G_SI * self.mass_kg / (C_SI ** 3)

    @property
    def angular_rg_rad(self) -> float:
        """Angular size of one gravitational radius at the specified distance."""
        return self.gravitational_radius_m / self.distance_m

    @property
    def angular_rg_microarcsec(self) -> float:
        return self.angular_rg_rad * (180.0 / math.pi) * 3600.0 * 1.0e6

    def code_radius_to_meters(self, r_over_m: float) -> float:
        return r_over_m * self.gravitational_radius_m

    def code_time_to_seconds(self, t_over_m: float) -> float:
        return t_over_m * self.gravitational_time_s

    def angular_size_microarcsec(self, radius_over_m: float) -> float:
        return radius_over_m * self.angular_rg_microarcsec

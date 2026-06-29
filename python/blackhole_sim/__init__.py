"""Black-hole simulation algorithms.

The package now includes:

1. A legacy Schwarzschild educational renderer kept for comparison.
2. A Kerr spacetime ray tracer using the full Boyer-Lindquist metric,
   ZAMO camera tetrads, orbital mechanics, and relativistic redshift.
3. Relativistic Boris/Vay particle pushers and a guiding-center validation path.
"""

from .geodesics import critical_impact_parameter, trace_null_geodesic
from .renderer import Camera, DiskModel, RenderConfig, render_image
from .kerr import (
    LocalCamera,
    camera_ray_initial_state,
    circular_orbit_four_velocity,
    horizon_radius,
    isco_radius,
    keplerian_omega,
    kerr_metric_covariant,
    kerr_metric_contravariant,
    orbital_period_code,
    trace_kerr_null_geodesic,
)
from .kerr_renderer import KerrDiskModel, KerrRenderConfig, render_kerr_image
from .plasma import boris_push, guiding_center_push
from .relativistic_plasma import (
    RelativisticGuidingCenterState,
    RelativisticParticleState,
    push_guiding_center,
    push_particles,
    relativistic_boris_step,
    relativistic_guiding_center_from_particle,
    vay_step,
)
from .physics import BlackHoleSystem
from .calibration import PhysicalScaling, calibrate_flux_scale
from .grmhd_adapters import load_harm_hdf5, load_bhac_hdf5, load_koral_hdf5
from .polarized_transfer import integrate_polarized_kerr_grrt, stokes_step_exact
from .synchrotron import HybridSynchrotronCoefficients, ThermalSynchrotronCoefficients, NonthermalPowerLawSynchrotronCoefficients

from .accelerated_renderer import (
    AcceleratedRenderConfig,
    render_stokes_image_bricks,
    render_progressive_stokes_bricks,
    sample_coefficient_brick,
)
from .native_kernels import stokes_rk2_brick, stokes_rk2_brick_reference, native_stokes_rk2_available

__all__ = [
    "critical_impact_parameter",
    "trace_null_geodesic",
    "Camera",
    "DiskModel",
    "RenderConfig",
    "render_image",
    "LocalCamera",
    "camera_ray_initial_state",
    "circular_orbit_four_velocity",
    "horizon_radius",
    "isco_radius",
    "keplerian_omega",
    "kerr_metric_covariant",
    "kerr_metric_contravariant",
    "orbital_period_code",
    "trace_kerr_null_geodesic",
    "KerrDiskModel",
    "KerrRenderConfig",
    "render_kerr_image",
    "boris_push",
    "guiding_center_push",
    "RelativisticGuidingCenterState",
    "RelativisticParticleState",
    "push_guiding_center",
    "push_particles",
    "relativistic_boris_step",
    "relativistic_guiding_center_from_particle",
    "vay_step",
    "BlackHoleSystem",
    "PhysicalScaling",
    "calibrate_flux_scale",
    "load_harm_hdf5",
    "load_bhac_hdf5",
    "load_koral_hdf5",
    "integrate_polarized_kerr_grrt",
    "stokes_step_exact",
    "HybridSynchrotronCoefficients",
    "ThermalSynchrotronCoefficients",
    "NonthermalPowerLawSynchrotronCoefficients",
    "AcceleratedRenderConfig",
    "render_stokes_image_bricks",
    "render_progressive_stokes_bricks",
    "sample_coefficient_brick",
    "stokes_rk2_brick",
    "stokes_rk2_brick_reference",
    "native_stokes_rk2_available",
]

__version__ = "0.9.0"

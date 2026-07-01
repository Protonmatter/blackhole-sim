import numpy as np
import pytest

from blackhole_sim.grmhd import generate_analytic_grmhd_torus
from blackhole_sim.kerr import LocalCamera, camera_ray_initial_state, trace_kerr_null_geodesic
from blackhole_sim.radiative_transfer import ThermalSynchrotronFit, TransferConfig, integrate_kerr_grrt, invariant_redshift


def test_invariant_redshift_positive_for_valid_fluid_sample():
    snap = generate_analytic_grmhd_torus(spin_a=0.4, nr=16, ntheta=10, nphi=8)
    sample = snap.sample(float(snap.r[8]), float(snap.theta[5]), float(snap.phi[3]))
    p_cov = np.array([-1.0, 0.0, 0.0, 2.0])
    g = invariant_redshift(p_cov, sample.u_con)
    assert g >= 0.0


def test_grrt_integrator_accumulates_nonnegative_intensity():
    snap = generate_analytic_grmhd_torus(spin_a=0.4, nr=18, ntheta=12, nphi=10)
    cam = LocalCamera.from_degrees(r=45, inclination_degrees=68, fov_y_degrees=28)
    y0, pt, pph = camera_ray_initial_state(cam, snap.spin_a, 0.0, 0.0, 16 / 9)
    trace = trace_kerr_null_geodesic(y0, pt, pph, snap.spin_a, step=0.09, max_steps=1200, escape_radius=120)
    result = integrate_kerr_grrt(
        trace,
        snap,
        coeffs=ThermalSynchrotronFit(emissivity_scale=2.0, absorptivity_scale=0.1),
        cfg=TransferConfig(path_length_scale=0.08),
    )
    assert result.valid_steps >= 0
    assert result.optical_depth >= 0.0
    assert np.all(result.intensity_rgb >= 0.0)


def test_validated_mode_rejects_default_proxy_coefficients():
    snap = generate_analytic_grmhd_torus(spin_a=0.4, nr=8, ntheta=6, nphi=5)
    cam = LocalCamera.from_degrees(r=35, inclination_degrees=62, fov_y_degrees=28)
    y0, pt, pph = camera_ray_initial_state(cam, snap.spin_a, 0.0, 0.0, 16 / 9)
    trace = trace_kerr_null_geodesic(y0, pt, pph, snap.spin_a, step=0.12, max_steps=8, escape_radius=80)
    with pytest.raises(ValueError, match="validated transfer mode"):
        integrate_kerr_grrt(trace, snap, cfg=TransferConfig(physics_mode="validated"))

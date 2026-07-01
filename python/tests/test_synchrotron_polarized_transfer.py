import numpy as np

from blackhole_sim.calibration import PhysicalScaling
from blackhole_sim.grmhd import FluidSample
from blackhole_sim.synchrotron import (
    LocalPlasmaFrame,
    NonthermalPowerLawSynchrotronCoefficients,
    PolarizedCoefficients,
    ThermalSynchrotronCoefficients,
    local_plasma_from_sample,
    magnetic_field_strength_code,
    magnetic_pitch_cosine,
)
from blackhole_sim.polarized_transfer import stokes_step_exact


def test_thermal_and_nonthermal_coefficients_have_full_stokes_terms():
    frame = LocalPlasmaFrame(n_e_cm3=1e5, theta_e=10.0, b_gauss=25.0, cos_pitch_to_los=0.2, evpa_rad=0.1)
    thermal = ThermalSynchrotronCoefficients(gamma_grid_size=48, pitch_nodes=6).coefficients(frame, 230e9)
    nonthermal = NonthermalPowerLawSynchrotronCoefficients(gamma_grid_size=48, pitch_nodes=6, gamma_max=1e3).coefficients(frame, 230e9)
    for c in (thermal, nonthermal):
        assert c.j_i >= 0
        assert c.alpha_i >= 0
        assert np.isfinite(c.propagation_matrix()).all()
    assert thermal.rho_v != 0.0
    assert thermal.rho_q >= 0.0


def test_faraday_rotation_converts_q_to_u():
    coeff = PolarizedCoefficients(rho_v=2.0)
    out = stokes_step_exact(np.array([0.0, 1.0, 0.0, 0.0]), coeff, ds_cm=0.1)
    assert np.isclose(out[0], 0.0, atol=1e-12)
    assert abs(out[2]) > 0.05
    assert np.isclose(np.linalg.norm(out[1:3]), 1.0, atol=1e-12)


def test_metric_magnetic_strength_replaces_coordinate_norm():
    r = 10.0
    f = 1.0 - 2.0 / r
    sample = FluidSample(
        r=r,
        theta=np.pi / 2.0,
        phi=0.0,
        rho=1.0,
        theta_e=10.0,
        pressure=1.0,
        b_con=np.array([0.0, np.sqrt(f), 0.0, 0.0]),
        u_con=np.array([1.0 / np.sqrt(f), 0.0, 0.0, 0.0]),
        valid=True,
    )
    assert np.linalg.norm(sample.b_con[1:]) != 1.0
    assert np.isclose(magnetic_field_strength_code(sample, spin_a=0.0), 1.0, atol=1e-12)


def test_pitch_angle_uses_invariant_fluid_frame_projection():
    r = 10.0
    f = 1.0 - 2.0 / r
    sample = FluidSample(
        r=r,
        theta=np.pi / 2.0,
        phi=0.0,
        rho=1.0,
        theta_e=10.0,
        pressure=1.0,
        b_con=np.array([0.0, np.sqrt(f), 0.0, 0.0]),
        u_con=np.array([1.0 / np.sqrt(f), 0.0, 0.0, 0.0]),
        valid=True,
    )
    radial_photon = np.array([-np.sqrt(f), 1.0 / np.sqrt(f), 0.0, 0.0])
    theta_photon = np.array([-np.sqrt(f), 0.0, r, 0.0])
    assert np.isclose(magnetic_pitch_cosine(sample, radial_photon, spin_a=0.0), 1.0, atol=1e-12)
    assert np.isclose(magnetic_pitch_cosine(sample, theta_photon, spin_a=0.0), 0.0, atol=1e-12)


def test_local_plasma_frame_applies_metric_field_before_scaling():
    r = 10.0
    f = 1.0 - 2.0 / r
    sample = FluidSample(
        r=r,
        theta=np.pi / 2.0,
        phi=0.0,
        rho=2.0,
        theta_e=4.0,
        pressure=1.0,
        b_con=np.array([0.0, np.sqrt(f), 0.0, 0.0]),
        u_con=np.array([1.0 / np.sqrt(f), 0.0, 0.0, 0.0]),
        valid=True,
    )
    scaling = PhysicalScaling(1.0, 1.0, rho_cgs_per_code=2.0, b_gauss_per_code=7.0)
    frame = local_plasma_from_sample(sample, scaling, spin_a=0.0)
    assert np.isclose(frame.b_gauss, 7.0, atol=1e-12)
    assert frame.n_e_cm3 > 0.0

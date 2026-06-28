import numpy as np

from blackhole_sim.synchrotron import LocalPlasmaFrame, ThermalSynchrotronCoefficients, NonthermalPowerLawSynchrotronCoefficients, PolarizedCoefficients
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

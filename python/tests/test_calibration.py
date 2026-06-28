import numpy as np

from blackhole_sim.calibration import PhysicalScaling, calibrate_flux_scale, fit_density_scale_from_flux_powerlaw
from blackhole_sim.physics import BlackHoleSystem


def test_physical_scaling_from_mdot_positive():
    s = PhysicalScaling.from_mdot(BlackHoleSystem.m87_star(0.5), mdot_msun_per_year=1e-4)
    assert s.rg_cm > 0
    assert s.rho_cgs_per_code > 0
    assert s.b_gauss_per_code > 0
    assert s.electron_number_density_cm3(1.0) > 0


def test_flux_calibration_and_density_fit():
    s = PhysicalScaling.from_black_hole_system(BlackHoleSystem.sgr_a_star(), 1e-18, 30.0)
    img = np.ones((4,4)) * 1e-20
    result = calibrate_flux_scale(s, img, target_flux_jy=2.4, pixel_solid_angle_sr=1e-20)
    assert result.measured_flux_jy > 0
    assert result.multiplicative_scale > 0
    fitted = fit_density_scale_from_flux_powerlaw(s, measured_flux_jy=1.0, target_flux_jy=8.0)
    assert fitted.rho_cgs_per_code > s.rho_cgs_per_code
    assert fitted.b_gauss_per_code > s.b_gauss_per_code

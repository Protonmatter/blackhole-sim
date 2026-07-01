import math

import numpy as np

from blackhole_sim.kerr import (
    LocalCamera,
    camera_ray_initial_state,
    circular_orbit_four_velocity,
    hamiltonian_null,
    horizon_radius,
    isco_radius,
    kerr_metric_covariant,
    kerr_metric_contravariant,
    orbital_period_code,
    static_limit_radius,
    trace_kerr_null_geodesic,
    zamo_tetrad,
)


def test_kerr_horizon_and_isco_limits():
    assert np.isclose(horizon_radius(0.0), 2.0)
    assert np.isclose(horizon_radius(0.9), 1.0 + math.sqrt(1.0 - 0.9**2))
    assert np.isclose(static_limit_radius(math.pi / 2.0, 0.9), 2.0)
    assert np.isclose(static_limit_radius(0.0, 0.9), horizon_radius(0.9))
    assert np.isclose(isco_radius(0.0, prograde=True), 6.0)
    assert isco_radius(0.9, prograde=True) < 3.0
    assert isco_radius(0.9, prograde=False) > 8.0


def test_kerr_metric_reduces_to_schwarzschild_at_zero_spin():
    r, th, a = 11.0, 1.2, 0.0
    f = 1.0 - 2.0 / r
    s2 = math.sin(th) ** 2

    gcov = kerr_metric_covariant(r, th, a)
    expected_cov = np.diag([-f, 1.0 / f, r * r, r * r * s2])
    assert np.allclose(gcov, expected_cov, atol=1e-12)

    gcon = kerr_metric_contravariant(r, th, a)
    expected_con = np.diag([-1.0 / f, f, 1.0 / (r * r), 1.0 / (r * r * s2)])
    assert np.allclose(gcon, expected_con, atol=1e-12)


def test_metric_inverse_identity():
    r, th, a = 8.0, 1.1, 0.72
    gcov = kerr_metric_covariant(r, th, a)
    gcon = kerr_metric_contravariant(r, th, a)
    assert np.allclose(gcov @ gcon, np.eye(4), atol=1e-11)


def test_zamo_tetrad_is_orthonormal():
    r, th, a = 12.0, 1.0, 0.7
    tetrad = zamo_tetrad(r, th, a)
    gcov = kerr_metric_covariant(r, th, a)
    gram = tetrad @ gcov @ tetrad.T
    assert np.allclose(gram, np.diag([-1.0, 1.0, 1.0, 1.0]), atol=1e-11)


def test_camera_ray_is_null():
    cam = LocalCamera.from_degrees(r=40.0, inclination_degrees=62.0, fov_y_degrees=30.0)
    y0, pt, pphi = camera_ray_initial_state(cam, a=0.6, ndc_x=0.1, ndc_y=-0.2, aspect=16/9)
    assert abs(hamiltonian_null(y0, 0.6, pt, pphi)) < 1e-11


def test_short_geodesic_preserves_null_hamiltonian():
    cam = LocalCamera.from_degrees(r=35.0, inclination_degrees=60.0, fov_y_degrees=28.0)
    y0, pt, pphi = camera_ray_initial_state(cam, a=0.5, ndc_x=0.25, ndc_y=0.15, aspect=16/9)
    tr = trace_kerr_null_geodesic(y0, pt, pphi, a=0.5, step=0.02, max_steps=50, escape_radius=80.0)
    assert tr.hamiltonian_error_max < 1e-6
    assert tr.states.shape[1] == 6


def test_circular_orbit_four_velocity_timelike_and_period_positive():
    u = circular_orbit_four_velocity(8.0, a=0.4, prograde=True)
    g = kerr_metric_covariant(8.0, math.pi / 2, 0.4)
    assert np.isclose(u @ g @ u, -1.0, atol=1e-12)
    assert orbital_period_code(8.0, a=0.4, prograde=True) > 0.0

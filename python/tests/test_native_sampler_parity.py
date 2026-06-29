import math

import numpy as np
import pytest

from blackhole_sim.native_kernels import (
    SAMPLE_BRICK_TRILINEAR_ATOL,
    SAMPLE_BRICK_TRILINEAR_RTOL,
    STOKES_RK2_ATOL,
    STOKES_RK2_RTOL,
    native_sample_and_step_stokes_available,
    native_sample_brick_trilinear_available,
    sample_and_step_stokes,
    sample_and_step_stokes_reference,
    sample_brick_trilinear,
    sample_brick_trilinear_reference,
    stokes_rk2_brick_reference,
)


def _linear_coeff_fixture(nphi: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_grid = np.array([0.0, 1.0], dtype=np.float64)
    theta_grid = np.array([0.0, 1.0], dtype=np.float64)
    if nphi == 2:
        phi_grid = np.array([0.0, 1.0], dtype=np.float64)
    else:
        phi_grid = np.linspace(0.0, 2.0 * math.pi, nphi, endpoint=False, dtype=np.float64)
    coeffs = np.empty((2, 2, nphi, 11), dtype=np.float64)
    for ir in range(2):
        for itheta in range(2):
            for iphi in range(nphi):
                base = (100.0 * ir) + (10.0 * itheta) + float(iphi)
                coeffs[ir, itheta, iphi] = base + np.arange(11, dtype=np.float64) * 0.125
    return coeffs, r_grid, theta_grid, phi_grid


def _sample_points() -> np.ndarray:
    return np.array(
        [
            [0.25, 0.50, 0.75],
            [1.00, 1.00, 0.00],
            [-0.10, 0.50, 0.75],
            [0.25, 1.10, 0.75],
        ],
        dtype=np.float64,
    )


def _initial(point_count: int) -> np.ndarray:
    initial = np.zeros((point_count, 4), dtype=np.float64)
    initial[:, 0] = 1.0e-2
    initial[:, 1] = 1.0e-3
    return initial


def test_sample_brick_trilinear_reference_interpolates_in_bounds():
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture()
    point = np.array([[0.25, 0.50, 0.75]], dtype=np.float64)

    out = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, point)
    expected = 30.75 + np.arange(11, dtype=np.float64) * 0.125

    assert out.shape == (1, 11)
    np.testing.assert_allclose(out[0], expected, rtol=SAMPLE_BRICK_TRILINEAR_RTOL, atol=SAMPLE_BRICK_TRILINEAR_ATOL)


def test_sample_brick_trilinear_reference_returns_nan_outside_r_theta():
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture()
    points = np.array(
        [
            [-0.10, 0.50, 0.50],
            [1.10, 0.50, 0.50],
            [0.50, -0.10, 0.50],
            [0.50, 1.10, 0.50],
        ],
        dtype=np.float64,
    )

    out = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)

    assert out.shape == (4, 11)
    assert np.all(np.isnan(out))


def test_sample_brick_trilinear_reference_wraps_periodic_phi():
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture(nphi=4)
    base_point = np.array([[0.50, 0.50, math.pi / 4.0]], dtype=np.float64)
    wrapped_point = np.array([[0.50, 0.50, (2.0 * math.pi) + (math.pi / 4.0)]], dtype=np.float64)

    base = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, base_point)
    wrapped = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, wrapped_point)

    np.testing.assert_allclose(wrapped, base, rtol=SAMPLE_BRICK_TRILINEAR_RTOL, atol=SAMPLE_BRICK_TRILINEAR_ATOL)


def test_sample_brick_trilinear_fallback_matches_reference():
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture()
    points = _sample_points()

    reference = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)
    fallback = sample_brick_trilinear(coeffs, r_grid, theta_grid, phi_grid, points, prefer_native=False)

    np.testing.assert_allclose(
        fallback,
        reference,
        rtol=SAMPLE_BRICK_TRILINEAR_RTOL,
        atol=SAMPLE_BRICK_TRILINEAR_ATOL,
        equal_nan=True,
    )


def test_native_sample_brick_trilinear_matches_reference_when_installed():
    if not native_sample_brick_trilinear_available():
        pytest.skip("blackhole_native.sample_brick_trilinear is not installed")
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture(nphi=4)
    points = np.array(
        [
            [0.25, 0.50, math.pi / 4.0],
            [0.50, 0.25, (2.0 * math.pi) + (math.pi / 4.0)],
            [-0.10, 0.50, math.pi / 4.0],
            [0.50, 1.10, math.pi / 4.0],
        ],
        dtype=np.float64,
    )

    reference = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)
    native = sample_brick_trilinear(coeffs, r_grid, theta_grid, phi_grid, points, prefer_native=True)

    np.testing.assert_allclose(
        native,
        reference,
        rtol=SAMPLE_BRICK_TRILINEAR_RTOL,
        atol=SAMPLE_BRICK_TRILINEAR_ATOL,
        equal_nan=True,
    )


def test_sample_and_step_stokes_reference_matches_sampler_then_rk2():
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture()
    points = _sample_points()
    initial = _initial(points.shape[0])

    sampled = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)
    expected = np.full((points.shape[0], 4), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(sampled), axis=1)
    expected[valid] = stokes_rk2_brick_reference(sampled[valid], ds_cm=0.05, initial=initial[valid])
    out = sample_and_step_stokes_reference(coeffs, r_grid, theta_grid, phi_grid, points, ds_cm=0.05, initial=initial)

    np.testing.assert_allclose(out, expected, rtol=STOKES_RK2_RTOL, atol=STOKES_RK2_ATOL, equal_nan=True)


def test_native_sample_and_step_stokes_matches_reference_when_installed():
    if not native_sample_and_step_stokes_available():
        pytest.skip("blackhole_native.sample_and_step_stokes is not installed")
    coeffs, r_grid, theta_grid, phi_grid = _linear_coeff_fixture(nphi=4)
    points = np.array(
        [
            [0.25, 0.50, math.pi / 4.0],
            [0.50, 0.25, (2.0 * math.pi) + (math.pi / 4.0)],
            [-0.10, 0.50, math.pi / 4.0],
        ],
        dtype=np.float64,
    )
    initial = _initial(points.shape[0])

    reference = sample_and_step_stokes_reference(coeffs, r_grid, theta_grid, phi_grid, points, ds_cm=0.05, initial=initial)
    native = sample_and_step_stokes(coeffs, r_grid, theta_grid, phi_grid, points, ds_cm=0.05, initial=initial, prefer_native=True)

    np.testing.assert_allclose(native, reference, rtol=STOKES_RK2_RTOL, atol=STOKES_RK2_ATOL, equal_nan=True)

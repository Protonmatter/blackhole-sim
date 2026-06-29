import numpy as np
import pytest

from blackhole_sim.native_kernels import (
    STOKES_RK2_ATOL,
    STOKES_RK2_RTOL,
    deterministic_stokes_coefficients,
    native_stokes_rk2_available,
    stokes_rk2_brick,
    stokes_rk2_brick_reference,
)


def _initial(shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape + (4,), dtype=np.float64)
    out[..., 0] = 1.0e-2
    out[..., 1] = 1.0e-3
    return out


def test_reference_stokes_rk2_brick_shape_and_finiteness():
    coeffs = deterministic_stokes_coefficients(3, 2, 4)
    out = stokes_rk2_brick_reference(coeffs, ds_cm=0.05, initial=_initial(coeffs.shape[:-1]))

    assert out.shape == (3, 2, 4, 4)
    assert np.all(np.isfinite(out))


def test_stokes_rk2_brick_fallback_matches_reference():
    coeffs = deterministic_stokes_coefficients(3, 2, 4)
    initial = _initial(coeffs.shape[:-1])
    reference = stokes_rk2_brick_reference(coeffs, ds_cm=0.05, initial=initial)
    fallback = stokes_rk2_brick(coeffs, ds_cm=0.05, initial=initial, prefer_native=False)

    np.testing.assert_allclose(fallback, reference, rtol=STOKES_RK2_RTOL, atol=STOKES_RK2_ATOL)


def test_native_stokes_rk2_brick_matches_reference_when_installed():
    if not native_stokes_rk2_available():
        pytest.skip("blackhole_native.stokes_rk2_brick is not installed")
    coeffs = deterministic_stokes_coefficients(4, 3, 5)
    initial = _initial(coeffs.shape[:-1])
    reference = stokes_rk2_brick_reference(coeffs, ds_cm=0.05, initial=initial)
    native = stokes_rk2_brick(coeffs, ds_cm=0.05, initial=initial, prefer_native=True)

    np.testing.assert_allclose(native, reference, rtol=STOKES_RK2_RTOL, atol=STOKES_RK2_ATOL)


def test_stokes_rk2_brick_rejects_invalid_shapes():
    coeffs = deterministic_stokes_coefficients(2, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        stokes_rk2_brick_reference(coeffs[..., :10], ds_cm=0.05)
    with pytest.raises(ValueError, match="initial"):
        stokes_rk2_brick_reference(coeffs, ds_cm=0.05, initial=np.zeros((2, 4)))

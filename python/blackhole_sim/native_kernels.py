"""Reference and native CPU kernels for first hot-loop parity targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from .native_loader import load_native_module

STOKES_RK2_RTOL = 1.0e-12
STOKES_RK2_ATOL = 1.0e-12


def deterministic_stokes_coefficients(
    nr: int = 4,
    ntheta: int = 3,
    nphi: int = 4,
    dtype: str | np.dtype[Any] = "float64",
) -> np.ndarray:
    """Return a small deterministic coefficient brick for parity tests.

    The coefficients use the same 11-value ordering as ``COEFF_NAMES``:
    emission, absorption, and Faraday terms. Values are deliberately small so
    the RK2 step remains well-conditioned across platforms.
    """

    if min(int(nr), int(ntheta), int(nphi)) < 1:
        raise ValueError("nr, ntheta, and nphi must be positive")
    r = np.linspace(0.0, 1.0, int(nr), dtype=np.float64)
    theta = np.linspace(0.0, 1.0, int(ntheta), dtype=np.float64)
    phi = np.linspace(0.0, 1.0, int(nphi), dtype=np.float64)
    rr, tt, pp = np.meshgrid(r, theta, phi, indexing="ij")
    coeffs = np.empty((int(nr), int(ntheta), int(nphi), 11), dtype=np.float64)
    coeffs[..., 0] = 1.0e-3 * (1.0 + rr)
    coeffs[..., 1] = 2.0e-4 * (tt - 0.5)
    coeffs[..., 2] = 1.5e-4 * (pp - 0.5)
    coeffs[..., 3] = 1.0e-4 * (rr - tt)
    coeffs[..., 4] = 1.0e-2 + 1.0e-3 * rr
    coeffs[..., 5] = 1.0e-4 * (tt - 0.5)
    coeffs[..., 6] = 1.0e-4 * (pp - 0.5)
    coeffs[..., 7] = 8.0e-5 * (rr - 0.5)
    coeffs[..., 8] = 2.0e-3 * (pp - 0.5)
    coeffs[..., 9] = 1.0e-3 * (rr - tt)
    coeffs[..., 10] = 1.0e-3 * (tt - pp)
    return coeffs.astype(dtype, copy=False)


def _coerce_coefficients(coeffs: np.ndarray | Any) -> tuple[np.ndarray, tuple[int, ...]]:
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    if coeff_arr.ndim < 1 or coeff_arr.shape[-1] != 11:
        raise ValueError("coeffs must have shape (..., 11)")
    if coeff_arr.size == 0:
        raise ValueError("coeffs must contain at least one cell")
    if not np.all(np.isfinite(coeff_arr)):
        raise ValueError("coeffs must contain only finite values")
    return np.ascontiguousarray(coeff_arr), tuple(int(v) for v in coeff_arr.shape[:-1])


def _coerce_initial(initial: np.ndarray | Any | None, prefix_shape: tuple[int, ...]) -> np.ndarray:
    if initial is None:
        return np.zeros(prefix_shape + (4,), dtype=np.float64)
    initial_arr = np.asarray(initial, dtype=np.float64)
    if not np.all(np.isfinite(initial_arr)):
        raise ValueError("initial must contain only finite values")
    if initial_arr.shape == (4,):
        return np.ascontiguousarray(np.broadcast_to(initial_arr, prefix_shape + (4,)))
    if initial_arr.shape != prefix_shape + (4,):
        raise ValueError(f"initial must have shape (4,) or {prefix_shape + (4,)}")
    return np.ascontiguousarray(initial_arr)


def _validate_ds(ds_cm: float) -> float:
    ds_value = float(ds_cm)
    if not np.isfinite(ds_value) or ds_value < 0.0:
        raise ValueError("ds_cm must be finite and non-negative")
    return ds_value


def _stokes_rhs(stokes: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    out = np.empty_like(stokes, dtype=np.float64)
    j_i, j_q, j_u, j_v = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    alpha_i, alpha_q, alpha_u, alpha_v = coeffs[:, 4], coeffs[:, 5], coeffs[:, 6], coeffs[:, 7]
    rho_v, rho_q, rho_u = coeffs[:, 8], coeffs[:, 9], coeffs[:, 10]
    out[:, 0] = j_i - (alpha_i * stokes[:, 0] + alpha_q * stokes[:, 1] + alpha_u * stokes[:, 2] + alpha_v * stokes[:, 3])
    out[:, 1] = j_q - (alpha_q * stokes[:, 0] + alpha_i * stokes[:, 1] + rho_v * stokes[:, 2] - rho_u * stokes[:, 3])
    out[:, 2] = j_u - (alpha_u * stokes[:, 0] - rho_v * stokes[:, 1] + alpha_i * stokes[:, 2] + rho_q * stokes[:, 3])
    out[:, 3] = j_v - (alpha_v * stokes[:, 0] + rho_u * stokes[:, 1] - rho_q * stokes[:, 2] + alpha_i * stokes[:, 3])
    return out


def stokes_rk2_brick_reference(
    coeffs: np.ndarray | Any,
    ds_cm: float,
    initial: np.ndarray | Any | None = None,
) -> np.ndarray:
    """Vectorized Python reference for one RK2 Stokes step over a coefficient brick."""

    coeff_arr, prefix_shape = _coerce_coefficients(coeffs)
    ds_value = _validate_ds(ds_cm)
    initial_arr = _coerce_initial(initial, prefix_shape)
    coeff_flat = coeff_arr.reshape((-1, 11))
    stokes_flat = initial_arr.reshape((-1, 4))
    k1 = _stokes_rhs(stokes_flat, coeff_flat)
    mid = stokes_flat + 0.5 * ds_value * k1
    k2 = _stokes_rhs(mid, coeff_flat)
    return np.ascontiguousarray((stokes_flat + ds_value * k2).reshape(prefix_shape + (4,)))


def native_stokes_rk2_available() -> bool:
    module = load_native_module()
    return module is not None and callable(getattr(module, "stokes_rk2_brick", None))


def stokes_rk2_brick(
    coeffs: np.ndarray | Any,
    ds_cm: float,
    initial: np.ndarray | Any | None = None,
    *,
    prefer_native: bool = True,
) -> np.ndarray:
    """Run one RK2 Stokes step through native Rust when available, else Python."""

    coeff_arr, prefix_shape = _coerce_coefficients(coeffs)
    ds_value = _validate_ds(ds_cm)
    initial_arr = _coerce_initial(initial, prefix_shape)
    if prefer_native:
        module = load_native_module()
        native_fn = getattr(module, "stokes_rk2_brick", None) if module is not None else None
        if callable(native_fn):
            coeff_flat = coeff_arr.reshape((-1, 11))
            initial_flat = initial_arr.reshape((-1, 4))
            raw = native_fn(coeff_flat.ravel().tolist(), ds_value, initial_flat.ravel().tolist())
            return np.asarray(raw, dtype=np.float64).reshape(prefix_shape + (4,))
    return stokes_rk2_brick_reference(coeff_arr, ds_value, initial_arr)

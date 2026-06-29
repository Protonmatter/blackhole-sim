"""Reference and native CPU kernels for first hot-loop parity targets."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .native_loader import load_native_module

STOKES_RK2_RTOL = 1.0e-12
STOKES_RK2_ATOL = 1.0e-12
SAMPLE_BRICK_TRILINEAR_RTOL = 1.0e-12
SAMPLE_BRICK_TRILINEAR_ATOL = 1.0e-12
InvalidSamplePolicy = Literal["nan", "initial", "zero"]


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


def _coerce_grid(name: str, grid: np.ndarray | Any) -> np.ndarray:
    grid_arr = np.asarray(grid, dtype=np.float64)
    if grid_arr.ndim != 1 or grid_arr.size < 2:
        raise ValueError(f"{name} grid must contain at least 2 values")
    if not np.all(np.isfinite(grid_arr)):
        raise ValueError(f"{name} grid must contain only finite values")
    if np.any(np.diff(grid_arr) <= 0.0):
        raise ValueError(f"{name} grid must be strictly increasing")
    return np.ascontiguousarray(grid_arr)


def _coerce_points(points: np.ndarray | Any) -> np.ndarray:
    point_arr = np.asarray(points, dtype=np.float64)
    if point_arr.ndim == 1:
        if point_arr.size % 3 != 0:
            raise ValueError("points must contain r/theta/phi triples")
        point_arr = point_arr.reshape((-1, 3))
    if point_arr.ndim != 2 or point_arr.shape[1] != 3:
        raise ValueError("points must have shape (n, 3) or be flat r/theta/phi triples")
    if not np.all(np.isfinite(point_arr)):
        raise ValueError("points must contain only finite values")
    return np.ascontiguousarray(point_arr)


def _coerce_sampler_inputs(
    coeffs: np.ndarray | Any,
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    phi_grid: np.ndarray | Any,
    points: np.ndarray | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeff_arr, prefix_shape = _coerce_coefficients(coeffs)
    if len(prefix_shape) != 3:
        raise ValueError("coeffs must have shape (r, theta, phi, 11)")
    rg = _coerce_grid("r", r_grid)
    tg = _coerce_grid("theta", theta_grid)
    pg = _coerce_grid("phi", phi_grid)
    expected_shape = (int(rg.size), int(tg.size), int(pg.size))
    if prefix_shape != expected_shape:
        raise ValueError(f"coeffs grid shape must be {expected_shape}; got {prefix_shape}")
    return coeff_arr, rg, tg, pg, _coerce_points(points)


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


def _coerce_invalid_sample_policy(policy: InvalidSamplePolicy) -> InvalidSamplePolicy:
    if policy not in {"nan", "initial", "zero"}:
        raise ValueError("invalid_policy must be 'nan', 'initial', or 'zero'")
    return policy


def _wrap_phi(phi: float, base: float = 0.0) -> float:
    return float(((phi - base) % (2.0 * math.pi)) + base)


def _bracket_linear_grid(grid: np.ndarray, x: float) -> tuple[int, int, float] | None:
    if x < grid[0] or x > grid[-1]:
        return None
    if x == grid[-1]:
        return int(grid.size - 2), int(grid.size - 1), 1.0
    i0 = int(np.searchsorted(grid, x, side="right") - 1)
    i0 = max(0, min(i0, int(grid.size - 2)))
    i1 = i0 + 1
    w = (x - grid[i0]) / max(grid[i1] - grid[i0], 1.0e-300)
    return i0, i1, float(w)


def _bracket_periodic_phi_grid(grid: np.ndarray, phi: float) -> tuple[int, int, float]:
    p = _wrap_phi(phi, float(grid[0]))
    n = int(grid.size)
    i0 = int(np.searchsorted(grid, p, side="right") - 1)
    if i0 < 0:
        i0 = n - 1
    if i0 >= n:
        i0 = n - 1
    i1 = (i0 + 1) % n
    hi = grid[i1] if i1 > i0 else grid[0] + 2.0 * math.pi
    pp = p if p >= grid[i0] else p + 2.0 * math.pi
    w = (pp - grid[i0]) / max(hi - grid[i0], 1.0e-300)
    return i0, i1, float(w)


def _sample_one(coeffs: np.ndarray, r_grid: np.ndarray, theta_grid: np.ndarray, phi_grid: np.ndarray, point: np.ndarray) -> np.ndarray:
    rb = _bracket_linear_grid(r_grid, float(point[0]))
    tb = _bracket_linear_grid(theta_grid, float(point[1]))
    if rb is None or tb is None:
        return np.full((11,), np.nan, dtype=np.float64)
    r0, r1, wr = rb
    t0, t1, wt = tb
    p0, p1, wp = _bracket_periodic_phi_grid(phi_grid, float(point[2]))
    c000 = coeffs[r0, t0, p0]
    c001 = coeffs[r0, t0, p1]
    c010 = coeffs[r0, t1, p0]
    c011 = coeffs[r0, t1, p1]
    c100 = coeffs[r1, t0, p0]
    c101 = coeffs[r1, t0, p1]
    c110 = coeffs[r1, t1, p0]
    c111 = coeffs[r1, t1, p1]
    c00 = (1.0 - wp) * c000 + wp * c001
    c01 = (1.0 - wp) * c010 + wp * c011
    c10 = (1.0 - wp) * c100 + wp * c101
    c11 = (1.0 - wp) * c110 + wp * c111
    c0 = (1.0 - wt) * c00 + wt * c01
    c1 = (1.0 - wt) * c10 + wt * c11
    return np.ascontiguousarray((1.0 - wr) * c0 + wr * c1)


def sample_brick_valid_mask(
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    points: np.ndarray | Any,
) -> np.ndarray:
    """Return True for sample points inside the nonperiodic r/theta domain.

    ``phi`` is intentionally ignored here because sampler ``phi`` coordinates
    are periodic. The returned mask is the integration contract downstream
    renderers should use before deciding whether to keep, terminate, or replace
    an invalid sample.
    """

    rg = _coerce_grid("r", r_grid)
    tg = _coerce_grid("theta", theta_grid)
    point_arr = _coerce_points(points)
    return np.ascontiguousarray(
        (point_arr[:, 0] >= rg[0])
        & (point_arr[:, 0] <= rg[-1])
        & (point_arr[:, 1] >= tg[0])
        & (point_arr[:, 1] <= tg[-1])
    )


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


def sample_brick_trilinear_reference(
    coeffs: np.ndarray | Any,
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    phi_grid: np.ndarray | Any,
    points: np.ndarray | Any,
) -> np.ndarray:
    """Python reference for trilinear coefficient sampling.

    Samples outside the nonperiodic ``r`` or ``theta`` grids return NaN
    11-coefficient vectors so invalid domain access cannot be confused with
    physical zero coefficients. ``phi`` is periodic over a 2*pi domain anchored
    at ``phi_grid[0]`` to match the accelerated renderer's existing helper.
    """

    coeff_arr, rg, tg, pg, point_arr = _coerce_sampler_inputs(coeffs, r_grid, theta_grid, phi_grid, points)
    out = np.zeros((int(point_arr.shape[0]), 11), dtype=np.float64)
    for idx, point in enumerate(point_arr):
        out[idx] = _sample_one(coeff_arr, rg, tg, pg, point)
    return np.ascontiguousarray(out)


def native_sample_brick_trilinear_available() -> bool:
    module = load_native_module()
    return module is not None and callable(getattr(module, "sample_brick_trilinear", None))


def sample_brick_trilinear(
    coeffs: np.ndarray | Any,
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    phi_grid: np.ndarray | Any,
    points: np.ndarray | Any,
    *,
    prefer_native: bool = True,
) -> np.ndarray:
    """Sample coefficient vectors through native Rust when available, else Python."""

    coeff_arr, rg, tg, pg, point_arr = _coerce_sampler_inputs(coeffs, r_grid, theta_grid, phi_grid, points)
    if prefer_native:
        module = load_native_module()
        native_fn = getattr(module, "sample_brick_trilinear", None) if module is not None else None
        if callable(native_fn):
            raw = native_fn(
                coeff_arr.ravel().tolist(),
                rg.ravel().tolist(),
                tg.ravel().tolist(),
                pg.ravel().tolist(),
                point_arr.ravel().tolist(),
            )
            return np.asarray(raw, dtype=np.float64).reshape((int(point_arr.shape[0]), 11))
    return sample_brick_trilinear_reference(coeff_arr, rg, tg, pg, point_arr)


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


def sample_and_step_stokes_reference(
    coeffs: np.ndarray | Any,
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    phi_grid: np.ndarray | Any,
    points: np.ndarray | Any,
    ds_cm: float,
    initial: np.ndarray | Any | None = None,
    invalid_policy: InvalidSamplePolicy = "nan",
) -> np.ndarray:
    """Python reference for sampling a coefficient brick and applying one RK2 Stokes step."""

    coeff_arr, rg, tg, pg, point_arr = _coerce_sampler_inputs(coeffs, r_grid, theta_grid, phi_grid, points)
    ds_value = _validate_ds(ds_cm)
    policy = _coerce_invalid_sample_policy(invalid_policy)
    initial_arr = _coerce_initial(initial, (int(point_arr.shape[0]),))
    sampled = sample_brick_trilinear_reference(coeff_arr, rg, tg, pg, point_arr)
    out = np.full((int(point_arr.shape[0]), 4), np.nan, dtype=np.float64)
    valid = sample_brick_valid_mask(rg, tg, point_arr)
    if np.any(valid):
        out[valid] = stokes_rk2_brick_reference(sampled[valid], ds_value, initial_arr[valid])
    if np.any(~valid):
        if policy == "initial":
            out[~valid] = initial_arr[~valid]
        elif policy == "zero":
            out[~valid] = 0.0
    return np.ascontiguousarray(out)


def native_sample_and_step_stokes_available() -> bool:
    module = load_native_module()
    return module is not None and callable(getattr(module, "sample_and_step_stokes", None))


def sample_and_step_stokes(
    coeffs: np.ndarray | Any,
    r_grid: np.ndarray | Any,
    theta_grid: np.ndarray | Any,
    phi_grid: np.ndarray | Any,
    points: np.ndarray | Any,
    ds_cm: float,
    initial: np.ndarray | Any | None = None,
    *,
    prefer_native: bool = True,
    invalid_policy: InvalidSamplePolicy = "nan",
) -> np.ndarray:
    """Sample a coefficient brick and step Stokes values through native Rust when available."""

    coeff_arr, rg, tg, pg, point_arr = _coerce_sampler_inputs(coeffs, r_grid, theta_grid, phi_grid, points)
    ds_value = _validate_ds(ds_cm)
    policy = _coerce_invalid_sample_policy(invalid_policy)
    initial_arr = _coerce_initial(initial, (int(point_arr.shape[0]),))
    valid = sample_brick_valid_mask(rg, tg, point_arr)
    if prefer_native:
        module = load_native_module()
        native_fn = getattr(module, "sample_and_step_stokes", None) if module is not None else None
        if callable(native_fn):
            raw = native_fn(
                coeff_arr.ravel().tolist(),
                rg.ravel().tolist(),
                tg.ravel().tolist(),
                pg.ravel().tolist(),
                point_arr.ravel().tolist(),
                ds_value,
                initial_arr.ravel().tolist(),
            )
            out = np.asarray(raw, dtype=np.float64).reshape((int(point_arr.shape[0]), 4))
            if np.any(~valid):
                if policy == "initial":
                    out[~valid] = initial_arr[~valid]
                elif policy == "zero":
                    out[~valid] = 0.0
            return np.ascontiguousarray(out)
    return sample_and_step_stokes_reference(
        coeff_arr,
        rg,
        tg,
        pg,
        point_arr,
        ds_value,
        initial_arr,
        invalid_policy=policy,
    )

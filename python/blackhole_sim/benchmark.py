"""Deterministic micro-benchmarks for native hot-loop migration planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import platform
import time
from typing import Any, Literal, cast

import numpy as np

from .calibration import PhysicalScaling
from .coefficient_bricks import precompute_coefficient_bricks
from .grmhd import generate_analytic_grmhd_torus
from .native_kernels import (
    SAMPLE_BRICK_TRILINEAR_ATOL,
    SAMPLE_BRICK_TRILINEAR_RTOL,
    STOKES_RK2_ATOL,
    STOKES_RK2_RTOL,
    deterministic_stokes_coefficients,
    native_sample_brick_trilinear_available,
    native_stokes_rk2_available,
    sample_brick_trilinear,
    sample_brick_trilinear_reference,
    stokes_rk2_brick,
    stokes_rk2_brick_reference,
)
from .platform_probe import runtime_arch_report


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    iterations: int
    items: int
    items_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def coefficient_brick_benchmark(
    nr: int = 8,
    ntheta: int = 6,
    nphi: int = 8,
    iterations: int = 1,
    dtype: str = "float32",
) -> BenchmarkResult:
    """Benchmark coefficient precompute throughput on a deterministic fixture."""

    if min(nr, ntheta, nphi, iterations) < 1:
        raise ValueError("nr, ntheta, nphi, and iterations must be positive")
    if dtype not in {"float32", "float16", "float64"}:
        raise ValueError("dtype must be float32, float16, or float64")
    dtype_value = cast(Literal["float32", "float16", "float64"], dtype)
    snapshot = generate_analytic_grmhd_torus(nr=nr, ntheta=ntheta, nphi=nphi)
    scaling = PhysicalScaling(
        mass_bh_g=1.0,
        distance_cm=1.0,
        rho_cgs_per_code=1.0e-18,
        b_gauss_per_code=30.0,
    )
    cells = int(nr) * int(ntheta) * int(nphi)
    start = time.perf_counter()
    for _ in range(int(iterations)):
        precompute_coefficient_bricks(snapshot, scaling, dtype=dtype_value)
    elapsed = max(time.perf_counter() - start, 1.0e-12)
    total = cells * int(iterations)
    arch = runtime_arch_report()
    return BenchmarkResult(
        name="coefficient_brick_precompute",
        seconds=elapsed,
        iterations=int(iterations),
        items=total,
        items_per_second=total / elapsed,
        metadata={
            "grid": [int(nr), int(ntheta), int(nphi)],
            "dtype": dtype,
            "python": platform.python_version(),
            "process_arch": arch["process_arch"],
            "emulation_detected": arch["emulation_detected"],
        },
    )


def _sampler_fixture(
    nr: int,
    ntheta: int,
    nphi: int,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeffs = deterministic_stokes_coefficients(nr=nr, ntheta=ntheta, nphi=nphi)
    r_grid = np.linspace(0.0, 1.0, int(nr), dtype=np.float64)
    theta_grid = np.linspace(0.0, 1.0, int(ntheta), dtype=np.float64)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, int(nphi), endpoint=False, dtype=np.float64)
    idx = np.arange(int(point_count), dtype=np.float64)
    denom = max(int(point_count) - 1, 1)
    points = np.empty((int(point_count), 3), dtype=np.float64)
    points[:, 0] = 0.05 + 0.90 * ((idx % denom) / denom)
    points[:, 1] = 0.10 + 0.80 * (((idx * 3.0) % (denom + 1)) / max(denom, 1))
    points[:, 2] = phi_grid[0] + (idx * 0.6180339887498949 * 2.0 * np.pi)
    return coeffs, r_grid, theta_grid, phi_grid, points


def sample_brick_trilinear_parity_benchmark(
    nr: int = 8,
    ntheta: int = 6,
    nphi: int = 8,
    point_count: int = 64,
    iterations: int = 1,
) -> dict[str, Any]:
    """Benchmark Python reference versus native Rust coefficient-brick sampling."""

    if min(nr, ntheta, nphi, point_count, iterations) < 1:
        raise ValueError("nr, ntheta, nphi, point_count, and iterations must be positive")

    coeffs, r_grid, theta_grid, phi_grid, points = _sampler_fixture(nr, ntheta, nphi, point_count)
    total = int(point_count) * int(iterations)
    arch = runtime_arch_report()

    reference_once = sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)
    start = time.perf_counter()
    for _ in range(int(iterations)):
        sample_brick_trilinear_reference(coeffs, r_grid, theta_grid, phi_grid, points)
    reference_elapsed = max(time.perf_counter() - start, 1.0e-12)

    native_available = native_sample_brick_trilinear_available()
    native_result: BenchmarkResult | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    allclose: bool | None = None
    if native_available:
        native_once = sample_brick_trilinear(coeffs, r_grid, theta_grid, phi_grid, points, prefer_native=True)
        diff = np.abs(native_once - reference_once)
        max_abs_diff = float(np.max(diff))
        denom = np.maximum(np.abs(reference_once), SAMPLE_BRICK_TRILINEAR_ATOL)
        max_rel_diff = float(np.max(diff / denom))
        allclose = bool(
            np.allclose(
                native_once,
                reference_once,
                rtol=SAMPLE_BRICK_TRILINEAR_RTOL,
                atol=SAMPLE_BRICK_TRILINEAR_ATOL,
            )
        )
        start = time.perf_counter()
        for _ in range(int(iterations)):
            sample_brick_trilinear(coeffs, r_grid, theta_grid, phi_grid, points, prefer_native=True)
        native_elapsed = max(time.perf_counter() - start, 1.0e-12)
        native_result = BenchmarkResult(
            name="sample_brick_trilinear_native",
            seconds=native_elapsed,
            iterations=int(iterations),
            items=total,
            items_per_second=total / native_elapsed,
            metadata={
                "backend": "rust-cpu",
                "grid": [int(nr), int(ntheta), int(nphi)],
                "point_count": int(point_count),
                "process_arch": arch["process_arch"],
                "emulation_detected": arch["emulation_detected"],
            },
        )

    reference_result = BenchmarkResult(
        name="sample_brick_trilinear_reference",
        seconds=reference_elapsed,
        iterations=int(iterations),
        items=total,
        items_per_second=total / reference_elapsed,
        metadata={
            "backend": "python-numpy",
            "grid": [int(nr), int(ntheta), int(nphi)],
            "point_count": int(point_count),
            "process_arch": arch["process_arch"],
            "emulation_detected": arch["emulation_detected"],
        },
    )
    return {
        "name": "sample_brick_trilinear_parity",
        "reference": reference_result.to_json_dict(),
        "native": native_result.to_json_dict() if native_result is not None else None,
        "parity": {
            "native_available": native_available,
            "allclose": allclose,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "rtol": SAMPLE_BRICK_TRILINEAR_RTOL,
            "atol": SAMPLE_BRICK_TRILINEAR_ATOL,
        },
        "metadata": {
            "workload": "sample_brick_trilinear",
            "grid": [int(nr), int(ntheta), int(nphi)],
            "point_count": int(point_count),
            "python": platform.python_version(),
            "process_arch": arch["process_arch"],
            "emulation_detected": arch["emulation_detected"],
        },
    }


def stokes_rk2_brick_parity_benchmark(
    nr: int = 8,
    ntheta: int = 6,
    nphi: int = 8,
    iterations: int = 1,
    ds_cm: float = 0.05,
) -> dict[str, Any]:
    """Benchmark Python reference versus native Rust Stokes RK2 brick stepping."""

    if min(nr, ntheta, nphi, iterations) < 1:
        raise ValueError("nr, ntheta, nphi, and iterations must be positive")
    if not np.isfinite(float(ds_cm)) or float(ds_cm) < 0.0:
        raise ValueError("ds_cm must be finite and non-negative")

    coeffs = deterministic_stokes_coefficients(nr=nr, ntheta=ntheta, nphi=nphi)
    initial = np.zeros(coeffs.shape[:-1] + (4,), dtype=np.float64)
    initial[..., 0] = 1.0e-2
    initial[..., 1] = 1.0e-3
    cells = int(nr) * int(ntheta) * int(nphi)
    total = cells * int(iterations)
    arch = runtime_arch_report()

    reference_once = stokes_rk2_brick_reference(coeffs, ds_cm, initial)
    start = time.perf_counter()
    for _ in range(int(iterations)):
        stokes_rk2_brick_reference(coeffs, ds_cm, initial)
    reference_elapsed = max(time.perf_counter() - start, 1.0e-12)

    native_available = native_stokes_rk2_available()
    native_result: BenchmarkResult | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    allclose: bool | None = None
    if native_available:
        native_once = stokes_rk2_brick(coeffs, ds_cm, initial, prefer_native=True)
        diff = np.abs(native_once - reference_once)
        max_abs_diff = float(np.max(diff))
        denom = np.maximum(np.abs(reference_once), STOKES_RK2_ATOL)
        max_rel_diff = float(np.max(diff / denom))
        allclose = bool(np.allclose(native_once, reference_once, rtol=STOKES_RK2_RTOL, atol=STOKES_RK2_ATOL))
        start = time.perf_counter()
        for _ in range(int(iterations)):
            stokes_rk2_brick(coeffs, ds_cm, initial, prefer_native=True)
        native_elapsed = max(time.perf_counter() - start, 1.0e-12)
        native_result = BenchmarkResult(
            name="stokes_rk2_brick_native",
            seconds=native_elapsed,
            iterations=int(iterations),
            items=total,
            items_per_second=total / native_elapsed,
            metadata={
                "backend": "rust-cpu",
                "grid": [int(nr), int(ntheta), int(nphi)],
                "process_arch": arch["process_arch"],
                "emulation_detected": arch["emulation_detected"],
            },
        )

    reference_result = BenchmarkResult(
        name="stokes_rk2_brick_reference",
        seconds=reference_elapsed,
        iterations=int(iterations),
        items=total,
        items_per_second=total / reference_elapsed,
        metadata={
            "backend": "python-numpy",
            "grid": [int(nr), int(ntheta), int(nphi)],
            "process_arch": arch["process_arch"],
            "emulation_detected": arch["emulation_detected"],
        },
    )
    return {
        "name": "stokes_rk2_brick_parity",
        "reference": reference_result.to_json_dict(),
        "native": native_result.to_json_dict() if native_result is not None else None,
        "parity": {
            "native_available": native_available,
            "allclose": allclose,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "rtol": STOKES_RK2_RTOL,
            "atol": STOKES_RK2_ATOL,
        },
        "metadata": {
            "workload": "stokes_rk2_brick",
            "grid": [int(nr), int(ntheta), int(nphi)],
            "ds_cm": float(ds_cm),
            "python": platform.python_version(),
            "process_arch": arch["process_arch"],
            "emulation_detected": arch["emulation_detected"],
        },
    }


def benchmark_suite(
    nr: int = 8,
    ntheta: int = 6,
    nphi: int = 8,
    point_count: int = 64,
    iterations: int = 1,
    dtype: str = "float32",
) -> dict[str, Any]:
    return {
        "schema": "blackhole_sim.benchmark.v2",
        "benchmarks": [
            coefficient_brick_benchmark(nr=nr, ntheta=ntheta, nphi=nphi, iterations=iterations, dtype=dtype).to_json_dict(),
            sample_brick_trilinear_parity_benchmark(
                nr=nr,
                ntheta=ntheta,
                nphi=nphi,
                point_count=point_count,
                iterations=iterations,
            ),
            stokes_rk2_brick_parity_benchmark(nr=nr, ntheta=ntheta, nphi=nphi, iterations=iterations),
        ],
    }

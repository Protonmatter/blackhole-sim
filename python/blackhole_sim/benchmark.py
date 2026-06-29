"""Deterministic micro-benchmarks for native hot-loop migration planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import platform
import time
from typing import Any, Literal, cast

from .calibration import PhysicalScaling
from .coefficient_bricks import precompute_coefficient_bricks
from .grmhd import generate_analytic_grmhd_torus
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

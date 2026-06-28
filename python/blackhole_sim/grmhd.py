"""GRMHD snapshot ingestion, validation, and interpolation.

This module provides the data boundary between a Kerr geodesic/radiative-transfer
renderer and fluid simulations. It is intentionally explicit about schema and
units so external GRMHD outputs can be adapted without hiding assumptions.

Coordinate convention
---------------------
- Coordinates are Boyer-Lindquist-like spherical coordinates (r, theta, phi).
- r is measured in GM/c^2.
- theta is polar angle in radians.
- phi is periodic in radians.
- fluid four-velocity is contravariant u^mu = (u^t, u^r, u^theta, u^phi).
- magnetic field is stored as a contravariant four-vector b^mu when available.

The included analytic torus generator is a deterministic fixture and a useful
smoke-test dataset. Production work should load a real GRMHD dump through
``load_grmhd_hdf5`` or an adapter that returns ``GRMHDSnapshot``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .kerr import circular_orbit_four_velocity, horizon_radius, isco_radius, kerr_metric_covariant


class SnapshotSchemaError(ValueError):
    """Raised when a file cannot be mapped into the GRMHD snapshot schema."""


@dataclass(frozen=True)
class FluidSample:
    r: float
    theta: float
    phi: float
    rho: float
    theta_e: float
    pressure: float
    b_con: np.ndarray
    u_con: np.ndarray
    valid: bool = True

    @property
    def magnetization_proxy(self) -> float:
        return float(np.dot(self.b_con, self.b_con) / max(self.rho, 1.0e-300))


@dataclass(frozen=True)
class GRMHDSnapshot:
    """Structured GRMHD snapshot in a renderer-friendly schema."""

    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rho: np.ndarray
    theta_e: np.ndarray
    pressure: np.ndarray
    b_con: np.ndarray
    u_con: np.ndarray
    spin_a: float
    time_m: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", np.asarray(self.r, dtype=float))
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=float))
        object.__setattr__(self, "phi", np.asarray(self.phi, dtype=float))
        for name in ("r", "theta", "phi"):
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.size < 2 or not np.all(np.diff(arr) > 0):
                raise SnapshotSchemaError(f"{name} grid must be a strictly increasing 1-D array")
        shape = (self.r.size, self.theta.size, self.phi.size)
        for name in ("rho", "theta_e", "pressure"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != shape:
                raise SnapshotSchemaError(f"{name} must have shape {shape}; got {arr.shape}")
            object.__setattr__(self, name, arr)
        b = np.asarray(self.b_con, dtype=float)
        u = np.asarray(self.u_con, dtype=float)
        if b.shape != shape + (4,):
            raise SnapshotSchemaError(f"b_con must have shape {shape + (4,)}; got {b.shape}")
        if u.shape != shape + (4,):
            raise SnapshotSchemaError(f"u_con must have shape {shape + (4,)}; got {u.shape}")
        object.__setattr__(self, "b_con", b)
        object.__setattr__(self, "u_con", u)
        if abs(float(self.spin_a)) >= 1.0:
            raise SnapshotSchemaError("spin_a must satisfy |a| < 1")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.r.size, self.theta.size, self.phi.size)

    @property
    def radial_bounds(self) -> tuple[float, float]:
        return float(self.r[0]), float(self.r[-1])

    def in_domain(self, r: float, theta: float) -> bool:
        return bool(self.r[0] <= r <= self.r[-1] and self.theta[0] <= theta <= self.theta[-1])

    def sample(self, r: float, theta: float, phi: float) -> FluidSample:
        """Trilinearly interpolate the snapshot at one coordinate.

        Phi is periodic. r/theta are finite-domain coordinates; samples outside
        them return ``valid=False`` and zero-valued physical fields.
        """
        if not self.in_domain(r, theta) or not np.isfinite([r, theta, phi]).all():
            return _invalid_sample(r, theta, phi)
        rho = float(_interp3(self.r, self.theta, self.phi, self.rho, r, theta, phi))
        theta_e = float(_interp3(self.r, self.theta, self.phi, self.theta_e, r, theta, phi))
        pressure = float(_interp3(self.r, self.theta, self.phi, self.pressure, r, theta, phi))
        b = np.asarray(_interp3(self.r, self.theta, self.phi, self.b_con, r, theta, phi), dtype=float)
        u = np.asarray(_interp3(self.r, self.theta, self.phi, self.u_con, r, theta, phi), dtype=float)
        if rho <= 0.0 or theta_e <= 0.0 or not np.all(np.isfinite(u)):
            return _invalid_sample(r, theta, phi)
        return FluidSample(float(r), float(theta), _wrap_phi(phi, self.phi[0]), rho, theta_e, pressure, b, u, True)

    def to_npz(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            r=self.r,
            theta=self.theta,
            phi=self.phi,
            rho=self.rho,
            theta_e=self.theta_e,
            pressure=self.pressure,
            b_con=self.b_con,
            u_con=self.u_con,
            spin_a=np.array(self.spin_a),
            time_m=np.array(self.time_m),
        )

    @classmethod
    def from_npz(cls, path: str | Path, metadata: Mapping[str, Any] | None = None) -> "GRMHDSnapshot":
        with np.load(path) as z:
            return cls(
                r=z["r"],
                theta=z["theta"],
                phi=z["phi"],
                rho=z["rho"],
                theta_e=z["theta_e"],
                pressure=z["pressure"],
                b_con=z["b_con"],
                u_con=z["u_con"],
                spin_a=float(z["spin_a"]),
                time_m=float(z["time_m"]) if "time_m" in z else 0.0,
                metadata=dict(metadata or {}),
            )

    def to_hdf5(self, path: str | Path) -> None:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("h5py is required for HDF5 output; install blackhole-sim[hdf5]") from exc
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            f.attrs["schema"] = "blackhole_sim.grmhd.v1"
            f.attrs["spin_a"] = self.spin_a
            f.attrs["time_m"] = self.time_m
            for key in ("r", "theta", "phi", "rho", "theta_e", "pressure", "b_con", "u_con"):
                f.create_dataset(key, data=getattr(self, key), compression="gzip", shuffle=True)


_FIELD_ALIASES = {
    "r": ("r", "radius", "x1", "R"),
    "theta": ("theta", "th", "x2", "Theta"),
    "phi": ("phi", "ph", "x3", "Phi"),
    "rho": ("rho", "density", "RHO", "fluid/rho"),
    "theta_e": ("theta_e", "Thetae", "electron_temperature", "Te", "fluid/theta_e"),
    "pressure": ("pressure", "press", "P", "fluid/pressure"),
    "b_con": ("b_con", "bcon", "Bcon", "magnetic_b_con", "fluid/b_con"),
    "u_con": ("u_con", "ucon", "Ucon", "fluid_u_con", "fluid/u_con"),
}


def load_grmhd_hdf5(path: str | Path, aliases: Mapping[str, tuple[str, ...]] | None = None) -> GRMHDSnapshot:
    """Load an HDF5 GRMHD dump into the project schema.

    The loader supports this project's native schema and common alias names.
    For simulation codes with different coordinates or primitive variables,
    write a small adapter that produces the canonical arrays and call the
    ``GRMHDSnapshot`` constructor directly.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 input; install blackhole-sim[hdf5]") from exc

    merged = {k: tuple(v) for k, v in _FIELD_ALIASES.items()}
    if aliases:
        for k, v in aliases.items():
            merged[k] = tuple(v) + merged.get(k, ())

    def read_any(f: Any, logical: str) -> np.ndarray:
        for name in merged[logical]:
            if name in f:
                return np.asarray(f[name])
        raise SnapshotSchemaError(f"Could not find field {logical!r}; tried {merged[logical]}")

    with h5py.File(path, "r") as f:
        spin = float(f.attrs.get("spin_a", f.attrs.get("a", 0.0)))
        time_m = float(f.attrs.get("time_m", f.attrs.get("t", 0.0)))
        arrays = {key: read_any(f, key) for key in _FIELD_ALIASES}
    return GRMHDSnapshot(spin_a=spin, time_m=time_m, metadata={"source": str(path)}, **arrays)


def _invalid_sample(r: float, theta: float, phi: float) -> FluidSample:
    return FluidSample(
        r=float(r),
        theta=float(theta),
        phi=float(phi),
        rho=0.0,
        theta_e=0.0,
        pressure=0.0,
        b_con=np.zeros(4),
        u_con=np.array([1.0, 0.0, 0.0, 0.0]),
        valid=False,
    )


def _wrap_phi(phi: float, base: float = 0.0) -> float:
    return float(((phi - base) % (2.0 * math.pi)) + base)


def _bracket_linear(grid: np.ndarray, x: float) -> tuple[int, int, float] | None:
    if x < grid[0] or x > grid[-1]:
        return None
    if x == grid[-1]:
        return grid.size - 2, grid.size - 1, 1.0
    i0 = int(np.searchsorted(grid, x, side="right") - 1)
    i0 = max(0, min(i0, grid.size - 2))
    i1 = i0 + 1
    w = (x - grid[i0]) / max(grid[i1] - grid[i0], 1.0e-300)
    return i0, i1, float(w)


def _bracket_periodic_phi(grid: np.ndarray, phi: float) -> tuple[int, int, float]:
    p = _wrap_phi(phi, float(grid[0]))
    n = grid.size
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


def _lerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * a + w * b


def _interp3(rg: np.ndarray, tg: np.ndarray, pg: np.ndarray, values: np.ndarray, r: float, theta: float, phi: float) -> np.ndarray:
    rb = _bracket_linear(rg, r)
    tb = _bracket_linear(tg, theta)
    if rb is None or tb is None:
        raise ValueError("sample outside nonperiodic domain")
    r0, r1, wr = rb
    t0, t1, wt = tb
    p0, p1, wp = _bracket_periodic_phi(pg, phi)
    c000 = values[r0, t0, p0]
    c001 = values[r0, t0, p1]
    c010 = values[r0, t1, p0]
    c011 = values[r0, t1, p1]
    c100 = values[r1, t0, p0]
    c101 = values[r1, t0, p1]
    c110 = values[r1, t1, p0]
    c111 = values[r1, t1, p1]
    c00 = _lerp(c000, c001, wp)
    c01 = _lerp(c010, c011, wp)
    c10 = _lerp(c100, c101, wp)
    c11 = _lerp(c110, c111, wp)
    c0 = _lerp(c00, c01, wt)
    c1 = _lerp(c10, c11, wt)
    return _lerp(c0, c1, wr)


def normalize_timelike_u(r: float, theta: float, a: float, u_spatial: np.ndarray) -> np.ndarray:
    """Construct a future-directed timelike four-velocity from spatial components."""
    ur, uth, uph = map(float, u_spatial)
    g = kerr_metric_covariant(r, theta, a)
    A = g[0, 0]
    B = 2.0 * g[0, 3] * uph
    C = g[1, 1] * ur * ur + g[2, 2] * uth * uth + g[3, 3] * uph * uph + 1.0
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        raise ValueError("Requested spatial four-velocity cannot be normalized")
    roots = [(-B + math.sqrt(disc)) / (2.0 * A), (-B - math.sqrt(disc)) / (2.0 * A)]
    ut = max(root for root in roots if root > 0.0)
    return np.array([ut, ur, uth, uph], dtype=float)



def angular_velocity_four_velocity(r: float, theta: float, a: float, omega: float, ur: float = 0.0, uth: float = 0.0) -> np.ndarray:
    """Construct a timelike four-velocity with prescribed angular velocity.

    For ur=uth=0 this gives u^phi = omega u^t. Small radial/polar
    components are included in the normalization equation.
    """
    g = kerr_metric_covariant(r, theta, a)
    A = g[0, 0] + 2.0 * omega * g[0, 3] + omega * omega * g[3, 3]
    C = g[1, 1] * ur * ur + g[2, 2] * uth * uth + 1.0
    if A >= 0.0:
        raise ValueError("Requested angular velocity is not timelike at this coordinate")
    ut = math.sqrt(max(C / (-A), 0.0))
    return np.array([ut, ur, uth, omega * ut], dtype=float)


def generate_analytic_grmhd_torus(
    spin_a: float = 0.85,
    nr: int = 72,
    ntheta: int = 40,
    nphi: int = 48,
    r_min: float | None = None,
    r_max: float = 60.0,
    density_peak_radius: float | None = None,
    scale_height: float = 0.32,
    electron_temp_peak: float = 18.0,
    magnetization: float = 0.08,
    time_m: float = 0.0,
) -> GRMHDSnapshot:
    """Generate a deterministic torus-shaped GRMHD-like fixture.

    This is not a substitute for a GRMHD solver dump. It exists so the full
    renderer, interpolation, and transfer pipeline can run without downloading a
    research dataset. The fields satisfy the project's schema and timelike
    four-velocity normalization checks.
    """
    r_plus = horizon_radius(spin_a)
    rin = max(isco_radius(spin_a, prograde=True), r_plus * 1.15)
    if r_min is None:
        r_min = max(r_plus * 1.08, rin * 0.75)
    r0 = density_peak_radius if density_peak_radius is not None else max(8.0, 2.7 * rin)
    r = np.geomspace(r_min, r_max, nr)
    theta = np.linspace(0.045, math.pi - 0.045, ntheta)
    phi = np.linspace(0.0, 2.0 * math.pi, nphi, endpoint=False)
    R, TH, PH = np.meshgrid(r, theta, phi, indexing="ij")

    log_width = 0.46
    vertical = (TH - math.pi / 2.0) / scale_height
    radial = np.log(R / r0) / log_width
    spiral = 1.0 + 0.08 * np.sin(2.0 * PH + 0.7 * np.log(R) - 0.03 * time_m)
    turbulence = 1.0 + 0.035 * np.sin(5.0 * PH + 3.0 * TH + 0.17 * R)
    rho = np.exp(-0.5 * (radial * radial + vertical * vertical)) * spiral * turbulence
    rho = np.maximum(rho, 1.0e-8)

    theta_e = electron_temp_peak * (np.maximum(R, rin) / rin) ** -0.84 * (0.82 + 0.18 * np.exp(-0.5 * vertical * vertical))
    theta_e = np.maximum(theta_e, 1.0e-3)
    pressure = rho * theta_e

    b_con = np.zeros(R.shape + (4,), dtype=float)
    b_strength = magnetization * np.sqrt(np.maximum(rho, 0.0)) / np.maximum(R, 1.0)
    b_con[..., 3] = b_strength / np.maximum(np.sin(TH), 1.0e-3)  # toroidal field proxy
    b_con[..., 1] = 0.04 * b_strength * np.sin(PH)
    b_con[..., 2] = 0.03 * b_strength * np.cos(PH)

    u_con = np.zeros(R.shape + (4,), dtype=float)
    for i, rr in enumerate(r):
        for j, th in enumerate(theta):
            for k, ph in enumerate(phi):
                omega = 1.0 / (max(rr, rin) ** 1.5 + spin_a)
                ur = 0.0 if (abs(th - math.pi / 2.0) < 0.42 and rr > rin) else -0.012 * math.exp(-((rr - rin) / max(rin, 1.0)) ** 2)
                try:
                    u_con[i, j, k] = angular_velocity_four_velocity(float(rr), float(th), spin_a, omega=omega, ur=ur, uth=0.0)
                except ValueError:
                    u_con[i, j, k] = np.array([1.0, 0.0, 0.0, 0.0])

    return GRMHDSnapshot(
        r=r,
        theta=theta,
        phi=phi,
        rho=rho,
        theta_e=theta_e,
        pressure=pressure,
        b_con=b_con,
        u_con=u_con,
        spin_a=spin_a,
        time_m=time_m,
        metadata={"generator": "analytic_torus_fixture", "not_grmhd_solver_output": True},
    )


def assert_four_velocity_normalization(snapshot: GRMHDSnapshot, atol: float = 1.0e-6, samples: int = 128) -> float:
    """Return worst |u_mu u^mu + 1| over a deterministic sample of cells."""
    nr, nt, np_ = snapshot.shape
    max_err = 0.0
    count = min(samples, nr * nt * np_)
    for n in range(count):
        i = (n * 37) % nr
        j = (n * 17) % nt
        k = (n * 29) % np_
        r = float(snapshot.r[i])
        th = float(snapshot.theta[j])
        u = snapshot.u_con[i, j, k]
        g = kerr_metric_covariant(r, th, snapshot.spin_a)
        err = abs(float(u @ g @ u) + 1.0)
        max_err = max(max_err, err)
    if max_err > atol:
        raise AssertionError(f"four-velocity normalization error {max_err:.3e} > {atol:.3e}")
    return max_err

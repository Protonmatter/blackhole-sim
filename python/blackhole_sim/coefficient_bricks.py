"""Precompute polarized transfer coefficient bricks for GPU renderers.

The GPU path should not integrate the expensive synchrotron distribution at every
ray step. Instead, the CPU loads/calibrates a GRMHD dump, evaluates transfer
coefficients on the fluid grid or a resampled grid, and uploads compact bricks to
WebGPU/CUDA/Metal/ROCm kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .calibration import PhysicalScaling
from .grmhd import GRMHDSnapshot, FluidSample
from .synchrotron import HybridSynchrotronCoefficients, LocalPlasmaFrame, local_plasma_from_sample

COEFF_NAMES: tuple[str, ...] = (
    "j_i", "j_q", "j_u", "j_v",
    "alpha_i", "alpha_q", "alpha_u", "alpha_v",
    "rho_v", "rho_q", "rho_u",
)


@dataclass(frozen=True)
class CoefficientBrickGrid:
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    coeffs: np.ndarray
    nu_hz: float
    spin_a: float
    dtype_name: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", np.asarray(self.r, dtype=np.float32))
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=np.float32))
        object.__setattr__(self, "phi", np.asarray(self.phi, dtype=np.float32))
        c = np.asarray(self.coeffs)
        if c.shape != (self.r.size, self.theta.size, self.phi.size, len(COEFF_NAMES)):
            raise ValueError(f"coeffs must have shape {(self.r.size, self.theta.size, self.phi.size, len(COEFF_NAMES))}; got {c.shape}")
        if not np.all(np.isfinite(c)):
            raise ValueError("coefficient brick contains non-finite values")
        object.__setattr__(self, "coeffs", c)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.coeffs.shape

    @property
    def bytes(self) -> int:
        return int(self.coeffs.nbytes + self.r.nbytes + self.theta.nbytes + self.phi.nbytes)

    def save_npz(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            r=self.r,
            theta=self.theta,
            phi=self.phi,
            coeffs=self.coeffs,
            coeff_names=np.array(COEFF_NAMES),
            nu_hz=np.array(self.nu_hz),
            spin_a=np.array(self.spin_a),
            dtype_name=np.array(self.dtype_name),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "CoefficientBrickGrid":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["r"], z["theta"], z["phi"], z["coeffs"], float(z["nu_hz"]), float(z["spin_a"]), str(z["dtype_name"]))

    def save_hdf5(self, path: str | Path) -> None:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("h5py is required for HDF5 coefficient bricks") from exc
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            f.attrs["schema"] = "blackhole_sim.coefficient_bricks.v1"
            f.attrs["nu_hz"] = self.nu_hz
            f.attrs["spin_a"] = self.spin_a
            f.attrs["dtype_name"] = self.dtype_name
            f.create_dataset("r", data=self.r, compression="gzip", shuffle=True)
            f.create_dataset("theta", data=self.theta, compression="gzip", shuffle=True)
            f.create_dataset("phi", data=self.phi, compression="gzip", shuffle=True)
            f.create_dataset("coeffs", data=self.coeffs, compression="gzip", shuffle=True, chunks=True)
            f.create_dataset("coeff_names", data=np.asarray(COEFF_NAMES, dtype="S"))


def _cell_sample(snapshot: GRMHDSnapshot, i: int, j: int, k: int) -> FluidSample:
    return FluidSample(
        r=float(snapshot.r[i]),
        theta=float(snapshot.theta[j]),
        phi=float(snapshot.phi[k]),
        rho=float(snapshot.rho[i, j, k]),
        theta_e=float(snapshot.theta_e[i, j, k]),
        pressure=float(snapshot.pressure[i, j, k]),
        b_con=np.asarray(snapshot.b_con[i, j, k], dtype=float),
        u_con=np.asarray(snapshot.u_con[i, j, k], dtype=float),
        valid=True,
    )


def _fast_frame(sample: FluidSample, scaling: PhysicalScaling, spin_a: float) -> LocalPlasmaFrame:
    return local_plasma_from_sample(sample, scaling, spin_a=spin_a)


def precompute_coefficient_bricks(
    snapshot: GRMHDSnapshot,
    scaling: PhysicalScaling,
    nu_hz: float = 230e9,
    dtype: Literal["float32", "float16", "float64"] = "float32",
    coeff_model: HybridSynchrotronCoefficients | None = None,
    stride: int = 1,
) -> CoefficientBrickGrid:
    """Evaluate polarized transfer coefficients on a snapshot grid.

    ``stride`` supports quick preview bricks: stride=2 keeps every second cell.
    For final renders use stride=1 or a deliberate resampling stage.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    coeff_model = coeff_model or HybridSynchrotronCoefficients()
    r_idx = np.arange(0, snapshot.r.size, stride)
    t_idx = np.arange(0, snapshot.theta.size, stride)
    p_idx = np.arange(0, snapshot.phi.size, stride)
    out_dtype = np.dtype(dtype)
    coeffs = np.zeros((len(r_idx), len(t_idx), len(p_idx), len(COEFF_NAMES)), dtype=out_dtype)
    for ii, i in enumerate(r_idx):
        for jj, j in enumerate(t_idx):
            for kk, k in enumerate(p_idx):
                sample = _cell_sample(snapshot, int(i), int(j), int(k))
                frame = _fast_frame(sample, scaling, float(snapshot.spin_a))
                c = coeff_model.coefficients(frame, nu_hz)
                coeffs[ii, jj, kk] = np.asarray([
                    c.j_i, c.j_q, c.j_u, c.j_v,
                    c.alpha_i, c.alpha_q, c.alpha_u, c.alpha_v,
                    c.rho_v, c.rho_q, c.rho_u,
                ], dtype=out_dtype)
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=np.finfo(out_dtype).max if out_dtype.kind == "f" else 0.0, neginf=0.0)
    return CoefficientBrickGrid(snapshot.r[r_idx], snapshot.theta[t_idx], snapshot.phi[p_idx], coeffs, float(nu_hz), float(snapshot.spin_a), dtype)


def estimate_brick_memory(nr: int, ntheta: int, nphi: int, dtype: str = "float32") -> dict[str, float]:
    bytes_per = np.dtype(dtype).itemsize
    coeff_bytes = int(nr) * int(ntheta) * int(nphi) * len(COEFF_NAMES) * bytes_per
    grid_bytes = (int(nr) + int(ntheta) + int(nphi)) * 4
    total = coeff_bytes + grid_bytes
    return {"coeff_bytes": float(coeff_bytes), "grid_bytes": float(grid_bytes), "total_bytes": float(total), "total_mib": total / (1024.0 * 1024.0)}

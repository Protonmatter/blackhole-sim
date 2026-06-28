"""Adapters for external GRMHD dump formats.

The canonical renderer schema is :class:`blackhole_sim.grmhd.GRMHDSnapshot`.
This module turns common HDF5 layouts into that schema while retaining explicit
mapping choices. Real simulation codes use different coordinate maps and velocity
variables, so the adapter returns validation metadata and never silently claims
that a non-canonical primitive vector is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .grmhd import GRMHDSnapshot, SnapshotSchemaError, normalize_timelike_u


@dataclass(frozen=True)
class HARMPrimitiveMap:
    """Mapping for HARM/iharm-style primitive vector arrays.

    Default order follows the common [rho, internal energy, u1, u2, u3, B1, B2,
    B3] convention. If the file already contains u_con or b_con, prefer those
    over reconstructing from primitives.
    """

    rho: int = 0
    internal_energy: int = 1
    u1: int = 2
    u2: int = 3
    u3: int = 4
    b1: int = 5
    b2: int = 6
    b3: int = 7
    theta_e_field: str | None = None
    gamma_adiabatic: float = 13.0 / 9.0
    electron_temperature_floor: float = 1.0e-3


@dataclass(frozen=True)
class AdapterReport:
    adapter_name: str
    source: str
    field_map: dict[str, str]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptedSnapshot:
    snapshot: GRMHDSnapshot
    report: AdapterReport


def list_hdf5_datasets(path: str | Path) -> dict[str, tuple[int, ...]]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required to inspect HDF5 dumps") from exc
    out: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if hasattr(obj, "shape"):
                out[name] = tuple(int(x) for x in obj.shape)
        f.visititems(visit)
    return out


def _read_first(f: Any, names: Sequence[str]) -> tuple[str, np.ndarray]:
    for name in names:
        if name in f:
            return name, np.asarray(f[name])
    # Recursive fallback by final component.
    matches: list[str] = []
    def visit(n: str, obj: Any) -> None:
        if hasattr(obj, "shape") and n.split("/")[-1] in names:
            matches.append(n)
    f.visititems(visit)
    if matches:
        n = sorted(matches)[0]
        return n, np.asarray(f[n])
    raise SnapshotSchemaError(f"Could not find any of: {names}")


def _read_attr_float(f: Any, names: Sequence[str], default: float) -> float:
    for name in names:
        if name in f.attrs:
            return float(f.attrs[name])
    return float(default)


def _maybe_transpose_field(arr: np.ndarray, nr: int, nt: int, nphi: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (nr, nt, nphi):
        return arr
    if arr.shape == (nphi, nt, nr):
        return np.transpose(arr, (2, 1, 0))
    if arr.shape == (nt, nr, nphi):
        return np.transpose(arr, (1, 0, 2))
    raise SnapshotSchemaError(f"Cannot map field shape {arr.shape} to {(nr, nt, nphi)}")


def _maybe_transpose_vec(arr: np.ndarray, nr: int, nt: int, nphi: int, ncomp: int = 4) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (nr, nt, nphi, ncomp):
        return arr
    if arr.shape == (ncomp, nr, nt, nphi):
        return np.moveaxis(arr, 0, -1)
    if arr.shape == (nphi, nt, nr, ncomp):
        return np.transpose(arr, (2, 1, 0, 3))
    raise SnapshotSchemaError(f"Cannot map vector field shape {arr.shape} to {(nr, nt, nphi, ncomp)}")


def load_harm_hdf5(
    path: str | Path,
    primitive_map: HARMPrimitiveMap | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    spin_default: float = 0.0,
) -> AdaptedSnapshot:
    """Load a HARM/iharm/KORAL-like HDF5 dump into the canonical schema.

    Supported canonical paths include explicit ``r/theta/phi/rho/u_con/b_con``
    arrays, or a primitive vector dataset named ``prims``, ``P``, ``dump/prims``
    using :class:`HARMPrimitiveMap`.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required for HARM HDF5 adapters") from exc
    pmap = primitive_map or HARMPrimitiveMap()
    default_aliases = {
        "r": ("r", "R", "x1", "X1", "grid/r", "geom/r"),
        "theta": ("theta", "th", "x2", "X2", "grid/theta", "geom/th"),
        "phi": ("phi", "ph", "x3", "X3", "grid/phi", "geom/ph"),
        "rho": ("rho", "RHO", "fluid/rho"),
        "theta_e": ("theta_e", "Thetae", "Te", "fluid/theta_e"),
        "pressure": ("pressure", "press", "Pgas", "fluid/pressure"),
        "u_con": ("u_con", "ucon", "Ucon", "fluid/u_con"),
        "b_con": ("b_con", "bcon", "Bcon", "fluid/b_con"),
        "prims": ("prims", "P", "dump/prims", "fluid/prims"),
    }
    if aliases:
        for key, names in aliases.items():
            default_aliases[key] = tuple(names) + tuple(default_aliases.get(key, ()))

    field_map: dict[str, str] = {}
    warnings: list[str] = []
    with h5py.File(path, "r") as f:
        r_name, r = _read_first(f, default_aliases["r"])
        th_name, theta = _read_first(f, default_aliases["theta"])
        ph_name, phi = _read_first(f, default_aliases["phi"])
        field_map.update({"r": r_name, "theta": th_name, "phi": ph_name})
        r = np.asarray(r, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        phi = np.asarray(phi, dtype=float).reshape(-1)
        nr, nt, np_ = len(r), len(theta), len(phi)
        spin = _read_attr_float(f, ("spin_a", "a", "a/M", "bhspin"), spin_default)
        time_m = _read_attr_float(f, ("time_m", "t", "dump_time"), 0.0)

        try:
            rho_name, rho = _read_first(f, default_aliases["rho"])
            rho = _maybe_transpose_field(rho, nr, nt, np_)
            field_map["rho"] = rho_name
            try:
                te_name, theta_e = _read_first(f, default_aliases["theta_e"])
                theta_e = _maybe_transpose_field(theta_e, nr, nt, np_)
                field_map["theta_e"] = te_name
            except SnapshotSchemaError:
                pressure_name, pressure_tmp = _read_first(f, default_aliases["pressure"])
                pressure_tmp = _maybe_transpose_field(pressure_tmp, nr, nt, np_)
                theta_e = np.maximum(pressure_tmp / np.maximum(rho, 1.0e-300), pmap.electron_temperature_floor)
                field_map["theta_e"] = f"derived:{pressure_name}/rho"
        except SnapshotSchemaError:
            prim_name, prims = _read_first(f, default_aliases["prims"])
            field_map["prims"] = prim_name
            prims = np.asarray(prims, dtype=float)
            if prims.shape[:3] != (nr, nt, np_):
                if prims.shape[-3:] == (nr, nt, np_):
                    prims = np.moveaxis(prims, 0, -1)
                else:
                    raise SnapshotSchemaError(f"Cannot map primitive shape {prims.shape}")
            rho = np.maximum(prims[..., pmap.rho], 1.0e-300)
            uu = np.maximum(prims[..., pmap.internal_energy], 0.0)
            theta_e = np.maximum((pmap.gamma_adiabatic - 1.0) * uu / rho, pmap.electron_temperature_floor)
            field_map["rho"] = f"{prim_name}[{pmap.rho}]"
            field_map["theta_e"] = f"derived:({pmap.gamma_adiabatic}-1)*uu/rho"

        try:
            pressure_name, pressure = _read_first(f, default_aliases["pressure"])
            pressure = _maybe_transpose_field(pressure, nr, nt, np_)
            field_map["pressure"] = pressure_name
        except SnapshotSchemaError:
            pressure = rho * theta_e
            field_map["pressure"] = "derived:rho*theta_e"

        try:
            u_name, u_con = _read_first(f, default_aliases["u_con"])
            u_con = _maybe_transpose_vec(u_con, nr, nt, np_, 4)
            field_map["u_con"] = u_name
        except SnapshotSchemaError:
            prim_name, prims = _read_first(f, default_aliases["prims"])
            if prims.shape[:3] != (nr, nt, np_):
                prims = np.moveaxis(prims, 0, -1)
            u_con = np.zeros((nr, nt, np_, 4), dtype=float)
            for i, rr in enumerate(r):
                for j, th in enumerate(theta):
                    spatial = prims[i, j, :, [pmap.u1, pmap.u2, pmap.u3]]
                    for k in range(np_):
                        u_con[i, j, k] = normalize_timelike_u(float(rr), float(th), spin, spatial[k])
            field_map["u_con"] = f"derived:normalize(prims[{pmap.u1}:{pmap.u3}])"
            warnings.append("u_con reconstructed from primitive spatial components; verify the source code's velocity convention.")

        try:
            b_name, b_con = _read_first(f, default_aliases["b_con"])
            b_con = _maybe_transpose_vec(b_con, nr, nt, np_, 4)
            field_map["b_con"] = b_name
        except SnapshotSchemaError:
            prim_name, prims = _read_first(f, default_aliases["prims"])
            if prims.shape[:3] != (nr, nt, np_):
                prims = np.moveaxis(prims, 0, -1)
            b_con = np.zeros((nr, nt, np_, 4), dtype=float)
            b_con[..., 1] = prims[..., pmap.b1]
            b_con[..., 2] = prims[..., pmap.b2]
            b_con[..., 3] = prims[..., pmap.b3]
            field_map["b_con"] = f"derived:prims[{pmap.b1},{pmap.b2},{pmap.b3}] with b^t=0"
            warnings.append("b_con reconstructed with b^t=0; prefer a dump-provided comoving magnetic four-vector when available.")

    snap = GRMHDSnapshot(
        r=r,
        theta=theta,
        phi=phi,
        rho=rho,
        theta_e=theta_e,
        pressure=pressure,
        b_con=b_con,
        u_con=u_con,
        spin_a=spin,
        time_m=time_m,
        metadata={"adapter": "harm_hdf5", "source": str(path), "field_map": field_map, "warnings": tuple(warnings)},
    )
    return AdaptedSnapshot(
        snap,
        AdapterReport("harm_hdf5", str(path), field_map=field_map, warnings=tuple(warnings), metadata={"time_m": time_m}),
    )


def load_bhac_hdf5(path: str | Path, **kwargs: Any) -> AdaptedSnapshot:
    """Load a BHAC HDF5 dump when it exposes canonical grid/primitive arrays.

    BHAC deployments can write multiple HDF5 layouts. This adapter handles the
    common case where the dump contains cell-centered spherical coordinates plus
    primitive variables; callers can pass ``aliases=`` to match local naming.
    """
    aliases = kwargs.pop("aliases", None) or {}
    merged = {
        "r": ("grid/r", "coordinates/r", "r", "x1"),
        "theta": ("grid/theta", "coordinates/theta", "theta", "x2"),
        "phi": ("grid/phi", "coordinates/phi", "phi", "x3"),
        "prims": ("prims", "primitive", "w", "data/prims"),
    }
    for k, v in aliases.items():
        merged[k] = tuple(v) + tuple(merged.get(k, ()))
    adapted = load_harm_hdf5(path, aliases=merged, **kwargs)
    return AdaptedSnapshot(
        adapted.snapshot,
        AdapterReport("bhac_hdf5", str(path), adapted.report.field_map, adapted.report.warnings, adapted.report.metadata),
    )


def load_koral_hdf5(path: str | Path, **kwargs: Any) -> AdaptedSnapshot:
    """Load KORAL-like HDF5 output when exported with HARM-compatible fields."""
    aliases = kwargs.pop("aliases", None) or {}
    merged = {
        "r": ("r", "geom/r", "grid/r", "x1"),
        "theta": ("th", "theta", "geom/th", "grid/theta", "x2"),
        "phi": ("ph", "phi", "geom/ph", "grid/phi", "x3"),
        "prims": ("prims", "P", "dump/prims"),
    }
    for k, v in aliases.items():
        merged[k] = tuple(v) + tuple(merged.get(k, ()))
    adapted = load_harm_hdf5(path, aliases=merged, **kwargs)
    return AdaptedSnapshot(
        adapted.snapshot,
        AdapterReport("koral_hdf5", str(path), adapted.report.field_map, adapted.report.warnings, adapted.report.metadata),
    )

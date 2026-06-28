"""CLI for GPU coefficient brick precomputation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import PhysicalScaling
from .coefficient_bricks import estimate_brick_memory, precompute_coefficient_bricks
from .grmhd import GRMHDSnapshot, generate_analytic_grmhd_torus, load_grmhd_hdf5
from .physics import BlackHoleSystem


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute Stokes transfer coefficient bricks for GPU renderers")
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--spin", type=float, default=0.85)
    p.add_argument("--nu", type=float, default=230e9)
    p.add_argument("--mass", choices=["m87", "sgr-a"], default="m87")
    p.add_argument("--mdot-msun-year", type=float, default=1e-4)
    p.add_argument("--rho-scale", type=float, default=None)
    p.add_argument("--b-scale", type=float, default=None)
    p.add_argument("--dtype", choices=["float32", "float16", "float64"], default="float32")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--format", choices=["npz", "hdf5"], default="npz")
    p.add_argument("--output", type=Path, default=Path("out/coefficient_bricks.npz"))
    p.add_argument("--estimate-only", action="store_true")
    return p


def _load_snapshot(path: Path | None, spin: float) -> GRMHDSnapshot:
    if path is None:
        return generate_analytic_grmhd_torus(spin_a=spin, nr=32, ntheta=20, nphi=24)
    if path.suffix.lower() == ".npz":
        return GRMHDSnapshot.from_npz(path)
    return load_grmhd_hdf5(path)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    snap = _load_snapshot(args.snapshot, args.spin)
    nr = (snap.r.size + args.stride - 1) // args.stride
    nt = (snap.theta.size + args.stride - 1) // args.stride
    np_ = (snap.phi.size + args.stride - 1) // args.stride
    mem = estimate_brick_memory(nr, nt, np_, args.dtype)
    if args.estimate_only:
        print(f"grid={nr}x{nt}x{np_} coeffs={mem['total_mib']:.3f} MiB dtype={args.dtype}")
        return
    system = BlackHoleSystem.m87_star(snap.spin_a) if args.mass == "m87" else BlackHoleSystem.sgr_a_star(snap.spin_a)
    if args.rho_scale is not None and args.b_scale is not None:
        scaling = PhysicalScaling.from_black_hole_system(system, args.rho_scale, args.b_scale)
    else:
        scaling = PhysicalScaling.from_mdot(system, args.mdot_msun_year)
    bricks = precompute_coefficient_bricks(snap, scaling, args.nu, args.dtype, stride=args.stride)
    if args.format == "hdf5" or args.output.suffix.lower() in {".h5", ".hdf5"}:
        bricks.save_hdf5(args.output)
    else:
        bricks.save_npz(args.output)
    print(f"wrote {args.output}; shape={bricks.shape}; bytes={bricks.bytes}; dtype={args.dtype}")


if __name__ == "__main__":
    main()
